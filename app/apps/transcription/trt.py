import json
import logging
import os
import re
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile
import tensorrt_llm
import tensorrt_llm.logger as logger
import torch
import torch.nn.functional as F
from singleton import Singleton
from tensorrt_llm._utils import str_dtype_to_torch, str_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from tensorrt_llm.runtime.session import Session, TensorInfo
from whisper.tokenizer import get_tokenizer

from server.config import Settings

from .schemas import ClientAudio

SAMPLE_RATE = Settings.sample_rate
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def pad_or_trim(
    array: Union[np.ndarray, torch.Tensor], length: int = N_SAMPLES, *, axis: int = -1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac",
        "1", "-acodec", "pcm_s16le", "-ar",
        str(sr), "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def load_audio_wav_format(wav_path: str) -> Tuple[np.ndarray, int]:
    # make sure audio in .wav format
    assert wav_path.endswith(".wav"), f"Only support .wav format, but got {wav_path}"
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return waveform, sample_rate


@lru_cache(maxsize=None)
def mel_filters(
    device: str, n_mels: int, mel_filters_dir: Optional[str] = None
) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if mel_filters_dir is None:
        mel_filters_path = os.path.join(Settings.base_dir, "assets", "mel_filters.npz")
    else:
        mel_filters_path = os.path.join(mel_filters_dir, "mel_filters.npz")
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


class WhisperEncoding:
    def __init__(self, engine_dir: Path = Path(Settings.whisper_tensorrt_path)):
        self.session = self.get_session(engine_dir)
        self.dtype: str
        self.n_mels: int
        self.num_languages: int

    def get_session(self, engine_dir: Path) -> Session:
        config_path = engine_dir / "encoder_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        config["plugin_config"]["gpt_attention_plugin"]
        dtype = config["builder_config"]["precision"]
        n_mels = config["builder_config"]["n_mels"]
        num_languages = config["builder_config"]["num_languages"]

        self.dtype = dtype
        self.n_mels = n_mels
        self.num_languages = num_languages

        serialize_path = engine_dir / f"whisper_encoder_{self.dtype}_tp1_rank0.engine"

        with open(serialize_path, "rb") as f:
            session = Session.from_serialized_engine(f.read())

        return session

    def get_audio_features(self, mel: torch.Tensor) -> torch.Tensor:
        input_lengths = torch.tensor(
            [mel.shape[2] // 2 for _ in range(mel.shape[0])],
            dtype=torch.int32,
            device=mel.device,
        )

        inputs = OrderedDict()
        inputs["x"] = mel
        inputs["input_lengths"] = input_lengths

        output_list = [
            TensorInfo("x", str_dtype_to_trt(self.dtype), mel.shape),
            TensorInfo("input_lengths", str_dtype_to_trt("int32"), input_lengths.shape),
        ]

        output_info = self.session.infer_shapes(output_list)

        logger.debug(f"output info {output_info}")
        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )
            for t in output_info
        }
        logging.info(f'get_audio_features outputs: {outputs["output"].shape}')
        stream = torch.cuda.current_stream()
        logging.info(f"get_audio_features after stream: {stream}")
        ok = self.session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
        assert ok, "Engine execution failed"
        logging.info(f"get_audio_features after run: {ok}")
        stream.synchronize()
        audio_features = outputs["output"]
        return audio_features


class WhisperDecoding:
    def __init__(
        self,
        engine_dir: Path,
        runtime_mapping: tensorrt_llm.Mapping,
        debug_mode: bool = False,
    ):
        self.decoder_config = self.get_config(engine_dir)
        self.decoder_generation_session = self.get_session(
            engine_dir, runtime_mapping, debug_mode
        )

    def get_config(self, engine_dir: Path) -> OrderedDict:
        config_path = engine_dir / "decoder_config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        decoder_config = OrderedDict()
        decoder_config.update(config["plugin_config"])
        decoder_config.update(config["builder_config"])
        return decoder_config

    def get_session(
        self,
        engine_dir: Path,
        runtime_mapping: tensorrt_llm.Mapping,
        debug_mode: bool = False,
    ) -> tensorrt_llm.runtime.GenerationSession:
        dtype = self.decoder_config["precision"]
        serialize_path = engine_dir / f"whisper_decoder_{dtype}_tp1_rank0.engine"
        with open(serialize_path, "rb") as f:
            decoder_engine_buffer = f.read()

        decoder_model_config = ModelConfig(
            max_batch_size=self.decoder_config["max_batch_size"],
            max_beam_width=self.decoder_config["max_beam_width"],
            num_heads=self.decoder_config["num_heads"],
            num_kv_heads=self.decoder_config["num_heads"],
            hidden_size=self.decoder_config["hidden_size"],
            vocab_size=self.decoder_config["vocab_size"],
            num_layers=self.decoder_config["num_layers"],
            gpt_attention_plugin=self.decoder_config["gpt_attention_plugin"],
            remove_input_padding=self.decoder_config["remove_input_padding"],
            cross_attention=self.decoder_config["cross_attention"],
            has_position_embedding=self.decoder_config["has_position_embedding"],
            has_token_type_embedding=self.decoder_config["has_token_type_embedding"],
        )
        decoder_generation_session = tensorrt_llm.runtime.GenerationSession(
            decoder_model_config,
            decoder_engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode,
        )

        return decoder_generation_session

    def generate(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_outputs: torch.Tensor,
        eot_id: int,
        max_new_tokens: int = 40,
        num_beams: int = 1,
    ) -> List[List[int]]:
        encoder_input_lengths = torch.tensor(
            [encoder_outputs.shape[1] for x in range(encoder_outputs.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )

        decoder_input_lengths = torch.tensor(
            [decoder_input_ids.shape[-1] for _ in range(decoder_input_ids.shape[0])],
            dtype=torch.int32,
            device="cuda",
        )
        decoder_max_input_length = torch.max(decoder_input_lengths).item()
        cross_attention_mask = (
            torch.ones([encoder_outputs.shape[0], 1, encoder_outputs.shape[1]])
            .int()
            .cuda()
        )
        # generation config
        sampling_config = SamplingConfig(
            end_id=eot_id, pad_id=eot_id, num_beams=num_beams
        )
        self.decoder_generation_session.setup(
            decoder_input_lengths.size(0),
            decoder_max_input_length,
            max_new_tokens,
            beam_width=num_beams,
            encoder_max_input_length=encoder_outputs.shape[1],
        )

        torch.cuda.synchronize()

        decoder_input_ids = decoder_input_ids.type(torch.int32).cuda()
        output_ids = self.decoder_generation_session.decode(
            decoder_input_ids,
            decoder_input_lengths,
            sampling_config,
            encoder_output=encoder_outputs,
            encoder_input_lengths=encoder_input_lengths,
            cross_attention_mask=cross_attention_mask,
        )
        torch.cuda.synchronize()

        # get the list of int from output_ids tensor
        output_ids = output_ids.cpu().numpy().tolist()
        return output_ids


class WhisperTRTLLM:
    def __init__(
        self,
        engine_dir: Path = Settings.whisper_tensorrt_path,
        debug_mode: bool = False,
        assets_dir: Optional[Path] = Settings.base_dir / "assets",
        device: Optional[str] = "cuda",
    ):
        world_size = 1
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)
        engine_dir = Path(engine_dir)

        self.encoder = WhisperEncoding(engine_dir)
        self.decoder = WhisperDecoding(
            engine_dir, runtime_mapping, debug_mode=debug_mode
        )
        self.n_mels = self.encoder.n_mels
        self.tokenizer = get_tokenizer(
            False,
            # num_languages=self.encoder.num_languages,
            language="en",
            task="transcribe",
        )
        self.device = device
        self.filters = mel_filters(
            self.device, self.encoder.n_mels, str(assets_dir) if assets_dir else None
        )

    def log_mel_spectrogram(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        padding: int = 0,
        return_duration: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Compute the log-Mel spectrogram of

        Parameters
        ----------
        audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
            The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

        n_mels: int
            The number of Mel-frequency filters, only 80 and 128 are supported

        padding: int
            Number of zero samples to pad to the right

        device: Optional[Union[str, torch.device]]
            If given, the audio tensor is moved to this device before STFT

        Returns
        -------
        torch.Tensor, shape = (80 or 128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        if not torch.is_tensor(audio):
            if isinstance(audio, str):
                if audio.endswith(".wav"):
                    audio, _ = load_audio_wav_format(audio)
                else:
                    audio = load_audio(audio)
            assert isinstance(
                audio, np.ndarray
            ), f"Unsupported audio type: {type(audio)}"
            duration = audio.shape[-1] / SAMPLE_RATE
            audio = pad_or_trim(audio, N_SAMPLES)
            audio = audio.astype(np.float32)
            audio = torch.from_numpy(audio)

        if self.device is not None:
            audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        window = torch.hann_window(N_FFT).to(audio.device)
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        mel_spec = self.filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        if return_duration:
            return log_spec, duration
        else:
            return log_spec

    def process_batch(
        self,
        mel: torch.Tensor,
        text_prefix: str = "",
        num_beams: int = 1,
    ) -> List[str]:
        prompt_id = self.tokenizer.encode(
            text_prefix, allowed_special=set(self.tokenizer.special_tokens.keys())
        )

        prompt_id = torch.tensor(prompt_id)
        batch_size = mel.shape[0]
        decoder_input_ids = prompt_id.repeat(batch_size, 1)

        logging.info(f"process_batch: {mel.shape}")
        encoder_output = self.encoder.get_audio_features(mel)
        logging.info(f"encoder_output: {encoder_output.shape}")
        output_ids = self.decoder.generate(
            decoder_input_ids,
            encoder_output,
            self.tokenizer.eot,
            max_new_tokens=96,
            num_beams=num_beams,
        )
        texts = []
        for i in range(len(output_ids)):
            text = self.tokenizer.decode(output_ids[i][0]).strip()
            texts.append(text)
        return texts

    def transcribe(
        self,
        mel: torch.Tensor,
        text_prefix: str = "",
        dtype: str = "float16",
        batch_size: int = 1,
        num_beams: int = 1,
    ) -> str:
        mel = mel.type(str_dtype_to_torch(dtype))
        mel = mel.unsqueeze(0)
        predictions = self.process_batch(mel, text_prefix, num_beams)
        prediction = predictions[0]

        # remove all special tokens in the prediction
        prediction = re.sub(r"<\|.*?\|>", "", prediction)
        return prediction.strip()


class TranscriptionService(metaclass=Singleton):
    def __init__(self):
        self.whisper = WhisperTRTLLM()

    def transcribe(self, client_audio: ClientAudio) -> str:
        audio = client_audio.last_30_seconds
        mel, _ = self.whisper.log_mel_spectrogram(audio.copy())
        transcription = self.whisper.transcribe(mel)
        logging.info(f"Transcription: {transcription}")
        return transcription


if __name__ == "__main__":
    whisper = WhisperTRTLLM()
    mel, _ = whisper.log_mel_spectrogram("assets/audio.wav")
    print(mel.shape)

    transcription = whisper.transcribe(mel)
    print(transcription)
