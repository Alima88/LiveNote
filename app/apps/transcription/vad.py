# original: https://github.com/snakers4/silero-vad/blob/v4.0/utils_vad.py
import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime
import requests
import torch
from server.config import Settings
from singleton import Singleton
from torch import Tensor


class VoiceActivityDetection(metaclass=Singleton):
    def __init__(self, force_onnx_cpu: bool = True) -> None:
        logging.info("downloading ONNX model...")
        path: Path = self.download()
        logging.info("loading session")

        opts: onnxruntime.SessionOptions = onnxruntime.SessionOptions()
        opts.log_severity_level = 3

        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        logging.info("loading onnx model")
        if (
            force_onnx_cpu
            and "CPUExecutionProvider" in onnxruntime.get_available_providers()
        ):
            self.session: onnxruntime.InferenceSession = onnxruntime.InferenceSession(
                str(path), providers=["CPUExecutionProvider"], sess_options=opts
            )
        else:
            self.session = onnxruntime.InferenceSession(
                str(path), providers=["CUDAExecutionProvider"], sess_options=opts
            )

        logging.info("reset states")
        self.reset_states()
        self.sample_rates: list[int] = [8000, 16000]

    def _validate_input(self, x: Tensor, sr: int) -> Tuple[Tensor, int]:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        if sr != 16000 and (sr % 16000 == 0):
            step: int = sr // 16000
            x = x[:, ::step]
            sr = 16000

        if sr not in self.sample_rates:
            raise ValueError(
                f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)"
            )

        if sr / x.shape[1] > 31.25:  # 512 samples
            raise ValueError("Input audio chunk is too short")

        return x, sr

    def reset_states(self, batch_size: int = 1) -> None:
        self._h: np.ndarray = np.zeros((2, batch_size, 64)).astype("float32")
        self._c: np.ndarray = np.zeros((2, batch_size, 64)).astype("float32")
        self._last_sr: int = 0
        self._last_batch_size: int = 0

    def __call__(self, x: Tensor, sr: int) -> Tensor:
        x, sr = self._validate_input(x, sr)
        batch_size: int = x.shape[0]

        if not self._last_batch_size:
            self.reset_states(batch_size)
        if (self._last_sr) and (self._last_sr != sr):
            self.reset_states(batch_size)
        if (self._last_batch_size) and (self._last_batch_size != batch_size):
            self.reset_states(batch_size)

        if sr in [8000, 16000]:
            ort_inputs: dict[str, np.ndarray] = {
                "input": x.numpy(),
                "h": self._h,
                "c": self._c,
                "sr": np.array(sr, dtype="int64"),
            }
            ort_outs: list[np.ndarray] = self.session.run(None, ort_inputs)
            out: np.ndarray
            out, self._h, self._c = ort_outs
        else:
            raise ValueError()

        self._last_sr = sr
        self._last_batch_size = batch_size

        out_tensor: Tensor = torch.tensor(out)
        return out_tensor

    def audio_forward(self, x: Tensor, sr: int, num_samples: int = 512) -> Tensor:
        outs: list[Tensor] = []
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples:
            pad_num: int = num_samples - (x.shape[1] % num_samples)
            x = torch.nn.functional.pad(x, (0, pad_num), "constant", value=0.0)

        self.reset_states(x.shape[0])
        for i in range(0, x.shape[1], num_samples):
            wavs_batch: Tensor = x[:, i : i + num_samples]
            out_chunk: Tensor = self.__call__(wavs_batch, sr)
            outs.append(out_chunk)

        stacked: Tensor = torch.cat(outs, dim=1)
        return stacked.cpu()

    @staticmethod
    def download(
        model_url: str = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx",
    ) -> Path:
        target_dir: Path = Settings.base_dir / "ml" / "whisper-live"

        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)

        # Define the target file path
        model_filename: Path = target_dir / "silero_vad.onnx"

        # Check if the model file already exists
        if not os.path.exists(model_filename):
            logging.info("Downloading VAD ONNX model...")
            try:
                with open(model_filename, "wb") as f:
                    f.write(requests.get(model_url).content)
            except Exception:
                logging.error("Failed to download the VAD model using requests.")
        return model_filename
