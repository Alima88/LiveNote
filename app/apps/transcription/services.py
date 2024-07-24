import asyncio
import atexit
import logging
from multiprocessing import Process, Queue

import numpy as np
import torch
from singleton import Singleton

from server.config import Settings

from .schemas import ClientAudio
from .trt import TranscriptionService
from .vad import VoiceActivityDetection


class VoiceReceiverService(metaclass=Singleton):
    def __init__(self) -> None:
        self.clients_audios: dict[str, ClientAudio] = {}
        self.result_queue = Queue()
        self.vad_queue = Queue()

        self.vad_process = Process(
            target=self.run_vad, args=(self.vad_queue, self.result_queue)
        )
        self.vad_process.start()
        atexit.register(self.shutdown)

    def run_vad(self, vad_queue: Queue, result_queue: Queue) -> None:
        while True:
            new_task: tuple[str, np.ndarray] = vad_queue.get()
            client_id, frame_np = new_task
            if client_id is None:
                break

            try:
                if Settings.sample_rate / frame_np.shape[0] > 31.25:
                    speech_prob = 0.0
                else:
                    speech_prob = VoiceActivityDetection()(
                        torch.from_numpy(frame_np.copy()), Settings.sample_rate
                    ).item()
            except Exception as e:
                logging.error(f"Error in VAD: {e}")
                speech_prob = 0.0

            result_queue.put((client_id, speech_prob))

    def add_audio_frame(self, client_id: str, frame_np: np.ndarray) -> None:
        if self.clients_audios.get(client_id) is None:
            self.clients_audios[client_id] = ClientAudio(
                client_id=client_id, data=frame_np
            )
            return

        client_audio = self.clients_audios[client_id]
        client_audio.no_voice_activity_chunks = 0
        client_audio.add_frames(frame_np)

    def no_voice_activity(self, client_id: str, max_silnet_chunks: int = 3) -> None:
        if self.clients_audios.get(client_id) is None:
            return

        client_audio = self.clients_audios[client_id]
        client_audio.no_voice_activity_chunks += 1

        if client_audio.no_voice_activity_chunks < max_silnet_chunks:
            return

        logging.info(f"no_voice_activity for {client_id}, {max_silnet_chunks}")

        client_audio.eos = True
        self.save_audio(client_audio)
        # return

        # send to process
        try:
            text = TranscriptionService().transcribe(client_audio.data)
        except Exception as e:
            logging.error(f"Error in transcription: {e}")
            text = ""

        logging.info(f"Transcription for {client_id}: {text}")
        # self.transcribe(client_audio)

    def save_audio(self, client_audio: ClientAudio) -> None:
        # save audio to disk
        import wave

        data: np.ndarray = client_audio.data

        int_frames: np.ndarray = (data * 32767).astype(np.int16)
        with wave.open(f"logs/{client_audio.client_id}.wav", "wb") as wav_file:
            num_channels = 1
            sample_width = 2  # 16 bits = 2 bytes
            frame_rate = Settings.sample_rate

            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(frame_rate)
            wav_file.writeframes(int_frames.tobytes())

    async def process_audio_frame(self, data: bytes, client_id: str) -> None:
        frame_np = np.frombuffer(data, dtype=np.float32)
        self.vad_queue.put((client_id, frame_np))
        asyncio.create_task(self.handle_vad_result(frame_np))
        # await self.handle_vad_result(frame_np)

    async def handle_vad_result(self, frame_np: np.ndarray) -> None:
        # while not self.result_queue.empty():
        client_id, speech_prob = self.result_queue.get()

        if speech_prob > Settings.vad_threshold:
            self.add_audio_frame(client_id, frame_np)
        else:
            self.no_voice_activity(client_id)

    def shutdown(self) -> None:
        self.vad_queue.put((None, None))
        self.vad_process.join()
