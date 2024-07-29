import asyncio
import logging
from datetime import datetime
from multiprocessing import Queue

import numpy as np
import torch
from server.config import Settings
from singleton import Singleton

from .llm import LLMService
from .schemas import ClientAudio
from .trt import TranscriptionService
from .vad import VoiceActivityDetection


def run_vad(
    clients: dict[str, ClientAudio], source_queue: Queue, result_queue: Queue
) -> None:
    Settings.config_logger()
    vad = VoiceActivityDetection()

    while True:
        new_task: tuple[str, np.ndarray, datetime] = source_queue.get()
        client_id, frame_np, _ = new_task
        if client_id is None or client_id == "terminate":
            break

        # client_audio: ClientAudio = clients.get(client_id)
        # if client_audio is None:
        #     continue

        # logging.info(f"VAD for {client_id}")

        # frame_np: np.ndarray = np.frombuffer(client_audio.data, dtype=np.float32)
        # frame_np: np.ndarray = client_audio.data

        try:
            if Settings.sample_rate / frame_np.shape[0] > 31.25:
                speech_prob = 0.0
            else:
                speech_prob = vad(
                    torch.from_numpy(frame_np.copy()), Settings.sample_rate
                ).item()
        except Exception as e:
            logging.error(f"Error in VAD: {e}")
            speech_prob = 0.0

        result_queue.put((client_id, speech_prob))
        # logging.info(f"VAD completed for {client_id}, {speech_prob}")


def run_transcription(
    clients: dict[str, ClientAudio], source_queue: Queue, result_queue: Queue
) -> None:
    Settings.config_logger()

    while True:
        new_task: tuple[str, np.ndarray, datetime] = source_queue.get()
        client_id, frame_np, at = new_task
        if client_id is None or client_id == "terminate":
            break

        client_audio: ClientAudio = clients.get(client_id)
        if client_audio is None:
            logging.info(f"Client {client_id} not found")
            continue

        if at and at < client_audio.transcribed_at:
            logging.info(f"Skipping transcription for {client_id} for {at}")
            continue
        client_audio.transcribed_at = datetime.now()

        frame_np: np.ndarray = client_audio.segments_data[-1]
        logging.debug(f"Transcription for {client_id} {frame_np.shape}")

        try:
            text = TranscriptionService().transcribe(frame_np)
        except Exception as e:
            logging.error(f"Error in transcription: {e}")
            text = ""

        client_audio.text = text
        client_audio.segments.append(text)
        clients[client_id] = client_audio
        result_queue.put((client_id, text))
        logging.debug(f"Transcription completed for {client_id}: {text}")


def run_llm(
    clients: dict[str, ClientAudio], source_queue: Queue, result_queue: Queue
) -> None:
    Settings.config_logger()

    while True:
        new_task: tuple[str, str, datetime] = source_queue.get()
        client_id, text, at = new_task
        if client_id is None or client_id == "terminate":
            break

        client_audio: ClientAudio = clients.get(client_id)
        if client_audio is None:
            logging.info(f"Client {client_id} not found")
            continue

        if at and at < client_audio.summerized_at:
            logging.info(f"Skipping LLM for {client_id} for {at}")
            continue
        client_audio.summerized_at = datetime.now()

        logging.debug(f"LLM for {client_id}")

        text: str = "\n".join(client_audio.segments)
        try:
            summary = LLMService().summerize(text)
        except Exception as e:
            logging.error(f"Error in LLM: {e}")
            summary = ""

        client_audio.summary = summary
        clients[client_id] = client_audio
        result_queue.put((client_id, summary))
        logging.debug(f"LLM completed for {client_id}: {summary}")


class VoiceReceiverService(metaclass=Singleton):
    def __init__(self) -> None:
        self.clients_audios: dict[str, ClientAudio] = Settings().clients
        self.vad_queue: Queue = Settings().vad_queues["source"]
        self.vad_result_queue: Queue = Settings().vad_queues["result"]
        self.transcription_queue: Queue = Settings().transcription_queues["source"]
        self.transcription_queue_result: Queue = Settings().transcription_queues[
            "result"
        ]
        self.llm_queue: Queue = Settings().llm_queues["source"]
        self.llm_queue_result: Queue = Settings().llm_queues["result"]

        self.transcription_sent_queue: asyncio.Queue = (
            Settings().transcription_sent_queue
        )
        self.summary_sent_queue: asyncio.Queue = Settings().summary_sent_queue

    def save_audio(self, client_audio: ClientAudio) -> None:
        # save audio to disk
        import wave

        data: np.ndarray = np.concatenate(client_audio.segments_data, axis=0)
        logging.debug(f"save audio data shape {data.shape}")

        try:
            int_frames: np.ndarray = (data * 32767).astype(np.int16)
            with wave.open(f"logs/{client_audio.client_id}.wav", "wb") as wav_file:
                num_channels = 1
                sample_width = 2  # 16 bits = 2 bytes
                frame_rate = Settings.sample_rate

                wav_file.setnchannels(num_channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(frame_rate)
                wav_file.writeframes(int_frames.tobytes())
        except Exception as e:
            logging.error(f"Error saving audio: {e}")

    async def add_audio_frame(self, client_id: str, frame_np: np.ndarray) -> None:
        if self.clients_audios.get(client_id) is None:
            client_audio = ClientAudio(client_id=client_id, data=frame_np)
        else:
            client_audio = self.clients_audios[client_id]

        client_audio.eos = False
        client_audio.no_voice_activity_chunks = 0

        client_audio.add_frames(frame_np)
        self.clients_audios[client_id] = client_audio

    async def process_audio_frame(self, data: bytes, client_id: str) -> None:
        frame_np = np.frombuffer(data, dtype=np.float32)
        self.vad_queue.put((client_id, frame_np, datetime.now()))
        asyncio.create_task(self.handle_vad_result(frame_np))
        # await self.handle_vad_result(frame_np)

    async def handle_vad_result(self, frame_np: np.ndarray) -> None:
        client_id, speech_prob = self.vad_result_queue.get()

        if speech_prob > Settings.vad_threshold:
            await self.add_audio_frame(client_id, frame_np)
        else:
            await self.no_voice_activity(client_id)

    async def no_voice_activity(
        self, client_id: str, max_silnet_chunks: int = 3
    ) -> None:
        if self.clients_audios.get(client_id) is None:
            logging.debug(f"no_voice_activity for {client_id}, no audio data")
            return

        client_audio = self.clients_audios[client_id]
        client_audio.no_voice_activity_chunks += 1

        if client_audio.no_voice_activity_chunks < max_silnet_chunks:
            logging.debug(
                f"no_voice_activity for {client_id}, {client_audio.no_voice_activity_chunks}"
            )
            return

        if client_audio.eos:
            return

        logging.debug(f"no_voice_activity start process for {client_id}, {max_silnet_chunks}")
        client_audio.eos = True
        client_audio.segments_data.append(client_audio.data)
        client_audio.data = None
        self.clients_audios[client_id] = client_audio

        # self.save_audio(client_audio)
        self.transcription_queue.put((client_id, client_audio.segments_data[-1], datetime.now()))

        asyncio.create_task(self.handle_transcription_result(client_id))

    async def handle_transcription_result(self, client_id: str) -> None:
        client_id, text = self.transcription_queue_result.get()
        self.llm_queue.put((client_id, text, datetime.now()))
        await self.transcription_sent_queue.put((client_id, text, datetime.now()))

        asyncio.create_task(self.handle_llm_result(client_id))

    async def handle_llm_result(self, client_id: str) -> None:
        client_id, summary = self.llm_queue_result.get()
        await self.summary_sent_queue.put((client_id, summary, datetime.now()))
