import asyncio
import logging
import json
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import websockets
from pydub import AudioSegment

start_time = 0


async def print_message(ws: websockets.WebSocketClientProtocol):
    try:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            logging.info(
                "\n".join(
                    [
                        "-" * 50,
                        "",
                        f"{time.time() - start_time:.2f}s",
                        "",
                        f"{data['type']}:",
                        f"{data['data']}",
                        "",
                        "-" * 50,
                    ]
                )
            )
    except asyncio.CancelledError:
        pass


def save_wav(data: np.ndarray, sample_rate: int = 16000, num_channels: int = 1):
    int_frames: np.ndarray = (data * 32767).astype(np.int16)
    with wave.open(f"logs/save.wav", "wb") as wav_file:
        sample_width = 2  # 16 bits = 2 bytes

        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int_frames.tobytes())


def mp4_to_numpy_array(file_path):
    # Load the audio from the MP4 file
    audio = AudioSegment.from_file(file_path, format="mp4")

    # Get the raw audio data as an array of bytes
    raw_data = audio.raw_data

    # Convert raw audio data to numpy array
    audio_array = np.frombuffer(raw_data, dtype=np.int16)
    num_channels = audio.channels

    logging.info(
        ", ".join(
            [
                f"Audio array shape: {audio_array.shape}",
                f"{audio.frame_rate}",
                f"{num_channels}",
                f"{audio_array.shape[0] / audio.frame_rate}",
            ]
        )
    )

    # Reshape the array to match the number of channels
    audio_array = np.reshape(audio_array / audio_array.max(), (-1, audio.channels))

    save_wav(audio_array, audio.frame_rate)

    return audio_array


async def send_audio(wav_path: str, ws_uri: str):
    global start_time
    if isinstance(wav_path, Path):
        wav_path = str(wav_path)

    # Convert to wav if necessary
    if not wav_path.endswith(".wav"):
        wav_path = mp4_to_numpy_array(wav_path)

    async with websockets.connect(ws_uri) as websocket:
        recv_task = asyncio.create_task(print_message(websocket))
        with wave.open(wav_path, "rb") as wav_file:
            # Get basic information about the .wav file
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()

            logging.info(
                f"Sample rate: {sample_rate}, Number of frames: {num_frames}, Sample width: {sample_width}, "
            )

            # Read and send frames in chunks
            start_time = time.time()
            chunk_size = 1024
            for _ in range(0, num_frames, chunk_size):
                frames = wav_file.readframes(chunk_size)

                # Convert frames to numpy array
                if sample_width == 1:  # 8-bit audio
                    dtype = np.uint8
                elif sample_width == 2:  # 16-bit audio
                    dtype = np.int16
                elif sample_width == 4:  # 32-bit audio
                    dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")

                frame_np = np.frombuffer(frames, dtype=dtype).astype(np.float32)

                # Normalize the audio data
                frame_np = frame_np / np.max(np.abs(frame_np))

                # Send frames to WebSocket server
                await websocket.send(frame_np.tobytes())
                logging.info(
                    f"Sent {_/sample_rate:.2f}, {len(frame_np.tobytes())} bytes"
                )
                await asyncio.sleep(chunk_size / sample_rate)

        await asyncio.sleep(5)
        recv_task.cancel()
        await websocket.send(b"close()")


async def save_audio(wav_path: str, ws_uri: str):
    global start_time
    if isinstance(wav_path, Path):
        wav_path = str(wav_path)

    async with websockets.connect(ws_uri) as websocket:
        recv_task = asyncio.create_task(print_message(websocket))
        with wave.open(wav_path, "rb") as wav_file:
            # Get basic information about the .wav file
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()

            logging.info(
                f"Sample rate: {sample_rate}, Number of frames: {num_frames}, Sample width: {sample_width}, "
            )

            # Read and send frames in chunks
            start_time = time.time()
            chunk_size = 1024
            for _ in range(0, num_frames, chunk_size):
                frames = wav_file.readframes(chunk_size)

                # Convert frames to numpy array
                if sample_width == 1:  # 8-bit audio
                    dtype = np.uint8
                elif sample_width == 2:  # 16-bit audio
                    dtype = np.int16
                elif sample_width == 4:  # 32-bit audio
                    dtype = np.int32
                else:
                    raise ValueError(f"Unsupported sample width: {sample_width}")

                frame_np = np.frombuffer(frames, dtype=dtype).astype(np.float32)

                # Normalize the audio data
                frame_np = frame_np / np.max(np.abs(frame_np))

                # Send frames to WebSocket server
                await websocket.send(frame_np.tobytes())
                logging.info(
                    f"Sent {_/sample_rate:.2f}, {len(frame_np.tobytes())} bytes"
                )
                await asyncio.sleep(chunk_size / sample_rate)

        await asyncio.sleep(5)
        recv_task.cancel()
        await websocket.send(b"close()")


async def main():
    client_id = uuid.UUID("11112222-3333-4444-5555-666677778888")  # uuid.uuid4()
    logging.info(f"Client ID: {client_id}")
    s_url = "s://wln.inbeet.tech"

    ws_uri = f"ws{s_url}/ws/{client_id}"
    # ws_uri = f"ws://localhost:8000/transcription/ws/{client_id}"

    audio_file_path = Path(__file__).parent.parent / "assets" / "audios" / "Georgia.mp4"
    mp4_to_numpy_array(audio_file_path)
    # await save_audio(audio_file_path, ws_uri)
    # await send_audio(audio_file_path, ws_uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
