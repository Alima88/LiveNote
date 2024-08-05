import asyncio
import logging
import json
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import websockets

start_time = 0

async def print_message(ws: websockets.WebSocketClientProtocol):
    try:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            logging.info(
                "\n".join(
                    [
                        "-" * 50,"",
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


async def send_audio(wav_path: str, ws_uri: str):
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
                logging.info(f"Sent {_/sample_rate}, {len(frame_np.tobytes())} bytes")
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

    audio_file_path = Path(__file__).parent.parent / "assets" / "audios" / "audio.wav"
    await send_audio(audio_file_path, ws_uri)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    asyncio.run(main())
