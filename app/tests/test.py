import asyncio
import uuid
import wave
from pathlib import Path

import numpy as np
import websockets


async def test(ws_uri: str):
    i = 0
    async with websockets.connect(ws_uri) as websocket:
        await websocket.send(f"Hello, world! {i}".encode("utf-8"))
        print(i)
        await websocket.recv()
        await websocket.send(b"close()")


async def send_audio(wav_path: str, ws_uri: str):
    if isinstance(wav_path, Path):
        wav_path = str(wav_path)

    async with websockets.connect(ws_uri) as websocket:
        with wave.open(wav_path, "rb") as wav_file:
            # Get basic information about the .wav file
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()

            print(
                f"Sample rate: {sample_rate}, Number of frames: {num_frames}, Sample width: {sample_width}, "
            )

            # Read and send frames in chunks
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

        await websocket.send(b"close()")


async def main():
    client_id = uuid.uuid4()
    print(f"Client ID: {client_id}")
    # ws_uri = f"wss://wln.inbeet.tech/transcription/ws/{client_id}"
    ws_uri = f"ws://localhost:8000/transcription/ws/{client_id}"

    audio_file_path = Path(__file__).parent.parent / "assets" / "audio.wav"
    await send_audio(audio_file_path, ws_uri)


if __name__ == "__main__":
    asyncio.run(main())
