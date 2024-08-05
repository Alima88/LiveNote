import logging
import wave

import numpy as np

wav_path = "assets/audios/audio.wav"
frame_arr = []

with wave.open(wav_path, "rb") as wav_file:
    # Get basic information about the .wav file
    sample_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    sample_width = wav_file.getsampwidth()

    logging.info(
        f"Sample rate: {sample_rate}, Number of frames: {num_frames}, Sample width: {sample_width}, "
    )

    # Read and send frames in chunks
    chunk_size = 2048
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
        frame_arr.append(frame_np)

audio = np.concatenate(frame_arr)

from io import BytesIO

from scipy.io.wavfile import write

audio_byte = BytesIO()
write(audio_byte, 16000, audio.astype(np.int16))
audio_byte.seek(0)
with open("output.wav", "wb") as f:
    f.write(audio_byte.read())
