import dataclasses

import numpy as np

from server.config import Settings


@dataclasses.dataclass
class ClientAudio:
    client_id: str
    data: np.ndarray
    prompt: str | None = None
    offset: float = 0.0
    eos: bool = False
    no_voice_activity_chunks: int = 0

    @property
    def duration(self) -> float:
        return self.data.shape[0] / Settings.sample_rate

    @property
    def is_empty(self) -> bool:
        return self.data is None

    @property
    def last_30_seconds(self) -> np.ndarray:
        return self.data[-30 * Settings.sample_rate :]

    @property
    def last_30_seconds_duration(self) -> np.ndarray:
        return self.last_30_seconds.shape[0] / Settings.sample_rate

    def add_frames(self, frame_np: np.ndarray):
        """
        Add audio frames to the ongoing audio stream buffer.

        This method is responsible for maintaining the audio stream buffer, allowing the continuous addition
        of audio frames as they are received. It also ensures that the buffer does not exceed a specified size
        to prevent excessive memory usage.

        If the buffer size exceeds a threshold (45 seconds of audio data), it discards the oldest 30 seconds
        of audio data to maintain a reasonable buffer size. If the buffer is empty, it initializes it with the provided
        audio frame. The audio stream buffer is used for real-time processing of audio data for transcription.

        Args:
            frame_np (numpy.ndarray): The audio frame data as a NumPy array.

        """

        if self.data is None:
            self.data = frame_np.copy()
            return

        # if self.data.shape[0] > 45 * Settings.sample_rate:
        #     self.offset += 30.0
        #     self.data = self.data[int(30 * Settings.sample_rate) :]

        self.data = np.concatenate((self.data, frame_np), axis=0)
