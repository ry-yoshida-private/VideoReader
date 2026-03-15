from __future__ import annotations

import os
import cv2
import numpy as np
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Callable

from .buffer import FrameBuffer


@dataclass
class VideoReader:
    """
    VideoReader is the reader for the video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    iter_start_frame : int, optional
        The frame number to start reading from (default is 0).
    freq : int, optional
        The step size for frame iteration (default is 1).
    freq_th : int, optional
        Threshold for freq; when freq > freq_th, seek-based read is used (default is 10).
    use_queue : bool, optional
        If True, prefetch frames in a background thread (default is False).
    queue_size : int, optional
        Max size of the prefetch queue when use_queue is True (default is 2).

    Attributes
    ----------
    video_path: str
        The path to the video file.
    iter_start_frame: int
        The frame number to start reading from.
    freq: int
        The step size for frame iteration.
    cap: cv2.VideoCapture
        The video capture object.
    _read_next_frame: Callable[[], tuple[bool, np.ndarray | None]]
        The function to read the next frame.
    total_frame: int
        The total number of frames in the video.
    _next_frame_id: int
        The current frame number.
    use_queue: bool
        If True, frames are prefetched in a background thread and consumed from a queue.
    queue_size: int
        Max number of frames to prefetch when use_queue is True.
    freq_th: int
        Threshold for freq; above this value, seek-based read is used instead of loop.
    _next_impl: Callable[[], np.ndarray]
        Bound implementation for __next__ (either _next_from_queue or _next_from_cap).
    """

    video_path: str
    iter_start_frame: int = 0
    freq: int = 1
    freq_th: int = 10
    use_queue: bool = False
    queue_size: int = 2

    cap: cv2.VideoCapture = field(init=False)
    _frame_reader_function: Callable[[], tuple[bool, np.ndarray | None]] = field(init=False)
    total_frame: int = field(init=False)
    _next_frame_id: int = field(init=False)
    _buffer: FrameBuffer | None = field(init=False, default=None)
    _current_frame_id_queue: int | None = field(init=False, default=None)
    _next_impl: Callable[[], np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        """
        Initialize the VideoReader (open video, set reader function, etc.).

        Raises
        ------
        FileNotFoundError
            If the video file does not exist.
        ValueError
            If the video file cannot be opened.
        """
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Error: The video file does not exist: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open the video file: {self.video_path}")

        self._frame_reader_function = self.define_frame_reader_function()
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next_frame_id = self.iter_start_frame
        self._next_impl = self._next_from_cap

    def define_frame_reader_function(self) -> Callable[[], tuple[bool, np.ndarray | None]]:
        """
        Define the function to read the next frame.

        NOTE:
        ------
        If the step size is large, it is better to use the _read_with_set function.
        If the step size is small, it is better to use the _read_with_loop function.

        Returns:
        ----------
        Callable[[], tuple[bool, np.ndarray | None]]: The function to read the next frame.
        """
        if self.freq > self.freq_th:
            return self._read_with_set
        else:
            return self._read_with_loop

    def _read_with_loop(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame with loop.

        It takes a short time when step size is small because set the frame position is not needed.

        Returns:
        ----------
        tuple[bool, np.ndarray | None]: The next frame and a boolean indicating if the end of the video is reached.
        """
        ret = False
        frame = None
        for _ in range(self.freq):
            ret, frame = self.cap.read()
            if not ret:
                return False, None
        return ret, frame

    def _read_with_set(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame with set.
        It takes a short time when step size is large because all frame are not read from the video file.

        Returns:
        ----------
        tuple[bool, np.ndarray | None]: The next frame and a boolean indicating if the end of the video is reached.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self._next_frame_id)
        ret, frame = self.cap.read()
        return ret, frame

    def _next_from_queue(self) -> np.ndarray:
        """Get next frame from buffer. Used when use_queue is True."""
        frame_id, frame = self._buffer.__next__()
        self._current_frame_id_queue = frame_id
        return frame

    def _next_from_cap(self) -> np.ndarray:
        """Get next frame from VideoCapture. Used when use_queue is False."""
        if self._next_frame_id > self.total_frame:
            raise StopIteration
        ret, frame = self._frame_reader_function()
        self._next_frame_id += self.freq
        if frame is None or not ret:
            raise StopIteration
        return frame

    def _iterate_frames(
        self,
        cap: cv2.VideoCapture | None = None,
        ) -> Iterator[tuple[int, np.ndarray]]:
        """
        Yield (frame_id, frame). Uses cap if given.
        freq <= freq_th: sequential read (no seek per frame).

        Parameters
        ----------
        cap : cv2.VideoCapture | None
            The video capture object. If None, a new video capture object is created.

        Returns
        -------
        Iterator[tuple[int, np.ndarray]]: An iterator yielding (frame_id, frame).
        """
        own_cap = cap is None
        if cap is None:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return
        try:
            total = self.total_frame if not own_cap else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            next_id = self.iter_start_frame
            if self.freq <= self.freq_th:
                # Like _read_with_loop: set position once, then only read() in loop (fast).
                start_pos = max(0, next_id - self.freq + 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
                num_reads_first = next_id - start_pos + 1
                ret, frame = False, None
                for _ in range(num_reads_first):
                    ret, frame = cap.read()
                    if not ret:
                        return
                if frame is None:
                    return
                yield (next_id, frame)
                next_id += self.freq
                while next_id <= total:
                    ret, frame = False, None
                    for _ in range(self.freq):
                        ret, frame = cap.read()
                        if not ret:
                            return
                    if frame is None:
                        return
                    yield (next_id, frame)
                    next_id += self.freq
            else:
                # Like _read_with_set: seek per frame.
                while next_id <= total:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_id)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        return
                    yield (next_id, frame)
                    next_id += self.freq
        finally:
            if own_cap:
                cap.release()

    def _create_frame_iterator_factory(
        self,
        ) -> Callable[[], Iterator[tuple[int, np.ndarray]]]:
        """
        Create a frame iterator factory.

        Returns
        -------
        Callable[[], Iterator[tuple[int, np.ndarray]]]: A function that returns an iterator yielding (frame_id, frame).
        """
        return lambda: self._iterate_frames(self.cap)

    def _stop_buffer(self) -> None:
        """Stop the frame buffer if used. No-op otherwise."""
        if self._buffer is not None:
            self._buffer.release()
            self._buffer = None

    @property
    def is_reach_end_of_video(self) -> bool:
        return self._next_frame_id > self.total_frame

    def extract_frame(
        self,
        frame_number: int
        ) -> np.ndarray:
        """
        Extracts and saves a specific frame from the video.

        When use_queue is True, uses a temporary VideoCapture so it is safe to call during iteration.

        Parameters
        ----------
        frame_number : int
            The frame number to extract.

        Returns
        -------
        np.ndarray
            The extracted frame.

        Raises
        ------
        ValueError
            If the frame could not be read.
        """
        if self.use_queue and self._buffer is not None and self._buffer.is_running():
            cap = cv2.VideoCapture(self.video_path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret or frame is None:
                    raise ValueError(f"Failed to read frame {frame_number} from {self.video_path}")
                return frame
            finally:
                cap.release()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise ValueError(f"Failed to read frame {frame_number} from {self.video_path}")
        return frame

    def __iter__(self) -> VideoReader:
        """
        Resets the video position to the specified iter_start_frame and returns an iterator over frames.

        When use_queue is True, starts a background thread that prefetches frames into a queue.

        Returns
        -------
        VideoReader
            Iterator object for reading frames sequentially from the iter_start_frame.
        """
        self._current_frame_id_queue = None
        if self.use_queue:
            self._stop_buffer()
            self._buffer = FrameBuffer(
                self._create_frame_iterator_factory(),
                queue_size=self.queue_size,
            )
            self._buffer.start()
            self._next_impl = self._next_from_queue
        else:
            # So that _read_with_loop's first `freq` reads yield frame at iter_start_frame
            start_pos = max(0, self.iter_start_frame - self.freq + 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)
            self._next_frame_id = self.iter_start_frame
            self._next_impl = self._next_from_cap
        return self

    def __len__(self) -> int:
        """
        Returns the total number of frames in the video.

        Returns
        -------
        int
            Total number of frames in the video.
        """
        return self.total_frame

    def __next__(self) -> np.ndarray:
        """ Retrieves the next frame in the video, based on the specified frequency.

        Returns
        -------
        numpy.ndarray
            The next frame as a NumPy array.

        Raises
        ------
        StopIteration
            When the video reaches the end.
        """
        return self._next_impl()

    def release(self) -> None:
        """
        Releases the video file and stops the frame buffer if running.
        """
        self._stop_buffer()
        self.cap.release()

    @property
    def frame_id(self) -> int:
        """
        Returns the current frame id (last yielded when using queue).

        Returns
        -------
        int
            Current frame id.
            When use_queue is True, returns the frame id of the last frame yielded from the queue.
            Otherwise, POS_FRAMES - 1 (next frame to be read).
        """
        if self.use_queue and self._current_frame_id_queue is not None:
            return self._current_frame_id_queue
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __str__(self) -> str:
        return f"VideoReader(video_path={self.video_path}, total_frame={self.total_frame})"
