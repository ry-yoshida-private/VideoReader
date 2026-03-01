from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Callable

class VideoReader:
    """
    VideoReader is the reader for the video.

    Attributes:
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
    """
    def __init__(
        self, 
        video_path: str, 
        iter_start_frame: int = 0, 
        freq: int = 1
        ):
        """
        Initialize the VideoReader.

        Parameters
        ----------
        video_path : str
            Path to the video file.
        iter_start_frame : int, optional
            The frame number to start reading from (default is 0).
        freq : int, optional
            The step size for frame iteration (default is 1, meaning read every frame).

        Raises
        ------
        ValueError
            If the video file cannot be opened.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Error: The video file does not exist: {video_path}")
        self.video_path = video_path
        self.iter_start_frame = iter_start_frame
        self.freq = freq 

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Failed to open the video file: {video_path}")
        
        self._read_next_frame = self.define_read_next_frame()
        self.total_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next_frame_id = self.iter_start_frame

    def define_read_next_frame(self) -> Callable[[], tuple[bool, np.ndarray | None]]:
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
        if self.freq > 2:
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

    @property
    def is_reach_end_of_video(self) -> bool:
        return self._next_frame_id > self.total_frame

    def extract_frame(
        self, 
        frame_number: int
        ) -> np.ndarray:
        """
        Extracts and saves a specific frame from the video.

        Parameters
        ----------
        frame_number : int
            The frame number to extract.

        Returns
        -------
        np.ndarray
            The extracted frame.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame

    def __iter__(self) -> VideoReader:
        """
        Resets the video position to the specified iter_start_frame and returns an iterator over frames.

        Returns
        -------
        VideoReader
            Iterator object for reading frames sequentially from the iter_start_frame.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.iter_start_frame)
        self._next_frame_id = self.iter_start_frame
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
        if self.is_reach_end_of_video:
            raise StopIteration
        
        ret, frame = self._read_next_frame()
        self._next_frame_id += self.freq
        if frame is None or not ret:
            raise StopIteration
        return frame

    def release(self) -> None:
        """ Releases the video file.
        """
        self.cap.release()

    @property
    def frame_id(self) -> int:
        """ Returns the current number of frames in the video.

        Returns
        -------
        int
            Current number of frames in the video.
            * Subtract 1 because POS_FRAMES returns the next frame to be read.
        """
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))-1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    def __str__(self) -> str:
        return f"VideoReader(video_path={self.video_path}, total_frame={self.total_frame})"
