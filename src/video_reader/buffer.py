from __future__ import annotations

import queue as queue_module
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

# Sentinel for end of stream in queue
_QUEUE_END: Any = object()


class FrameBuffer:
    """
    Buffers (frame_id, frame) from a frame iterator in a background thread.

    Consumes an iterator produced by the given factory and puts items into
    a bounded queue. Callers iterate via __iter__ / __next__ and get
    (frame_id, np.ndarray). Call release() when done to stop the producer.
    """

    def __init__(
        self,
        frame_iterator_factory: Callable[[], Iterator[tuple[int, np.ndarray]]],
        queue_size: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        frame_iterator_factory : callable
            No-arg callable that returns an iterator yielding (frame_id, frame).
        queue_size : int
            Max size of the internal queue.
        """
        self._frame_iterator_factory = frame_iterator_factory
        self._queue_size = queue_size
        self._queue: queue_module.Queue[tuple[int, np.ndarray] | Any] | None = None
        self._producer_stop: threading.Event | None = None
        self._producer_thread: threading.Thread | None = None

    def _producer_loop(self) -> None:
        try:
            it = self._frame_iterator_factory()
            for frame_id, frame in it:
                if self._producer_stop is not None and self._producer_stop.is_set():
                    break
                if self._queue is not None:
                    self._queue.put((frame_id, frame))
        finally:
            if self._queue is not None:
                self._queue.put(_QUEUE_END)

    def start(self) -> None:
        """Start the producer thread. Idempotent if already started."""
        if self._producer_thread is not None and self._producer_thread.is_alive():
            return
        self._queue = queue_module.Queue(maxsize=self._queue_size)
        self._producer_stop = threading.Event()
        self._producer_thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._producer_thread.start()

    def release(self) -> None:
        """Stop the producer thread and clear state."""
        if self._producer_stop is not None:
            self._producer_stop.set()
        if self._producer_thread is not None and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5.0)
        self._producer_thread = None
        self._producer_stop = None
        self._queue = None

    def is_running(self) -> bool:
        """Return True if the producer thread is running."""
        return (
            self._producer_thread is not None
            and self._producer_thread.is_alive()
        )

    def __iter__(self) -> FrameBuffer:
        self.start()
        return self

    def __next__(self) -> tuple[int, np.ndarray]:
        if self._queue is None:
            raise StopIteration
        item = self._queue.get()
        if item is _QUEUE_END:
            raise StopIteration
        return item
