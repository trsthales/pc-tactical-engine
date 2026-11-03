import threading

from src.capture.frame_queue import LatestQueue


def test_get_returns_latest_and_counts_drops() -> None:
    queue: LatestQueue[int] = LatestQueue()
    for i in range(5):
        queue.put(i)
    assert queue.get(timeout=0) == 4
    assert queue.dropped == 4
    assert queue.get(timeout=0) is None


def test_get_wakes_up_on_put_from_other_thread() -> None:
    queue: LatestQueue[str] = LatestQueue()
    threading.Timer(0.05, queue.put, args=("frame",)).start()
    assert queue.get(timeout=1.0) == "frame"


def test_none_is_a_valid_item() -> None:
    queue: LatestQueue[None] = LatestQueue()
    queue.put(None)
    assert queue.dropped == 0
    queue.put(None)
    assert queue.dropped == 1
