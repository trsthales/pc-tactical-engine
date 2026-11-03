import sys
import threading
import types

import numpy as np
import pytest

from config.settings import ROIConfig
from src.capture.screen_capture import ScreenCapture


class FakeCamera:
    """Imita o dxcam: get_latest_frame espera um frame novo ou o stop()."""

    def __init__(self) -> None:
        self.new_frame = threading.Event()
        self.capturing = False
        self.released = False

    def start(self, target_fps: int) -> None:
        if self.capturing:
            raise RuntimeError("Capture is already running")
        self.capturing = True

    def stop(self) -> None:
        self.capturing = False
        self.new_frame.set()

    def release(self) -> None:
        self.released = True

    def get_latest_frame(self, copy: bool = False) -> np.ndarray:
        self.new_frame.wait()
        self.new_frame.clear()
        return np.zeros((1080, 1920, 3), np.uint8)


@pytest.fixture
def camera(monkeypatch: pytest.MonkeyPatch) -> FakeCamera:
    fake = FakeCamera()
    module = types.SimpleNamespace(create=lambda **_: fake)
    monkeypatch.setitem(sys.modules, "dxcam", module)
    return fake


def test_stop_unblocks_a_static_screen_and_restart_works(camera: FakeCamera) -> None:
    capture = ScreenCapture(ROIConfig())
    for _ in range(2):  # a tela nunca muda: só o stop() destrava o get_latest_frame
        capture.start()
        camera.new_frame.set()
        assert capture.frames.get(timeout=1.0) is not None
        capture.stop()
        assert not capture.running
        assert not camera.capturing
        assert camera.released
