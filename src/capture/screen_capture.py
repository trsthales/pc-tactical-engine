"""Captura da tela via DXGI Desktop Duplication (dxcam) e recorte das ROIs.

Benchmark no Windows, com o jogo aberto:
    python -m src.capture.screen_capture --seconds 10
"""

import argparse
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from config.settings import TARGET_FPS, ROIConfig
from src.capture.frame_queue import LatestQueue
from src.capture.win32_dpi import enable_dpi_awareness

type Image = NDArray[np.uint8]

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    frame_id: int
    timestamp: float  # time.perf_counter() quando a thread recebeu o frame do dxcam
    crops: Mapping[str, Image]  # BGR, por nome de ROI


@dataclass(slots=True)
class CaptureStats:
    frames: int = 0
    busy_seconds: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    def record(self, seconds: float) -> None:
        self.frames += 1
        self.busy_seconds += seconds

    @property
    def fps(self) -> float:
        return self.frames / max(time.perf_counter() - self.started_at, 1e-9)

    @property
    def avg_ms(self) -> float:
        return 1000 * self.busy_seconds / max(self.frames, 1)


def crop_rois(frame: Image, rois: ROIConfig) -> dict[str, Image]:
    """Recorta cada ROI do frame inteiro. Copia para não segurar o buffer da GPU."""
    height, width = frame.shape[:2]
    crops: dict[str, Image] = {}
    for name, roi in rois.named().items():
        left, top, right, bottom = roi.to_pixels(width, height)
        crops[name] = frame[top:bottom, left:right].copy()
    return crops


class ScreenCapture:
    """Thread produtora: frame da GPU → recortes das ROIs → `self.frames` (drop-oldest).

    Uso:
        with ScreenCapture(ROIConfig.load()) as capture:
            frame = capture.frames.get(timeout=1)
    """

    def __init__(
        self, rois: ROIConfig, *, fps: int = TARGET_FPS, output_idx: int | None = None
    ) -> None:
        self.rois = rois
        self.fps = fps
        self.output_idx = output_idx
        self.frames: LatestQueue[CapturedFrame] = LatestQueue()
        self.stats = CaptureStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera: Any = None

    @property
    def running(self) -> bool:
        """Falso se a captura parou ou a thread morreu (o erro vai para o log)."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        enable_dpi_awareness()
        import dxcam  # só instala no Windows

        camera = dxcam.create(output_idx=self.output_idx, output_color="BGR")
        if camera is None:
            raise RuntimeError("dxcam não conseguiu abrir o monitor (DXGI)")
        camera.start(target_fps=self.fps)
        self._camera = camera
        self.stats = CaptureStats()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(camera,), name="screen-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        camera, self._camera = self._camera, None
        if camera is None:
            return
        # Parar a câmera aqui destrava o get_latest_frame, que espera um frame novo
        # e com a tela parada (pausa, menu) não recebe nenhum.
        camera.stop()
        if self._thread:
            self._thread.join(timeout=1.0)
        camera.release()

    def _run(self, camera: Any) -> None:
        frame_id = 0
        try:
            while not self._stop.is_set():
                frame = camera.get_latest_frame(copy=False)  # bloqueia até o próximo frame
                if frame is None or self._stop.is_set():
                    break
                received = time.perf_counter()
                crops = crop_rois(frame, self.rois)
                self.frames.put(CapturedFrame(frame_id, received, crops))
                self.stats.record(time.perf_counter() - received)
                frame_id += 1
        except Exception:
            log.exception("Captura interrompida por erro")

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark da captura de tela.")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--monitor", type=int, default=None, help="índice do monitor (dxcam)")
    args = parser.parse_args()

    with ScreenCapture(ROIConfig.load(), output_idx=args.monitor) as capture:
        time.sleep(args.seconds)
        stats, dropped = capture.stats, capture.frames.dropped

    print(f"FPS capturado: {stats.fps:.1f} (alvo {TARGET_FPS})")
    print(f"Recorte das ROIs: {stats.avg_ms:.2f} ms/frame (meta < 4 ms)")
    print(f"Frames descartados sem consumidor: {dropped}")


if __name__ == "__main__":
    main()
