"""Salva prints da tela inteira em intervalos, para calibrar a visão (Marco 2).

Com o jogo aberto em 1080p:
    python -m tools.grab_samples --every 3 --count 20
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.capture.win32_dpi import enable_dpi_awareness

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "tests" / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description="Salva prints da tela em intervalos.")
    parser.add_argument("--every", type=float, default=3.0, help="segundos entre prints")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--monitor", type=int, default=None, help="índice do monitor (dxcam)")
    args = parser.parse_args()

    enable_dpi_awareness()
    import dxcam  # só instala no Windows

    camera = dxcam.create(output_idx=args.monitor, output_color="BGR")
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    try:
        for i in range(args.count):
            while (frame := camera.grab()) is None:  # None = tela não mudou desde o último grab
                time.sleep(0.01)
            path = args.out / f"{stamp}_{i:03}.png"
            cv2.imwrite(str(path), frame)
            print(f"[{i + 1}/{args.count}] {path}")
            time.sleep(args.every)
    finally:
        camera.release()


if __name__ == "__main__":
    main()
