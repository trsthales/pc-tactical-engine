"""Frames de um vídeo gravado, no mesmo formato da captura de tela (`CapturedFrame`).

Serve para desenvolver e validar a visão sem o jogo aberto. O tempo de cada frame é o do
vídeo (índice / FPS), não o do relógio da máquina: a análise é a mesma em qualquer ritmo.
"""

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from config.settings import ROIConfig
from src.capture import CapturedFrame, crop_rois


class VideoSource:
    def __init__(self, path: Path, rois: ROIConfig, start_s: float = 0.0) -> None:
        self.path, self.rois, self.start_s = path, rois, start_s
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Não consegui abrir o vídeo {path}")
        self.fps = float(capture.get(cv2.CAP_PROP_FPS)) or 60.0
        capture.release()

    def __iter__(self) -> Iterator[CapturedFrame]:
        capture = cv2.VideoCapture(str(self.path))
        try:
            first = round(self.start_s * self.fps)
            if first:
                capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            index = first
            while True:
                ok, image = capture.read()
                if not ok:
                    return
                frame = np.asarray(image, np.uint8)
                yield CapturedFrame(index, index / self.fps, crop_rois(frame, self.rois))
                index += 1
        finally:
            capture.release()
