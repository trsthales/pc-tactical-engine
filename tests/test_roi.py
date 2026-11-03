from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from config.settings import ROI, ROIConfig
from src.capture.screen_capture import crop_rois

RESOLUTIONS = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]


def _synthetic_screen(width: int, height: int) -> np.ndarray:
    """Tela com uma grade de pontos em coordenadas relativas, como o HUD do jogo escala.

    A grade não depende das ROIs: um recorte deslocado pega outros pontos e falha.
    """
    screen = np.zeros((height, width, 3), np.uint8)
    radius = max(2, height // 200)
    for fx in np.arange(0.01, 1.0, 0.02):
        for fy in np.arange(0.01, 1.0, 0.03):
            center = (round(fx * width), round(fy * height))
            cv2.circle(screen, center, radius, (255, 255, 255), -1)
    return screen


@pytest.mark.parametrize("name", list(ROIConfig().named()))
@pytest.mark.parametrize(("width", "height"), RESOLUTIONS)
def test_crop_is_resolution_invariant(width: int, height: int, name: str) -> None:
    size = (200, 110)
    reference = cv2.resize(crop_rois(_synthetic_screen(1920, 1080), ROIConfig())[name], size)
    crop = cv2.resize(crop_rois(_synthetic_screen(width, height), ROIConfig())[name], size)
    # Mesmo conteúdo, só reamostrado (≤ 7); um recorte 0,5% fora do lugar passa de 25.
    assert np.abs(crop.astype(int) - reference.astype(int)).mean() < 10


def test_pixels_round_trip() -> None:
    box = (760, 820, 1160, 1040)
    assert ROI.from_pixels(box, 1920, 1080).to_pixels(1920, 1080) == box


def test_rejects_inverted_roi() -> None:
    with pytest.raises(ValidationError):
        ROI(left=0.6, top=0.1, right=0.4, bottom=0.2)


def test_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "rois.json"
    custom = ROIConfig(radar=ROI(left=0.1, top=0.1, right=0.3, bottom=0.3))
    custom.save(path)
    assert ROIConfig.load(path) == custom
    assert ROIConfig.load(tmp_path / "inexistente.json") == ROIConfig()
