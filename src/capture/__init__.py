"""Módulo de captura de baixa latência via DirectX Desktop Duplication API (DXCam)."""

from src.capture.frame_queue import LatestQueue
from src.capture.screen_capture import CapturedFrame, ScreenCapture, crop_rois

__all__ = ["CapturedFrame", "LatestQueue", "ScreenCapture", "crop_rois"]
