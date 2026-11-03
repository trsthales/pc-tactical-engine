"""Gera o fundo do radar desta máquina (`config/radar_background.png`).

O detector tira tudo do fundo médio do radar: cantos do campo, homografia, máscaras e
opacidade. O que vem no projeto é do vídeo de referência (EA FC 27, radar padrão). Se o
radar estiver com outro tamanho ou transparência nas opções do jogo, gere o seu:

    python -m tools.calibrate_radar --screen              # com uma partida rolando
    python -m tools.calibrate_radar --video partida.mp4

Junta recortes do radar com o jogo rolando, espaçados no tempo, e tira a mediana pixel a
pixel: os marcadores se mexem e somem; o desenho do radar fica.
"""

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from config.settings import RADAR_BACKGROUND_FILE, ROIConfig
from src.vision.radar_geometry import DEFAULT_BACKGROUND, Image, RadarReference, fit_pitch_corners
from src.vision.video_source import VideoSource

_MIN_GAP_S = 0.5  # recortes muito próximos repetem os marcadores no mesmo lugar
_MIN_OPACITY, _MIN_SIMILARITY = 0.8, 0.5


def _video_crops(path: Path) -> Iterator[tuple[float, Image]]:
    for frame in VideoSource(path, ROIConfig.load()):
        yield frame.timestamp, frame.crops["radar"]


def _screen_crops() -> Iterator[tuple[float, Image]]:
    from src.capture.screen_capture import ScreenCapture

    with ScreenCapture(ROIConfig.load()) as capture:
        while capture.running:
            if (frame := capture.frames.get(timeout=1.0)) is not None:
                yield time.perf_counter(), frame.crops["radar"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra o fundo do radar.")
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument("--screen", action="store_true")
    origin.add_argument("--video", type=Path)
    parser.add_argument("--count", type=int, default=300, help="recortes na mediana")
    parser.add_argument("--out", type=Path, default=RADAR_BACKGROUND_FILE)
    args = parser.parse_args()

    reference = RadarReference.load(DEFAULT_BACKGROUND)  # o do projeto: o antigo pode estar ruim
    width, height = reference.size
    picked: list[Image] = []
    last = -np.inf
    crops = _video_crops(args.video) if args.video else _screen_crops()
    for t, crop in crops:
        if t - last < _MIN_GAP_S:
            continue
        crop = np.asarray(cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA), np.uint8)
        if reference.opacity(crop) < _MIN_OPACITY or reference.similarity(crop) < _MIN_SIMILARITY:
            continue  # radar fora da tela ou em fade
        picked.append(crop)
        last = t
        print(f"\r{len(picked)}/{args.count}", end="", flush=True)
        if len(picked) >= args.count:
            break
    print()
    if len(picked) < args.count // 2:
        raise SystemExit(f"Só {len(picked)} recortes com o radar visível: jogue mais um pouco.")

    background = np.median(np.array(picked), axis=0).astype(np.uint8)
    fit_pitch_corners(background)  # falha aqui se a borda do campo não aparece
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), background)
    print(f"Fundo salvo em {args.out}")


if __name__ == "__main__":
    main()
