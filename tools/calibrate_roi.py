"""Ajuste visual das ROIs (radar, placar, HUD) sobre um print do jogo.

    python -m tools.calibrate_roi tests/samples/meu_print.png
    python -m tools.calibrate_roi            # sem arquivo: captura a tela agora

Teclas: 1/2/3 escolhem a ROI, S salva em config/rois.json, Q ou Esc sai.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from config.settings import ROI, ROI_FILE, ROIConfig
from src.capture.win32_dpi import enable_dpi_awareness

WINDOW, PREVIEW = "Calibrar ROI", "Recorte"
EDGES = ("left", "top", "right", "bottom")
SCALE = 1000  # trackbar trabalha com inteiros: posição em milésimos da tela
SELECTED, OTHER = (0, 255, 255), (160, 160, 160)


def grab_screen() -> NDArray[np.uint8]:
    enable_dpi_awareness()
    import dxcam  # só instala no Windows

    camera = dxcam.create(output_color="BGR")
    try:
        while (frame := camera.grab()) is None:  # None = tela não mudou desde o último grab
            time.sleep(0.01)
        return frame  # type: ignore[no-any-return]
    finally:
        camera.release()


def set_trackbars(roi: ROI) -> None:
    for edge in EDGES:
        cv2.setTrackbarPos(edge, WINDOW, round(getattr(roi, edge) * SCALE))


def read_trackbars() -> ROI | None:
    try:
        return ROI(**{edge: cv2.getTrackbarPos(edge, WINDOW) / SCALE for edge in EDGES})
    except ValidationError:  # bordas cruzadas no meio do arraste
        return None


def draw(image: NDArray[np.uint8], rois: dict[str, ROI], selected: str) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    canvas = image.copy()
    for name, roi in rois.items():
        left, top, right, bottom = roi.to_pixels(width, height)
        color, thickness = (SELECTED, 3) if name == selected else (OTHER, 1)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
        cv2.putText(canvas, name, (left, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibração visual das ROIs.")
    parser.add_argument("image", type=Path, nargs="?", help="print do jogo (PNG/JPG)")
    parser.add_argument("--out", type=Path, default=ROI_FILE)
    args = parser.parse_args()

    loaded = cv2.imread(str(args.image)) if args.image else grab_screen()
    if loaded is None:
        raise SystemExit(f"Não consegui abrir {args.image}")
    image = np.asarray(loaded, np.uint8)
    height, width = image.shape[:2]

    rois = ROIConfig.load(args.out).named()
    names = list(rois)
    selected = names[0]

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for edge in EDGES:
        cv2.createTrackbar(edge, WINDOW, 0, SCALE, lambda _: None)
    set_trackbars(rois[selected])

    while True:
        if roi := read_trackbars():
            rois[selected] = roi
        cv2.imshow(WINDOW, draw(image, rois, selected))
        left, top, right, bottom = rois[selected].to_pixels(width, height)
        if right > left and bottom > top:  # ROI menor que 1 px some no arredondamento
            cv2.imshow(PREVIEW, image[top:bottom, left:right])

        match chr(cv2.waitKey(30) & 0xFF).lower():
            case key if key.isdecimal() and 0 < int(key) <= len(names):
                selected = names[int(key) - 1]
                set_trackbars(rois[selected])
            case "s":
                ROIConfig.model_validate(rois).save(args.out)
                print(f"Salvo em {args.out}")
            case "q" | "\x1b":
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
