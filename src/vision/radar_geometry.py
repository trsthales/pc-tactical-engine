"""Geometria do radar do FC: de pixel do recorte para coordenada do campo em [0, 1]².

O radar é desenhado em perspectiva (trapézio mais estreito em cima) e as linhas internas
são estilizadas, fora de escala. A referência confiável é a borda do trapézio escuro: ela é
a linha lateral e de fundo (a bola do pontapé inicial cai no cruzamento das diagonais e a
linha do meio, em x = 0,5, em qualquer altura).

Tudo sai de uma única imagem: o fundo médio do radar (mediana de centenas de frames, em
que os marcadores somem). Dela vêm os cantos, a homografia, as máscaras e a referência de
opacidade. Calibrar outra máquina = gerar outro fundo (`tools/calibrate_radar.py`).
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from config.settings import RADAR_BACKGROUND_FILE

type Image = NDArray[np.uint8]
type Points = NDArray[np.float64]

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_BACKGROUND = ASSETS_DIR / "radar_background.png"

_OUTSIDE_V, _INSIDE_V = 143.0, 91.0  # brilho (V) do gramado em volta e do fundo do radar
_EDGE_RUN = 5  # pixels seguidos abaixo do meio-termo para aceitar a borda


def _value(image: Image) -> NDArray[np.float64]:
    v = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float64)
    return np.asarray(cv2.GaussianBlur(v, (0, 0), 0.7), np.float64)


def _crossing(profile: NDArray[np.float64], level: float) -> float | None:
    """Primeira descida abaixo de `level` que se mantém, com interpolação subpixel."""
    below = profile < level
    for i in range(1, len(profile) - _EDGE_RUN):
        if below[i] and not below[i - 1] and np.all(profile[i : i + _EDGE_RUN] < level + 8):
            return float(i - 1 + (profile[i - 1] - level) / (profile[i - 1] - profile[i]))
    return None


def _fit_line(points: list[tuple[float, float]]) -> NDArray[np.float64]:
    """Reta robusta (Huber) em forma homogênea a·x + b·y + c = 0."""
    if len(points) < 10:
        raise ValueError("borda do radar não encontrada no fundo")
    vx, vy, x0, y0 = cv2.fitLine(np.array(points, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    normal = np.array([-vy[0], vx[0]], np.float64)
    return np.array([normal[0], normal[1], -normal @ [x0[0], y0[0]]])


def fit_pitch_corners(background: Image) -> Points:
    """Cantos do campo no recorte, em pixels: superior esq., superior dir., inferior dir.,
    inferior esq. Ajusta as quatro bordas por subpixel e evita os cantos arredondados.
    """
    v = _value(background)
    height, width = v.shape
    level = (_OUTSIDE_V + _INSIDE_V) / 2
    mid_x, mid_y = width // 2, height // 2
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    top: list[tuple[float, float]] = []
    bottom: list[tuple[float, float]] = []
    for y in range(round(0.15 * height), round(0.84 * height)):
        if (lx := _crossing(v[y, :mid_x], level)) is not None:
            left.append((lx, y))
        if (rx := _crossing(v[y, mid_x:][::-1], level)) is not None:
            right.append((width - 1 - rx, y))
    for x in range(round(0.18 * width), round(0.82 * width)):
        if (ty := _crossing(v[:mid_y, x], level)) is not None:
            top.append((x, ty))
        # embaixo há uma faixa de traços brancos: a busca começa um pouco acima dela
        lower = v[mid_y : round(0.94 * height), x][::-1]
        if (by := _crossing(lower, level)) is not None:
            bottom.append((x, mid_y + len(lower) - 1 - by))
    edges = {"left": _fit_line(left), "right": _fit_line(right),
             "top": _fit_line(top), "bottom": _fit_line(bottom)}  # fmt: skip

    def meet(a: str, b: str) -> NDArray[np.float64]:
        p = np.cross(edges[a], edges[b])
        return np.asarray(p[:2] / p[2], np.float64)

    return np.array([meet("top", "left"), meet("top", "right"),
                     meet("bottom", "right"), meet("bottom", "left")])  # fmt: skip


@dataclass(frozen=True)
class RadarReference:
    """Tudo o que o detector precisa saber sobre o radar, no tamanho do recorte de referência."""

    background: Image

    @classmethod
    def load(cls, path: Path | None = None) -> "RadarReference":
        """Fundo calibrado nesta máquina, se existir; senão, o do vídeo de referência (FC 27)."""
        chosen = path or (RADAR_BACKGROUND_FILE if RADAR_BACKGROUND_FILE.exists() else None)
        image = cv2.imread(str(chosen or DEFAULT_BACKGROUND), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(chosen or DEFAULT_BACKGROUND)
        return cls(np.asarray(image, np.uint8))

    @property
    def size(self) -> tuple[int, int]:
        """(largura, altura) do recorte de referência."""
        return self.background.shape[1], self.background.shape[0]

    @cached_property
    def corners(self) -> Points:
        return fit_pitch_corners(self.background)

    @cached_property
    def _to_pitch(self) -> NDArray[np.float64]:
        unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        return np.asarray(
            cv2.getPerspectiveTransform(self.corners.astype(np.float32), unit), np.float64
        )

    def to_pitch(self, pixels: Points) -> Points:
        """Pixels do recorte (n, 2) → campo normalizado (n, 2): x = comprimento, y = largura."""
        if not len(pixels):
            return np.empty((0, 2))
        projected = cv2.perspectiveTransform(
            pixels.reshape(-1, 1, 2).astype(np.float64), self._to_pitch
        )
        return np.asarray(projected, np.float64).reshape(-1, 2)

    @cached_property
    def pitch_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros(self.background.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.rint(self.corners).astype(np.int32), 1)
        return mask.astype(bool)

    @cached_property
    def search_mask(self) -> NDArray[np.uint8]:
        """Campo com folga: o marcador de quem está na linha lateral fica metade para fora."""
        return np.asarray(
            cv2.dilate(self.pitch_mask.astype(np.uint8), np.ones((9, 9), np.uint8)), np.uint8
        )

    @cached_property
    def _opacity_masks(self) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Fundo do radar sem as linhas desenhadas, e um anel de gramado logo fora dele."""
        pitch = self.pitch_mask.astype(np.uint8)
        inner = cv2.erode(pitch, np.ones((9, 9), np.uint8)).astype(bool)
        inner &= _value(self.background) < (_OUTSIDE_V + _INSIDE_V) / 2 - 15
        ring = (cv2.dilate(pitch, np.ones((13, 13), np.uint8))
                - cv2.dilate(pitch, np.ones((5, 5), np.uint8))).astype(bool)  # fmt: skip
        ring[round(0.9 * ring.shape[0]) :, :] = False  # faixa de traços brancos embaixo
        # amostra esparsa: a mediana fica igual e custa uma fração do tempo
        sparse = np.zeros_like(inner)
        sparse[::3, ::3] = True
        return inner & sparse, ring

    def _contrast(self, value: NDArray[np.float64]) -> float:
        inner, ring = self._opacity_masks
        return float(np.median(value[ring]) - np.median(value[inner]))

    @cached_property
    def _full_contrast(self) -> float:
        return self._contrast(_value(self.background))

    def opacity(self, crop: Image) -> float:
        """Quanto o radar escurece o gramado: ~1 visível, ~0 ausente, no meio = fade."""
        return self._contrast(_value(crop)) / self._full_contrast

    @cached_property
    def _gray_reference(self) -> NDArray[np.float32]:
        gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray -= gray.mean()
        return np.asarray(gray / np.linalg.norm(gray), np.float32)

    def similarity(self, crop: Image) -> float:
        """Correlação com o fundo: separa radar de verdade de cenas que só escurecem ali."""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray -= gray.mean()
        return float((gray * self._gray_reference).sum() / (np.linalg.norm(gray) + 1e-9))
