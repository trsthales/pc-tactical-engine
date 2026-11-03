"""Leitura do placar e do relógio do HUD do EA FC 27, e detecção do menu de pausa.

O placar (canto superior esquerdo) tem o gol de cada time num quadro azul-marinho (texto
branco) e o relógio mm:ss numa faixa branca (texto escuro, fonte monoespaçada: posições
fixas). As duas fontes são a mesma: um único conjunto de modelos (`assets/digits.png`, a
média de milhares de dígitos do vídeo de referência, rotulados à mão) lê os dois.

Cada dígito é recortado pela caixa justa dos pixels de texto, normalizado para 12x16 e
comparado com os 10 modelos por correlação. Dígito com nota baixa ou ambíguo (animação,
transição) é recusado: a leitura inteira vira None e quem chama mantém a anterior.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from config.settings import VisionConfig
from src.vision.radar_geometry import ASSETS_DIR, Image

type Box = tuple[int, int, int, int]  # (x0, y0, x1, y1) no recorte de referência

DIGITS_FILE = ASSETS_DIR / "digits.png"
BACKGROUND_FILE = ASSETS_DIR / "scoreboard_background.png"
_GLYPH = (12, 16)  # (largura, altura) do dígito normalizado
_CLOCK_SLOTS: tuple[Box, ...] = ((84, 74, 99, 92), (99, 74, 114, 92),
                                 (122, 74, 137, 92), (138, 74, 153, 92))  # fmt: skip
# gols do time da linha de cima e da linha de baixo
_SCORE_BOXES: tuple[Box, Box] = ((175, 18, 212, 43), (175, 43, 212, 67))
_MIN_TEXT_PIXELS = 10


@dataclass(frozen=True, slots=True)
class HudReading:
    visible: bool  # placar na tela
    menu: bool  # menu de pausa / intervalo
    score: tuple[int, int] | None = None  # (linha de cima, linha de baixo)
    clock: tuple[int, int] | None = None  # (minutos, segundos) do jogo


def _normalize(patch: NDArray[np.float32]) -> NDArray[np.float32] | None:
    """Dígito claro sobre fundo escuro → 12x16 com média zero e norma 1."""
    text = (patch > (patch.min() + patch.max()) / 2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
    if count < 2:
        return None
    # cada dígito desta fonte é uma peça só: a maior descarta a borda do vizinho que invade
    biggest = 1 + int(np.argmax(np.asarray(stats)[1:, cv2.CC_STAT_AREA]))
    digit = np.asarray(labels) == biggest
    if digit.sum() < _MIN_TEXT_PIXELS:
        return None
    # caixa pelas linhas/colunas com pelo menos 2 pixels: um pixel solto não estica o dígito
    rows, cols = np.nonzero(digit.sum(axis=1) >= 2)[0], np.nonzero(digit.sum(axis=0) >= 2)[0]
    if not len(rows) or not len(cols):
        return None
    ys, xs = rows[[0, -1]], cols[[0, -1]]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    height = ys.max() - ys.min() + 1
    width = 0.9 * height  # caixa proporcional: o "1" estreito continua estreito
    gw, gh = _GLYPH
    warp = np.array([[gw / width, 0, gw / 2 - cx * gw / width],
                       [0, gh / height, gh / 2 - cy * gh / height]], np.float32)  # fmt: skip
    glyph = np.asarray(
        cv2.warpAffine(patch, warp, _GLYPH, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE),
        np.float32,
    )
    glyph -= glyph.mean()
    return np.asarray(glyph / (np.linalg.norm(glyph) + 1e-9), np.float32)


class HudReader:
    def __init__(self, cfg: VisionConfig | None = None) -> None:
        self.cfg = cfg or VisionConfig()

    @cached_property
    def _background(self) -> Image:
        return _load(BACKGROUND_FILE, cv2.IMREAD_COLOR)

    @cached_property
    def _templates(self) -> NDArray[np.float32]:
        strip = _load(DIGITS_FILE, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        gw, gh = _GLYPH
        glyphs = []
        for d in range(10):
            g = strip[:gh, d * gw : (d + 1) * gw].copy()
            g -= g.mean()
            glyphs.append(g / np.linalg.norm(g))
        return np.array(glyphs)

    @cached_property
    def _gray_reference(self) -> NDArray[np.float32]:
        gray = cv2.cvtColor(self._background, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray -= gray.mean()
        return np.asarray(gray / np.linalg.norm(gray), np.float32)

    def read(self, crop: Image) -> HudReading:
        height, width = self._background.shape[:2]
        if crop.shape[:2] != (height, width):
            crop = np.asarray(
                cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR), np.uint8
            )
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if not self._is_scoreboard(gray):
            return HudReading(visible=False, menu=self._is_menu(crop))
        clock = self._digits([gray[y0:y1, x0:x1] for x0, y0, x1, y1 in _CLOCK_SLOTS], dark=True)
        top, bottom = (self._number(gray[y0:y1, x0:x1]) for x0, y0, x1, y1 in _SCORE_BOXES)
        score = None if top is None or bottom is None else (top, bottom)
        reading = None
        if clock is not None and clock[2] < 6:  # segundos: dezena de 0 a 5
            reading = (10 * clock[0] + clock[1], 10 * clock[2] + clock[3])
        return HudReading(visible=True, menu=False, score=score, clock=reading)

    def _is_scoreboard(self, gray: NDArray[np.float32]) -> bool:
        centered = gray - gray.mean()
        similarity = (centered * self._gray_reference).sum() / (np.linalg.norm(centered) + 1e-9)
        return bool(similarity >= self.cfg.min_scoreboard_similarity)

    def _is_menu(self, crop: Image) -> bool:
        """Menu de pausa/intervalo: fundo azul-marinho liso com o título em branco.

        Medido no vídeo: menus têm ~91% de azul-marinho, 4,5 a 5,5% de texto branco e ~78%
        de área lisa; cenas de corte escuras (teto do estádio) não têm o texto branco.
        """
        hsv = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), np.uint8)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        navy = ((hue >= 100) & (hue <= 125) & (val < 100)).mean()
        white = ((sat < 40) & (val > 200)).mean()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = np.asarray(cv2.blur(gray, (5, 5)), np.float32)
        spread = np.asarray(cv2.blur(gray * gray, (5, 5)), np.float32) - mean * mean
        flat = (spread < 4.0).mean()  # desvio padrão local < 2
        low, high = self.cfg.menu_title_fraction
        return bool(navy >= self.cfg.min_menu_navy and low <= white <= high
                    and flat >= self.cfg.min_menu_flat)  # fmt: skip

    def _classify(self, patch: NDArray[np.float32]) -> int | None:
        glyph = _normalize(patch)
        if glyph is None:
            return None
        scores = np.tensordot(self._templates, glyph, axes=([1, 2], [0, 1]))
        best, second = np.argsort(-scores)[:2]
        sure = (scores[best] >= self.cfg.min_digit_score
                and scores[best] - scores[second] >= self.cfg.min_digit_margin)  # fmt: skip
        return int(best) if sure else None

    def _digits(self, patches: list[NDArray[np.float32]], dark: bool) -> list[int] | None:
        digits = [self._classify(255 - p if dark else p) for p in patches]
        known = [d for d in digits if d is not None]
        return known if len(known) == len(digits) else None

    def _number(self, patch: NDArray[np.float32]) -> int | None:
        """Gols de um time: um ou dois dígitos brancos, separados por componente."""
        text = (patch > (patch.min() + patch.max()) / 2).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(text, connectivity=8)
        tall = sorted(
            (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])
            for i in range(1, count)
            if stats[i, cv2.CC_STAT_HEIGHT] >= 0.4 * patch.shape[0]
        )
        if not tall or len(tall) > 2:
            return None
        if len(tall) == 1:
            digits = self._digits([patch], dark=False)
        else:
            cut = (tall[0][1] + tall[1][0]) // 2
            digits = self._digits([patch[:, :cut], patch[:, cut:]], dark=False)
        return None if digits is None else int("".join(map(str, digits)))


def _load(path: Path, flags: int) -> Image:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return np.asarray(image, np.uint8)
