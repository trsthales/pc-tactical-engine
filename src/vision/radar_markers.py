"""Marcadores do radar num frame: triângulos vermelhos, círculos azuis e a bola ("+").

Segmentação HSV dentro do campo e componentes conexos. Cada componente é comparado com a
área esperada de um marcador naquela profundidade (o radar é em perspectiva: embaixo os
marcadores são maiores):
- pedaço pequeno demais é ruído, exceto um pedaço de triângulo encostado num círculo azul
  ou na bola: o radar desenha esses por cima dos triângulos e, na marcação colada, do
  triângulo sobra só a ponta;
- componente grande é um marcador só ou vários fundidos:
  - círculos têm sempre o mesmo tamanho: a área diz quantos são (k-means separa);
  - triângulos não: o do jogador controlado tem ~2,3 vezes a área normal (e não tem a
    borda creme). Neles decide a forma: triângulos fundidos se ligam por "pescoços" finos
    que somem com uma erosão de 1 px, e cada pedaço que sobra é um marcador;
- mancha maior que 3 marcadores (camisa de jogador, banner de gol) é descartada;
- sem a borda clara do sprite é cena, não marcador (calção azul de um jogador 3D passando
  atrás do radar). Círculo sempre tem anel; triângulo, só fora do campo é exigido (dentro,
  o do jogador controlado e os pedaços cobertos não têm borda).

Posição de cada marcador = centro do sprite. No círculo, é o centroide. No triângulo
(apontando para cima), o centroide fica 1/6 da altura abaixo do centro do sprite: a
detecção corrige isso. Hipótese: o jogo centraliza o sprite na posição do jogador.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from config.settings import VisionConfig
from src.vision.radar_geometry import Image, Points, RadarReference

type Marker = tuple[float, float, float]  # (x, y, qualidade): maior = mais típico

_MAX_MARKERS_PER_BLOB = 3
_SPLIT_FROM_RATIO = 1.3  # a partir daqui o componente pode ser mais de um marcador
_MIN_PIECE_PX = 2  # pedaço que sobra da erosão e ainda conta como marcador
_MAX_BALL_AREA_RATIO = 3.5  # bola alta fica maior; mais que isso do esperado não é a bola
_RING_PX = 1.5  # anel claro em volta do círculo azul (e borda creme do triângulo)
_MIN_OCCLUDED_RATIO = 0.12  # pedaço visível de um triângulo coberto
_OCCLUDER_GAP_PX = 3.0  # distância extra entre o pedaço e a borda do que o cobre
_LIGHT_MAX_SAT, _LIGHT_MIN_VAL = 110, 150  # borda creme / anel branco
_MIN_RIM = 0.2  # fração clara em volta: sprites reais >= 0,32 (p0,1 no vídeo), cena < 0,2
_CROSS = np.ones((3, 3), np.uint8)
_RIM_KERNEL = np.ones((5, 5), np.uint8)


def _no_points() -> Points:
    return np.empty((0, 2))


@dataclass(frozen=True, slots=True)
class RadarDetections:
    """Posições em pixels do recorte de referência (já redimensionado)."""

    opacity: float
    similarity: float
    visible: bool
    red: Points = field(default_factory=_no_points)  # triângulos
    blue: Points = field(default_factory=_no_points)  # círculos
    ball_candidates: Points = field(default_factory=_no_points)  # maior primeiro


def _hue_mask(hue: NDArray[np.uint8], bounds: tuple[int, int]) -> NDArray[np.bool_]:
    low, high = bounds
    if low <= high:
        return (hue >= low) & (hue <= high)
    return (hue >= low) | (hue <= high)  # faixa que passa pelo zero (vermelho)


def expected_area(model: tuple[float, float], y: float) -> float:
    a, b = model
    return a + b * y


def sprite_height(cfg: VisionConfig, y: float) -> float:
    """Altura do sprite do marcador naquela profundidade: diâmetro do círculo + anel."""
    return 2 * float(np.sqrt(expected_area(cfg.blue_area, y) / np.pi)) + 2 * _RING_PX


def _near_occluder(x: float, y: float, occluders: Points, cfg: VisionConfig) -> bool:
    if not len(occluders):
        return False
    reach = sprite_height(cfg, y) + _OCCLUDER_GAP_PX  # raio do que cobre + meio triângulo
    return bool(np.min(np.hypot(occluders[:, 0] - x, occluders[:, 1] - y)) <= reach)


def _pieces(blob: NDArray[np.uint8]) -> list[NDArray[np.float64]]:
    """Centros dos marcadores num componente: um por pedaço que resiste à erosão."""
    core = cv2.erode(blob, _CROSS)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(core, connectivity=4)
    stats, centroids = np.asarray(stats), np.asarray(centroids, np.float64)
    return [centroids[j] for j in range(1, count) if stats[j, cv2.CC_STAT_AREA] >= _MIN_PIECE_PX]


def _kmeans_seeds(
    blob: NDArray[np.bool_], markers: int, x0: int, y0: int
) -> list[tuple[float, float]]:
    ys, xs = np.nonzero(blob)
    pixels = np.column_stack([xs + x0, ys + y0]).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    cv2.setRNGSeed(0)  # determinístico: mesmo frame, mesma divisão
    _, _, centers = cv2.kmeans(
        pixels, markers, np.empty(0, np.int32), criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    return [(float(cx), float(cy)) for cx, cy in np.asarray(centers, np.float64)]


def _rim(blob: NDArray[np.uint8], light: NDArray[np.bool_]) -> float:
    """Fração de pixels claros (borda do sprite) num anel de 2 px em volta do componente."""
    ring = cv2.dilate(blob, _RIM_KERNEL).astype(bool) & ~blob.astype(bool)
    return float(light[ring].mean()) if ring.any() else 0.0


def _split_blobs(
    mask: NDArray[np.bool_],
    model: tuple[float, float],
    cfg: VisionConfig,
    triangles: bool,
    rim_required: NDArray[np.bool_],
    light: NDArray[np.bool_],
    occluders: Points | None = None,
) -> list[Marker]:
    """`rim_required`: onde um componente só vale com a borda clara em volta."""
    count, raw_labels, raw_stats, raw_centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    labels = np.asarray(raw_labels, np.int32)
    stats = np.asarray(raw_stats, np.int32)
    centroids = np.asarray(raw_centroids, np.float64)
    found: list[Marker] = []
    for i in range(1, count):
        cx, cy = centroids[i]
        x, y, w, h, area = stats[i]
        ratio = area / expected_area(model, cy)
        if ratio < cfg.min_area_ratio:
            near = occluders is not None and _near_occluder(cx, cy, occluders, cfg)
            if triangles and near and ratio >= _MIN_OCCLUDED_RATIO:
                found.append((cx, cy, 0.0))  # posição aproximada: só um pedaço aparece
            continue
        if ratio > _MAX_MARKERS_PER_BLOB + 0.5:
            continue
        if rim_required[min(round(cy), mask.shape[0] - 1), min(round(cx), mask.shape[1] - 1)]:
            pad = 2  # o anel sai 2 px da caixa
            y0, x0 = max(y - pad, 0), max(x - pad, 0)
            window = (labels[y0 : y + h + pad, x0 : x + w + pad] == i).astype(np.uint8)
            if _rim(window, light[y0 : y + h + pad, x0 : x + w + pad]) < _MIN_RIM:
                continue
        quality = 1.0 - min(abs(ratio - round(ratio)), 0.5)
        seeds: list[tuple[float, float]] = []
        if triangles and ratio >= _SPLIT_FROM_RATIO:
            blob = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
            seeds = [(sx + x, sy + y) for sx, sy in _pieces(blob)][:_MAX_MARKERS_PER_BLOB]
        elif not triangles and round(ratio) >= 2:
            seeds = _kmeans_seeds(labels[y : y + h, x : x + w] == i, round(ratio), x, y)
        if len(seeds) <= 1:
            # marcador único; no triângulo o centro do sprite fica h/6 acima do centroide
            shift = max(h, sprite_height(cfg, cy) - 2 * _RING_PX) / 6 if triangles else 0.0
            found.append((cx, cy - shift, quality))
            continue
        ys, xs = np.nonzero(labels[y : y + h, x : x + w] == i)
        pixels = np.column_stack([xs + x, ys + y]).astype(np.float64)
        centers = np.array(seeds)
        owner = np.argmin(np.linalg.norm(pixels[:, None, :] - centers[None], axis=2), axis=1)
        for k in range(len(centers)):
            px, py = pixels[owner == k].mean(axis=0)
            shift = sprite_height(cfg, py) / 6 if triangles else 0.0
            found.append((float(px), float(py) - shift, quality))
    return found


def _strongest(found: list[Marker], limit: int) -> Points:
    """No máximo `limit` marcadores (um time tem 11): fica com os de área mais típica."""
    best = sorted(found, key=lambda m: -m[2])[:limit]
    return np.array([(x, y) for x, y, _ in best]).reshape(-1, 2)


def detect_markers(
    crop: Image, ref: RadarReference, cfg: VisionConfig, was_visible: bool = False
) -> RadarDetections:
    """`was_visible`: o radar estava na tela no frame anterior (aplica a histerese)."""
    width, height = ref.size
    if crop.shape[1] != width or crop.shape[0] != height:
        crop = np.asarray(
            cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR), np.uint8
        )
    opacity, similarity = ref.opacity(crop), ref.similarity(crop)
    if was_visible:
        visible = opacity >= cfg.keep_radar_opacity and similarity >= cfg.keep_radar_similarity
    else:
        visible = opacity >= cfg.min_radar_opacity and similarity >= cfg.min_radar_similarity
    if not visible:
        return RadarDetections(opacity, similarity, visible=False)

    hsv = np.asarray(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), np.uint8)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside = ref.search_mask.astype(bool)
    vivid = inside & (sat >= cfg.min_saturation) & (val >= cfg.min_value)
    light = (sat < _LIGHT_MAX_SAT) & (val > _LIGHT_MIN_VAL)
    blue_mask = vivid & _hue_mask(hue, cfg.blue_hue)
    everywhere = np.ones_like(inside)
    blue = _strongest(
        _split_blobs(blue_mask, cfg.blue_area, cfg, False, everywhere, light),
        cfg.players_per_team,
    )

    ball_mask = (inside & _hue_mask(hue, cfg.ball_hue)
                 & (sat >= cfg.min_ball_saturation) & (val >= cfg.min_value))  # fmt: skip
    count, _, raw_stats, raw_centroids = cv2.connectedComponentsWithStats(
        ball_mask.astype(np.uint8), connectivity=8
    )
    stats, centroids = np.asarray(raw_stats), np.asarray(raw_centroids, np.float64)
    balls = sorted(
        (
            (int(stats[i, cv2.CC_STAT_AREA]), float(centroids[i][0]), float(centroids[i][1]))
            for i in range(1, count)
            if cfg.min_ball_area
            <= stats[i, cv2.CC_STAT_AREA]
            <= _MAX_BALL_AREA_RATIO * expected_area(cfg.blue_area, centroids[i][1])
        ),
        reverse=True,
    )
    ball = np.array([(x, y) for _, x, y in balls]).reshape(-1, 2)

    red_mask = vivid & _hue_mask(hue, cfg.red_hue)
    covers = np.vstack([blue, ball[:1]])  # desenhados por cima dos triângulos
    red = _strongest(
        _split_blobs(red_mask, cfg.red_area, cfg, True, ~ref.pitch_mask, light, covers),
        cfg.players_per_team,
    )
    return RadarDetections(opacity, similarity, True, red, blue, ball)
