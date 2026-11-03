"""Geometria tática por frame: dominância territorial, forma dos times e linhas de passe.

Tudo é calculado em metros (o radar normalizado distorce distâncias, já que o campo não é
quadrado) e vetorizado com NumPy/Shapely.
"""

import numpy as np
import shapely
from numpy.typing import NDArray

from config.settings import PITCH_SIZE_M, TacticalThresholds
from src.models import FrameTelemetry, PassLane, Polygon, TacticalSnapshot, Team, TeamShape

type Points = NDArray[np.float64]  # (n, 2), em metros
type Mask = NDArray[np.bool_]

LENGTH_M, WIDTH_M = PITCH_SIZE_M
_SCALE = np.array(PITCH_SIZE_M)
_PITCH = shapely.box(0.0, 0.0, LENGTH_M, WIDTH_M)
_DEFAULT = TacticalThresholds()
_KEEPER_LEAD_M = 3.0  # o goleiro fica atrás da linha; sem essa folga, é um zagueiro


def progress_m[T: (float, NDArray[np.float64])](x_m: T, attacks_right: bool) -> T:
    """Distância até a própria linha de fundo: cresce rumo ao gol adversário."""
    return x_m if attacks_right else LENGTH_M - x_m


def voronoi_cells(xy: Points) -> NDArray[np.object_]:
    """Célula de Voronoi de cada ponto, recortada no campo e na mesma ordem da entrada."""
    if len(xy) < 2:
        return np.array([_PITCH] * len(xy), dtype=object)
    # O GEOS rejeita pontos repetidos (dois jogadores sobrepostos no radar): desempata em µm.
    jitter = np.arange(len(xy))[:, None] * 1e-6
    diagram = shapely.voronoi_polygons(
        shapely.multipoints(xy + jitter), extend_to=_PITCH, ordered=True
    )
    return shapely.intersection(shapely.get_parts(diagram), _PITCH)  # type: ignore[no-any-return]


def team_shape(xy: Points, attacks_right: bool) -> TeamShape:
    """Profundidade do bloco e espaço entre os zagueiros centrais.

    Goleiro é o jogador mais recuado, se estiver a pelo menos `_KEEPER_LEAD_M` atrás do
    seguinte; senão (goleiro fora do radar), todos são de linha. Zaga = os 4 mais recuados.
    """
    if len(xy) < 3:
        return TeamShape(depth_m=0.0, cb_gap_m=0.0)
    prog = progress_m(xy[:, 0], attacks_right)
    order = np.argsort(prog)
    has_keeper = prog[order[1]] - prog[order[0]] >= _KEEPER_LEAD_M
    outfield = order[1:] if has_keeper else order
    back_line = outfield[:4]
    center_backs = back_line[np.argsort(np.abs(xy[back_line, 1] - WIDTH_M / 2))[:2]]
    return TeamShape(
        depth_m=float(np.ptp(prog[outfield])),
        cb_gap_m=float(np.linalg.norm(xy[center_backs[0]] - xy[center_backs[1]])),
    )


def find_carrier(xy: Points, ball_m: Points | None, radius_m: float) -> int | None:
    """Índice do jogador mais perto da bola, se estiver dentro do raio de posse."""
    if ball_m is None or not len(xy):
        return None
    distances = np.linalg.norm(xy - ball_m, axis=1)
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] <= radius_m else None


def open_passes(
    xy: Points,
    ids: NDArray[np.int64],
    teammates: Mask,
    passer: int,
    attacks_right: bool,
    th: TacticalThresholds = _DEFAULT,
) -> tuple[PassLane, ...]:
    """Passes do portador sem marcador no cone de interceptação.

    Um adversário bloqueia o passe se estiver a menos de `lane_radius_m + lane_growth * d`
    do segmento, onde `d` é a distância do ponto mais próximo do segmento ao passador.
    """
    receivers = np.flatnonzero(teammates)
    receivers = receivers[receivers != passer]
    vectors = xy[receivers] - xy[passer]
    lengths = np.linalg.norm(vectors, axis=1)
    in_range = (lengths > 0) & (lengths <= th.max_pass_m)
    receivers, vectors, lengths = receivers[in_range], vectors[in_range], lengths[in_range]
    if not len(receivers):
        return ()

    rivals = xy[~teammates] - xy[passer]  # (k, 2), relativo ao passador
    along = np.clip(rivals @ vectors.T / lengths**2, 0.0, 1.0)  # (k, m): projeção no segmento
    gaps = np.linalg.norm(rivals[:, None, :] - along[..., None] * vectors, axis=2)
    blocked = (gaps < th.lane_radius_m + th.lane_growth * along * lengths).any(axis=0)

    prog = progress_m(xy[:, 0], attacks_right)
    rival_prog = np.sort(prog[~teammates])
    second_last = rival_prog[-2] if len(rival_prog) >= 2 else LENGTH_M
    offside_line = max(LENGTH_M / 2, second_last, prog[passer])  # a bola está com o passador

    gains = prog[receivers] - prog[passer]
    onside = prog[receivers] <= offside_line
    behind_passer = np.searchsorted(rival_prog, prog[passer], side="right")
    bypassed = np.maximum(np.searchsorted(rival_prog, prog[receivers]) - behind_passer, 0)
    through = onside & (gains >= th.min_pass_gain_m) & (bypassed >= th.min_bypassed)

    return tuple(
        PassLane(
            passer_id=int(ids[passer]),
            receiver_id=int(ids[receivers[i]]),
            length_m=float(lengths[i]),
            gain_m=float(gains[i]),
            bypassed=int(bypassed[i]),
            onside=bool(onside[i]),
            through=bool(through[i]),
        )
        for i in np.flatnonzero(~blocked)
    )


def _to_unit_polygon(cell: shapely.Polygon) -> Polygon:
    coords = np.round(shapely.get_coordinates(cell.exterior) / _SCALE, 4)
    return tuple((float(x), float(y)) for x, y in coords)


def analyze(frame: FrameTelemetry, th: TacticalThresholds = _DEFAULT) -> TacticalSnapshot:
    players = frame.players
    xy = np.array([(p.x, p.y) for p in players], dtype=np.float64).reshape(-1, 2) * _SCALE
    ids = np.array([p.id for p in players], dtype=np.int64)
    is_a = np.array([p.team is Team.A for p in players], dtype=np.bool_)

    cells = voronoi_cells(xy)
    control_a = float(shapely.area(cells[is_a]).sum() / _PITCH.area) if len(cells) else 0.5

    ball_m = None if frame.ball is None else np.array(frame.ball) * _SCALE
    carrier = find_carrier(xy, ball_m, th.carrier_radius_m)
    carrier_id: int | None = None
    possession: Team | None = None
    passes: tuple[PassLane, ...] = ()
    if carrier is not None:
        owner = players[carrier]
        carrier_id, possession = owner.id, owner.team
        attacks_right = frame.attacks_right(owner.team)
        passes = open_passes(xy, ids, is_a == is_a[carrier], carrier, attacks_right, th)

    return TacticalSnapshot(
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        control_a=control_a,
        regions=tuple(_to_unit_polygon(cell) for cell in cells),
        shapes={
            Team.A: team_shape(xy[is_a], frame.attacks_right(Team.A)),
            Team.B: team_shape(xy[~is_a], frame.attacks_right(Team.B)),
        },
        carrier_id=carrier_id,
        possession=possession,
        open_passes=passes,
    )
