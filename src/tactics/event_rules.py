"""Regras que transformam as métricas do frame em eventos táticos para o time do usuário (A)."""

from config.settings import TacticalThresholds
from src.models import (
    EventType,
    FrameTelemetry,
    GameState,
    PassLane,
    TacticalEvent,
    TacticalSnapshot,
    Team,
)
from src.tactics.spatial_engine import LENGTH_M, progress_m

_DEFAULT = TacticalThresholds()


def _ball_in_own_half(frame: FrameTelemetry, team: Team) -> bool:
    if frame.ball is None:
        return False
    return progress_m(frame.ball[0] * LENGTH_M, frame.attacks_right(team)) < LENGTH_M / 2


def best_through_pass(snapshot: TacticalSnapshot) -> PassLane | None:
    """O passe de ruptura livre que mais avança."""
    candidates = [lane for lane in snapshot.open_passes if lane.through]
    return max(candidates, key=lambda lane: lane.gain_m, default=None)


def detect_events(
    frame: FrameTelemetry, snapshot: TacticalSnapshot, th: TacticalThresholds = _DEFAULT
) -> list[TacticalEvent]:
    if frame.state is not GameState.EM_ANDAMENTO:
        return []

    def event(kind: EventType, value: float, target_id: int | None = None) -> TacticalEvent:
        return TacticalEvent(
            type=kind,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            value=value,
            target_id=target_id,
        )

    shape = snapshot.shapes[Team.A]
    events: list[TacticalEvent] = []
    if shape.cb_gap_m > th.max_cb_gap_m and _ball_in_own_half(frame, Team.A):
        events.append(event(EventType.CENTRO_EXPOSTO, shape.cb_gap_m))
    if shape.depth_m > th.max_depth_m:
        events.append(event(EventType.TIME_ESPACADO, shape.depth_m))
    if snapshot.possession is Team.A and (lane := best_through_pass(snapshot)):
        events.append(event(EventType.PASSE_RUPTURA, lane.gain_m, lane.receiver_id))
    return events
