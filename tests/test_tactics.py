import itertools
import time
from collections import Counter

import numpy as np
import pytest
import shapely

from config.settings import PITCH_SIZE_M
from src.models import EventType, FrameTelemetry, GameState, Player, Team
from src.simulation import RadarSimulator
from src.tactics import analyze, detect_events
from src.tactics.spatial_engine import voronoi_cells

LENGTH, WIDTH = PITCH_SIZE_M

type Spot = tuple[Team, float, float]  # (time, x em metros, y em metros)


def make_frame(
    spots: list[Spot],
    ball_m: tuple[float, float] | None = None,
    state: GameState = GameState.EM_ANDAMENTO,
) -> FrameTelemetry:
    return FrameTelemetry(
        frame_id=1,
        timestamp=0.0,
        state=state,
        players=tuple(
            Player(id=i, team=team, x=x / LENGTH, y=y / WIDTH)
            for i, (team, x, y) in enumerate(spots)
        ),
        ball=None if ball_m is None else (ball_m[0] / LENGTH, ball_m[1] / WIDTH),
    )


def defense(team: Team, x: float, cb_gap: float) -> list[Spot]:
    """Goleiro + linha de 4 com os zagueiros centrais separados por `cb_gap` metros."""
    goal_x = 2.0 if team is Team.A else LENGTH - 2.0
    half = cb_gap / 2
    return [
        (team, goal_x, 34.0),
        (team, x, 8.0),
        (team, x, 34.0 - half),
        (team, x, 34.0 + half),
        (team, x, 60.0),
    ]


def test_symmetric_teams_split_the_pitch() -> None:
    snapshot = analyze(make_frame([(Team.A, 26.25, 34.0), (Team.B, 78.75, 34.0)]))
    assert snapshot.control_a == pytest.approx(0.5)


def test_voronoi_cells_follow_input_order_and_survive_duplicates() -> None:
    rng = np.random.default_rng(7)
    xy = rng.random((22, 2)) * PITCH_SIZE_M
    xy[1] = xy[0]  # dois jogadores sobrepostos no radar
    cells = voronoi_cells(xy)
    assert len(cells) == 22
    points = shapely.points(xy)
    assert all(shapely.distance(cells, points) < 1e-3)


def test_team_shape_measures_center_back_gap_and_depth() -> None:
    spots = [*defense(Team.A, 20.0, cb_gap=24.0), (Team.A, 60.0, 34.0)]
    shape = analyze(make_frame(spots)).shapes[Team.A]
    assert shape.cb_gap_m == pytest.approx(24.0)
    assert shape.depth_m == pytest.approx(40.0)


def test_team_shape_without_keeper_on_the_radar_keeps_the_back_line() -> None:
    spots = [*defense(Team.A, 20.0, cb_gap=10.0)[1:], (Team.A, 60.0, 34.0)]
    shape = analyze(make_frame(spots)).shapes[Team.A]
    assert shape.cb_gap_m == pytest.approx(10.0)
    assert shape.depth_m == pytest.approx(40.0)


@pytest.mark.parametrize(("rival_y", "is_open"), [(34.0, False), (36.0, False), (50.0, True)])
def test_rival_inside_the_lane_blocks_the_pass(rival_y: float, is_open: bool) -> None:
    spots = [(Team.A, 40.0, 34.0), (Team.A, 60.0, 34.0), (Team.B, 50.0, rival_y)]
    snapshot = analyze(make_frame(spots, ball_m=(40.0, 34.0)))
    assert snapshot.carrier_id == 0
    assert snapshot.possession is Team.A
    assert any(lane.receiver_id == 1 for lane in snapshot.open_passes) is is_open


def test_receiver_behind_second_last_defender_is_offside() -> None:
    spots = [
        (Team.A, 50.0, 34.0),
        (Team.A, 85.0, 20.0),  # atrás da zaga: impedido
        (Team.A, 75.0, 55.0),  # antes da zaga: legal
        *defense(Team.B, 80.0, cb_gap=10.0),
    ]
    lanes = {
        lane.receiver_id: lane for lane in analyze(make_frame(spots, (50.0, 34.0))).open_passes
    }
    assert not lanes[1].onside
    assert lanes[2].onside


def test_exposed_center_only_counts_when_ball_is_in_own_half() -> None:
    spots = [*defense(Team.A, 20.0, cb_gap=25.0), (Team.B, 70.0, 34.0)]
    own_half = detect_events(frame := make_frame(spots, (30.0, 34.0)), analyze(frame))
    rival_half = detect_events(frame := make_frame(spots, (70.0, 34.0)), analyze(frame))
    assert [e.type for e in own_half] == [EventType.CENTRO_EXPOSTO]
    assert rival_half == []


def test_no_events_outside_live_play() -> None:
    spots = [*defense(Team.A, 20.0, cb_gap=25.0), (Team.A, 90.0, 34.0)]
    frame = make_frame(spots, (30.0, 34.0), state=GameState.REPLAY)
    assert detect_events(frame, analyze(frame)) == []


def test_open_forward_pass_raises_through_ball_event() -> None:
    midfield = [(Team.B, 50.0, 5.0), (Team.B, 50.0, 60.0)]  # longe da linha do passe
    spots = [(Team.A, 40.0, 34.0), (Team.A, 62.0, 20.0), *midfield, *defense(Team.B, 70.0, 12.0)]
    frame = make_frame(spots, (40.0, 34.0))
    [event] = detect_events(frame, analyze(frame))
    assert event.type is EventType.PASSE_RUPTURA
    assert event.target_id == 1
    assert event.value == pytest.approx(22.0)


def test_long_pass_that_breaks_no_line_is_not_a_through_ball() -> None:
    spots = [(Team.A, 5.0, 34.0), (Team.A, 25.0, 34.0), *defense(Team.B, 70.0, cb_gap=12.0)]
    [lane] = analyze(make_frame(spots, (5.0, 34.0))).open_passes
    assert lane.gain_m == pytest.approx(20.0)
    assert lane.bypassed == 0
    assert not lane.through


def test_simulator_triggers_every_event_type() -> None:
    counts: Counter[EventType] = Counter()
    for frame in itertools.islice(RadarSimulator(fps=10), 10 * 120):
        counts.update(event.type for event in detect_events(frame, analyze(frame)))
    assert set(counts) == set(EventType)


def test_analysis_fits_the_frame_budget() -> None:
    frames = list(itertools.islice(RadarSimulator(), 300))
    start = time.perf_counter()
    for frame in frames:
        detect_events(frame, analyze(frame))
    avg_ms = 1000 * (time.perf_counter() - start) / len(frames)
    assert avg_ms < 5.0, f"{avg_ms:.2f} ms por frame"
