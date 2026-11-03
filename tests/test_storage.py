import itertools
import time
from pathlib import Path

import duckdb
import pytest

from src.models import FrameTelemetry, GameState, Player, Team
from src.simulation import RadarSimulator
from src.storage import TelemetryWriter
from src.storage.post_match_reports import export, heatmap, latest_match, open_db, summarize
from src.tactics import analyze, detect_events


@pytest.fixture(scope="module")
def recorded_match(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    """~5 min de partida simulada a 60 FPS: 18 mil frames, mais de 400 mil linhas."""
    path = tmp_path_factory.mktemp("db") / "telemetry.duckdb"
    frames = 18_000
    with TelemetryWriter(path, match_id="sim", max_pending=0) as writer:  # lote sem ritmo
        for frame in itertools.islice(RadarSimulator(), frames):
            if frame.frame_id % 6 == 0:  # análise completa só em parte dos frames: teste rápido
                snapshot = analyze(frame)
                writer.submit(frame, snapshot, detect_events(frame, snapshot))
            else:
                writer.submit(frame)
    return path, frames


def test_every_frame_and_entity_is_stored(recorded_match: tuple[Path, int]) -> None:
    path, frames = recorded_match
    with open_db(path) as conn:
        assert conn.execute("SELECT count(*) FROM frames").fetchone() == (frames,)
        assert conn.execute("SELECT count(*) FROM telemetry").fetchone() == (frames * 23,)
        assert latest_match(conn) == "sim"


def test_summary_and_heatmap(recorded_match: tuple[Path, int]) -> None:
    path, frames = recorded_match
    with open_db(path) as conn:
        summary = summarize(conn, "sim")
        grid = heatmap(conn, "sim", Team.A)
        (live,) = conn.execute(
            "SELECT count(*) FROM frames WHERE state = 'EM_ANDAMENTO'"
        ).fetchone() or (0,)
    assert summary.frames == frames
    assert summary.duration_s == pytest.approx((frames - 1) / 60)
    assert 0.3 < (summary.control_a or 0) < 0.7
    assert summary.events  # o simulador gera alertas
    assert 0 < live < frames  # o simulador tem bola parada
    assert grid.sum() == live * 11  # heatmap só conta bola rolando


def test_queries_fit_the_budget(recorded_match: tuple[Path, int]) -> None:
    path, _ = recorded_match
    with open_db(path) as conn:
        for query in (lambda: summarize(conn, "sim"), lambda: heatmap(conn, "sim", Team.B)):
            query()  # aquece o cache
            start = time.perf_counter()
            query()
            assert (time.perf_counter() - start) * 1000 < 50


def test_export_parquet_and_csv(recorded_match: tuple[Path, int], tmp_path: Path) -> None:
    path, frames = recorded_match
    with open_db(path) as conn:
        parquet = export(conn, "sim", tmp_path, "parquet")
        csv = export(conn, "sim", tmp_path, "csv")
    assert [p.name for p in parquet] == [
        "sim_frames.parquet",
        "sim_telemetry.parquet",
        "sim_events.parquet",
    ]
    assert duckdb.execute("SELECT count(*) FROM read_parquet(?)", [str(parquet[0])]).fetchone() == (
        frames,
    )
    assert all(p.stat().st_size > 0 for p in csv)


def test_writer_flushes_while_running(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "t.duckdb", flush_every_s=0.05)
    writer.start()
    for frame in itertools.islice(RadarSimulator(), 10):
        writer.submit(frame)
    time.sleep(0.3)
    flushed_before_close = writer.rows_written
    writer.close()
    assert flushed_before_close == 10 * 24  # 10 frames + 230 posições


def _frame(tick: int, x: float, y: float, attacks_right: bool) -> FrameTelemetry:
    return FrameTelemetry(
        frame_id=tick,
        timestamp=tick / 60,
        state=GameState.EM_ANDAMENTO,
        players=(Player(id=1, team=Team.A, x=x, y=y),),
        team_a_attacks_right=attacks_right,
    )


def test_heatmap_turns_the_second_half_around(tmp_path: Path) -> None:
    path = tmp_path / "t.duckdb"
    with TelemetryWriter(path, match_id="m") as writer:
        writer.submit(_frame(1, 0.15, 0.25, attacks_right=True))  # 1º tempo, perto do próprio gol
        writer.submit(_frame(2, 0.85, 0.75, attacks_right=False))  # 2º tempo, mesmo lugar do time
    with open_db(path) as conn:
        grid = heatmap(conn, "m", Team.A, bins=(10, 10))
    assert grid[2, 1] == 2  # as duas posições caem na mesma célula
    assert grid.sum() == 2


def test_writer_drops_frames_instead_of_growing_without_bound(tmp_path: Path) -> None:
    writer = TelemetryWriter(tmp_path / "t.duckdb", max_pending=5)  # thread não iniciada
    for frame in itertools.islice(RadarSimulator(), 8):
        writer.submit(frame)
    assert writer.dropped == 3
    writer.close()  # sem start(): não pode levantar


def test_writer_drops_a_batch_that_keeps_failing(tmp_path: Path) -> None:
    path = tmp_path / "t.duckdb"
    with TelemetryWriter(path, flush_every_s=0.02) as writer:
        writer._frames.sql = "INSERT INTO tabela_que_nao_existe VALUES (1)"  # flush sempre falha
        for frame in itertools.islice(RadarSimulator(), 10):
            writer.submit(frame)
        deadline = time.perf_counter() + 2.0
        while writer.dropped < 10 and time.perf_counter() < deadline:
            time.sleep(0.02)
        assert writer.dropped == 10
        assert len(writer._frames) == 0  # buffer liberado
        writer.submit(next(iter(RadarSimulator())))  # a thread segue viva
        time.sleep(0.1)
        assert writer._thread.is_alive()
    writer.submit(next(iter(RadarSimulator())))  # depois do close: ignorado, não enche a fila
    assert writer._queue.empty()
