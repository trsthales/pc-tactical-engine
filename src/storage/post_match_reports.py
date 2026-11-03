"""Relatórios pós-jogo sobre o DuckDB: resumo, mapa de calor e exportação.

python -m src.storage.post_match_reports                       # última partida
python -m src.storage.post_match_reports --match ID --export parquet
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import duckdb
import numpy as np
from numpy.typing import NDArray

from config.settings import DATA_DIR
from src.models import GameState, Team
from src.storage.telemetry_db import DB_FILE

type ExportFormat = Literal["parquet", "csv"]

TABLES = ("frames", "telemetry", "events")
SHADES = " .:-=+*#%@"


@dataclass(frozen=True, slots=True)
class MatchSummary:
    match_id: str
    frames: int
    duration_s: float
    control_a: float | None  # média da fração do campo dominada pelo time A (bola rolando)
    possession_a: float | None  # fração dos frames com posse definida em que a bola era do A
    events: dict[str, int]


def open_db(path: Path = DB_FILE) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True)


def latest_match(conn: duckdb.DuckDBPyConnection) -> str | None:
    row = conn.execute("SELECT match_id FROM matches ORDER BY started_at DESC LIMIT 1").fetchone()
    return None if row is None else str(row[0])


def summarize(conn: duckdb.DuckDBPyConnection, match_id: str) -> MatchSummary:
    # Replay, pausa e bola parada não contam para domínio e posse.
    row = conn.execute(
        """
        SELECT count(*), coalesce(max(ts) - min(ts), 0),
               avg(control_a) FILTER (state = $live),
               avg(CASE possession WHEN 'A' THEN 1.0 WHEN 'B' THEN 0.0 END) FILTER (state = $live)
        FROM frames WHERE match_id = $match
        """,
        {"match": match_id, "live": GameState.EM_ANDAMENTO},
    ).fetchone()
    assert row is not None  # count(*) sempre devolve uma linha
    frames, duration_s, control_a, possession_a = row
    events = conn.execute(
        "SELECT type, count(*) FROM events WHERE match_id = ? GROUP BY type ORDER BY 2 DESC",
        [match_id],
    ).fetchall()
    return MatchSummary(match_id, frames, duration_s, control_a, possession_a, dict(events))


def heatmap(
    conn: duckdb.DuckDBPyConnection, match_id: str, team: Team, bins: tuple[int, int] = (21, 14)
) -> NDArray[np.int64]:
    """Presença por célula do campo com o time sempre atacando para a direita.

    `grid[linha_y, coluna_x]`. Só conta bola rolando. Quando o time ataca para a esquerda
    (troca de lado no intervalo), as posições são giradas 180°, que é o que a troca faz.
    """
    nx, ny = bins
    cells = conn.execute(
        """
        WITH pos AS (
            SELECT CASE WHEN f.a_attacks_right = ($team = 'A') THEN t.x ELSE 1 - t.x END AS x,
                   CASE WHEN f.a_attacks_right = ($team = 'A') THEN t.y ELSE 1 - t.y END AS y
            FROM telemetry t
            JOIN frames f USING (match_id, tick_id)
            WHERE t.match_id = $match AND t.team = $team AND f.state = $live
        )
        SELECT least(floor(x * $nx), $nx - 1)::INTEGER AS i,
               least(floor(y * $ny), $ny - 1)::INTEGER AS j,
               count(*)
        FROM pos GROUP BY i, j
        """,
        {"nx": nx, "ny": ny, "match": match_id, "team": team, "live": GameState.EM_ANDAMENTO},
    ).fetchall()
    grid = np.zeros((ny, nx), dtype=np.int64)
    for i, j, count in cells:
        grid[j, i] = count
    return grid


def export(
    conn: duckdb.DuckDBPyConnection, match_id: str, out_dir: Path, fmt: ExportFormat = "parquet"
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for table in TABLES:
        path = out_dir / Path(f"{match_id}_{table}.{fmt}").name  # `--match ../x` fica em out_dir
        relation = conn.sql(
            f"SELECT * FROM {table} WHERE match_id = $match ORDER BY tick_id",
            params={"match": match_id},
        )
        match fmt:
            case "parquet":
                relation.write_parquet(str(path))
            case "csv":
                relation.write_csv(str(path))
        paths.append(path)
    return paths


def render_heatmap(grid: NDArray[np.int64]) -> str:
    levels = np.rint(grid / max(int(grid.max()), 1) * (len(SHADES) - 1)).astype(int)
    border = "+" + "-" * grid.shape[1] + "+"
    rows = ("|" + "".join(SHADES[level] for level in row) + "|" for row in levels)
    return "\n".join([border, *rows, border])


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatório pós-jogo.")
    parser.add_argument("--db", type=Path, default=DB_FILE)
    parser.add_argument("--match", help="match_id (padrão: a partida mais recente)")
    parser.add_argument("--export", choices=get_args(ExportFormat.__value__))
    parser.add_argument("--out", type=Path, default=DATA_DIR / "exports")
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"Banco não encontrado: {args.db}. Grave uma partida primeiro.")
    with open_db(args.db) as conn:
        match_id = args.match or latest_match(conn)
        if match_id is None:
            raise SystemExit("Nenhuma partida gravada ainda.")
        summary = summarize(conn, match_id)
        print(f"Partida {summary.match_id}: {summary.frames} frames, {summary.duration_s:.0f} s")
        print(f"Domínio territorial do time A: {_percent(summary.control_a)}")
        print(f"Posse do time A: {_percent(summary.possession_a)}")
        for kind, count in summary.events.items():
            print(f"  {kind}: {count}")
        print("\nMapa de calor do time A (atacando para a direita, bola rolando):")
        print(render_heatmap(heatmap(conn, match_id, Team.A)))
        if args.export:
            for path in export(conn, match_id, args.out, args.export):
                print(f"Exportado: {path}")


if __name__ == "__main__":
    main()
