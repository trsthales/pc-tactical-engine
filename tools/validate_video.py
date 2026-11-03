"""Roda a visão num vídeo inteiro e confere invariantes que não dependem de gabarito.

    python -m tools.validate_video partida.mp4 [--every 1] [--save resultado.npz]

O que é conferido (tudo vem do próprio jogo, sem anotação manual):
- contagem: com o jogo rolando, quantos frames têm 11 jogadores de cada time;
- física: nenhum jogador rastreado anda mais rápido que um velocista (~12 m/s), medido em
  janelas de 0,25 s (de um frame para o outro, 1 px de ruído já daria 24 m/s);
- relógio: só anda para a frente, exceto a volta para 45:00 no início do 2º tempo;
- placar: só sobe, de 1 em 1;
- lado de ataque: troca uma vez, no intervalo;
- estados: quanto tempo em cada um, trocas e trechos curtos demais (pisca-pisca);
- tempo de processamento por frame (meta do Marco 2: radar < 3 ms).
"""

import argparse
import itertools
import time
from collections import Counter
from pathlib import Path

import numpy as np

from config.settings import PITCH_SIZE_M, ROIConfig
from src.models import FrameTelemetry, GameState, Team
from src.vision.pipeline import TeamSetup, VisionPipeline
from src.vision.video_source import VideoSource

_SPRINT_M_S = 12.0
_WINDOW_S = 0.25
_SECOND_HALF_START = 45
_FLICKER_S = 0.5


def _speeds(frames: list[FrameTelemetry]) -> np.ndarray:
    """Velocidade de cada jogador entre frames ~0,25 s distantes, com o jogo rolando."""
    scale = np.array(PITCH_SIZE_M)
    speeds: list[float] = []
    j = 0
    for i, start in enumerate(frames):
        j = max(j, i)
        while j + 1 < len(frames) and frames[j + 1].timestamp - start.timestamp <= _WINDOW_S:
            j += 1
        end = frames[j]
        dt = end.timestamp - start.timestamp
        live = all(f.state is GameState.EM_ANDAMENTO for f in frames[i : j + 1])
        if dt < 0.8 * _WINDOW_S or not live:
            continue
        before = {p.id: np.array([p.x, p.y]) for p in start.players}
        for p in end.players:
            if p.id in before:
                moved = (np.array([p.x, p.y]) - before[p.id]) * scale
                speeds.append(float(np.linalg.norm(moved) / dt))
    return np.array(speeds)


def _report(frames: list[FrameTelemetry], spent: list[float], seconds: float) -> None:
    print(f"Frames: {len(frames)} ({seconds:.0f} s de vídeo)")
    print(f"Tempo por frame: média {1000 * np.mean(spent):.2f} ms, "
          f"p99 {1000 * np.percentile(spent, 99):.2f} ms")  # fmt: skip

    states = Counter(f.state for f in frames)
    print("Estados:", ", ".join(f"{s.value} {100 * n / len(frames):.1f}%"
                                for s, n in states.most_common()))  # fmt: skip
    runs = [
        (state, len(list(group))) for state, group in itertools.groupby(f.state for f in frames)
    ]
    step = (frames[-1].timestamp - frames[0].timestamp) / max(len(frames) - 1, 1)
    short = sum(n * step < _FLICKER_S for _, n in runs[1:-1])
    print(f"Trocas de estado: {len(runs) - 1}, trechos com menos de {_FLICKER_S} s: {short}")

    live = [f for f in frames if f.state is GameState.EM_ANDAMENTO]
    per_team = Counter(
        (sum(p.team is Team.A for p in f.players), sum(p.team is Team.B for p in f.players))
        for f in live
    )
    full = per_team[(11, 11)]
    print(f"Jogo rolando com 11 x 11 rastreados: {100 * full / max(len(live), 1):.1f}%")
    print("  (A, B):", ", ".join(f"{k}: {v}" for k, v in per_team.most_common(6)))
    with_ball = sum(f.ball is not None for f in live)
    print(f"Bola localizada com o jogo rolando: {100 * with_ball / max(len(live), 1):.1f}%")

    speeds = _speeds(frames)
    if len(speeds):
        p50, p99, p999 = np.percentile(speeds, [50, 99, 99.9])
        fast = 100 * float((speeds > _SPRINT_M_S).mean())
        print(f"Velocidade em {_WINDOW_S} s: mediana {p50:.1f}, p99 {p99:.1f}, "
              f"p99,9 {p999:.1f} m/s; acima de {_SPRINT_M_S:.0f} m/s: {fast:.2f}%")  # fmt: skip

    clocks = [(f.timestamp, f.minute) for f in frames if f.minute is not None]
    back = [
        (round(t), m0, m1)
        for (_, m0), (t, m1) in itertools.pairwise(clocks)
        if m1 < m0 and not (m1 == _SECOND_HALF_START and m0 >= _SECOND_HALF_START)
    ]
    print(f"Relógio voltou (fora o início do 2º tempo): {len(back)} vez(es) {back[:5]}")

    scores = [f.score for f in frames if f.score is not None]
    changes = [(a, b) for a, b in itertools.pairwise(scores) if a != b]
    bad = [(a, b) for a, b in changes if sum(b) - sum(a) != 1 or b[0] < a[0] or b[1] < a[1]]
    final = scores[-1] if scores else None
    print(f"Placar: {len(changes)} gol(s), final {final}, mudanças inválidas: {bad}")

    sides = [(f.timestamp, f.team_a_attacks_right) for f in live]
    flips = [round(t) for (_, a), (t, b) in itertools.pairwise(sides) if a != b]
    first = "direita" if sides and sides[0][1] else "esquerda"
    print(f"Lado de ataque do time A: começa para a {first}, trocas em {flips} s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida a visão num vídeo de partida.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--every", type=int, default=1, help="processa 1 a cada N frames")
    parser.add_argument("--team-a-blue", action="store_true", help="usuário = círculos azuis")
    parser.add_argument("--team-a-bottom", action="store_true", help="usuário = linha de baixo")
    parser.add_argument("--save", type=Path, help="salva posições e estados em .npz")
    args = parser.parse_args()

    source = VideoSource(args.video, ROIConfig.load())
    teams = TeamSetup(a_is_red=not args.team_a_blue, a_on_top=not args.team_a_bottom)
    pipeline = VisionPipeline(teams=teams)
    frames: list[FrameTelemetry] = []
    spent: list[float] = []
    for i, captured in enumerate(source):
        if i % args.every:
            continue
        start = time.perf_counter()
        frames.append(pipeline.process(captured))
        spent.append(time.perf_counter() - start)
    if not frames:
        raise SystemExit("Nenhum frame lido.")
    _report(frames, spent, frames[-1].timestamp - frames[0].timestamp)

    if args.save:
        padding = [(-1, np.nan, np.nan)]
        np.savez_compressed(
            args.save,
            t=[f.timestamp for f in frames],
            state=[f.state.value for f in frames],
            minute=[-1 if f.minute is None else f.minute for f in frames],
            score=[(-1, -1) if f.score is None else f.score for f in frames],
            right=[f.team_a_attacks_right for f in frames],
            ball=[(np.nan, np.nan) if f.ball is None else f.ball for f in frames],
            players=np.array(
                [
                    [(p.id, p.x, p.y) for p in f.players] + padding * (22 - len(f.players))
                    for f in frames
                ]
            ),
        )


if __name__ == "__main__":
    main()
