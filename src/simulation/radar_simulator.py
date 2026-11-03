"""Radar sintético para rodar e testar o sistema sem o jogo (até a visão do Marco 2 existir).

Dois times em 4-4-2 deslizam com a bola, que circula em passes e às vezes muda de dono.
A zaga e o ataque do time A "respiram" de propósito, para disparar os alertas táticos.
Simplificação: o time A ataca sempre para a direita (sem troca de lado no intervalo).
"""

import itertools
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass

from config.settings import PITCH_SIZE_M, TARGET_FPS
from src.models import FrameTelemetry, GameState, Player, Team

type XY = tuple[float, float]

# Time atacando para a direita, coordenadas normalizadas.
FORMATION_442: tuple[XY, ...] = (
    (0.04, 0.50),  # goleiro
    (0.20, 0.15), (0.18, 0.40), (0.18, 0.60), (0.20, 0.85),  # zaga
    (0.38, 0.15), (0.35, 0.40), (0.35, 0.60), (0.38, 0.85),  # meio
    (0.50, 0.40), (0.50, 0.60),  # ataque
)  # fmt: skip
CENTER_BACKS, STRIKERS = (2, 3), (9, 10)
PLAYERS_PER_TEAM = len(FORMATION_442)
BALL_SPEED_M_S = 20.0
SET_PIECE_EVERY_S, SET_PIECE_LENGTH_S = 75.0, 5.0


@dataclass(frozen=True, slots=True)
class _Pass:
    origin: XY
    receiver: int
    started: float
    duration: float


def _clamp(value: float) -> float:
    return min(max(value, 0.01), 0.99)


def _distance_m(a: XY, b: XY) -> float:
    return math.hypot((a[0] - b[0]) * PITCH_SIZE_M[0], (a[1] - b[1]) * PITCH_SIZE_M[1])


def _team(index: int) -> Team:
    return Team.A if index < PLAYERS_PER_TEAM else Team.B


class RadarSimulator:
    """Iterável infinito de `FrameTelemetry` com relógio próprio (`frame_id / fps`)."""

    def __init__(self, fps: int = TARGET_FPS, seed: int = 0) -> None:
        self.fps = fps
        self.seed = seed

    def _positions(self, t: float, ball: XY, phases: list[float]) -> list[XY]:
        breathe = max(0.0, math.sin(t / 9))  # abre os zagueiros do time A
        stretch = max(0.0, math.sin(t / 13))  # adianta os atacantes do time A
        shift_x, shift_y = 0.3 * (ball[0] - 0.5), 0.15 * (ball[1] - 0.5)
        positions: list[XY] = []
        for index, phase in enumerate(phases):
            slot = index % PLAYERS_PER_TEAM
            x, y = FORMATION_442[slot]
            if _team(index) is Team.A:
                if slot in CENTER_BACKS:
                    y += 0.12 * breathe * (1 if y > 0.5 else -1)
                if slot in STRIKERS:
                    x += 0.15 * stretch
            else:
                x = 1 - x  # time B defende o gol da direita
            if slot:  # goleiro não acompanha o bloco
                x, y = x + shift_x, y + shift_y
            wobble = 0.02 * math.sin(1.3 * t + phase), 0.02 * math.cos(0.9 * t + phase)
            positions.append((_clamp(x + wobble[0]), _clamp(y + wobble[1])))
        return positions

    def _next_receiver(self, rng: random.Random, holder: int, positions: list[XY]) -> int:
        """80% passe para companheiro, 20% perde para o rival mais próximo."""
        mates = [i for i in range(len(positions)) if _team(i) is _team(holder) and i != holder]
        rivals = [i for i in range(len(positions)) if _team(i) is not _team(holder)]
        if rng.random() < 0.2:
            return min(rivals, key=lambda i: _distance_m(positions[i], positions[holder]))
        near = [i for i in mates if _distance_m(positions[i], positions[holder]) < 35]
        return rng.choice(near or mates)

    def __iter__(self) -> Iterator[FrameTelemetry]:
        rng = random.Random(self.seed)
        phases = [rng.uniform(0, 2 * math.pi) for _ in range(2 * PLAYERS_PER_TEAM)]
        holder, next_pass_at = 6, 2.0
        flight: _Pass | None = None
        ball: XY = FORMATION_442[holder]
        previous: list[XY] | None = None

        for frame_id in itertools.count():
            t = frame_id / self.fps
            positions = self._positions(t, ball, phases)

            if flight is not None:
                target = positions[flight.receiver]
                done = min((t - flight.started) / flight.duration, 1.0)
                ball = (
                    flight.origin[0] + done * (target[0] - flight.origin[0]),
                    flight.origin[1] + done * (target[1] - flight.origin[1]),
                )
                if done >= 1.0:
                    holder, flight = flight.receiver, None
                    next_pass_at = t + rng.uniform(1.5, 3.0)
            elif t >= next_pass_at:
                receiver = self._next_receiver(rng, holder, positions)
                duration = max(0.3, _distance_m(ball, positions[receiver]) / BALL_SPEED_M_S)
                flight = _Pass(ball, receiver, t, duration)
            else:
                ball = positions[holder]

            previous = previous or positions
            players = tuple(
                Player(
                    id=index,
                    team=_team(index),
                    x=x,
                    y=y,
                    vx=(x - previous[index][0]) * self.fps,
                    vy=(y - previous[index][1]) * self.fps,
                )
                for index, (x, y) in enumerate(positions)
            )
            previous = positions
            in_set_piece = t % SET_PIECE_EVERY_S > SET_PIECE_EVERY_S - SET_PIECE_LENGTH_S
            yield FrameTelemetry(
                frame_id=frame_id,
                timestamp=t,
                state=GameState.BOLA_PARADA if in_set_piece else GameState.EM_ANDAMENTO,
                players=players,
                ball=ball,
                score=(0, 0),
                minute=min(int(t // 60), 90),
            )
