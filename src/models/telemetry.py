"""Contrato canônico entre visão, motor tático, armazenamento e segunda tela."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

type Unit = Annotated[float, Field(ge=0.0, le=1.0)]
"""Coordenada normalizada do radar: x = comprimento do campo, y = largura."""


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class GameState(StrEnum):
    EM_ANDAMENTO = "EM_ANDAMENTO"
    BOLA_PARADA = "BOLA_PARADA"
    REPLAY = "REPLAY"
    PAUSA = "PAUSA"


class Team(StrEnum):
    A = "A"  # time controlado pelo usuário: os alertas são para ele
    B = "B"

    @property
    def rival(self) -> "Team":
        return Team.B if self is Team.A else Team.A


class Player(Frozen):
    id: int
    team: Team
    x: Unit
    y: Unit
    vx: float = 0.0  # unidades normalizadas por segundo
    vy: float = 0.0


class FrameTelemetry(Frozen):
    frame_id: int
    timestamp: float  # segundos, relógio monotônico
    state: GameState
    players: tuple[Player, ...]
    ball: tuple[Unit, Unit] | None = None
    score: tuple[int, int] | None = None  # (time A, time B)
    minute: int | None = None
    team_a_attacks_right: bool = True  # inverte no intervalo

    def attacks_right(self, team: Team) -> bool:
        return self.team_a_attacks_right == (team is Team.A)
