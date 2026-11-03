"""Saída do motor tático: métricas por frame e eventos para o coach."""

from enum import StrEnum

from src.models.telemetry import Frozen, Team

type Polygon = tuple[tuple[float, float], ...]


class TeamShape(Frozen):
    depth_m: float  # do jogador de linha mais recuado ao mais avançado
    cb_gap_m: float  # distância entre os dois zagueiros centrais


class PassLane(Frozen):
    passer_id: int
    receiver_id: int
    length_m: float
    gain_m: float  # quanto o passe avança rumo ao gol adversário
    bypassed: int  # rivais que ficam para trás (entre passador e receptor)
    onside: bool
    through: bool  # passe de ruptura: legal, avança e quebra linha


class TacticalSnapshot(Frozen):
    frame_id: int
    timestamp: float
    control_a: float  # fração do campo dominada pelo time A (Voronoi)
    regions: tuple[Polygon, ...]  # célula de Voronoi de cada jogador, mesma ordem de `players`
    shapes: dict[Team, TeamShape]
    carrier_id: int | None  # jogador com a bola
    possession: Team | None
    open_passes: tuple[PassLane, ...]


class EventType(StrEnum):
    CENTRO_EXPOSTO = "CENTRO_EXPOSTO"
    TIME_ESPACADO = "TIME_ESPACADO"
    PASSE_RUPTURA = "PASSE_RUPTURA"


class TacticalEvent(Frozen):
    type: EventType
    frame_id: int
    timestamp: float
    value: float  # métrica que disparou o evento, em metros
    target_id: int | None = None  # receptor, no caso de passe
