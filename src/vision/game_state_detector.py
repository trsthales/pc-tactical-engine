"""Estado da partida e lado de ataque, a partir do que a interface do FC mostra.

Estado (medido no vídeo de referência, com radar, placar e menu frame a frame):
- radar visível → EM_ANDAMENTO (o jogo esconde o radar em escanteio, lateral, falta e
  depois de gol);
- sem radar, com menu de pausa/intervalo → PAUSA;
- sem radar, com placar → BOLA_PARADA (cobranças, comemoração, "continuando partida");
- sem radar nem placar → REPLAY (replays e cenas de corte).
Sair do jogo é imediato (o coach se cala no mesmo frame); voltar exige o radar por alguns
frames, para o fade de entrada não piscar.

Lado de ataque: cada goleiro é o jogador do time mais perto de uma linha de fundo. O voto
do frame compara as duas pontas para os dois times; uma média móvel com histerese só troca
de lado quando a evidência se sustenta (no intervalo).
"""

import math

import numpy as np
from numpy.typing import NDArray

from config.settings import VisionConfig
from src.models import GameState

_SIDE_HYSTERESIS = 0.05  # média do voto precisa passar disso para (re)decidir o lado
_SIDE_MIN_PLAYERS = 8
_SIDE_VOTE_CLIP = 0.3
_FIRST_STEP_S = 0.15
_MAX_STEP_S = 0.5


class GameStateDetector:
    def __init__(self, cfg: VisionConfig | None = None) -> None:
        self.cfg = cfg or VisionConfig()
        self.state = GameState.REPLAY
        self._live_frames = 0

    def update(self, radar_visible: bool, scoreboard_visible: bool, menu: bool) -> GameState:
        if radar_visible:
            self._live_frames += 1
            if self._live_frames >= self.cfg.frames_to_resume:
                self.state = GameState.EM_ANDAMENTO
            return self.state
        self._live_frames = 0
        if menu:
            self.state = GameState.PAUSA
        elif scoreboard_visible:
            self.state = GameState.BOLA_PARADA
        else:
            self.state = GameState.REPLAY
        return self.state


class AttackSideEstimator:
    """Decide se o time A ataca para a direita (x crescente)."""

    def __init__(self, cfg: VisionConfig | None = None) -> None:
        self.cfg = cfg or VisionConfig()
        self.attacks_right: bool | None = None  # None até ter evidência
        self._mean = 0.0
        self._last_t: float | None = None

    def update(
        self, team_a_x: NDArray[np.float64], team_b_x: NDArray[np.float64], t: float
    ) -> bool:
        if len(team_a_x) >= _SIDE_MIN_PLAYERS and len(team_b_x) >= _SIDE_MIN_PLAYERS:
            # > 0: A tem alguém colado na linha da esquerda (goleiro) e B na da direita
            vote = (np.min(1 - team_a_x) - np.min(team_a_x)) + (
                np.min(team_b_x) - np.min(1 - team_b_x)
            )
            # depois de um sumiço longo do radar, um frame só não pode decidir sozinho
            dt = (
                _FIRST_STEP_S
                if self._last_t is None
                else min(max(t - self._last_t, 0.0), _MAX_STEP_S)
            )
            weight = 1 - math.exp(-dt / self.cfg.side_time_constant_s)
            self._mean += weight * (
                float(np.clip(vote, -_SIDE_VOTE_CLIP, _SIDE_VOTE_CLIP)) - self._mean
            )
            self._last_t = t
            if self._mean > _SIDE_HYSTERESIS:
                self.attacks_right = True
            elif self._mean < -_SIDE_HYSTERESIS:
                self.attacks_right = False
        return True if self.attacks_right is None else self.attacks_right
