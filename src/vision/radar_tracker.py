"""Rastreamento dos jogadores e da bola entre frames: identidade estável, suavização e oclusão.

Por time, a cada frame:
1. prevê a posição de cada trilha (posição + velocidade·dt);
2. associa trilhas e detecções minimizando a distância total (algoritmo húngaro), com um
   portão físico: ninguém anda mais que `max_speed_m_s`;
3. corrige com um filtro alfa-beta (suavização exponencial da posição e da velocidade),
   que tira o tremido do radar sem o atraso de uma média simples;
4. trilha sem detecção:
   - encostada num marcador desenhado por cima dela (círculo azul ou bola cobrindo um
     triângulo) segue esse marcador: o jogador está atrás dele, não sumiu;
   - senão, continua na posição prevista por até `max_coast_s` e depois é descartada;
5. detecção sem trilha abre uma nova, se o time ainda não tem 11.

Trabalha em metros, para que velocidades e portões tenham sentido físico.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from config.settings import PITCH_SIZE_M, VisionConfig

type Vector = NDArray[np.float64]
type Points = NDArray[np.float64]

_SCALE = np.array(PITCH_SIZE_M)
_UNASSIGNED = 1e9
_OCCLUSION_RADIUS_M = 2.5  # trilha a essa distância de um marcador de cima: escondida atrás dele
_MAX_OCCLUDED_S = 3.0
_BALL_MAX_SPEED_M_S = 40.0  # chute forte
_BALL_COAST_S = 0.3


@dataclass(slots=True)
class Track:
    id: int
    pos: Vector  # metros
    vel: Vector  # metros/segundo
    last_seen: float
    occluded_by: Vector | None = None  # deslocamento até quem cobre, quando escondida


def _assign(predicted: Points, detected: Points, gates: Vector) -> list[tuple[int, int]]:
    if not len(predicted) or not len(detected):
        return []
    cost = np.linalg.norm(predicted[:, None, :] - detected[None, :, :], axis=2)
    cost = np.where(cost <= gates[:, None], cost, _UNASSIGNED)
    rows, cols = linear_sum_assignment(cost)
    return [(r, c) for r, c in zip(rows, cols, strict=True) if cost[r, c] < _UNASSIGNED]


class TeamTracker:
    """Trilhas de um time. `ids` = identificadores reservados para ele (11)."""

    def __init__(self, ids: range, cfg: VisionConfig) -> None:
        self.ids, self.cfg = ids, cfg
        self.tracks: list[Track] = []
        self._last_t: float | None = None

    def reset(self) -> None:
        self.tracks, self._last_t = [], None

    def update(self, detected: Points, t: float, occluders: Points) -> list[Track]:
        """Detecções e marcadores de cima em coordenadas normalizadas (n, 2)."""
        cfg = self.cfg
        if self._last_t is not None and not 0 < t - self._last_t <= cfg.max_coast_s:
            self.reset()  # radar sumiu por um tempo (bola parada, replay): começa do zero
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        found, covers = detected * _SCALE, occluders * _SCALE

        predicted = np.array([tr.pos + tr.vel * dt for tr in self.tracks]).reshape(-1, 2)
        gates = np.array(
            [cfg.max_speed_m_s * (t - tr.last_seen) + cfg.gate_margin_m for tr in self.tracks]
        )
        pairs = _assign(predicted, found, gates)
        matched_tracks = {r for r, _ in pairs}
        matched_found = {c for _, c in pairs}

        for r, c in pairs:
            track, guess = self.tracks[r], predicted[r]
            residual = found[c] - guess
            track.pos = guess + cfg.track_alpha * residual
            if dt > 0:
                track.vel = track.vel + cfg.track_beta * residual / dt
            track.last_seen, track.occluded_by = t, None

        kept: list[Track] = []
        for r, track in enumerate(self.tracks):
            if r in matched_tracks or self._hide(track, predicted[r], covers, t):
                kept.append(track)
        self.tracks = kept

        free = [i for i in self.ids if i not in {tr.id for tr in self.tracks}]
        for c in sorted(set(range(len(found))) - matched_found):
            if not free:
                break
            self.tracks.append(Track(free.pop(0), found[c].copy(), np.zeros(2), t))
        return self.tracks

    def _hide(self, track: Track, guess: Vector, covers: Points, t: float) -> bool:
        """Trilha sem detecção: segue quem a cobre, ou desliza um pouco e expira."""
        if len(covers):
            distances = np.linalg.norm(covers - guess, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= _OCCLUSION_RADIUS_M and t - track.last_seen <= _MAX_OCCLUDED_S:
                if track.occluded_by is None:
                    track.occluded_by = track.pos - covers[nearest]
                track.pos, track.vel = covers[nearest] + track.occluded_by, np.zeros(2)
                return True
        track.pos, track.vel = guess, track.vel * 0.5
        return t - track.last_seen <= self.cfg.max_coast_s


class BallTracker:
    """A bola: entre os candidatos, o mais coerente com a trajetória."""

    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg
        self.pos: Vector | None = None  # metros
        self.vel = np.zeros(2)
        self.last_seen = -np.inf

    def reset(self) -> None:
        self.pos, self.vel, self.last_seen = None, np.zeros(2), -np.inf

    def update(self, candidates: Points, t: float) -> Vector | None:
        """Candidatos normalizados (maior primeiro). Devolve a posição normalizada ou None."""
        found = candidates * _SCALE
        lost = self.pos is None or t - self.last_seen > _BALL_COAST_S
        if lost:
            self.reset()
            if not len(found):
                return None
            self.pos, self.last_seen = found[0].copy(), t
            return np.asarray(self.pos / _SCALE, np.float64)
        assert self.pos is not None
        dt = t - self.last_seen
        guess = self.pos + self.vel * dt
        if len(found):
            distances = np.linalg.norm(found - guess, axis=1)
            best = int(np.argmin(distances))
            if distances[best] <= _BALL_MAX_SPEED_M_S * dt + self.cfg.gate_margin_m:
                residual = found[best] - guess
                self.pos = guess + self.cfg.ball_alpha * residual
                if dt > 0:
                    self.vel = self.vel + self.cfg.track_beta * residual / dt
                self.last_seen = t
                return np.asarray(self.pos / _SCALE, np.float64)
        return np.asarray(guess / _SCALE, np.float64)  # breve sumiço: segue a trajetória
