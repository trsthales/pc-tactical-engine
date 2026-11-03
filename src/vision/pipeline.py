"""Visão completa: recortes do frame (radar e placar) → `FrameTelemetry` para o motor tático."""

from dataclasses import dataclass

import numpy as np

from config.settings import PITCH_SIZE_M, VisionConfig
from src.capture import CapturedFrame
from src.models import FrameTelemetry, Player, Team
from src.vision.game_state_detector import AttackSideEstimator, GameStateDetector
from src.vision.ocr_hud import HudReader
from src.vision.radar_geometry import RadarReference
from src.vision.radar_markers import detect_markers
from src.vision.radar_tracker import BallTracker, TeamTracker, Track

_IDS_A, _IDS_B = range(0, 11), range(11, 22)
_SCALE = np.array(PITCH_SIZE_M)


@dataclass(frozen=True, slots=True)
class TeamSetup:
    """Qual marcador e qual linha do placar são do usuário (time A).

    No vídeo de referência o usuário é o mandante: triângulos vermelhos, linha de cima.
    """

    a_is_red: bool = True
    a_on_top: bool = True


def _players(tracks: list[Track], team: Team) -> list[Player]:
    out = []
    for track in tracks:
        x, y = np.clip(track.pos / _SCALE, 0.0, 1.0)
        vx, vy = track.vel / _SCALE
        out.append(
            Player(id=track.id, team=team, x=float(x), y=float(y), vx=float(vx), vy=float(vy))
        )
    return out


class VisionPipeline:
    def __init__(
        self,
        reference: RadarReference | None = None,
        cfg: VisionConfig | None = None,
        teams: TeamSetup | None = None,
    ) -> None:
        self.cfg = cfg or VisionConfig()
        self.reference = reference or RadarReference.load()
        self.teams = teams or TeamSetup()
        self.hud = HudReader(self.cfg)
        self.states = GameStateDetector(self.cfg)
        self.side = AttackSideEstimator(self.cfg)
        self.team_a, self.team_b = TeamTracker(_IDS_A, self.cfg), TeamTracker(_IDS_B, self.cfg)
        self.ball = BallTracker(self.cfg)
        self.score: tuple[int, int] | None = None  # (time A, time B), última leitura válida
        self.minute: int | None = None
        self._radar_visible = False

    def process(self, frame: CapturedFrame) -> FrameTelemetry:
        t = frame.timestamp
        radar = detect_markers(frame.crops["radar"], self.reference, self.cfg, self._radar_visible)
        self._radar_visible = radar.visible
        hud = self.hud.read(frame.crops["scoreboard"])
        if hud.score is not None:
            top, bottom = hud.score
            self.score = (top, bottom) if self.teams.a_on_top else (bottom, top)
        if hud.clock is not None:
            self.minute = hud.clock[0]
        state = self.states.update(radar.visible, hud.visible, hud.menu)

        players: list[Player] = []
        ball: tuple[float, float] | None = None
        if radar.visible:
            to_pitch = self.reference.to_pitch
            red, blue = to_pitch(radar.red), to_pitch(radar.blue)
            balls = to_pitch(radar.ball_candidates)
            # o radar desenha os círculos (azul) e a bola por cima dos triângulos (vermelho)
            red_covers, blue_covers = np.vstack([blue, balls[:1]]), balls[:1]
            if self.teams.a_is_red:
                tracks_a = self.team_a.update(red, t, red_covers)
                tracks_b = self.team_b.update(blue, t, blue_covers)
            else:
                tracks_a = self.team_a.update(blue, t, blue_covers)
                tracks_b = self.team_b.update(red, t, red_covers)
            players = _players(tracks_a, Team.A) + _players(tracks_b, Team.B)
            if (spot := self.ball.update(balls, t)) is not None:
                bx, by = np.clip(spot, 0.0, 1.0)
                ball = (float(bx), float(by))
            xs = np.array([p.x for p in players])
            is_a = np.array([p.team is Team.A for p in players], bool)
            if len(players):
                self.side.update(xs[is_a], xs[~is_a], t)

        return FrameTelemetry(
            frame_id=frame.frame_id,
            timestamp=t,
            state=state,
            players=tuple(players),
            ball=ball,
            score=self.score,
            minute=self.minute,
            team_a_attacks_right=self.side.attacks_right is not False,
        )
