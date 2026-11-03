"""Visão (Marco 2): geometria, detecção, rastreamento, HUD, estados e o vídeo de referência.

Três níveis de verdade:
- sintético: posições conhecidas por construção (cantos, sprites, trajetórias);
- amostras reais (`tests/samples/fc27/`): gabarito anotado olhando as imagens, sem a saída
  do detector (`labels.json`);
- vídeo inteiro (opcional, `FC_VIDEO=caminho.mp4`): invariantes do próprio jogo.
"""

import itertools
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from config.settings import PITCH_SIZE_M, ROIConfig, VisionConfig
from src.capture import CapturedFrame
from src.models import GameState, Team
from src.vision.game_state_detector import AttackSideEstimator, GameStateDetector
from src.vision.ocr_hud import HudReader
from src.vision.pipeline import VisionPipeline
from src.vision.radar_geometry import RadarReference, fit_pitch_corners
from src.vision.radar_markers import detect_markers, expected_area
from src.vision.radar_tracker import BallTracker, TeamTracker
from src.vision.video_source import VideoSource

SAMPLES = Path(__file__).parent / "samples" / "fc27"
LABELS = {k: v for k, v in json.loads((SAMPLES / "labels.json").read_text()).items()
          if not k.startswith("_")}  # fmt: skip
CFG = VisionConfig()
SCALE = np.array(PITCH_SIZE_M)


@pytest.fixture(scope="module")
def ref() -> RadarReference:
    return RadarReference.load()


def _sample(name: str, crop: str) -> np.ndarray:
    image = cv2.imread(str(SAMPLES / f"{name}_{crop}.png"))
    assert image is not None, name
    return np.asarray(image, np.uint8)


# --- geometria ---------------------------------------------------------------------------


def test_fit_pitch_corners_recovers_a_known_trapezoid() -> None:
    corners = np.array([[100.3, 30.6], [300.8, 30.6], [352.2, 160.4], [47.7, 160.4]])
    big = 8  # desenha em alta resolução e reduz: bordas com antialias, cantos subpixel
    canvas = np.full((190 * big, 400 * big, 3), (60, 143, 60), np.uint8)
    cv2.fillConvexPoly(canvas, np.rint(corners * big).astype(np.int32), (40, 91, 40))
    background = np.asarray(cv2.resize(canvas, (400, 190), interpolation=cv2.INTER_AREA), np.uint8)
    # a redução desloca o centro do pixel em (big - 1) / (2 big)
    found = fit_pitch_corners(background) + (big - 1) / (2 * big)
    assert np.abs(found - corners).max() < 0.35


def test_reference_maps_its_corners_to_the_unit_square(ref: RadarReference) -> None:
    unit = ref.to_pitch(ref.corners)
    np.testing.assert_allclose(unit, [[0, 0], [1, 0], [1, 1], [0, 1]], atol=1e-6)
    top, bottom = (
        np.linalg.norm(ref.corners[1] - ref.corners[0]),
        np.linalg.norm(ref.corners[2] - ref.corners[3]),
    )
    assert top < bottom  # perspectiva: o lado de lá do campo é mais estreito


def test_background_is_fully_opaque_and_identical(ref: RadarReference) -> None:
    assert ref.opacity(ref.background) == pytest.approx(1.0)
    assert ref.similarity(ref.background) == pytest.approx(1.0)


def test_center_of_the_drawing_is_the_center_of_the_pitch(ref: RadarReference) -> None:
    """Diagonais do trapézio se cruzam no centro do campo (homografia, não interpolação)."""
    c = ref.corners
    d1, d2 = np.cross([*c[0], 1], [*c[2], 1]), np.cross([*c[1], 1], [*c[3], 1])
    meet = np.cross(d1, d2)
    center = ref.to_pitch(np.array([meet[:2] / meet[2]]))[0]
    np.testing.assert_allclose(center, [0.5, 0.5], atol=1e-6)


# --- detecção em sprites sintéticos ------------------------------------------------------

_RED, _CREAM = (40, 30, 215), (175, 215, 235)
_BLUE, _RING = (215, 80, 40), (245, 235, 225)
_BALL = (30, 170, 250)
_K = 8  # supersampling do desenho


def _circle(img: np.ndarray, x: float, y: float) -> None:
    r = float(np.sqrt(expected_area(CFG.blue_area, y) / np.pi))
    center = (round(x * _K), round(y * _K))
    cv2.circle(img, center, round((r + 1.5) * _K), _RING, -1, cv2.LINE_AA)
    cv2.circle(img, center, round(r * _K), _BLUE, -1, cv2.LINE_AA)


def _triangle(img: np.ndarray, x: float, y: float, scale: float = 1.0, rim: bool = True) -> None:
    """Triângulo para cima com a caixa centrada em (x, y), como o jogo desenha."""
    side = float(np.sqrt(4 * scale * expected_area(CFG.red_area, y) / np.sqrt(3)))
    h = side * np.sqrt(3) / 2

    def pts(s: float, hh: float) -> np.ndarray:
        tri = [(x, y - hh / 2), (x + s / 2, y + hh / 2), (x - s / 2, y + hh / 2)]
        return np.rint(np.array(tri) * _K).astype(np.int32)

    if rim:
        cv2.fillConvexPoly(img, pts(side + 4, h + 3.5), _CREAM, cv2.LINE_AA)
    cv2.fillConvexPoly(img, pts(side, h), _RED, cv2.LINE_AA)


def _ball(img: np.ndarray, x: float, y: float) -> None:
    c, arm = (round(x * _K), round(y * _K)), round(2.5 * _K)
    cv2.line(img, (c[0] - arm, c[1]), (c[0] + arm, c[1]), _BALL, 2 * _K)
    cv2.line(img, (c[0], c[1] - arm), (c[0], c[1] + arm), _BALL, 2 * _K)


def _scene(ref: RadarReference, draw: list) -> np.ndarray:  # type: ignore[type-arg]
    h, w = ref.background.shape[:2]
    canvas = cv2.resize(ref.background, (w * _K, h * _K), interpolation=cv2.INTER_NEAREST)
    for fn, *args in draw:
        fn(canvas, *args)
    return np.asarray(cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA), np.uint8)


def _shift(points: np.ndarray) -> np.ndarray:
    return points + (_K - 1) / (2 * _K)  # centro do pixel após a redução


def _match(found: np.ndarray, truth: list[tuple[float, float]]) -> float:
    """Maior erro, em pixels, casando cada verdade com a detecção mais próxima."""
    assert len(found) == len(truth), (found, truth)
    found = _shift(found)
    return max(float(np.min(np.linalg.norm(found - p, axis=1))) for p in truth)


def test_isolated_markers_are_found_where_they_were_drawn(ref: RadarReference) -> None:
    reds = [(120.0, 60.0), (200.0, 110.0), (280.0, 150.0)]
    blues = [(160.0, 50.0), (240.0, 95.0), (130.0, 140.0)]
    img = _scene(ref, [(_triangle, *p) for p in reds] + [(_circle, *p) for p in blues]
                 + [(_ball, 220.0, 70.0)])  # fmt: skip
    det = detect_markers(img, ref, CFG)
    assert det.visible
    assert _match(det.red, reds) < 1.0
    assert _match(det.blue, blues) < 0.6
    assert _match(det.ball_candidates[:1], [(220.0, 70.0)]) < 0.6


def test_overlapping_markers_are_split(ref: RadarReference) -> None:
    blues = [(200.0, 100.0), (209.0, 102.0)]  # círculos colados (marcação)
    reds = [(140.0, 120.0), (151.0, 120.0)]  # triângulos lado a lado
    img = _scene(ref, [(_circle, *p) for p in blues] + [(_triangle, *p) for p in reds])
    det = detect_markers(img, ref, CFG)
    assert len(det.blue) == 2 and _match(det.blue, blues) < 2.0
    assert len(det.red) == 2 and _match(det.red, reds) < 2.5


def test_controlled_player_big_bare_triangle_counts_once(ref: RadarReference) -> None:
    img = _scene(ref, [(_triangle, 200.0, 100.0, 2.3, False)])
    det = detect_markers(img, ref, CFG)
    assert len(det.red) == 1
    assert _match(det.red, [(200.0, 100.0)]) < 1.5


def test_triangle_mostly_under_a_circle_is_still_seen(ref: RadarReference) -> None:
    """O radar desenha os círculos por cima: da marcação colada sobra a ponta do triângulo."""
    img = _scene(ref, [(_triangle, 200.0, 100.0), (_circle, 200.0, 105.0)])
    det = detect_markers(img, ref, CFG)
    assert len(det.blue) == 1 and len(det.red) == 1
    assert _match(det.red, [(200.0, 100.0)]) < 4.0  # só um pedaço: posição aproximada


def test_big_red_overlay_is_not_a_player(ref: RadarReference) -> None:
    """Banner de gol/escudo vermelho por cima do radar não vira um bolo de jogadores."""
    big = 8 * expected_area(CFG.red_area, 100)
    side = float(np.sqrt(big))

    def banner(img: np.ndarray) -> None:
        cv2.rectangle(img, (round(180 * _K), round(80 * _K)),
                      (round((180 + side) * _K), round((80 + side) * _K)), _RED, -1)  # fmt: skip

    det = detect_markers(_scene(ref, [(banner,)]), ref, CFG)
    assert len(det.red) == 0


def test_blue_without_the_white_ring_is_scenery_not_a_marker(ref: RadarReference) -> None:
    """Calção azul-marinho de um jogador 3D passando atrás do radar: mesma cor, sem anel."""

    def shorts(img: np.ndarray) -> None:
        r = float(np.sqrt(expected_area(CFG.blue_area, 100) / np.pi))
        cv2.circle(img, (200 * _K, 100 * _K), round(r * _K), _BLUE, -1, cv2.LINE_AA)

    det = detect_markers(_scene(ref, [(shorts,), (_circle, 260.0, 100.0)]), ref, CFG)
    assert len(det.blue) == 1 and _match(det.blue, [(260.0, 100.0)]) < 0.6


def test_empty_radar_finds_nothing_and_hidden_radar_skips_detection(ref: RadarReference) -> None:
    det = detect_markers(ref.background, ref, CFG)
    assert det.visible and not len(det.red) and not len(det.blue) and not len(det.ball_candidates)
    grass = np.full_like(ref.background, (60, 143, 60))
    assert not detect_markers(grass, ref, CFG).visible


def test_detection_accepts_other_capture_resolutions(ref: RadarReference) -> None:
    """O recorte de uma tela 1440p chega maior: é reduzido ao tamanho de referência."""
    img = _scene(ref, [(_circle, 200.0, 100.0)])
    bigger = np.asarray(cv2.resize(img, (533, 253), interpolation=cv2.INTER_CUBIC), np.uint8)
    det = detect_markers(bigger, ref, CFG)
    assert len(det.blue) == 1 and _match(det.blue, [(200.0, 100.0)]) < 1.0


# --- amostras reais x gabarito -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(LABELS))
def test_radar_sample_matches_labels(name: str, ref: RadarReference) -> None:
    label = LABELS[name]
    det = detect_markers(_sample(name, "radar"), ref, CFG, label.get("was_visible", False))
    assert det.visible == label["radar"]
    if label["radar"]:
        low, high = label["red"]
        assert low <= len(det.red) <= high, f"vermelhos: {len(det.red)}"
        low, high = label["blue"]
        assert low <= len(det.blue) <= high, f"azuis: {len(det.blue)}"
        assert (len(det.ball_candidates) > 0) == label["ball"]


@pytest.mark.parametrize("name", sorted(LABELS))
def test_hud_sample_matches_labels(name: str) -> None:
    label = LABELS[name]
    hud = HudReader(CFG).read(_sample(name, "scoreboard"))
    assert hud.menu == (label["state"] == "PAUSA")
    if label.get("score") is not None:
        assert hud.score == tuple(label["score"])
    if label.get("clock") is not None:
        assert hud.clock == tuple(label["clock"])
    if label["state"] in ("PAUSA", "REPLAY"):
        assert not hud.visible


@pytest.mark.parametrize("name", sorted(LABELS))
def test_state_sample_matches_labels(name: str, ref: RadarReference) -> None:
    label = LABELS[name]
    det = detect_markers(_sample(name, "radar"), ref, CFG, label.get("was_visible", False))
    hud = HudReader(CFG).read(_sample(name, "scoreboard"))
    states = GameStateDetector(CFG)
    for _ in range(CFG.frames_to_resume):  # um trecho parado no mesmo frame
        state = states.update(det.visible, hud.visible, hud.menu)
    assert state.value == label["state"]


def test_marker_detection_is_fast(ref: RadarReference) -> None:
    crops = [_sample(n, "radar") for n, lab in LABELS.items() if lab["radar"]]
    for crop in crops:  # aquece caches
        detect_markers(crop, ref, CFG)
    spent = []
    for crop in crops * 5:
        start = time.perf_counter()
        detect_markers(crop, ref, CFG)
        spent.append(time.perf_counter() - start)
    assert np.median(spent) < 0.003  # meta do Marco 2


# --- rastreamento ------------------------------------------------------------------------


def _norm(meters: np.ndarray | list[list[float]]) -> np.ndarray:
    return np.asarray(meters, np.float64).reshape(-1, 2) / SCALE  # type: ignore[no-any-return]


def test_tracker_keeps_identities_through_noise_and_crossing() -> None:
    rng = np.random.default_rng(1)
    tracker = TeamTracker(range(11), CFG)
    start = np.column_stack([np.linspace(10, 90, 11), np.full(11, 34.0)])
    vel = np.zeros((11, 2))
    vel[3], vel[4] = (8.0, 1.0), (-8.0, -1.0)  # 3 e 4 se cruzam em ~0,7 s
    ids_of: dict[int, int] = {}
    for k in range(120):  # 2 s a 60 fps
        t = k / 60
        true = start + vel * t
        noisy = true + rng.normal(0, 0.15, true.shape)
        order = rng.permutation(11)
        tracks = tracker.update(_norm(noisy[order]), t, np.empty((0, 2)))
        assert len(tracks) == 11
        for tr in tracks:
            nearest = int(np.argmin(np.linalg.norm(true - tr.pos, axis=1)))
            assert ids_of.setdefault(tr.id, nearest) == nearest, f"troca de id em {t:.2f} s"
            assert np.linalg.norm(tr.pos - true[nearest]) < 0.6


def test_tracker_smooths_jitter() -> None:
    rng = np.random.default_rng(2)
    tracker = TeamTracker(range(1), CFG)
    errors = []
    for k in range(180):
        t = k / 60
        true = np.array([[30 + 5 * t, 20.0]])
        tracks = tracker.update(_norm(true + rng.normal(0, 0.3, (1, 2))), t, np.empty((0, 2)))
        if k > 30:
            errors.append(np.linalg.norm(tracks[0].pos - true[0]))
    assert np.sqrt(np.mean(np.square(errors))) < 0.3 * np.sqrt(2)  # menos tremido que a entrada


def test_occluded_player_follows_the_marker_on_top() -> None:
    tracker = TeamTracker(range(1), CFG)
    cover = np.array([[50.0, 30.0]])
    tracker.update(_norm(cover + np.array([0.5, 0])), 0.0, _norm(cover))
    for k in range(1, 120):  # 2 s escondido atrás de um círculo que anda
        t = k / 60
        moving = cover + np.array([6 * t, 0])
        tracks = tracker.update(np.empty((0, 2)), t, _norm(moving))
        assert len(tracks) == 1
        assert np.linalg.norm(tracks[0].pos - (moving[0] + [0.5, 0])) < 0.3


def test_lost_player_expires_and_long_gap_resets() -> None:
    tracker = TeamTracker(range(2), CFG)
    tracker.update(_norm([[10.0, 10.0], [50.0, 30.0]]), 0.0, np.empty((0, 2)))
    for k in range(1, 40):  # o segundo some por 0,65 s, longe de qualquer marcador
        tracks = tracker.update(_norm([[10.0, 10.0]]), k / 60, np.empty((0, 2)))
    assert [tr.id for tr in tracks] == [0]
    tracks = tracker.update(_norm([[80.0, 60.0]]), 5.0, np.empty((0, 2)))  # radar voltou
    assert len(tracks) == 1 and np.allclose(tracks[0].pos, [80.0, 60.0])


def test_teleport_is_a_new_player_not_the_same_one() -> None:
    tracker = TeamTracker(range(11), CFG)
    tracker.update(_norm([[10.0, 10.0]]), 0.0, np.empty((0, 2)))
    tracks = tracker.update(_norm([[60.0, 10.0]]), 1 / 60, np.empty((0, 2)))  # 50 m em 1 frame
    assert {tr.id for tr in tracks} == {0, 1}


def test_ball_tracker_ignores_far_false_candidate() -> None:
    ball = BallTracker(CFG)
    for k in range(30):
        ball.update(_norm([[40 + 0.2 * k, 30.0]]), k / 60)
    spot = ball.update(_norm([[95.0, 5.0], [46.1, 30.0]]), 30 / 60)
    assert spot is not None and np.linalg.norm(spot * SCALE - [46.0, 30.0]) < 0.3


# --- estado e lado de ataque -------------------------------------------------------------


def test_state_leaves_play_at_once_and_returns_after_a_few_frames() -> None:
    states = GameStateDetector(CFG)
    seen = [states.update(True, True, False) for _ in range(CFG.frames_to_resume)]
    assert seen[-1] is GameState.EM_ANDAMENTO and seen[0] is not GameState.EM_ANDAMENTO
    assert states.update(False, True, False) is GameState.BOLA_PARADA
    assert states.update(True, True, False) is GameState.BOLA_PARADA  # fade de entrada
    assert states.update(False, False, True) is GameState.PAUSA
    assert states.update(False, False, False) is GameState.REPLAY


def _lineup(keeper_x: float, rng: np.random.Generator) -> np.ndarray:
    field = rng.uniform(0.2, 0.8, 10)
    return np.append(field, keeper_x)


def test_attack_side_follows_the_keepers_and_flips_only_at_halftime() -> None:
    rng = np.random.default_rng(3)
    side = AttackSideEstimator(CFG)
    t = 0.0
    for _ in range(300):  # 1º tempo: goleiro de A na esquerda
        t += 1 / 30
        assert side.update(_lineup(0.03, rng), _lineup(0.97, rng), t)
    for _ in range(10):  # um lance isolado com A todo à direita não vira o lado
        t += 1 / 30
        side.update(_lineup(0.9, rng), _lineup(0.97, rng), t)
    assert side.attacks_right is True
    t += 600  # intervalo: radar fora por minutos
    for _ in range(300):
        t += 1 / 30
        side.update(_lineup(0.97, rng), _lineup(0.03, rng), t)
    assert side.attacks_right is False


# --- fonte de vídeo e pipeline -----------------------------------------------------------


def test_video_source_yields_crops_with_video_time(tmp_path: Path) -> None:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30, (320, 180))
    for k in range(12):
        writer.write(np.full((180, 320, 3), 20 * k, np.uint8))
    writer.release()
    source = VideoSource(path, ROIConfig(), start_s=0.2)
    frames = list(source)
    assert source.fps == pytest.approx(30)
    assert [f.frame_id for f in frames] == list(range(6, 12))
    assert frames[0].timestamp == pytest.approx(0.2)
    assert set(frames[0].crops) == set(ROIConfig().named())


def test_video_source_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "nada.mp4", ROIConfig())


def test_pipeline_turns_a_sample_into_telemetry(ref: RadarReference) -> None:
    pipeline = VisionPipeline(ref, CFG)
    crops = {"radar": _sample("t0030.0", "radar"), "scoreboard": _sample("t0030.0", "scoreboard")}
    for k in range(CFG.frames_to_resume):
        tel = pipeline.process(CapturedFrame(k, k / 60, crops))
    assert tel.state is GameState.EM_ANDAMENTO
    assert tel.score == (0, 0) and tel.minute == 3
    assert sum(p.team is Team.A for p in tel.players) == 11
    assert sum(p.team is Team.B for p in tel.players) == 11
    assert all(0 <= p.x <= 1 and 0 <= p.y <= 1 for p in tel.players)
    assert tel.ball is not None


# --- vídeo inteiro (opcional) ------------------------------------------------------------

VIDEO = os.environ.get("FC_VIDEO")


@pytest.mark.skipif(not VIDEO, reason="defina FC_VIDEO=partida.mp4 (vídeo de referência)")
def test_reference_video_invariants() -> None:
    """Vídeo de referência (EA FC 27, vitória por 4 x 5 do visitante): invariantes do jogo."""
    assert VIDEO
    pipeline = VisionPipeline()
    frames = [pipeline.process(f) for i, f in enumerate(VideoSource(Path(VIDEO), ROIConfig()))
              if i % 2 == 0]  # fmt: skip
    live = [f for f in frames if f.state is GameState.EM_ANDAMENTO]
    full = sum(len(f.players) == 22 for f in live) / len(live)
    assert full > 0.6
    assert sum(f.ball is not None for f in live) / len(live) > 0.99

    scores = [f.score for f in frames if f.score is not None]
    changes = [(a, b) for a, b in itertools.pairwise(scores) if a != b]
    assert all(sum(b) - sum(a) == 1 and b[0] >= a[0] and b[1] >= a[1] for a, b in changes)
    assert scores[-1] == (4, 5) and len(changes) == 9

    sides = [f.team_a_attacks_right for f in live]
    assert sum(a != b for a, b in itertools.pairwise(sides)) == 1
