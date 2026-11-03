import logging
import threading
import time

import pytest

from src.alerts import AudioDispatcher, CoachHeuristics
from src.alerts.audio_dispatcher import Speaker, log_speaker
from src.models import EventType, FrameTelemetry, GameState, TacticalEvent

CENTER, PASS, STRETCH = EventType.CENTRO_EXPOSTO, EventType.PASSE_RUPTURA, EventType.TIME_ESPACADO

type Timeline = list[tuple[float, list[EventType]]]


def frame(t: float, score: tuple[int, int] = (0, 0), minute: int = 10) -> FrameTelemetry:
    return FrameTelemetry(
        frame_id=round(t * 60),
        timestamp=t,
        state=GameState.EM_ANDAMENTO,
        players=(),
        score=score,
        minute=minute,
    )


def run(
    timeline: Timeline, score: tuple[int, int] = (0, 0), minute: int = 10
) -> list[tuple[float, EventType]]:
    """Alimenta um coach novo e devolve (instante, tipo) de cada alerta disparado."""
    coach, fired = CoachHeuristics(), []
    for t, kinds in timeline:
        events = [TacticalEvent(type=k, frame_id=0, timestamp=t, value=0.0) for k in kinds]
        if alert := coach.process(frame(t, score, minute), events):
            fired.append((t, alert.type))
    return fired


def every_frame(kinds: list[EventType], start: float, end: float) -> Timeline:
    return [(start + i / 60, kinds) for i in range(round((end - start) * 60))]


def test_debounce_waits_for_a_persistent_event() -> None:
    fired = run(every_frame([CENTER], 0.0, 0.5))
    assert len(fired) == 1
    assert 0.1 <= fired[0][0] < 0.12


def test_flicker_never_fires() -> None:
    timeline = [(i / 60, [CENTER] if i % 2 else []) for i in range(120)]
    assert run(timeline) == []


def test_cooldown_blocks_repeats_of_the_same_alert() -> None:
    fired = run(every_frame([CENTER], 0.0, 15.0))
    assert len(fired) == 2
    assert fired[1][0] - fired[0][0] >= 12.0


def test_priority_and_global_gap_between_different_alerts() -> None:
    fired = run(every_frame([STRETCH, CENTER], 0.0, 4.0))
    assert [kind for _, kind in fired] == [CENTER, STRETCH]
    assert fired[1][0] - fired[0][0] >= 3.0


def test_more_urgent_alert_skips_the_global_gap() -> None:
    timeline = [(t, [STRETCH, CENTER] if t >= 0.5 else [STRETCH]) for t, _ in every_frame([], 0, 4)]
    fired = run(timeline)
    assert [kind for _, kind in fired] == [STRETCH, CENTER]
    assert fired[1][0] < 0.65  # 0,5 s + debounce, sem esperar os 3 s do intervalo


def test_restarted_clock_does_not_silence_the_coach() -> None:
    coach = CoachHeuristics()
    event = [TacticalEvent(type=CENTER, frame_id=0, timestamp=0.0, value=0.0)]
    for t in (0.0, 0.2, 600.0, 600.2):  # 600 s de partida, depois uma partida nova
        coach.process(frame(t), event)
    assert coach.process(frame(0.0), event) is None
    assert coach.process(frame(0.2), event) is not None


def test_attack_alerts_are_suppressed_when_protecting_a_late_lead() -> None:
    late_win = run(every_frame([PASS], 0.0, 1.0), score=(2, 1), minute=88)
    late_draw = run(every_frame([PASS], 0.0, 1.0), score=(1, 1), minute=88)
    assert late_win == []
    assert [kind for _, kind in late_draw] == [PASS]


def test_dispatcher_speaks_fast_and_drops_stale_alerts() -> None:
    spoken: list[str] = []
    first_started = threading.Event()

    def slow_speaker(text: str) -> None:
        spoken.append(text)
        first_started.set()
        time.sleep(0.2)

    with AudioDispatcher(lambda: slow_speaker) as audio:
        start = time.perf_counter()
        audio.say("um")
        assert first_started.wait(1.0)
        assert audio.last_latency_ms is not None and audio.last_latency_ms < 100
        for text in ("dois", "três", "quatro"):  # enquanto "um" ainda toca
            audio.say(text)
        assert time.perf_counter() - start < 0.1  # say nunca bloqueia
        time.sleep(0.5)

    assert spoken == ["um", "quatro"]


def test_dispatcher_survives_missing_tts(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    def broken() -> Speaker:
        raise RuntimeError("sem SAPI")

    with AudioDispatcher(broken) as audio:
        audio.say("ainda funciona")
        deadline = time.perf_counter() + 1.0
        while "ainda funciona" not in caplog.text and time.perf_counter() < deadline:
            time.sleep(0.01)

    assert "TTS indisponível" in caplog.text
    assert "[coach] ainda funciona" in caplog.text


def test_dispatcher_stop_without_start() -> None:
    AudioDispatcher(lambda: log_speaker).stop()
