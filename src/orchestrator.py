"""Laço principal: telemetria → análise tática → coach, gravação e segunda tela."""

import logging
import threading
import time
from collections.abc import Iterable, Iterator
from typing import Protocol

from src.alerts import CoachHeuristics
from src.models import FrameTelemetry
from src.server import LiveAlert, LivePayload, TelemetryHub
from src.storage import TelemetryWriter
from src.tactics import analyze, detect_events

log = logging.getLogger(__name__)

FPS_WINDOW_S = 1.0
LOG_EVERY_S = 10.0


class Voice(Protocol):
    """Quem fala os alertas (ex.: `AudioDispatcher`). Não pode bloquear o motor."""

    def say(self, text: str) -> None: ...


def realtime[T](items: Iterable[T], fps: float) -> Iterator[T]:
    """Entrega os itens no ritmo de `fps`. Para fontes que não esperam a tela (simulador)."""
    period = 1 / fps
    next_tick = time.perf_counter()
    for item in items:
        yield item
        next_tick = max(next_tick + period, time.perf_counter() - period)  # não acumula atraso
        time.sleep(max(0.0, next_tick - time.perf_counter()))


class Engine:
    def __init__(
        self,
        source: Iterable[FrameTelemetry],
        hub: TelemetryHub,
        *,
        coach: CoachHeuristics | None = None,
        audio: Voice | None = None,
        writer: TelemetryWriter | None = None,
    ) -> None:
        self.source = source
        self.hub = hub
        self.coach = coach or CoachHeuristics()
        self.audio = audio
        self.writer = writer
        self.last_alert: LiveAlert | None = None
        self.fps = 0.0
        self.cycle_ms = 0.0  # média móvel exponencial

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        window_start, window_frames = time.perf_counter(), 0
        next_log = window_start + LOG_EVERY_S
        for frame in self.source:
            if stop.is_set():
                break
            self.step(frame)

            window_frames += 1
            now = time.perf_counter()
            if now - window_start >= FPS_WINDOW_S:
                self.fps = window_frames / (now - window_start)
                window_start, window_frames = now, 0
            if now >= next_log:
                log.info("motor: %.1f FPS, %.2f ms/frame", self.fps, self.cycle_ms)
                next_log = now + LOG_EVERY_S

    def step(self, frame: FrameTelemetry) -> None:
        started = time.perf_counter()
        snapshot = analyze(frame)
        events = detect_events(frame, snapshot)
        if alert := self.coach.process(frame, events):
            log.info("alerta: %s", alert.message)
            self.last_alert = LiveAlert(message=alert.message, timestamp=alert.timestamp)
            if self.audio:
                self.audio.say(alert.message)
        if self.writer:
            self.writer.submit(frame, snapshot, events)
        self.cycle_ms = 0.9 * self.cycle_ms + 0.1 * 1000 * (time.perf_counter() - started)

        payload = LivePayload(
            frame=frame,
            snapshot=snapshot,
            last_alert=self.last_alert,
            cycle_ms=self.cycle_ms,
            engine_fps=self.fps,
        )
        self.hub.publish(payload.model_dump_json())
