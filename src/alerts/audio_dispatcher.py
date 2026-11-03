"""Voz dos alertas numa thread própria: o motor tático nunca espera o áudio."""

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

from src.capture.frame_queue import LatestQueue

log = logging.getLogger(__name__)

type Speaker = Callable[[str], None]
"""Fala o texto e só retorna ao terminar."""


def pyttsx3_speaker() -> Speaker:
    """Voz do sistema (SAPI5 no Windows), em português se houver voz instalada."""
    if sys.platform == "win32":
        import comtypes  # SAPI5 é COM: cada thread que o usa precisa inicializar o COM

        comtypes.CoInitialize()
    import pyttsx3

    engine: Any = pyttsx3.init()
    engine.setProperty("rate", 200)
    for voice in engine.getProperty("voices"):
        if any(tag in f"{voice.id} {voice.name}".lower() for tag in ("pt-br", "portug", "brazil")):
            engine.setProperty("voice", voice.id)
            break

    def speak(text: str) -> None:
        engine.say(text)
        try:
            engine.runAndWait()
        except Exception:
            # O pyttsx3 não limpa o próprio estado quando a fala falha e recusaria todas as
            # seguintes ("run loop already started"). O init() devolve o mesmo engine, então
            # recriar não adianta: descarta os comandos pendentes e destrava aqui.
            engine.proxy._queue.clear()
            engine._inLoop = False
            engine.proxy.setBusy(False)
            raise

    return speak


def log_speaker(text: str) -> None:
    log.info("[coach] %s", text)


@dataclass(frozen=True, slots=True)
class _Utterance:
    text: str
    queued_at: float


class AudioDispatcher:
    """`say` nunca bloqueia. Se uma fala ainda está tocando, só o alerta mais novo espera.

    Recebe uma fábrica e não o `Speaker` pronto porque o pyttsx3 (COM) precisa ser criado
    na mesma thread que fala.
    """

    def __init__(self, speaker_factory: Callable[[], Speaker] = pyttsx3_speaker) -> None:
        self._factory = speaker_factory
        self._pending: LatestQueue[_Utterance | None] = LatestQueue()
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._stopped = threading.Event()
        self.last_latency_ms: float | None = None  # de `say` até o início da fala

    def start(self) -> None:
        self._thread.start()

    def say(self, text: str) -> None:
        self._pending.put(_Utterance(text, time.perf_counter()))

    def stop(self) -> None:
        self._stopped.set()
        self._pending.put(None)  # só acorda a thread; um `say` concorrente pode substituí-lo
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            speak = self._factory()
        except Exception:
            log.exception("TTS indisponível; alertas vão só para o log")
            speak = log_speaker

        while (utterance := self._pending.get()) is not None and not self._stopped.is_set():
            self.last_latency_ms = 1000 * (time.perf_counter() - utterance.queued_at)
            try:
                speak(utterance.text)
            except Exception:
                log.exception("Falha ao falar %r", utterance.text)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
