"""Segunda tela: FastAPI servindo o painel e transmitindo o estado por WebSocket."""

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from src.models import FrameTelemetry, TacticalSnapshot
from src.models.telemetry import Frozen

STATIC_DIR = Path(__file__).resolve().parent / "static"
STREAM_FPS = 30


class LiveAlert(Frozen):
    message: str
    timestamp: float  # mesmo relógio de `frame.timestamp`


class LivePayload(Frozen):
    """Mensagem enviada ao painel a cada frame transmitido."""

    frame: FrameTelemetry
    snapshot: TacticalSnapshot
    last_alert: LiveAlert | None  # persiste entre frames: o painel pode pular alguns
    cycle_ms: float  # tempo de análise do frame no motor
    engine_fps: float


class TelemetryHub:
    """Ponte entre a thread do motor e o loop asyncio: guarda só o estado mais recente."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0
        self._payload: str | None = None

    def publish(self, payload: str) -> None:
        with self._lock:
            self._version += 1
            self._payload = payload

    def latest(self) -> tuple[int, str | None]:
        with self._lock:
            return self._version, self._payload


async def _until_disconnect(websocket: WebSocket) -> None:
    while (await websocket.receive())["type"] != "websocket.disconnect":
        pass  # o painel não envia nada; qualquer mensagem é ignorada


def create_app(hub: TelemetryHub, stream_fps: int = STREAM_FPS) -> FastAPI:
    app = FastAPI(title="FC-Tactical-Engine")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        # Escuta o socket em paralelo: sem isso, um cliente que cai só é notado no próximo
        # envio, e com o motor parado (nada novo para enviar) o laço nunca termina.
        disconnected = asyncio.create_task(_until_disconnect(websocket))
        sent = 0
        try:
            while not disconnected.done():
                version, payload = hub.latest()
                if payload is not None and version != sent:
                    await websocket.send_text(payload)
                    sent = version
                await asyncio.wait({disconnected}, timeout=1 / stream_fps)
        except (WebSocketDisconnect, OSError):
            pass  # painel fechado ou rede caiu: o navegador reconecta sozinho
        finally:
            disconnected.cancel()

    return app
