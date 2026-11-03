import asyncio
import itertools
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.routing import WebSocketRoute

from src.orchestrator import Engine, realtime
from src.server import TelemetryHub, create_app
from src.simulation import RadarSimulator
from src.storage import TelemetryWriter


def test_hub_keeps_only_the_latest_payload() -> None:
    hub = TelemetryHub()
    assert hub.latest() == (0, None)
    hub.publish("a")
    hub.publish("b")
    assert hub.latest() == (2, "b")


def test_dashboard_page_is_served() -> None:
    response = TestClient(create_app(TelemetryHub())).get("/")
    assert response.status_code == 200
    assert "<canvas" in response.text


def test_stream_ends_when_the_client_leaves_even_without_new_frames() -> None:
    """Com o motor parado nada é enviado; antes, o handler nunca percebia a saída do cliente
    e cada painel fechado deixava uma task presa, travando o Ctrl+C do uvicorn."""

    class LeavingClient:
        async def accept(self) -> None:
            pass

        async def receive(self) -> dict[str, str]:
            return {"type": "websocket.disconnect"}

    app = create_app(TelemetryHub())
    stream = next(r.endpoint for r in app.routes if isinstance(r, WebSocketRoute))
    asyncio.run(asyncio.wait_for(stream(LeavingClient()), timeout=1.0))


def test_engine_feeds_dashboard_coach_and_database(tmp_path: Path) -> None:
    hub = TelemetryHub()
    spoken: list[str] = []

    class FakeAudio:
        def say(self, text: str) -> None:
            spoken.append(text)

    with TelemetryWriter(tmp_path / "t.duckdb") as writer:
        engine = Engine(
            itertools.islice(RadarSimulator(fps=10), 10 * 60),
            hub,
            audio=FakeAudio(),
            writer=writer,
        )
        engine.run()
    assert writer.rows_written >= 600 * 24
    assert spoken  # um minuto simulado sempre gera algum alerta

    with TestClient(create_app(hub)).websocket_connect("/ws") as ws:
        payload = json.loads(ws.receive_text())
    assert payload["frame"]["frame_id"] == 599
    assert len(payload["snapshot"]["regions"]) == 22
    assert payload["last_alert"]["message"] in spoken


def test_realtime_paces_the_source() -> None:
    start = time.perf_counter()
    assert list(realtime(range(6), fps=50)) == list(range(6))
    assert time.perf_counter() - start >= 0.1
