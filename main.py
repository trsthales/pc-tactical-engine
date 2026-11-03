"""FC-Tactical-Engine: coach por voz, gravação da partida e segunda tela no navegador.

    python main.py                       # abre o painel em http://localhost:8000
    python main.py --no-audio --no-db    # só a segunda tela

Até a visão (Marco 2) ficar pronta, a fonte de dados é o radar simulado.
"""

import argparse
import logging
import socket
import threading
import webbrowser
from contextlib import ExitStack

import uvicorn

from config.settings import TARGET_FPS
from src.alerts import AudioDispatcher
from src.orchestrator import Engine, realtime
from src.server import TelemetryHub, create_app
from src.simulation import RadarSimulator
from src.storage import TelemetryWriter

log = logging.getLogger("fc-tactical-engine")


def lan_ip() -> str | None:
    """IP desta máquina na rede local, para abrir o painel no tablet/celular."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 80))  # não envia nada: só escolhe a interface de saída
            return str(probe.getsockname()[0])
        except OSError:
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FC-Tactical-Engine")
    parser.add_argument("--host", default="0.0.0.0", help="use 127.0.0.1 para só este PC")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-audio", dest="audio", action="store_false", help="sem voz")
    parser.add_argument("--no-db", dest="db", action="store_false", help="não grava a partida")
    parser.add_argument("--no-browser", dest="browser", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hub, stop = TelemetryHub(), threading.Event()
    with ExitStack() as stack:
        audio = stack.enter_context(AudioDispatcher()) if args.audio else None
        writer = stack.enter_context(TelemetryWriter()) if args.db else None
        if writer:
            log.info("gravando a partida %s em %s", writer.match_id, writer.path)

        source = realtime(RadarSimulator(fps=TARGET_FPS), TARGET_FPS)
        engine = Engine(source, hub, audio=audio, writer=writer)
        config = uvicorn.Config(
            create_app(hub), host=args.host, port=args.port, log_level="warning"
        )
        server = uvicorn.Server(config)

        def run_engine() -> None:
            try:
                engine.run(stop)
            except Exception:
                log.exception("o motor parou com erro; encerrando")
            finally:
                server.should_exit = True  # painel sem motor só mostraria um frame congelado

        engine_thread = threading.Thread(target=run_engine, name="engine")
        engine_thread.start()

        local_url = f"http://localhost:{args.port}"
        log.info("painel: %s", local_url)
        if args.host == "0.0.0.0" and (ip := lan_ip()):
            log.info("no tablet/celular (mesma rede): http://%s:%d", ip, args.port)
        if args.browser:
            threading.Timer(1.0, webbrowser.open, args=(local_url,)).start()

        try:
            server.run()
        finally:
            stop.set()
            engine_thread.join(timeout=2.0)  # para o motor antes de fechar banco e áudio
            if engine_thread.is_alive():
                log.warning("o motor não parou em 2 s; frames depois daqui não são gravados")


if __name__ == "__main__":
    main()
