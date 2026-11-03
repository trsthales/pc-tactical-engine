"""FC-Tactical-Engine: coach por voz, gravação da partida e segunda tela no navegador.

    python main.py --screen              # jogo aberto (Windows): lê o radar da tela
    python main.py --video partida.mp4   # vídeo gravado, no ritmo do vídeo
    python main.py --video partida.mp4 --start 12:30 --speed 2   # pula até 12:30, 2x
    python main.py                       # radar simulado (sem jogo)
    python main.py --no-audio --no-db    # só a segunda tela

Times: por padrão o usuário (time A) é o triângulo vermelho e a linha de cima do placar
(mandante). Visitante: --team-a-blue --team-a-bottom.
"""

import argparse
import logging
import socket
import threading
import webbrowser
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path

import uvicorn

from config.settings import TARGET_FPS, ROIConfig
from src.alerts import AudioDispatcher
from src.capture import CapturedFrame
from src.capture.screen_capture import ScreenCapture
from src.models import FrameTelemetry
from src.orchestrator import Engine, realtime
from src.server import TelemetryHub, create_app
from src.simulation import RadarSimulator
from src.storage import TelemetryWriter
from src.vision.pipeline import TeamSetup, VisionPipeline
from src.vision.video_source import VideoSource

log = logging.getLogger("fc-tactical-engine")


def video_time(text: str) -> float:
    """Posição no vídeo: segundos (750), mm:ss (12:30) ou hh:mm:ss (1:02:30)."""
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        parts = []
    if not 1 <= len(parts) <= 3 or any(p < 0 for p in parts):
        raise argparse.ArgumentTypeError(f"tempo inválido: {text!r} (use 750, 12:30 ou 1:02:30)")
    seconds = 0.0
    for part in parts:
        seconds = 60 * seconds + part
    return seconds


def positive(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("precisa ser maior que zero")
    return value


def lan_ip() -> str | None:
    """IP desta máquina na rede local, para abrir o painel no tablet/celular."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 80))  # não envia nada: só escolhe a interface de saída
            return str(probe.getsockname()[0])
        except OSError:
            return None


def screen_frames(capture: ScreenCapture, stop: threading.Event) -> Iterator[CapturedFrame]:
    """Frames da tela até a captura parar. Tela parada (menu) não trava o encerramento."""
    while capture.running and not stop.is_set():
        if (frame := capture.frames.get(timeout=0.5)) is not None:
            yield frame
    if not stop.is_set():
        log.error("a captura de tela parou")


def telemetry_source(
    args: argparse.Namespace, stack: ExitStack, stop: threading.Event
) -> Iterable[FrameTelemetry]:
    if not (args.screen or args.video):
        return realtime(RadarSimulator(fps=TARGET_FPS), TARGET_FPS)
    rois = ROIConfig.load()
    teams = TeamSetup(a_is_red=not args.team_a_blue, a_on_top=not args.team_a_bottom)
    pipeline = VisionPipeline(teams=teams)
    frames: Iterable[CapturedFrame]
    if args.video:
        video = VideoSource(args.video, rois, start_s=args.start)
        frames = realtime(video, video.fps * args.speed)  # lento demais? roda o mais rápido que der
    else:
        frames = screen_frames(stack.enter_context(ScreenCapture(rois)), stop)
    return (pipeline.process(frame) for frame in frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FC-Tactical-Engine")
    parser.add_argument("--host", default="0.0.0.0", help="use 127.0.0.1 para só este PC")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-audio", dest="audio", action="store_false", help="sem voz")
    parser.add_argument("--no-db", dest="db", action="store_false", help="não grava a partida")
    parser.add_argument("--no-browser", dest="browser", action="store_false")
    origin = parser.add_mutually_exclusive_group()
    origin.add_argument("--screen", action="store_true", help="lê o jogo na tela (Windows)")
    origin.add_argument("--video", type=Path, help="lê um vídeo gravado da partida")
    parser.add_argument("--start", type=video_time, default=0.0, help="com --video: 12:30")
    parser.add_argument("--speed", type=positive, default=1.0, help="com --video: 2 = 2x")
    parser.add_argument("--team-a-blue", action="store_true", help="você é o círculo azul")
    parser.add_argument("--team-a-bottom", action="store_true", help="você é a linha de baixo")
    args = parser.parse_args()
    if not args.video and (args.start or args.speed != 1.0):
        parser.error("--start e --speed só valem com --video")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    hub, stop = TelemetryHub(), threading.Event()
    with ExitStack() as stack:
        audio = stack.enter_context(AudioDispatcher()) if args.audio else None
        writer = stack.enter_context(TelemetryWriter()) if args.db else None
        if writer:
            log.info("gravando a partida %s em %s", writer.match_id, writer.path)

        engine = Engine(telemetry_source(args, stack, stop), hub, audio=audio, writer=writer)
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
