"""Configurações globais do FC-Tactical-Engine (resoluções, thresholds, ROI)."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent
ROI_FILE = CONFIG_DIR / "rois.json"
RADAR_BACKGROUND_FILE = CONFIG_DIR / "radar_background.png"
DATA_DIR = CONFIG_DIR.parent / "data"

# Resolução base de referência (Full HD 1080p)
TARGET_RESOLUTION: tuple[int, int] = (1920, 1080)
TARGET_FPS: int = 60

type PixelBox = tuple[int, int, int, int]
"""Caixa em pixels no formato (left, top, right, bottom)."""


class ROI(BaseModel):
    """Região de interesse normalizada em [0.0, 1.0], independente de resolução.

    Assume tela 16:9: o HUD do jogo escala junto com a resolução nesse formato.
    """

    model_config = ConfigDict(frozen=True)

    left: float = Field(ge=0.0, le=1.0)
    top: float = Field(ge=0.0, le=1.0)
    right: float = Field(ge=0.0, le=1.0)
    bottom: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError(f"ROI vazia ou invertida: {self}")
        return self

    @classmethod
    def from_pixels(cls, box: PixelBox, width: int, height: int) -> Self:
        left, top, right, bottom = box
        return cls(left=left / width, top=top / height, right=right / width, bottom=bottom / height)

    def to_pixels(self, width: int, height: int) -> PixelBox:
        return (
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )


class ROIConfig(BaseModel):
    """Conjunto de ROIs capturadas a cada frame. Calibre com `python -m tools.calibrate_roi`."""

    model_config = ConfigDict(frozen=True)

    # Layout do HUD do EA FC 27, medido num vídeo 1920x1072 (radar embaixo no centro, placar
    # no canto superior esquerdo). Os detectores trabalham sobre esses recortes.
    radar: ROI = ROI.from_pixels((760, 845, 1160, 1035), 1920, 1072)
    scoreboard: ROI = ROI.from_pixels((80, 40, 320, 150), 1920, 1072)
    player_hud: ROI = ROI(left=0.0, top=0.85, right=0.25, bottom=1.0)

    def named(self) -> dict[str, ROI]:
        return {name: getattr(self, name) for name in type(self).model_fields}

    @classmethod
    def load(cls, path: Path = ROI_FILE) -> Self:
        """Lê a calibração salva; sem arquivo, usa os valores padrão."""
        return cls.model_validate_json(path.read_text("utf-8")) if path.exists() else cls()

    def save(self, path: Path = ROI_FILE) -> None:
        path.write_text(self.model_dump_json(indent=2), "utf-8")


PITCH_SIZE_M: tuple[float, float] = (105.0, 68.0)
"""Campo oficial (comprimento, largura). Converte o radar normalizado para metros."""


class TacticalThresholds(BaseModel):
    """Limiares do motor tático, em metros."""

    model_config = ConfigDict(frozen=True)

    carrier_radius_m: float = 2.0  # bola até essa distância = posse do jogador
    lane_radius_m: float = 1.5  # raio de interceptação junto ao passador
    lane_growth: float = 0.08  # o raio cresce isso por metro de passe (cone de sombra)
    max_pass_m: float = 40.0
    min_pass_gain_m: float = 15.0  # passe de ruptura precisa avançar pelo menos isso
    min_bypassed: int = 2  # ...e deixar pelo menos esse número de rivais para trás
    max_cb_gap_m: float = 18.0  # acima disso, centro exposto
    max_depth_m: float = 45.0  # acima disso, time espaçado


class CoachConfig(BaseModel):
    """Ritmo dos alertas de voz, para ajudar sem sobrecarregar o jogador."""

    model_config = ConfigDict(frozen=True)

    debounce_s: float = 0.1  # o evento precisa persistir isso antes de virar alerta
    cooldown_s: float = 12.0  # o mesmo alerta não se repete antes disso
    min_gap_s: float = 3.0  # intervalo mínimo entre dois alertas quaisquer
    late_minute: int = 85  # a partir daqui, vencendo, o coach para de pedir ataque


class VisionConfig(BaseModel):
    """Limiares da visão (Marco 2), medidos no radar e no placar do EA FC 27.

    Matiz (H) na escala do OpenCV, 0 a 179. Áreas em pixels do recorte de referência.
    """

    model_config = ConfigDict(frozen=True)

    # Radar presente: escurece o gramado (opacidade) e tem o desenho do radar (similaridade).
    # O fade de entrada/saída fica abaixo de 0,6; cenas escuras sem radar, abaixo de 0,4.
    # Histerese: já visível, só some abaixo dos limiares "keep" (linhas da cena por cima do
    # radar baixam a opacidade para ~0,55 sem ele sumir).
    min_radar_opacity: float = 0.6
    min_radar_similarity: float = 0.4
    keep_radar_opacity: float = 0.45
    keep_radar_similarity: float = 0.3
    min_scoreboard_similarity: float = 0.6
    # Menu de pausa: fundo azul-marinho liso + título branco no canto do placar.
    min_menu_navy: float = 0.8
    menu_title_fraction: tuple[float, float] = (0.03, 0.09)
    min_menu_flat: float = 0.65

    # Cores dos marcadores: time A = triângulos vermelhos, time B = círculos azuis, bola = "+".
    # vermelho passa pelo zero; até 11 cobre o marcador semitransparente no fade (mistura com
    # o verde puxa o matiz para o laranja). A bola começa em 12.
    red_hue: tuple[int, int] = (170, 11)
    blue_hue: tuple[int, int] = (100, 125)
    ball_hue: tuple[int, int] = (12, 30)
    min_saturation: int = 120
    min_ball_saturation: int = 130
    min_value: int = 110
    # Área do miolo colorido de um marcador cresce com a profundidade (perspectiva):
    # area(y) = a + b·y, y em pixels do recorte. Mediana medida em ~1400 frames.
    red_area: tuple[float, float] = (13.0, 0.18)
    blue_area: tuple[float, float] = (17.0, 0.34)
    min_area_ratio: float = 0.35  # menos que isso do esperado: ruído ou marcador quase coberto
    min_ball_area: int = 8
    players_per_team: int = 11

    # Leitura de dígitos do placar e do relógio (modelos em src/vision/assets/digits.png).
    min_digit_score: float = 0.8
    min_digit_margin: float = 0.05

    # Rastreamento: suavização alfa-beta, tempo máximo sem detecção e salto máximo.
    track_alpha: float = 0.35
    track_beta: float = 0.07  # ~ alfa²/(2 - alfa): resposta sem oscilar
    ball_alpha: float = 0.8
    max_coast_s: float = 0.5
    max_speed_m_s: float = 12.0
    gate_margin_m: float = 3.0
    # Entrar em jogo exige o radar por alguns frames (fade); sair é imediato.
    frames_to_resume: int = 3
    # Lado de ataque: média móvel do voto dos goleiros, com histerese.
    side_time_constant_s: float = 3.0
