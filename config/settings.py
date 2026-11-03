"""Configurações globais do FC-Tactical-Engine (resoluções, thresholds, ROI)."""

from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent
ROI_FILE = CONFIG_DIR / "rois.json"

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

    # Valores de partida medidos em 1080p; o radar original era (760, 820, 1160, 1040) px.
    radar: ROI = ROI.from_pixels((760, 820, 1160, 1040), *TARGET_RESOLUTION)
    scoreboard: ROI = ROI(left=0.03, top=0.03, right=0.25, bottom=0.09)
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
