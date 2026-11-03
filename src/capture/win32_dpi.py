"""Consciência de DPI no Win32, para a captura enxergar pixels físicos e não escalados."""

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

_PROCESS_PER_MONITOR_DPI_AWARE = 2
_E_ACCESSDENIED = -2147024891  # já definido antes (por outro módulo ou pelo manifesto)


def enable_dpi_awareness() -> bool:
    """Marca o processo como per-monitor DPI aware. Fora do Windows não faz nada."""
    if sys.platform != "win32":
        return False
    try:
        result: int = ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_PER_MONITOR_DPI_AWARE)
    except (AttributeError, OSError):
        # Windows antigo sem shcore: API legada, só "system aware".
        return bool(ctypes.windll.user32.SetProcessDPIAware())
    if result not in (0, _E_ACCESSDENIED):
        log.warning("SetProcessDpiAwareness falhou (HRESULT %d)", result)
        return False
    return True
