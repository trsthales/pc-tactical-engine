"""Módulo de geometria computacional e análise tática (Voronoi, linhas de passe)."""

from src.tactics.event_rules import detect_events
from src.tactics.spatial_engine import analyze

__all__ = ["analyze", "detect_events"]
