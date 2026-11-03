"""Módulo de alertas táticos e síntese de voz (TTS)."""

from src.alerts.audio_dispatcher import AudioDispatcher, log_speaker, pyttsx3_speaker
from src.alerts.heuristics_manager import Alert, CoachHeuristics

__all__ = ["Alert", "AudioDispatcher", "CoachHeuristics", "log_speaker", "pyttsx3_speaker"]
