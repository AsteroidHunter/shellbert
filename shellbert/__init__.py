"""
Shellbert - EA-optimized conversational AI agent with personality-driven responses.

This package provides a modular architecture for deploying Shellbert across
different platforms (Discord, web, CLI) while maintaining consistent personality
and safety standards.
"""

__version__ = "0.2.0"

# Core components
from .core.agent import ShellbertAgent
from .personality.personality_core import ShellbertPersonality
from .safety.safety_monitor import SafetyMonitor

# Configuration
from .config import validate, SettingsError
from .config.model_registry import get_model_config

__all__ = [
    "ShellbertAgent",
    "ShellbertPersonality", 
    "SafetyMonitor",
    "validate",
    "SettingsError",
    "get_model_config",
] 