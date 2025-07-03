"""
Shellbert Personality System

This module implements Shellbert's core personality traits and consistency mechanisms.
Based on research from Anthropic's Constitutional AI and OpenAI's latest personality engineering.
"""

from .personality_core import PersonalityCore, ShellbertPersonality
from .traits.shellbert_traits import SHELLBERT_TRAITS, ShellbertTraitManager
from .consistency.consistency_checker import PersonalityConsistencyChecker

__all__ = [
    'PersonalityCore',
    'ShellbertPersonality', 
    'SHELLBERT_TRAITS',
    'ShellbertTraitManager',
    'PersonalityConsistencyChecker'
] 