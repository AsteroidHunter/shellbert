"""
Shellbert Core Module

Core agent logic and platform-agnostic functionality.
"""

from .agent import ShellbertAgent
from .llm_interface import ShellbertLLM, get_shellbert_llm

__all__ = [
    'ShellbertAgent',
    'ShellbertLLM',
    'get_shellbert_llm'
] 