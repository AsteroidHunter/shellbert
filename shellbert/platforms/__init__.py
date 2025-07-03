"""
Shellbert Platform Adapters

Platform-specific implementations that use the core Shellbert agent.
Each platform adapter handles the specific requirements and interfaces of that platform.
"""

from .discord import DiscordAdapter

__all__ = [
    'DiscordAdapter'
] 