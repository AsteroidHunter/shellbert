"""
Discord Platform Adapter

Discord-specific implementation using the core Shellbert agent.
Handles Discord API interactions, message formatting, and Discord-specific features.
"""

from .discord_adapter import DiscordAdapter

__all__ = [
    'DiscordAdapter'
] 