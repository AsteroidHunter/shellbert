"""
Shellbert Settings

Environment configuration and settings management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Attempt to load .env from repository root
_root = Path(__file__).resolve().parent.parent.parent
_env_file = _root / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

# Import model configuration
from .model_registry import DEPLOY_MODEL_CONFIG, TEST_MODEL_CONFIG, DEPLOY_MODEL_KEY, TEST_MODEL_KEY

# -----------------------------------------------------------------------------
# Discord Bot Settings
# -----------------------------------------------------------------------------
DISCORD_BOT_TOKEN: str = os.getenv("DISCORD_BOT_TOKEN", "") or os.getenv("DISCORD_TOKEN", "")
DISCORD_CHANNEL: str = os.getenv("DISCORD_CHANNEL", "#shellbert-chat")
DISCORD_CHANNEL_ID: int = int(os.getenv("DISCORD_CHANNEL_ID", "0")) if os.getenv("DISCORD_CHANNEL_ID") else 0

# Discord bot memory settings
DISCORD_MEMORY_TIMEOUT_HOURS: int = int(os.getenv("DISCORD_MEMORY_TIMEOUT_HOURS", "2"))
DISCORD_MEMORY_MAX_MESSAGES: int = int(os.getenv("DISCORD_MEMORY_MAX_MESSAGES", "10"))

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
# HuggingFace authentication for gated models
HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "") or os.getenv("HF_TOKEN", "")

# Runtime mode selector: "test" (CPU-friendly) vs "deploy" (GPU)
RUN_MODE: str = os.getenv("RUN_MODE", "deploy").lower()

# Debug mode for performance testing (disables safety/personality)
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Export model configurations for backwards compatibility
DEPLOY_MODEL_NAME: str = DEPLOY_MODEL_CONFIG.huggingface_id
TEST_MODEL_NAME: str = TEST_MODEL_CONFIG.huggingface_id

# Current model selection based on run mode
CURRENT_MODEL_CONFIG = TEST_MODEL_CONFIG if RUN_MODE == "test" else DEPLOY_MODEL_CONFIG
CURRENT_MODEL_KEY = TEST_MODEL_KEY if RUN_MODE == "test" else DEPLOY_MODEL_KEY

# -----------------------------------------------------------------------------
# SLURM Configuration for Remote Deployment
# -----------------------------------------------------------------------------
SLURM_PARTITION: str = os.getenv("SLURM_PARTITION", "gpu")
SLURM_CPUS_PER_GPU: int = int(os.getenv("SLURM_CPUS_PER_GPU", "8"))
SLURM_MEM_PER_GPU: str = os.getenv("SLURM_MEM_PER_GPU", "11G")  # RTX 3060 has 12GB total
SLURM_TIME_LIMIT: str = os.getenv("SLURM_TIME_LIMIT", "12:00:00")
SLURM_AUTO_SUBMIT: bool = os.getenv("SLURM_AUTO_SUBMIT", "true").lower() == "true"

# -----------------------------------------------------------------------------
# Legacy Model Configuration
# -----------------------------------------------------------------------------
if RUN_MODE == "test":
    DEFAULT_MODEL: str = TEST_MODEL_NAME
    DEVICE_MAP: str | None = "cpu"
else:
    DEFAULT_MODEL: str = DEPLOY_MODEL_NAME
    DEVICE_MAP: str | None = "auto"

MODEL_PATH: str | None = os.getenv("MODEL_PATH") or None


class SettingsError(RuntimeError):
    """Missing or invalid environment configuration."""


def validate() -> None:
    """Validate that all required settings are present."""
    missing: list[str] = []
    warnings: list[str] = []
    
    # Only require Discord bot token for Discord functionality
    if not DISCORD_BOT_TOKEN:
        missing.append("DISCORD_BOT_TOKEN (or DISCORD_TOKEN)")
    
    # Warn about missing HF token for gated models (but don't fail - fallbacks available)
    if CURRENT_MODEL_CONFIG.requires_auth and not HUGGINGFACE_TOKEN:
        warnings.append(f"HUGGINGFACE_TOKEN missing - will fallback from {CURRENT_MODEL_CONFIG.name} to open models")
    
    # Print warnings but don't fail
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"⚠️  {warning}")
    
    # Only fail for truly required settings
    if missing:
        raise SettingsError("Missing required env vars: " + ", ".join(missing)) 