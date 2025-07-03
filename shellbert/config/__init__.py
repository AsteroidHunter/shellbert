"""
Configuration module for Shellbert.

Provides access to settings and validates required environment variables.
"""

# Import all settings and functions from the settings module
from .settings import (
    # Discord Settings
    DISCORD_BOT_TOKEN,
    DISCORD_CHANNEL,
    DISCORD_CHANNEL_ID,
    DISCORD_MEMORY_TIMEOUT_HOURS,
    DISCORD_MEMORY_MAX_MESSAGES,
    
    # Model Configuration
    HUGGINGFACE_TOKEN,
    RUN_MODE,
    DEBUG_MODE,
    DEPLOY_MODEL_NAME,
    TEST_MODEL_NAME,
    CURRENT_MODEL_CONFIG,
    CURRENT_MODEL_KEY,
    
    # SLURM Configuration
    SLURM_PARTITION,
    SLURM_CPUS_PER_GPU,
    SLURM_MEM_PER_GPU,
    SLURM_TIME_LIMIT,
    SLURM_AUTO_SUBMIT,
    
    # Legacy Model Configuration
    DEFAULT_MODEL,
    DEVICE_MAP,
    MODEL_PATH,
    
    # Functions and classes
    SettingsError,
    validate
)

# Import model registry
from .model_registry import (
    DEPLOY_MODEL_CONFIG,
    TEST_MODEL_CONFIG,
    DEPLOY_MODEL_KEY,
    TEST_MODEL_KEY
)

__all__ = [
    # Discord Settings
    'DISCORD_BOT_TOKEN',
    'DISCORD_CHANNEL',
    'DISCORD_CHANNEL_ID',
    'DISCORD_MEMORY_TIMEOUT_HOURS',
    'DISCORD_MEMORY_MAX_MESSAGES',
    
    # Model Configuration
    'HUGGINGFACE_TOKEN',
    'RUN_MODE',
    'DEBUG_MODE',
    'DEPLOY_MODEL_NAME',
    'TEST_MODEL_NAME',
    'CURRENT_MODEL_CONFIG',
    'CURRENT_MODEL_KEY',
    
    # SLURM Configuration
    'SLURM_PARTITION',
    'SLURM_CPUS_PER_GPU',
    'SLURM_MEM_PER_GPU',
    'SLURM_TIME_LIMIT',
    'SLURM_AUTO_SUBMIT',
    
    # Legacy Model Configuration
    'DEFAULT_MODEL',
    'DEVICE_MAP',
    'MODEL_PATH',
    
    # Model registry
    'DEPLOY_MODEL_CONFIG',
    'TEST_MODEL_CONFIG',
    'DEPLOY_MODEL_KEY',
    'TEST_MODEL_KEY',
    
    # Functions and classes
    'SettingsError',
    'validate'
] 