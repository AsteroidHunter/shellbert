"""
Model registry and configuration for Shellbert.
Professional ML engineering approach for model selection and GPU allocation.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any, List
import os
import platform
import torch


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    huggingface_id: str
    requires_auth: bool = False
    supports_vision: bool = False
    context_length: int = 4096
    memory_gb_estimate: int = 8  # Estimated memory usage without quantization
    memory_gb_quantized: int = 4  # Estimated memory usage with 4-bit quantization
    chat_template: str = "auto"  # "auto", "llama", "gemma", "custom"
    special_tokens: Optional[Dict[str, str]] = None
    quantization_recommended: bool = True
    fallback_model: Optional[str] = None
    notes: str = ""
    use_mistral_api: bool = False  # Use Mistral-specific API (legacy)
    use_gemma3n_api: bool = False  # Use Gemma 3n specific API
    # NEW: Multi-GPU configuration
    requires_multi_gpu: bool = False  # Whether this model needs multiple GPUs
    recommended_gpu_count: int = 1  # Recommended number of GPUs
    min_gpu_memory_gb: float = 6.0  # Minimum GPU memory per GPU


# Professional Model Registry with clear size-based selection
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "gemma-3n-e2b": ModelConfig(
        name="Gemma 3n E2B Instruct",
        huggingface_id="google/gemma-3n-E2B-it",
        requires_auth=True,
        supports_vision=True,
        context_length=32768,  # 32K context
        memory_gb_estimate=3,  # ~3GB in BF16 (2B effective footprint)
        memory_gb_quantized=2,  # ~2GB with 4-bit quantization
        chat_template="auto",
        quantization_recommended=False,  # Efficient architecture
        use_gemma3n_api=True,
        # Single GPU configuration
        requires_multi_gpu=False,
        recommended_gpu_count=1,
        min_gpu_memory_gb=4.0,  # Can work on smaller GPUs
        notes="Optimized for single GPU deployment (RTX 3060+). Multimodal with 6B params, 2B effective memory footprint."
    ),
    
    "gemma-3n-e4b": ModelConfig(
        name="Gemma 3n E4B Instruct", 
        huggingface_id="google/gemma-3n-E4B-it",
        requires_auth=True,
        supports_vision=True,
        context_length=32768,  # 32K context
        memory_gb_estimate=8,  # ~8GB across multiple GPUs (4B effective footprint)
        memory_gb_quantized=4,  # ~4GB with 4-bit quantization
        chat_template="auto",
        quantization_recommended=False,  # Efficient architecture
        use_gemma3n_api=True,
        # Multi-GPU configuration for better performance
        requires_multi_gpu=True,
        recommended_gpu_count=2,
        min_gpu_memory_gb=6.0,  # Needs larger GPUs when using 2
        notes="High-performance model optimized for dual-GPU deployment. Multimodal with 8B params, 4B effective memory footprint."
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key."""
    if model_key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available models: {available}")
    return MODEL_REGISTRY[model_key]


def get_model_key_from_size() -> str:
    """
    Professional model selection based on MODEL_SIZE environment variable.
    
    MODEL_SIZE options:
    - 'e2b' or 'small': Use E2B (optimized for single GPU, RTX 3060+)
    - 'e4b' or 'large': Use E4B (optimized for dual GPU, high performance)
    
    Falls back to other environment variables if MODEL_SIZE not set.
    
    Automatically selects appropriate model based on platform:
    - macOS/CPU-only: Always use E2B (smaller, CPU-friendly)
    - GPU systems: Respect user choice or default to E2B for compatibility
    """
    # Check platform capabilities
    is_macos = platform.system() == "Darwin"
    has_cuda = torch.cuda.is_available() if 'torch' in globals() else False
    
    # Primary: Use MODEL_SIZE for easy selection
    model_size = os.getenv("MODEL_SIZE", "").lower().strip()
    
    if model_size in ["e2b", "small", "2b"]:
        return "gemma-3n-e2b"
    elif model_size in ["e4b", "large", "4b"]:
        # Force E2B on macOS or CPU-only systems for compatibility
        if is_macos or not has_cuda:
            print(f"⚠️  Forcing E2B model on {'macOS' if is_macos else 'CPU-only'} system (E4B requires GPU)")
            return "gemma-3n-e2b"
        return "gemma-3n-e4b"
    
    # Fallback: Use specific model key if provided
    model_key = os.getenv("MODEL_KEY", "").strip()
    if model_key and model_key in MODEL_REGISTRY:
        # Still enforce platform restrictions
        if model_key == "gemma-3n-e4b" and (is_macos or not has_cuda):
            print(f"⚠️  Forcing E2B model on {'macOS' if is_macos else 'CPU-only'} system (E4B requires GPU)")
            return "gemma-3n-e2b"
        return model_key
    
    # Fallback: Use HuggingFace ID if provided
    hf_id = os.getenv("MODEL_NAME", "").strip()
    if hf_id:
        for key, config in MODEL_REGISTRY.items():
            if config.huggingface_id == hf_id:
                # Still enforce platform restrictions
                if key == "gemma-3n-e4b" and (is_macos or not has_cuda):
                    print(f"⚠️  Forcing E2B model on {'macOS' if is_macos else 'CPU-only'} system (E4B requires GPU)")
                    return "gemma-3n-e2b"
                return key
    
    # Smart default: E2B for best compatibility, especially on macOS/CPU
    if is_macos or not has_cuda:
        print(f"💡 Auto-selecting E2B model for {'macOS' if is_macos else 'CPU-only'} system")
    
    return "gemma-3n-e2b"


def get_default_models() -> tuple[str, str]:
    """Get default model keys for deploy and test modes."""
    # Professional approach: Support both deployment-specific and unified model selection
    
    # For deployment mode
    deploy_model = os.getenv("DEPLOY_MODEL_KEY")
    if not deploy_model:
        deploy_model = get_model_key_from_size()
    
    # For test mode (typically use smaller model)
    test_model = os.getenv("TEST_MODEL_KEY")
    if not test_model:
        # Default to smaller model for testing unless explicitly overridden
        test_size = os.getenv("TEST_MODEL_SIZE", "e2b").lower()
        if test_size in ["e2b", "small", "2b"]:
            test_model = "gemma-3n-e2b"
        elif test_size in ["e4b", "large", "4b"]:
            test_model = "gemma-3n-e4b"
        else:
            test_model = "gemma-3n-e2b"  # Safe default
    
    # Validate model keys
    def resolve_model_key(model_input: str) -> str:
        if model_input in MODEL_REGISTRY:
            return model_input
        
        # Try to find by HuggingFace ID
        for key, config in MODEL_REGISTRY.items():
            if config.huggingface_id == model_input:
                return key
        
        # Fallback with warning
        print(f"Warning: Unknown model '{model_input}', falling back to gemma-3n-e2b")
        return "gemma-3n-e2b"
    
    deploy_model = resolve_model_key(deploy_model)
    test_model = resolve_model_key(test_model)
    
    return deploy_model, test_model


def should_use_quantization(model_config: ModelConfig, available_memory_gb: float = 12.0) -> bool:
    """Determine if quantization should be used based on model and available memory."""
    if not model_config.quantization_recommended:
        return False
    
    # Use quantization if model's estimated memory usage exceeds 80% of available memory
    memory_threshold = available_memory_gb * 0.8
    return model_config.memory_gb_estimate > memory_threshold


def get_effective_memory_usage(model_config: ModelConfig, use_quantization: bool = False) -> int:
    """Get the effective memory usage considering quantization."""
    if use_quantization and hasattr(model_config, 'memory_gb_quantized'):
        return model_config.memory_gb_quantized
    return model_config.memory_gb_estimate


def get_chat_template_type(model_config: ModelConfig) -> str:
    """Get the chat template type for a model."""
    if model_config.chat_template == "auto":
        # Auto-detect based on model name
        model_name_lower = model_config.huggingface_id.lower()
        if "gemma" in model_name_lower:
            return "gemma"
        elif "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        else:
            return "generic"
    return model_config.chat_template


def list_available_models() -> str:
    """Get a formatted list of available models for display."""
    lines = ["Available models:"]
    for key, config in MODEL_REGISTRY.items():
        memory_str = f"{config.memory_gb_estimate}GB"
        if hasattr(config, 'memory_gb_quantized') and config.quantization_recommended:
            memory_str += f" ({config.memory_gb_quantized}GB quantized)"
        
        gpu_str = f" ({config.recommended_gpu_count} GPU)" if config.recommended_gpu_count == 1 else f" ({config.recommended_gpu_count} GPUs)"
        vision_str = " (Vision)" if config.supports_vision else ""
        auth_str = " (Requires Auth)" if config.requires_auth else ""
        
        lines.append(f"  {key}: {config.name} - {memory_str}{gpu_str}{vision_str}{auth_str}")
        if config.notes:
            lines.append(f"    Note: {config.notes}")
    return "\n".join(lines)


def get_gpu_requirements(model_config: ModelConfig) -> Dict[str, Any]:
    """Get GPU requirements for a model."""
    return {
        "requires_multi_gpu": model_config.requires_multi_gpu,
        "recommended_gpu_count": model_config.recommended_gpu_count,
        "min_gpu_memory_gb": model_config.min_gpu_memory_gb,
        "total_memory_estimate_gb": model_config.memory_gb_estimate
    }


# Export current model selection based on environment
DEPLOY_MODEL_KEY, TEST_MODEL_KEY = get_default_models()
DEPLOY_MODEL_CONFIG = get_model_config(DEPLOY_MODEL_KEY)
TEST_MODEL_CONFIG = get_model_config(TEST_MODEL_KEY) 