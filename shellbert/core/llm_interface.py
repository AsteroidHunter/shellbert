"""
Advanced LLM Interface for Shellbert

A comprehensive LLM interface that supports:
- Gemma 3n models with multimodal capabilities
- Standard transformers models with proper quantization
- Robust fallback system with error handling
- GPU memory optimization and device detection
"""

import os
import logging
import torch

# CRITICAL: Set memory optimization BEFORE any CUDA operations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# PERFORMANCE: Disable PyTorch compilation for faster inference
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Additional compilation disabling
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_JIT_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
# Disable additional optimizations that cause delays
torch.backends.cudnn.benchmark = False

from typing import Dict, List, Optional, Any, Tuple, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline
)
# Import Gemma3n specific class
try:
    from transformers import Gemma3nForConditionalGeneration
except ImportError:
    # Fallback for older transformers versions
    Gemma3nForConditionalGeneration = None
from transformers.pipelines import Pipeline
from datetime import datetime, timedelta
from PIL import Image
import io
import base64
import requests
from pathlib import Path

from ..config import (
    CURRENT_MODEL_CONFIG, 
    CURRENT_MODEL_KEY,
    RUN_MODE,
    HUGGINGFACE_TOKEN
)
from ..config.model_registry import get_model_config, should_use_quantization, get_effective_memory_usage, get_gpu_requirements

logger = logging.getLogger(__name__)


class ShellbertLLM:
    """
    Advanced LLM interface with professional multi-GPU support.
    
    Features:
    - Automatic single/multi-GPU selection based on model requirements
    - Shared server-safe GPU detection
    - Professional device mapping strategies
    - Robust fallback mechanisms
    """
    
    def __init__(self):
        # Initialize configuration
        self._model_config = CURRENT_MODEL_CONFIG
        self._model_key = CURRENT_MODEL_KEY
        self._gpu_requirements = get_gpu_requirements(self._model_config)
        
        logger.info(f"🤖 Initializing Shellbert LLM: {self._model_config.name}")
        logger.info(f"📊 GPU Requirements: {self._gpu_requirements}")
        
        # Initialize state tracking
        self._is_initialized = False
        self._is_gemma3n_model = self._model_config.use_gemma3n_api
        self._device = "cpu"  # Will be updated during device mapping
        
        # Initialize device and quantization settings
        self._device_map = self._setup_device_mapping()
        self._quantization_config = self._setup_quantization()
        
        # Update device based on device mapping
        self._device = self._get_device()
        
        # Model components
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._pipeline = None
        
        # Load model
        try:
            self._load_model()
            self._is_initialized = True
            logger.info(f"✅ ShellbertLLM initialization completed successfully")
        except Exception as e:
            self._is_initialized = False
            logger.error(f"❌ ShellbertLLM initialization failed: {e}")
            raise
    
    def _find_idle_gpus(self, required_count: int, min_memory_gb: float = 6.0, max_usage_mb: float = 100) -> List[int]:
        """
        Find the required number of completely idle GPUs for multi-GPU deployment.
        
        Args:
            required_count: Number of GPUs needed
            min_memory_gb: Minimum memory per GPU
            max_usage_mb: Maximum current usage to consider GPU idle
            
        Returns:
            List of GPU IDs that are completely idle, or empty list if insufficient GPUs
        """
        if not torch.cuda.is_available():
            logger.error("❌ No CUDA GPUs available")
            return []
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"🔍 Scanning {gpu_count} GPUs for {required_count} idle GPU(s)...")
        logger.info(f"📋 Criteria per GPU: <{max_usage_mb}MB used AND >{min_memory_gb:.1f}GB available")
        
        idle_gpus = []
        
        for gpu_id in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                allocated = torch.cuda.memory_allocated(device=gpu_id)
                reserved = torch.cuda.memory_reserved(device=gpu_id)
                used_memory = max(allocated, reserved)
                
                free_memory_gb = (props.total_memory - used_memory) / 1024**3
                used_memory_mb = used_memory / 1024**2
                
                # Check if GPU meets criteria
                is_idle = used_memory_mb <= max_usage_mb
                has_sufficient_memory = free_memory_gb >= min_memory_gb
                
                if is_idle and has_sufficient_memory:
                    idle_gpus.append(gpu_id)
                    logger.info(f"✅ GPU {gpu_id} ({props.name}): {free_memory_gb:.1f}GB free, {used_memory_mb:.0f}MB used - AVAILABLE")
                else:
                    reason = "insufficient memory" if not has_sufficient_memory else "in use"
                    logger.info(f"❌ GPU {gpu_id} ({props.name}): {free_memory_gb:.1f}GB free, {used_memory_mb:.0f}MB used - {reason}")
                    
            except Exception as e:
                logger.warning(f"⚠️  Could not check GPU {gpu_id}: {e}")
        
        if len(idle_gpus) >= required_count:
            selected_gpus = idle_gpus[:required_count]
            logger.info(f"🎯 Selected GPUs: {selected_gpus}")
            return selected_gpus
        else:
            logger.error(f"❌ Insufficient idle GPUs: Found {len(idle_gpus)}, need {required_count}")
            return []
    
    def _setup_device_mapping(self) -> Union[str, Dict[str, int], None]:
        """
        Setup professional device mapping based on model requirements.
        
        Device priority: CUDA (production) -> MPS (macOS debugging) -> CPU (fallback)
        
        Returns:
            Device mapping configuration for the model
        """
        requires_multi_gpu = self._gpu_requirements["requires_multi_gpu"]
        recommended_gpu_count = self._gpu_requirements["recommended_gpu_count"]
        min_gpu_memory_gb = self._gpu_requirements["min_gpu_memory_gb"]
        
        # Priority 1: CUDA for production deployment
        if torch.cuda.is_available():
            logger.info("🚀 CUDA available - using production GPU deployment")
            
            if requires_multi_gpu:
                logger.info(f"🔧 Model requires multi-GPU deployment ({recommended_gpu_count} GPUs)")
                
                # Find required number of idle GPUs
                idle_gpus = self._find_idle_gpus(
                    required_count=recommended_gpu_count,
                    min_memory_gb=min_gpu_memory_gb,
                    max_usage_mb=50  # Very strict for shared servers
                )
                
                if len(idle_gpus) >= recommended_gpu_count:
                    # Create explicit device mapping for multi-GPU
                    # Spread layers across available GPUs
                    device_map = {}
                    selected_gpus = idle_gpus[:recommended_gpu_count]
                    
                    # For Gemma 3n models, we can use simple round-robin or let transformers auto-allocate
                    if recommended_gpu_count == 2:
                        # Simple dual-GPU setup - let transformers handle the distribution
                        device_map = "auto"
                        # Set CUDA_VISIBLE_DEVICES to limit to our selected GPUs
                        cuda_visible = ",".join(str(gpu) for gpu in selected_gpus)
                        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible
                        logger.info(f"🎯 Multi-GPU setup: Using GPUs {selected_gpus} (CUDA_VISIBLE_DEVICES={cuda_visible})")
                    else:
                        # For other multi-GPU configurations, use explicit mapping
                        device_map = {f"cuda:{i}": selected_gpus[i] for i in range(len(selected_gpus))}
                        logger.info(f"🎯 Multi-GPU explicit mapping: {device_map}")
                    
                    return device_map
                else:
                    logger.warning(f"⚠️  Cannot find {recommended_gpu_count} idle GPUs, falling back to single GPU")
                    # Fall through to single GPU logic
            
            # Single GPU CUDA setup (either by design or fallback)
            logger.info("🔧 Setting up single GPU CUDA deployment")
            
            idle_gpus = self._find_idle_gpus(
                required_count=1,
                min_memory_gb=min_gpu_memory_gb,
                max_usage_mb=50  # Strict for shared servers
            )
            
            if idle_gpus:
                selected_gpu = idle_gpus[0]
                torch.cuda.set_device(selected_gpu)
                # Use explicit device mapping for single GPU
                device_map = {"": selected_gpu}
                logger.info(f"🎯 Single GPU setup: Using GPU {selected_gpu}")
                return device_map
            else:
                logger.warning("⚠️  No idle GPUs found, checking other device options...")
                # Fall through to MPS/CPU
        
        # Priority 2: MPS for macOS debugging
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 MPS (Metal Performance Shaders) available - using Apple Silicon GPU")
            
            if requires_multi_gpu:
                logger.warning("⚠️  MPS doesn't support multi-GPU, forcing single device deployment")
            
            # Check if model memory estimate is within MPS limits (avoid the 2^32 bytes error)
            model_memory_gb = self._gpu_requirements["total_memory_estimate_gb"]
            if model_memory_gb > 2.5:  # Conservative limit for MPS
                logger.warning(f"⚠️  Model too large for MPS ({model_memory_gb}GB > 2.5GB limit), falling back to CPU")
                logger.info("💡 Use CUDA system for full performance with larger models")
                # Fall through to CPU
            else:
                # MPS doesn't support device mapping like CUDA, so we use a simple string
                logger.info("🎯 MPS setup: Using Apple Silicon GPU for debugging")
                return "mps"
        
        # Priority 3: CPU fallback
        logger.warning("⚠️  No GPU acceleration available, falling back to CPU")
        if requires_multi_gpu:
            logger.warning("⚠️  Multi-GPU model running on CPU - expect very slow performance")
        
        return "cpu"
    
    def _get_device(self) -> str:
        """Get the primary device for tensor operations (legacy compatibility)."""
        if isinstance(self._device_map, dict):
            # For multi-GPU, return the first GPU
            for device_name in self._device_map.values():
                if isinstance(device_name, int):
                    return f"cuda:{device_name}"
                elif isinstance(device_name, str) and device_name.startswith("cuda"):
                    return device_name
            return "cuda:0"  # Fallback
        elif isinstance(self._device_map, str):
            if self._device_map == "auto":
                return "cuda:0"  # Assume first GPU for auto
            elif self._device_map == "mps":
                return "mps"  # Apple Silicon GPU
            elif self._device_map == "cpu":
                return "cpu"
            else:
                return self._device_map
        else:
            return "cpu"
    
    def _setup_quantization(self) -> Optional[BitsAndBytesConfig]:
        """Setup quantization configuration based on model and hardware."""
        device_str = self._get_device()
        
        # Disable quantization for CPU and MPS (bitsandbytes doesn't support them)
        if device_str == "cpu":
            logger.info("📊 Quantization: Disabled for CPU inference")
            return None
        elif device_str == "mps":
            logger.info("📊 Quantization: Disabled for MPS (Apple Silicon)")
            return None
        
        # For Gemma 3n models, quantization should rarely be needed due to efficient architecture
        if self._model_config.use_gemma3n_api:
            # Check if we have sufficient GPU memory across all GPUs
            if torch.cuda.is_available():
                total_gpu_memory_gb = 0
                if isinstance(self._device_map, str) and self._device_map == "auto":
                    # Multi-GPU auto - estimate based on visible devices
                    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
                    gpu_ids = [int(x) for x in cuda_visible.split(",")]
                    for gpu_id in gpu_ids:
                        total_gpu_memory_gb += torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                elif isinstance(self._device_map, dict):
                    # Explicit mapping
                    unique_gpus = set()
                    for device in self._device_map.values():
                        if isinstance(device, int):
                            unique_gpus.add(device)
                    for gpu_id in unique_gpus:
                        total_gpu_memory_gb += torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                else:
                    # Single GPU
                    total_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                logger.info(f"📊 Total GPU Memory: {total_gpu_memory_gb:.1f}GB available")
                
                # Gemma 3n architecture is efficient - only quantize for very limited VRAM
                memory_needed = self._model_config.memory_gb_estimate
                if total_gpu_memory_gb >= memory_needed * 1.2:  # 20% buffer
                    logger.info("📊 Quantization: Disabled - Sufficient GPU memory for native precision")
                    logger.info(f"💡 Model uses efficient architecture: {memory_needed}GB needed, {total_gpu_memory_gb:.1f}GB available")
                    return None
                else:
                    logger.info(f"📊 Quantization: Enabled - Limited VRAM ({total_gpu_memory_gb:.1f}GB < {memory_needed * 1.2:.1f}GB needed)")
            else:
                logger.info("📊 Quantization: Disabled - Could not determine GPU memory")
                return None
        
        # Setup quantization for models that need it
        if not self._model_config.quantization_recommended:
            logger.info(f"📊 Quantization: Disabled for {self._model_config.name}")
            return None
        
        logger.info("📊 Setting up 4-bit quantization...")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    def _load_model(self):
        """Load the model based on the device mapping and quantization configuration."""
        device_str = self._get_device()
        
        logger.info(f"🚀 Loading model: {self._model_config.name}")
        logger.info(f"📍 Device mapping: {self._device_map}")
        logger.info(f"⚙️  Quantization: {'Enabled' if self._quantization_config else 'Disabled'}")
        
        try:
            if self._model_config.use_gemma3n_api:
                self._load_gemma3n_model()
            else:
                self._load_standard_model()
            
            logger.info("🎉 Model loading completed successfully")
            
            # Clean up device-specific cache after loading
            if device_str.startswith("cuda"):
                torch.cuda.empty_cache()
                logger.info("🔒 GPU cache cleared")
            elif device_str == "mps":
                try:
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    logger.info("🔒 MPS cache cleared")
                except Exception as e:
                    logger.warning(f"⚠️  MPS cache clear failed: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error("💡 This might be due to insufficient GPU memory or hardware constraints")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _load_gemma3n_model(self):
        """Load Gemma 3n model using the official Gemma3nForConditionalGeneration API."""
        try:
            # Check if Gemma3nForConditionalGeneration is available
            if Gemma3nForConditionalGeneration is None:
                raise ImportError("Gemma3nForConditionalGeneration not available. Please upgrade transformers to >=4.53.0")
            
            logger.info("🔥 Loading Gemma 3n model with official implementation...")
            
            # Load processor for multimodal support
            processor_kwargs = {}
            if self._model_config.requires_auth and HUGGINGFACE_TOKEN:
                processor_kwargs["token"] = HUGGINGFACE_TOKEN
            
            self._processor = AutoProcessor.from_pretrained(
                self._model_config.huggingface_id,
                **processor_kwargs
            )
            
            # Setup model loading with configured device mapping
            device_str = self._get_device()
            
            # Choose appropriate dtype based on device
            if device_str == "cpu":
                torch_dtype = torch.float32  # CPU prefers float32
            elif device_str == "mps":
                torch_dtype = torch.float16  # MPS works well with float16
            else:
                torch_dtype = torch.bfloat16  # CUDA prefers bfloat16
            
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": self._device_map,  # Use our configured device mapping (single or multi-GPU)
                "low_cpu_mem_usage": True,
            }
            
            # Only add quantization if absolutely necessary (fallback)
            if self._quantization_config:
                logger.info("🔧 Adding quantization as fallback for limited VRAM")
                model_kwargs["quantization_config"] = self._quantization_config
                # Note: Gemma 3n is designed for efficient memory usage, quantization should rarely be needed
            else:
                if self._gpu_requirements["requires_multi_gpu"] and device_str.startswith("cuda"):
                    logger.info(f"💡 Using native efficient architecture across {self._gpu_requirements['recommended_gpu_count']} GPUs")
                elif device_str == "mps":
                    logger.info("💡 Using native efficient architecture on Apple Silicon")
                elif device_str == "cpu":
                    logger.info("💡 Using native efficient architecture on CPU")
                else:
                    logger.info("💡 Using native efficient architecture (single GPU deployment)")
            
            # Add authentication if required
            if self._model_config.requires_auth and HUGGINGFACE_TOKEN:
                model_kwargs["token"] = HUGGINGFACE_TOKEN
            
            self._model = Gemma3nForConditionalGeneration.from_pretrained(
                self._model_config.huggingface_id,
                **model_kwargs
            ).eval()
            
            # Log device allocation
            if hasattr(self._model, 'hf_device_map'):
                logger.info(f"🔍 Model device map: {self._model.hf_device_map}")
            else:
                model_device = next(self._model.parameters()).device
                logger.info(f"🔍 Model loaded on device: {model_device}")
            
            # Determine deployment type for logging
            if device_str.startswith("cuda") and self._gpu_requirements["requires_multi_gpu"]:
                deployment_type = "multi-GPU CUDA"
            elif device_str.startswith("cuda"):
                deployment_type = "single GPU CUDA"
            elif device_str == "mps":
                deployment_type = "Apple Silicon MPS"
            else:
                deployment_type = "CPU"
            
            logger.info(f"✅ Gemma 3n model loaded successfully on {deployment_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load Gemma 3n model: {e}")
            raise
    
    def _load_standard_model(self):
        """Load standard model using transformers."""
        logger.info("🤖 Loading standard model using transformers...")
        device_str = self._get_device()
        
        # Load tokenizer
        tokenizer_kwargs = {}
        if self._model_config.requires_auth and HUGGINGFACE_TOKEN:
            tokenizer_kwargs["token"] = HUGGINGFACE_TOKEN
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.huggingface_id,
            **tokenizer_kwargs
        )
        
        # Ensure pad token exists
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Choose appropriate dtype based on device
        if device_str == "cpu":
            torch_dtype = torch.float32  # CPU prefers float32
        elif device_str == "mps":
            torch_dtype = torch.float16  # MPS works well with float16
        else:
            torch_dtype = torch.bfloat16  # CUDA prefers bfloat16
        
        # Load model
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": self._device_map,
        }
        
        # Add authentication if required
        if self._model_config.requires_auth and HUGGINGFACE_TOKEN:
            model_kwargs["token"] = HUGGINGFACE_TOKEN
        
        # Add quantization if recommended
        if self._quantization_config:
            model_kwargs["quantization_config"] = self._quantization_config
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_config.huggingface_id,
            **model_kwargs
        )
        
        # Move to device if not using device_map and on CPU/MPS
        if self._device_map in ["cpu", "mps"]:
            self._model = self._model.to(device_str)
        
        logger.info(f"✅ Standard model loaded successfully on {device_str}")
    
    def generate_response(self, 
                         user_input: str, 
                         context: str = "", 
                         max_tokens: int = 512,
                         temperature: float = 0.7,
                         timeout_seconds: int = 30) -> str:
        """Generate a response using the loaded model with timeout protection."""
        if not self._model:
            return "I'm sorry, I'm not available right now due to a model loading issue. Please try again later."
        
        try:
            import asyncio
            import concurrent.futures
            
            def _sync_generate():
                if self._model_config.use_gemma3n_api:
                    return self._generate_gemma3n_response(user_input, context, max_tokens, temperature)
                else:
                    return self._generate_standard_response(user_input, context, max_tokens, temperature)
            
            # Run generation with timeout in thread pool to avoid blocking
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_sync_generate)
                    return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                logger.warning(f"⏰ Generation timed out after {timeout_seconds}s")
                return "I'm taking longer than expected to respond. Please try a simpler question or try again."
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "I encountered an error while generating a response. Please try again."
    
    def _generate_gemma3n_response(self, 
                                  user_input: str, 
                                  context: str = "",
                                  max_tokens: int = 512,
                                  temperature: float = 0.7) -> str:
        """Generate response using Gemma 3n model."""
        # Prepare messages in chat format
        messages = []
        
        # Add system message if context provided
        if context:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": context}]
            })
        
        # Add user message
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        })
        
        # Apply chat template and tokenize
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Ensure all input tensors are on the same device as the model
        model_device = next(self._model.parameters()).device
        if isinstance(inputs, dict):
            inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            inputs = inputs.to(model_device)
        
        logger.debug(f"Input tensors moved to device: {model_device}")
        
        input_len = inputs["input_ids"].shape[-1]
        
        # OPTIMIZED: Generate response with faster settings
        with torch.inference_mode():
            # Optimized generation parameters for speed and reliability
            generation_kwargs = {
                "max_new_tokens": min(max_tokens, 128),  # Further reduce for faster generation
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.8,  # Slightly more focused
                "top_k": 40,   # Reduced for faster sampling
                "pad_token_id": self._processor.tokenizer.eos_token_id,
                "use_cache": True,  # Enable KV caching
                "num_beams": 1,     # Disable beam search for speed
                "repetition_penalty": 1.1,  # Prevent repetition
            }
            
            # Additional optimizations to prevent blocking
            torch.set_grad_enabled(False)  # Ensure no gradients
            
            try:
                generation = self._model.generate(
                    **inputs,
                    **generation_kwargs
                )[0]
            except Exception as e:
                logger.error(f"❌ Generation failed: {e}")
                # Fallback with even simpler generation
                try:
                    simple_kwargs = {
                        "max_new_tokens": 50,
                        "do_sample": False,  # Greedy for reliability
                        "pad_token_id": self._processor.tokenizer.eos_token_id,
                    }
                    generation = self._model.generate(**inputs, **simple_kwargs)[0]
                except Exception as e2:
                    logger.error(f"❌ Fallback generation also failed: {e2}")
                    return "I'm experiencing technical difficulties. Please try again."
        
        # Decode only the new tokens
        response_tokens = generation[input_len:]
        response = self._processor.decode(response_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def _generate_standard_response(self, 
                                   user_input: str, 
                                   context: str = "",
                                   max_tokens: int = 512,
                                   temperature: float = 0.7) -> str:
        """Generate response using standard transformers."""
        # Prepare prompt
        if context:
            prompt = f"Context: {context}\n\nUser: {user_input}\nAssistant:"
        else:
            prompt = f"User: {user_input}\nAssistant:"
        
        # Generate using pipeline
        with torch.no_grad():
            response = self._pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )[0]['generated_text']
        
        return response.strip()
    
    def generate_multimodal_response(
        self, 
        text: str = None, 
        images: List[Union[str, Image.Image, bytes]] = None,
        audio_files: List[str] = None,
        video_files: List[str] = None,
        context: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a multimodal response using text, images, audio, and/or video.
        
        Args:
            text: Text input (optional if other modalities provided)
            images: List of images (file paths, PIL Images, or bytes)
            audio_files: List of audio file paths
            video_files: List of video file paths
            context: Additional context for the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response string
        """
        if not self.is_available():
            raise RuntimeError("🚫 Model is not available for inference")
        
        if not any([text, images, audio_files, video_files]):
            raise ValueError("❌ At least one input modality (text, images, audio, video) must be provided")
        
        try:
            # Process images if provided
            processed_images = []
            if images:
                logger.info(f"🖼️  Processing {len(images)} images...")
                for i, img in enumerate(images):
                    try:
                        processed_img = self._process_image(img)
                        processed_images.append(processed_img)
                        logger.info(f"✅ Processed image {i+1}: {processed_img.size}")
                    except Exception as e:
                        logger.error(f"❌ Failed to process image {i+1}: {e}")
                        continue
            
            # Handle multimodal inputs based on model type
            if self._is_gemma3n_model:
                return self._generate_gemma3n_multimodal_response(
                    text, processed_images, audio_files, video_files, 
                    context, max_tokens, temperature
                )
            else:
                # Fallback for standard models - text only
                if text:
                    return self.generate_response(text, context, max_tokens, temperature)
                else:
                    return "❌ This model only supports text input. Multimodal features require Gemma 3n."
                    
        except Exception as e:
            logger.error(f"❌ Error generating multimodal response: {e}")
            return f"Error: {str(e)}"

    def _generate_gemma3n_multimodal_response(
        self, 
        text: str,
        images: List[Image.Image],
        audio_files: List[str],
        video_files: List[str],
        context: str,
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate multimodal response using Gemma 3n model."""
        try:
            # Build conversation with multimodal content
            messages = []
            
            # Add context if provided
            if context:
                messages.append({"role": "system", "content": [{"type": "text", "text": context}]})
            
            # Construct user message content
            user_content = []
            
            # Add text if provided
            if text:
                user_content.append({"type": "text", "text": text})
            
            # Add images if provided
            for img in images:
                user_content.append({"type": "image", "image": img})
            
            # Add audio references (files) - Gemma 3n supports audio
            if audio_files:
                for audio_file in audio_files:
                    if Path(audio_file).exists():
                        user_content.append({"type": "audio", "audio": str(Path(audio_file).resolve())})
                        logger.info(f"🎵 Added audio file: {audio_file}")
                    else:
                        logger.warning(f"⚠️  Audio file not found: {audio_file}")
            
            # Add video references (files) - Gemma 3n supports video
            if video_files:
                for video_file in video_files:
                    if Path(video_file).exists():
                        user_content.append({"type": "video", "video": str(Path(video_file).resolve())})
                        logger.info(f"🎬 Added video file: {video_file}")
                    else:
                        logger.warning(f"⚠️  Video file not found: {video_file}")
            
            # Add user message
            messages.append({"role": "user", "content": user_content})
            
            # Process inputs using the processor
            logger.info(f"🔄 Processing multimodal input: {len(user_content)} content items")
            
            # Use processor for multimodal inputs
            if images and len(images) > 0:
                # For text + images, use the processor
                text_content = self._extract_text_from_conversation(messages)
                inputs = self._processor(
                    text=text_content,
                    images=images,
                    return_tensors="pt",
                    padding=True
                )
                # Ensure all tensors are on the same device as the model
                model_device = next(self._model.parameters()).device
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                # For text-only or unsupported modalities, use chat template
                inputs = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                # Ensure all tensors are on the same device as the model
                model_device = next(self._model.parameters()).device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                else:
                    inputs = inputs.to(model_device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate multimodal response
            with torch.inference_mode():
                generation_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": 0.8,
                    "top_k": 40,
                    "pad_token_id": self._processor.tokenizer.eos_token_id,
                    "use_cache": True,
                    "num_beams": 1,
                    "repetition_penalty": 1.1,
                }
                    
                outputs = self._model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode response (remove input tokens)
            response_tokens = outputs[0][input_len:]
            response = self._processor.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Error in Gemma 3n multimodal generation: {e}")
            # Fallback to text-only if multimodal fails
            if text:
                logger.info("🔄 Falling back to text-only generation...")
                return self._generate_gemma3n_response(text, context, max_tokens, temperature)
            else:
                return f"Error: {str(e)}"

    def _process_image(self, image_input: Union[str, Image.Image, bytes]) -> Image.Image:
        """Process various image input formats into PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, str):
            if image_input.startswith(('http://', 'https://')):
                # URL
                logger.info(f"📥 Downloading image from URL: {image_input[:50]}...")
                response = requests.get(image_input)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content))
            elif image_input.startswith('data:image'):
                # Base64 data URL
                logger.info("📥 Processing base64 image...")
                header, data = image_input.split(',', 1)
                return Image.open(io.BytesIO(base64.b64decode(data)))
            else:
                # File path
                logger.info(f"📥 Loading image from file: {image_input}")
                if not Path(image_input).exists():
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                return Image.open(image_input)
        elif isinstance(image_input, bytes):
            logger.info("📥 Processing image from bytes...")
            return Image.open(io.BytesIO(image_input))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    def _extract_text_from_conversation(self, messages):
        """Extract text content from conversation structure for fallback processing."""
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            
            if isinstance(content, str):
                text_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Extract text from multimodal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(f"{role}: {item['text']}")
            
        result = "\n".join(text_parts)
        if not result.endswith("assistant:"):
            result += "\nassistant:"
        return result
    
    def is_available(self) -> bool:
        """Check if the LLM is available for generation."""
        return self._is_initialized
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": self._model_config.name,
            "model_id": self._model_config.huggingface_id,
            "model_key": self._model_key,
            "device": getattr(self, '_device', 'unknown'),
            "device_map": getattr(self, '_device_map', 'unknown'),
            "is_initialized": getattr(self, '_is_initialized', False),
            "is_gemma3n_model": getattr(self, '_is_gemma3n_model', False),
            "requires_auth": self._model_config.requires_auth,
            "supports_vision": self._model_config.supports_vision,
            "context_length": self._model_config.context_length,
            "memory_estimate_gb": self._model_config.memory_gb_estimate,
            "quantization_enabled": self._quantization_config is not None,
            "quantization_recommended": self._model_config.quantization_recommended
        }
    
    def cleanup(self):
        """Clean up model resources."""
        if hasattr(self, '_model') and self._model is not None:
            del self._model
        if hasattr(self, '_tokenizer') and self._tokenizer is not None:
            del self._tokenizer
        if hasattr(self, '_processor') and self._processor is not None:
            del self._processor
        if hasattr(self, '_pipeline') and self._pipeline is not None:
            del self._pipeline
        
        # Clear device-specific cache
        if hasattr(self, '_device') and self._device:
            if self._device.startswith("cuda"):
                try:
                    # Extract GPU ID from device string
                    if ":" in self._device:
                        gpu_id = int(self._device.split(":")[1])
                    else:
                        gpu_id = 0
                    
                    original_device = torch.cuda.current_device()
                    torch.cuda.set_device(gpu_id)
                    torch.cuda.empty_cache()
                    torch.cuda.set_device(original_device)  # Restore original device
                    logger.info(f"🧹 LLM resources cleaned up (GPU {gpu_id})")
                except Exception as e:
                    logger.warning(f"⚠️  Error during GPU cleanup: {e}")
                    torch.cuda.empty_cache()  # Fallback cleanup
                    logger.info("🧹 LLM resources cleaned up (fallback)")
            elif self._device == "mps":
                try:
                    # Clear MPS cache if available
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    logger.info("🧹 LLM resources cleaned up (MPS)")
                except Exception as e:
                    logger.warning(f"⚠️  Error during MPS cleanup: {e}")
                    logger.info("🧹 LLM resources cleaned up (MPS fallback)")
            else:
                logger.info("🧹 LLM resources cleaned up (CPU)")
        else:
            logger.info("🧹 LLM resources cleaned up")
        
        # Reset initialization state
        self._is_initialized = False

    def _is_rtx_3060(self):
        """Check if running on RTX 3060 (11.6GB VRAM)."""
        if not torch.cuda.is_available():
            return False
        
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / 1024**3
        
        # RTX 3060 has approximately 11.64GB total memory
        return 11.5 <= total_memory_gb <= 12.0 and "3060" in props.name


# Global instance for singleton pattern
_shellbert_llm_instance: Optional[ShellbertLLM] = None


def get_shellbert_llm() -> ShellbertLLM:
    """Get the global ShellbertLLM instance (singleton pattern)."""
    global _shellbert_llm_instance
    if _shellbert_llm_instance is None:
        _shellbert_llm_instance = ShellbertLLM()
    return _shellbert_llm_instance


def cleanup_shellbert_llm():
    """Clean up the global LLM instance."""
    global _shellbert_llm_instance
    if _shellbert_llm_instance is not None:
        _shellbert_llm_instance.cleanup()
        _shellbert_llm_instance = None 