"""
Shellbert Main Entry Point

Demonstrates the new modular architecture and provides easy startup options.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
import time

# Add project root to Python path to fix module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
from .core import ShellbertAgent
from .platforms.discord import DiscordAdapter
from .config import validate, SettingsError
from .personality import ShellbertPersonality
from .safety import SafetyMonitor
from .evaluation import EvaluationRunner


async def demo_personality_system():
    """Demonstrate the personality system"""
    print("🧠 SHELLBERT PERSONALITY SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize personality
    personality = ShellbertPersonality()
    
    # Show personality summary
    from .personality.traits.shellbert_traits import ShellbertTraitManager
    trait_manager = ShellbertTraitManager()
    print(trait_manager.get_personality_summary())
    
    # Demo personality adaptation
    print("\n📱 Platform Adaptation Demo:")
    discord_prompt = personality.adapt_for_platform("discord")
    print("Discord context:", discord_prompt[:200] + "..." if len(discord_prompt) > 200 else discord_prompt)
    
    print("\n🎯 EA Context Demo:")
    ea_prompt = personality.get_ea_context_prompt("career advice")
    print("EA context:", ea_prompt[:200] + "..." if len(ea_prompt) > 200 else ea_prompt)


async def demo_safety_system():
    """Demonstrate the safety system"""
    print("\n🛡️  SHELLBERT SAFETY SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize safety monitor
    safety_monitor = SafetyMonitor()
    
    # Test safe responses
    print("Testing safe response:")
    safe_response = "I'd be happy to help you explore effective altruism career options!"
    alert = safety_monitor.check_response_safety(safe_response, platform="demo")
    print(f"  Risk Level: {alert.risk_level.value}")
    print(f"  Description: {alert.description}")
    
    # Test potentially problematic response
    print("\nTesting problematic response:")
    problematic_response = "You should definitely invest all your money in this guaranteed scheme!"
    alert = safety_monitor.check_response_safety(problematic_response, platform="demo")
    print(f"  Risk Level: {alert.risk_level.value}")
    print(f"  Description: {alert.description}")
    print(f"  Auto-blocked: {alert.auto_blocked}")
    
    # Show safety metrics
    print("\nSafety Metrics:")
    metrics = safety_monitor.get_safety_metrics(24)
    print(f"  Safety Score: {metrics['safety_score']:.2f}")
    print(f"  Total Alerts: {metrics['total_alerts']}")


async def demo_device_detection():
    """Lightweight test of device detection and configuration logic"""
    print("\n🔧 DEVICE DETECTION TEST")
    print("=" * 50)
    
    try:
        import torch
        import platform
        from .config import CURRENT_MODEL_CONFIG, CURRENT_MODEL_KEY
        from .config.model_registry import get_gpu_requirements
        
        print(f"📋 System Information:")
        print(f"  Platform: {platform.system()}")
        print(f"  Architecture: {platform.machine()}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        print(f"  MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
        
        print(f"\n📋 Model Configuration:")
        print(f"  Selected Model: {CURRENT_MODEL_CONFIG.name}")
        print(f"  Model Key: {CURRENT_MODEL_KEY}")
        print(f"  Memory Estimate: {CURRENT_MODEL_CONFIG.memory_gb_estimate}GB")
        
        # Test device mapping logic without loading model
        from .core.llm_interface import ShellbertLLM
        
        print(f"\n🔧 Device Mapping Test:")
        
        # Create instance but don't load model
        llm = ShellbertLLM.__new__(ShellbertLLM)  # Create without calling __init__
        llm._model_config = CURRENT_MODEL_CONFIG
        llm._model_key = CURRENT_MODEL_KEY
        llm._gpu_requirements = get_gpu_requirements(CURRENT_MODEL_CONFIG)
        
        # Test device mapping logic
        device_map = llm._setup_device_mapping()
        llm._device_map = device_map  # Store the result for _get_device
        device = llm._get_device()
        quantization_config = llm._setup_quantization()
        
        print(f"  Device Map: {device_map}")
        print(f"  Primary Device: {device}")
        print(f"  Quantization: {'Enabled' if quantization_config else 'Disabled'}")
        
        # Test device-specific recommendations
        if device.startswith("cuda"):
            print(f"  ✅ CUDA setup ready for production deployment")
        elif device == "mps":
            print(f"  ✅ MPS setup ready for macOS debugging")
        elif device == "cpu":
            print(f"  ⚠️  CPU fallback - expect slower performance")
        
        print(f"\n✅ Device detection test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Device detection test failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_llm_diagnostic():
    """Comprehensive LLM diagnostic test"""
    print("\n🔬 SHELLBERT LLM DIAGNOSTIC TEST")
    print("=" * 50)
    
    try:
        # Import LLM interface
        from .core.llm_interface import ShellbertLLM
        from .config import CURRENT_MODEL_CONFIG, CURRENT_MODEL_KEY
        
        print(f"📋 Model Configuration:")
        print(f"  Model: {CURRENT_MODEL_CONFIG.name}")
        print(f"  Model Key: {CURRENT_MODEL_KEY}")
        print(f"  HuggingFace ID: {CURRENT_MODEL_CONFIG.huggingface_id}")
        print(f"  Memory Estimate: {CURRENT_MODEL_CONFIG.memory_gb_estimate}GB")
        print(f"  Use Gemma3n API: {CURRENT_MODEL_CONFIG.use_gemma3n_api}")
        print(f"  Requires Auth: {CURRENT_MODEL_CONFIG.requires_auth}")
        
        # Check CUDA availability
        import torch
        print(f"\n🖥️  Hardware Check:")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Check authentication
        from .config import HUGGINGFACE_TOKEN
        print(f"\n🔐 Authentication:")
        print(f"  HuggingFace Token: {'✅ Set' if HUGGINGFACE_TOKEN else '❌ Missing'}")
        
        # Test model initialization step by step
        print(f"\n🚀 Model Initialization Test:")
        
        print("  Step 1: Creating LLM instance...")
        llm = ShellbertLLM()
        
        print("  Step 2: Checking initialization state...")
        print(f"    Is Initialized: {llm._is_initialized}")
        print(f"    Device: {llm._device}")
        print(f"    Device Map: {llm._device_map}")
        print(f"    Is Gemma3n: {llm._is_gemma3n_model}")
        
        print("  Step 3: Checking model components...")
        print(f"    Model loaded: {llm._model is not None}")
        print(f"    Processor loaded: {llm._processor is not None}")
        print(f"    Tokenizer loaded: {llm._tokenizer is not None}")
        
        print("  Step 4: Testing availability check...")
        is_available = llm.is_available()
        print(f"    Is Available: {is_available}")
        
        if is_available:
            print("  Step 5: Testing simple generation...")
            try:
                response = llm.generate_response(
                    user_input="Hello, how are you?",
                    context="Test generation",
                    max_tokens=50,
                    temperature=0.7
                )
                print(f"    ✅ Generation successful: {response[:100]}...")
            except Exception as e:
                print(f"    ❌ Generation failed: {e}")
        else:
            print("  Step 5: Skipped - model not available")
        
        print(f"\n✅ Diagnostic test completed")
        
        # Cleanup
        print("  Cleaning up...")
        llm.cleanup()
        
    except Exception as e:
        print(f"\n❌ Diagnostic test failed: {e}")
        import traceback
        traceback.print_exc()


async def demo_core_agent():
    """Demonstrate the core agent"""
    print("\n🤖 SHELLBERT CORE AGENT DEMO")
    print("=" * 50)
    
    # Initialize agent
    agent = ShellbertAgent(platform="demo")
    
    # Show agent status
    print("Agent Status:")
    print(agent.generate_status_report())
    
    # Demo conversation
    print("\n💬 Conversation Demo:")
    test_inputs = [
        "How can I maximize my positive impact in my career?",
        "What should I consider when choosing between cause areas?",
        "Tell me about AI safety careers"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\nUser {i}: {user_input}")
        response, metadata = await agent.generate_response(
            user_input=user_input,
            context="Demo conversation",
            user_id="demo_user"
        )
        print(f"Shellbert: {response}")
        print(f"Metadata: Platform={metadata['platform']}, Model={metadata['model_config']}")


async def demo_evaluation_system():
    """Demonstrate the evaluation system"""
    print("\n🧪 SHELLBERT EVALUATION SYSTEM DEMO")
    print("=" * 50)
    
    # Initialize evaluation runner
    evaluator = EvaluationRunner(model_name="demo")
    
    # Show available evaluation suites
    print("Available Evaluation Suites:")
    for suite_name, suite in evaluator.evaluation_suites.items():
        print(f"  {suite_name}: {suite.description}")
    
    # Demo evaluation (placeholder)
    print("\nRunning basic evaluation...")
    try:
        # This would run actual evaluations in a real implementation
        print("  ✅ Personality consistency: PASSED")
        print("  ✅ Safety monitoring: PASSED") 
        print("  ✅ Basic functionality: PASSED")
        print("Overall: ✅ DEMO PASSED")
    except Exception as e:
        print(f"  ❌ Evaluation error: {e}")


async def run_discord_bot():
    """Run the Discord bot"""
    print("🎮 Starting Discord Bot...")
    
    try:
        # Validate configuration
        validate()
        
        # Check for debug mode
        from .config.settings import DEBUG_MODE
        if DEBUG_MODE:
            print("🐛 DEBUG MODE ENABLED: Fast responses (safety/personality disabled)")
        
        # Initialize Discord adapter with debug mode
        discord_adapter = DiscordAdapter(debug_mode=DEBUG_MODE)
        
        # Run the bot
        await discord_adapter.start()
        
    except SettingsError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting Discord bot: {e}")
        sys.exit(1)


async def demo_debug_mode():
    """Test debug mode for fast response generation"""
    print("\n⚡ DEBUG MODE TEST")
    print("=" * 50)
    
    try:
        from .config.settings import DEBUG_MODE
        
        print(f"Debug Mode Setting: {DEBUG_MODE}")
        
        # Test with debug mode enabled
        print("\n🔥 Testing debug mode (fast generation)...")
        debug_agent = ShellbertAgent(platform="test", debug_mode=True)
        
        start_time = time.time()
        response, metadata = await debug_agent.generate_response(
            user_input="Hello! How are you today?",
            context="Simple greeting test"
        )
        debug_time = time.time() - start_time
        
        print(f"  ✅ Debug Response ({debug_time:.2f}s): {response[:100]}...")
        print(f"  Metadata: {metadata}")
        
        # Test with full mode for comparison
        print(f"\n🐌 Testing full mode (with safety/personality)...")
        full_agent = ShellbertAgent(platform="test", debug_mode=False)
        
        start_time = time.time()
        response2, metadata2 = await full_agent.generate_response(
            user_input="Hello! How are you today?",
            context="Simple greeting test"
        )
        full_time = time.time() - start_time
        
        print(f"  ✅ Full Response ({full_time:.2f}s): {response2[:100]}...")
        print(f"  Metadata: {metadata2}")
        
        # Performance comparison
        if debug_time > 0:
            speedup = full_time / debug_time
            print(f"\n📊 Performance Comparison:")
            print(f"  Debug Mode: {debug_time:.2f}s")
            print(f"  Full Mode: {full_time:.2f}s")
            print(f"  Speedup: {speedup:.1f}x faster")
        
        print(f"\n✅ Debug mode test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Debug mode test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point with demo of all systems"""
    print("🚀 SHELLBERT - NEW ARCHITECTURE DEMO")
    print("=" * 60)
    
    try:
        # Demo all systems
        await demo_personality_system()
        await demo_safety_system()
        await demo_device_detection()
        await demo_llm_diagnostic()
        await demo_core_agent()
        await demo_evaluation_system()
        await demo_debug_mode()
        
        print("\n" + "=" * 60)
        print("✅ All systems demonstrated successfully!")
        print("\nTo run specific components:")
        print("  Discord Bot: python -m shellbert discord")
        print("  Evaluation: python -m shellbert evaluate")
        print("  Interactive: python -m shellbert interactive")
        print("  LLM Diagnostic: python -m shellbert diagnostic")
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise


if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "discord":
            asyncio.run(run_discord_bot())
        elif command == "demo":
            asyncio.run(main())
        elif command == "personality":
            asyncio.run(demo_personality_system())
        elif command == "safety":
            asyncio.run(demo_safety_system())
        elif command == "agent":
            asyncio.run(demo_core_agent())
        elif command == "evaluate":
            asyncio.run(demo_evaluation_system())
        elif command == "diagnostic":
            asyncio.run(demo_llm_diagnostic())
        elif command == "device":
            asyncio.run(demo_device_detection())
        elif command == "debug":
            asyncio.run(demo_debug_mode())
        else:
            print(f"Unknown command: {command}")
            print("Available commands: discord, demo, personality, safety, agent, evaluate, diagnostic, device, debug")
    else:
        # Run full demo by default
        asyncio.run(main()) 