#!/usr/bin/env python3
"""
Shellbert - Main entry point for module execution
"""

if __name__ == "__main__":
    from .main import *
    
    # Handle command line arguments exactly like main.py
    import sys
    import asyncio
    
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
        elif command == "--help" or command == "-h":
            print("🤖 Shellbert - AI Assistant")
            print("\nUsage: python -m shellbert [command]")
            print("\nAvailable commands:")
            print("  discord      - Run Discord bot")
            print("  demo         - Run full system demo")
            print("  personality  - Demo personality system")
            print("  safety       - Demo safety system")
            print("  agent        - Demo core agent")
            print("  evaluate     - Demo evaluation system")
            print("  diagnostic   - Run LLM diagnostic test")
            print("  device       - Run device detection test")
            print("  debug        - Test debug mode for fast responses")
            print("\nFor Discord bot, ensure .env is configured with:")
            print("  DISCORD_BOT_TOKEN")
            print("  DISCORD_CHANNEL")
            print("  RUN_MODE (test/deploy)")
        else:
            print(f"Unknown command: {command}")
            print("Available commands: discord, demo, personality, safety, agent, evaluate, diagnostic, device, debug")
            print("Use --help for more information")
    else:
        # Run full demo by default
        asyncio.run(main()) 