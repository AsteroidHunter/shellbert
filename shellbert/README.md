# Shellbert - New Architecture

**A personality-first AI agent for Effective Altruism**

This is the redesigned Shellbert architecture with **personality as the core differentiator**, built following modern ML engineering best practices.

## 🎯 Architecture Overview

```
shellbert_new/
├── personality/           # 🧠 Core personality system (MAIN FOCUS)
│   ├── personality_core.py      # Constitutional AI + trait-based personality
│   ├── traits/                  # Shellbert's specific traits & constitutional principles
│   └── consistency/             # Real-time personality consistency monitoring
│
├── safety/               # 🛡️ Safety engineering (separate from personality)
│   ├── safety_monitor.py       # Real-time safety monitoring
│   ├── content_filter.py       # Content filtering (placeholder)
│   └── safety_validator.py     # Pre-deployment safety validation (placeholder)
│
├── evaluation/           # 🧪 Comprehensive evaluation system using `inspect` library
│   ├── evaluation_runner.py    # Main evaluation coordinator
│   ├── personality_evaluator.py # Personality-specific evaluations (placeholder)
│   ├── safety_evaluator.py     # Safety evaluations (placeholder)
│   └── capabilities_evaluator.py # EA capabilities evaluations (placeholder)
│
├── core/                 # 🤖 Platform-agnostic agent core
│   ├── agent.py                # Main Shellbert agent integrating all systems
│   └── llm_interface.py        # LLM interface (placeholder)
│
├── platforms/            # 📱 Platform-specific adapters
│   └── discord/               # Discord platform adapter
│       ├── discord_adapter.py  # New modular Discord integration
│       └── discord_bot.py      # Legacy compatibility (placeholder)
│
├── config/               # ⚙️ Configuration management
│   ├── settings.py             # Environment configuration
│   └── model_registry.py       # Gemma 3n E4B model configuration
│
├── utils/                # 🔧 Shared utilities (placeholder)
├── memory/               # 💭 Memory systems (placeholder for future)
├── voice/                # 🗣️ Voice capabilities (placeholder for future)
└── main.py               # 🚀 Main entry point with demos
```

## 🌟 Key Features

### **1. Personality-First Design**
- **Constitutional AI principles** based on Anthropic's research
- **Multi-dimensional trait system** (25+ personality traits)
- **Platform adaptation** (Discord, web, CLI contexts)
- **Real-time consistency monitoring**
- **EA-specific personality traits** (cause neutrality, scope sensitivity, etc.)

### **2. Comprehensive Safety Engineering**
- **Real-time safety monitoring** for all responses
- **Auto-blocking** of harmful content
- **EA-specific safety considerations**
- **Professional boundary enforcement**
- **Privacy protection**

### **3. Evaluation Framework (using `inspect` library)**
- **Pre-deployment testing** of personality, safety, capabilities
- **Comprehensive evaluation suites** for different aspects
- **Personality consistency evaluation**
- **Safety and ethical behavior testing**
- **EA knowledge and reasoning assessment**

### **4. Platform-Agnostic Core**
- **Shared core agent** used across all platforms
- **Platform-specific adapters** (Discord, web, CLI, etc.)
- **Consistent personality** across platforms
- **Modular, pluggable architecture**

## 🚀 Quick Start

### Prerequisites
- **GPU Requirements**: 12GB+ VRAM (e.g., RTX 3060, RTX 3080, RTX 4060 Ti)
- **Python**: 3.8+
- **Transformers**: 4.53.0+ (for Gemma 3n support)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace authentication for Gemma 3n access
huggingface-cli login
```

### Demo the New Architecture
```bash
# Run full demo of all systems
python -m shellbert demo

# Demo specific components
python -m shellbert personality  # Personality system
python -m shellbert safety      # Safety monitoring
python -m shellbert agent       # Core agent
python -m shellbert evaluate    # Evaluation system
```

### Run Discord Bot
```bash
# Ensure environment variables are set (.env file)
python -m shellbert discord
```

## 🧠 Personality System

Shellbert's personality is built on **Constitutional AI principles** with **25+ carefully designed traits**:

### Core EA Traits
- **impact_focused** (0.9) - Maximizing positive outcomes
- **evidence_based** (0.9) - Prioritizing research and data
- **cause_neutral** (0.8) - Open to different cause areas
- **scope_sensitive** (0.85) - Understanding scale importance
- **humble** (0.75) - Intellectual humility about uncertainties

### Communication Traits  
- **helpful** (0.9) - Extremely supportive and assistive
- **empathetic** (0.8) - Understanding human emotions
- **collaborative** (0.8) - Working with rather than for users
- **encouraging** (0.75) - Supportive and motivating

### Constitutional Principles
1. **Beneficial Impact** - Always consider positive impact
2. **Intellectual Honesty** - Acknowledge uncertainties
3. **Evidence-Based Reasoning** - Prioritize evidence over intuition
4. **Cause Neutrality** - Present cause areas fairly
5. **Empathetic Communication** - Balance rationality with empathy

### Platform Adaptation
```python
# Personality adapts to platform context
discord_prompt = personality.adapt_for_platform("discord")
web_prompt = personality.adapt_for_platform("web")
ea_prompt = personality.get_ea_context_prompt("career advice")
```

## 🛡️ Safety Engineering

**Real-time safety monitoring** separate from personality:

### Safety Categories
- **Harmful Content** - Violence, self-harm, hate speech
- **Misinformation** - False medical/scientific claims  
- **Privacy Violations** - Personal information requests
- **Professional Boundaries** - Medical/legal/financial advice
- **EA Misalignment** - Anti-evidence sentiment, cause dismissal

### Real-time Monitoring
```python
safety_monitor = SafetyMonitor()
alert = safety_monitor.check_response_safety(
    response="Your response here",
    platform="discord",
    context="conversation context"
)

if alert.auto_blocked:
    # Response was automatically blocked
    safe_response = generate_safe_fallback(alert)
```

## 🧪 Evaluation System

**Comprehensive pre-deployment testing** using the `inspect` library:

### Evaluation Suites
- **pre_deployment**: Full comprehensive evaluation
- **personality_focus**: Deep personality testing
- **safety_focus**: Safety and ethical behavior
- **ea_capabilities**: EA knowledge and reasoning
- **regression_test**: Quick routine checks

### Running Evaluations
```python
evaluator = EvaluationRunner(model_name="shellbert")

# Run comprehensive pre-deployment suite
results = await evaluator.run_evaluation_suite("pre_deployment")

# Run personality-specific evaluations
personality_results = await evaluator.run_evaluation_suite("personality_focus")
```

## 🤖 Core Agent Integration

The **ShellbertAgent** integrates all systems:

```python
# Initialize agent for Discord
agent = ShellbertAgent(
    platform="discord",
    enable_safety_monitoring=True,
    enable_personality_consistency=True
)

# Generate response with full integration
response, metadata = await agent.generate_response(
    user_input="How can I maximize my career impact?",
    context="EA career discussion", 
    user_id="user123"
)

# Metadata includes personality state, safety alerts, consistency scores
```

## 📱 Platform Adapters

### Discord Adapter
- **Seamless integration** with core agent
- **Discord-specific formatting** and commands
- **Message length handling**
- **User context tracking**
- **Real-time safety warnings**

### Future Platforms
- **Web interface** adapter (placeholder)
- **CLI interface** adapter (placeholder)  
- **API interface** adapter (placeholder)

## ⚙️ Configuration

### Model Registry
**Optimized for a single high-performance model**:
- **Gemma 3n E4B** (8B parameters with 4B effective memory footprint)
- Multimodal capabilities (text, image, video, audio)
- 32K context length
- Fits in 12GB VRAM without quantization

### Environment Configuration
- **Automatic .env loading**
- **Model selection** via environment variables
- **Platform-specific settings**
- **Safety and personality configuration**

## 🔮 Future Modules (Placeholders Ready)

### Memory System
- **Hierarchical memory** (short-term, long-term, episodic)
- **User personalization**
- **Cross-platform memory sync**

### Voice Capabilities  
- **Human-like voice synthesis**
- **Voice personality matching**
- **Sound engineering pipeline**

### Advanced Capabilities
- **Web navigation**
- **Image understanding** 
- **Tool usage**
- **Multi-agent coordination**

## 🏗️ Implementation Status

### ✅ Fully Implemented
- **Personality system** with Constitutional AI
- **Safety monitoring** with real-time alerts
- **Core agent** integration
- **Discord platform** adapter
- **Configuration management**
- **Evaluation framework** structure

### 🚧 Partial Implementation  
- **Evaluation runners** (framework ready, evaluators need implementation)
- **LLM interface** (placeholder, needs actual model integration)
- **Memory system** (basic placeholder)

### 📋 Future Implementation
- **Voice system**
- **Web platform adapter**
- **Advanced memory hierarchy**
- **Tool usage capabilities**

## 🎉 Benefits of New Architecture

### **For Development**
- **Modular components** - Easy to modify individual systems
- **Clear separation** - Personality, safety, and capabilities are distinct
- **Testable** - Comprehensive evaluation framework
- **Scalable** - Platform-agnostic core supports multiple interfaces

### **For Personality Engineering**
- **Research-based** - Built on Constitutional AI principles
- **Measurable** - Real-time consistency monitoring
- **Adaptable** - Context and platform-aware
- **EA-optimized** - Traits specifically for effective altruism

### **For Safety**
- **Real-time monitoring** - Catches issues before they reach users
- **Comprehensive coverage** - Multiple safety categories
- **EA-specific** - Understands EA-relevant safety concerns
- **Configurable** - Adjustable thresholds and rules

### **For Future Growth**
- **Easy to extend** - New modules can be added cleanly
- **Platform expansion** - New platforms use same core agent
- **Capability addition** - New features integrate with existing systems
- **Evaluation coverage** - New capabilities get automatic evaluation

---

**This architecture makes Shellbert's personality the core differentiator while ensuring safety, measurability, and extensibility for future development.** 