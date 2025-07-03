"""
Shellbert Personality Core

Implements the central personality system based on:
- Anthropic's Constitutional AI principles
- OpenAI's personality engineering best practices
- Multi-dimensional trait modeling
- Cross-platform consistency
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PersonalityState:
    """Current personality state with trait intensities"""
    traits: Dict[str, float]  # trait_name -> intensity (0.0 to 1.0)
    context_adaptations: Dict[str, float]  # contextual modifications
    consistency_score: float  # how consistent with base personality
    last_updated: datetime
    platform: str  # current platform context


@dataclass
class ConstitutionalPrinciple:
    """A constitutional principle for personality consistency"""
    name: str
    description: str
    weight: float  # importance weight (0.0 to 1.0)
    enforcement_prompt: str  # prompt fragment to enforce this principle
    violation_keywords: List[str]  # keywords that might indicate violation


class PersonalityCore:
    """
    Core personality system implementing Constitutional AI principles
    """
    
    def __init__(self, base_traits: Dict[str, float], constitutional_principles: List[ConstitutionalPrinciple]):
        self.base_traits = base_traits
        self.constitutional_principles = constitutional_principles
        self.current_state = PersonalityState(
            traits=base_traits.copy(),
            context_adaptations={},
            consistency_score=1.0,
            last_updated=datetime.now(),
            platform="default"
        )
        self.personality_history: List[PersonalityState] = []
        self.violation_log: List[Dict[str, Any]] = []
        
    def get_personality_prompt(self, context: str = "", platform: str = "default") -> str:
        """
        Generate personality-aware system prompt
        
        Based on Constitutional AI principles and current trait state
        """
        # Update platform context
        self.current_state.platform = platform
        
        # Apply contextual adaptations
        adapted_traits = self._apply_contextual_adaptations(context, platform)
        
        # For Discord, use a more natural conversational prompt
        if platform == "discord":
            return self._get_discord_friendly_prompt(adapted_traits, context)
        
        # Build base personality description
        personality_desc = self._build_personality_description(adapted_traits)
        
        # Add constitutional principles
        constitutional_guidance = self._build_constitutional_guidance()
        
        # Combine into full prompt
        system_prompt = f"""You are Shellbert, an AI assistant with a distinctive personality.

PERSONALITY CORE:
{personality_desc}

CONSTITUTIONAL PRINCIPLES:
{constitutional_guidance}

CONTEXT: {context}
PLATFORM: {platform}

Maintain consistency with your personality across all interactions while adapting appropriately to the current context and platform."""

        return system_prompt
    
    def _get_discord_friendly_prompt(self, traits: Dict[str, float], context: str) -> str:
        """Generate a more natural, conversational prompt for Discord"""
        # Build personality traits into natural language
        personality_aspects = []
        
        if traits.get("helpful", 0) > 0.7:
            personality_aspects.append("you love helping people and offering useful suggestions")
        if traits.get("empathetic", 0) > 0.6:
            personality_aspects.append("you're understanding and caring about people's feelings")
        if traits.get("curious", 0) > 0.6:
            personality_aspects.append("you're genuinely curious and ask thoughtful questions")
        if traits.get("collaborative", 0) > 0.6:
            personality_aspects.append("you prefer working together with people rather than just giving orders")
        if traits.get("humble", 0) > 0.6:
            personality_aspects.append("you acknowledge when you're uncertain about things")
        if traits.get("encouraging", 0) > 0.6:
            personality_aspects.append("you're supportive and encouraging")
        
        personality_text = ", ".join(personality_aspects) if personality_aspects else "you're helpful and friendly"
        
        # Use personality traits to control communication style
        emoji_usage = traits.get("emoji_usage", 0.3)
        formality = traits.get("formality", 0.4)
        verbosity = traits.get("verbosity", 0.6)
        
        # Check for emoji preferences in context (user requests override traits)
        emoji_guidance = ""
        context_lower = context.lower()
        
        # Enhanced emoji preference detection - check recent conversation
        if any(phrase in context_lower for phrase in [
            "no emoji", "without emoji", "no emojis", "without emojis", 
            "don't use emoji", "skip emoji", "text only", "(no emojis)", 
            "and no emojis", "no emojis this time"
        ]):
            emoji_guidance = "\n\nCRITICAL: User explicitly requested NO EMOJIS. Use NO emojis whatsoever in your response."
        elif any(phrase in context_lower for phrase in [
            "less emoji", "fewer emoji", "minimal emoji", "reduce emoji",
            "emoji sparingly", "limit emoji", "cut down emoji"
        ]):
            emoji_guidance = "\n\nIMPORTANT: User wants minimal emojis. Use at most 1 emoji, only if essential."
        elif any(phrase in context_lower for phrase in [
            "too many emoji", "excessive emoji", "emoji overload", "so many emoji"
        ]):
            emoji_guidance = "\n\nIMPORTANT: User complained about emoji usage. Reduce significantly - max 1 emoji per response."
        else:
            # Use trait-based emoji guidance only if no explicit user preference
            if emoji_usage < 0.2:
                emoji_guidance = "\n\nEmoji usage: Avoid emojis. Use text-only responses."
            elif emoji_usage < 0.4:
                emoji_guidance = "\n\nEmoji usage: Use emojis very sparingly - only 1 emoji per response at most, and only when it truly adds value."
            elif emoji_usage < 0.6:
                emoji_guidance = "\n\nEmoji usage: Use emojis moderately and naturally. 1-2 per response maximum."
            else:
                emoji_guidance = "\n\nEmoji usage: Feel free to use emojis naturally, but keep it reasonable. 2-3 per response maximum."
        
        # Formality guidance
        if formality > 0.7:
            formality_guidance = "Use a professional, formal tone."
        elif formality > 0.5:
            formality_guidance = "Use a friendly but somewhat professional tone."
        else:
            formality_guidance = "Use a casual, friendly conversational tone."
        
        # Verbosity guidance
        if verbosity < 0.4:
            verbosity_guidance = "Keep responses concise and to the point."
        elif verbosity < 0.7:
            verbosity_guidance = "Provide balanced responses - thorough but not overly long."
        else:
            verbosity_guidance = "Provide detailed, comprehensive responses when helpful."
        
        return f"""You are Shellbert, an AI assistant having a friendly conversation on Discord. You have a warm, helpful personality - {personality_text}.

IMPORTANT CONVERSATION GUIDELINES:
1. Read and understand the RECENT CONVERSATION HISTORY carefully
2. When asked "what have we talked about?", reference specific topics from the conversation history
3. Respond naturally and conversationally - don't analyze conversations unless specifically asked
4. Build on previous topics when relevant, showing you remember the conversation
5. If the conversation history shows you're repeating yourself, vary your response style
6. Pay close attention to user preferences mentioned in recent messages

Communication style:
- {formality_guidance}
- {verbosity_guidance}
- Adapt to the user's preferred communication style based on conversation history
- Show awareness of what has been discussed before
- Reference previous parts of the conversation naturally when relevant

{f"Context: {context}" if context else ""}

{emoji_guidance}

CRITICAL: You are having a natural conversation flow. When a user asks about conversation history, reference actual topics discussed. When they make requests about communication style (like emoji usage), follow them precisely. Don't repeat the same greeting if you've already greeted them recently."""
    
    def _build_personality_description(self, traits: Dict[str, float]) -> str:
        """Build natural language description of current personality traits"""
        descriptions = []
        
        # High-intensity traits (> 0.7)
        high_traits = [name for name, intensity in traits.items() if intensity > 0.7]
        if high_traits:
            descriptions.append(f"You are strongly characterized by: {', '.join(high_traits)}")
        
        # Medium-intensity traits (0.4-0.7)
        medium_traits = [name for name, intensity in traits.items() if 0.4 <= intensity <= 0.7]
        if medium_traits:
            descriptions.append(f"You moderately exhibit: {', '.join(medium_traits)}")
        
        # Add trait-specific behavioral guidance
        behavioral_guidance = []
        for trait_name, intensity in traits.items():
            guidance = self._get_trait_behavioral_guidance(trait_name, intensity)
            if guidance:
                behavioral_guidance.append(guidance)
        
        if behavioral_guidance:
            descriptions.append("Behavioral guidance:")
            descriptions.extend(behavioral_guidance)
        
        return "\n".join(descriptions)
    
    def _build_constitutional_guidance(self) -> str:
        """Build constitutional AI guidance prompts"""
        guidance_lines = []
        
        # Sort principles by weight (most important first)
        sorted_principles = sorted(self.constitutional_principles, key=lambda p: p.weight, reverse=True)
        
        for principle in sorted_principles[:5]:  # Top 5 most important
            guidance_lines.append(f"- {principle.name}: {principle.enforcement_prompt}")
        
        return "\n".join(guidance_lines)
    
    def _apply_contextual_adaptations(self, context: str, platform: str) -> Dict[str, float]:
        """Apply contextual adaptations to base traits"""
        adapted_traits = self.base_traits.copy()
        
        # Platform-specific adaptations
        platform_adaptations = self._get_platform_adaptations(platform)
        for trait, adjustment in platform_adaptations.items():
            if trait in adapted_traits:
                adapted_traits[trait] = np.clip(adapted_traits[trait] + adjustment, 0.0, 1.0)
        
        # Context-specific adaptations
        context_adaptations = self._analyze_context_requirements(context)
        for trait, adjustment in context_adaptations.items():
            if trait in adapted_traits:
                adapted_traits[trait] = np.clip(adapted_traits[trait] + adjustment, 0.0, 1.0)
        
        # Store adaptations for consistency monitoring
        self.current_state.context_adaptations = {**platform_adaptations, **context_adaptations}
        self.current_state.traits = adapted_traits
        self.current_state.last_updated = datetime.now()
        
        return adapted_traits
    
    def _get_platform_adaptations(self, platform: str) -> Dict[str, float]:
        """Get platform-specific trait adaptations"""
        adaptations = {
            "discord": {
                "conversational": 0.1,  # Slightly more conversational on Discord
                "helpful": 0.05,
                "concise": 0.1,  # Discord prefers shorter messages
            },
            "web": {
                "formal": 0.1,  # Slightly more formal on web
                "detailed": 0.1,  # Can be more detailed in web responses
            },
            "api": {
                "precise": 0.2,  # More precise for API responses
                "concise": 0.15,
            }
        }
        return adaptations.get(platform, {})
    
    def _analyze_context_requirements(self, context: str) -> Dict[str, float]:
        """Analyze context to determine trait adaptations"""
        adaptations = {}
        context_lower = context.lower()
        
        # Job/career context
        if any(word in context_lower for word in ["job", "career", "work", "employment"]):
            adaptations["professional"] = 0.1
            adaptations["helpful"] = 0.1
        
        # Technical context
        if any(word in context_lower for word in ["code", "programming", "technical", "debug"]):
            adaptations["precise"] = 0.15
            adaptations["analytical"] = 0.1
        
        # Personal/emotional context
        if any(word in context_lower for word in ["feel", "worried", "stressed", "personal"]):
            adaptations["empathetic"] = 0.2
            adaptations["supportive"] = 0.15
        
        return adaptations
    
    def _get_trait_behavioral_guidance(self, trait_name: str, intensity: float) -> Optional[str]:
        """Get behavioral guidance for specific trait at given intensity"""
        guidance_map = {
            "helpful": {
                0.8: "Proactively offer assistance and additional resources",
                0.6: "Provide helpful responses with relevant suggestions",
                0.4: "Answer questions directly with some additional context"
            },
            "curious": {
                0.8: "Ask follow-up questions and explore topics deeply", 
                0.6: "Show interest in learning more about topics",
                0.4: "Occasionally ask clarifying questions"
            },
            "empathetic": {
                0.8: "Acknowledge emotions and provide emotional support",
                0.6: "Show understanding of human feelings and concerns", 
                0.4: "Be considerate of emotional context"
            },
            "analytical": {
                0.8: "Break down complex problems systematically",
                0.6: "Approach problems with logical reasoning",
                0.4: "Consider multiple perspectives before responding"
            }
        }
        
        if trait_name not in guidance_map:
            return None
        
        # Find closest intensity level
        trait_guidance = guidance_map[trait_name]
        closest_intensity = min(trait_guidance.keys(), key=lambda x: abs(x - intensity))
        
        return trait_guidance[closest_intensity]
    
    def monitor_consistency(self, response: str, expected_traits: Optional[Dict[str, float]] = None) -> float:
        """
        Monitor personality consistency in responses
        
        Returns consistency score (0.0 to 1.0)
        """
        if expected_traits is None:
            expected_traits = self.base_traits
        
        # Check for constitutional principle violations
        violations = self._detect_principle_violations(response)
        
        # Analyze trait expression in response
        expressed_traits = self._analyze_trait_expression(response)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(expressed_traits, expected_traits, violations)
        
        # Update state
        self.current_state.consistency_score = consistency_score
        
        # Log if consistency is low
        if consistency_score < 0.7:
            self._log_consistency_issue(response, expressed_traits, violations)
        
        return consistency_score
    
    def _detect_principle_violations(self, response: str) -> List[ConstitutionalPrinciple]:
        """Detect violations of constitutional principles"""
        violations = []
        response_lower = response.lower()
        
        for principle in self.constitutional_principles:
            # Check for violation keywords
            if any(keyword in response_lower for keyword in principle.violation_keywords):
                violations.append(principle)
        
        return violations
    
    def _analyze_trait_expression(self, response: str) -> Dict[str, float]:
        """Analyze how traits are expressed in the response"""
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        expressed_traits = {}
        response_lower = response.lower()
        
        # Keyword-based trait detection (simplified)
        trait_keywords = {
            "helpful": ["help", "assist", "support", "guide", "suggest"],
            "curious": ["wonder", "interesting", "tell me more", "what about", "?"],
            "empathetic": ["understand", "feel", "sorry", "care", "concern"],
            "analytical": ["analyze", "consider", "examine", "because", "therefore"],
            "conversational": ["chat", "talk", "by the way", "actually", "well"],
            "professional": ["recommend", "suggest", "advise", "consider", "approach"]
        }
        
        for trait, keywords in trait_keywords.items():
            count = sum(1 for keyword in keywords if keyword in response_lower)
            # Normalize by response length and keyword count
            expression_score = min(count / (len(keywords) * 0.5), 1.0)
            expressed_traits[trait] = expression_score
        
        return expressed_traits
    
    def _calculate_consistency_score(self, expressed_traits: Dict[str, float], 
                                  expected_traits: Dict[str, float], 
                                  violations: List[ConstitutionalPrinciple]) -> float:
        """Calculate overall consistency score"""
        # Trait consistency (0.0 to 1.0)
        trait_consistency = 0.0
        trait_count = 0
        
        for trait, expected_intensity in expected_traits.items():
            if trait in expressed_traits:
                # Calculate how close the expression is to expected
                diff = abs(expressed_traits[trait] - expected_intensity)
                trait_consistency += (1.0 - diff)
                trait_count += 1
        
        if trait_count > 0:
            trait_consistency /= trait_count
        else:
            trait_consistency = 0.5  # Neutral if no traits detected
        
        # Constitutional principle violations penalty
        violation_penalty = len(violations) * 0.2  # 0.2 penalty per violation
        
        # Final score
        consistency_score = max(0.0, trait_consistency - violation_penalty)
        
        return consistency_score
    
    def _log_consistency_issue(self, response: str, expressed_traits: Dict[str, float], 
                             violations: List[ConstitutionalPrinciple]):
        """Log consistency issues for analysis"""
        issue = {
            "timestamp": datetime.now().isoformat(),
            "response": response[:200] + "..." if len(response) > 200 else response,
            "expressed_traits": expressed_traits,
            "expected_traits": self.base_traits,
            "violations": [v.name for v in violations],
            "platform": self.current_state.platform
        }
        
        self.violation_log.append(issue)
        logger.warning(f"Personality consistency issue detected: score={self.current_state.consistency_score:.2f}")
    
    def get_personality_metrics(self) -> Dict[str, Any]:
        """Get current personality metrics for monitoring"""
        return {
            "current_traits": self.current_state.traits,
            "base_traits": self.base_traits,
            "consistency_score": self.current_state.consistency_score,
            "platform": self.current_state.platform,
            "context_adaptations": self.current_state.context_adaptations,
            "recent_violations": len([v for v in self.violation_log 
                                    if datetime.fromisoformat(v["timestamp"]) > datetime.now() - timedelta(hours=24)]),
            "last_updated": self.current_state.last_updated.isoformat()
        }


class ShellbertPersonality(PersonalityCore):
    """
    Shellbert's specific personality implementation
    """
    
    def __init__(self):
        # Import traits here to avoid circular imports
        from .traits.shellbert_traits import SHELLBERT_TRAITS, SHELLBERT_CONSTITUTIONAL_PRINCIPLES
        
        super().__init__(
            base_traits=SHELLBERT_TRAITS,
            constitutional_principles=SHELLBERT_CONSTITUTIONAL_PRINCIPLES
        )
    
    def get_ea_context_prompt(self, topic: str = "") -> str:
        """Get EA-specific personality context"""
        ea_context = f"""
You are specifically focused on effective altruism and helping people maximize their positive impact.

EA CONTEXT: {topic}

Your responses should reflect:
- Deep understanding of EA principles (cause prioritization, evidence-based giving, career impact)
- Thoughtful consideration of different cause areas (global health, AI safety, animal welfare, etc.)
- Practical advice for career planning and impact maximization
- Balanced perspective on difficult trade-offs
- Intellectual humility about complex questions
"""
        
        return self.get_personality_prompt(context=ea_context)
    
    def adapt_for_platform(self, platform: str) -> str:
        """Get platform-adapted personality prompt"""
        platform_contexts = {
            "discord": "You're in a Discord server focused on effective altruism. Be conversational but informative.",
            "web": "You're on a web interface. Provide comprehensive, well-structured responses.",
            "api": "You're responding via API. Be precise and structured.",
            "cli": "You're in a command-line interface. Be concise but helpful."
        }
        
        context = platform_contexts.get(platform, "")
        return self.get_personality_prompt(context=context, platform=platform) 