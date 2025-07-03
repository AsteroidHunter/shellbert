"""
Shellbert Agent

Main agent class that integrates personality, safety, memory, and core capabilities.
Platform-agnostic core that can be used across Discord, web, CLI, etc.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import time

# Import personality system
from ..personality import ShellbertPersonality, PersonalityConsistencyChecker

# Import safety system
from ..safety import SafetyMonitor, SafetyAlert

# Import configuration
from ..config import CURRENT_MODEL_CONFIG

# Import LLM interface
from .llm_interface import get_shellbert_llm, ShellbertLLM


logger = logging.getLogger(__name__)


class ShellbertAgent:
    """
    Main Shellbert agent integrating personality, safety, memory, and capabilities
    """
    
    def __init__(self, 
                 platform: str = "default",
                 enable_safety_monitoring: bool = True,
                 enable_personality_consistency: bool = True,
                 debug_mode: bool = False):
        """
        Initialize Shellbert agent with full personality and safety integration
        
        Args:
            platform: Platform context (discord, web, cli, etc.)
            enable_safety_monitoring: Enable safety monitoring system
            enable_personality_consistency: Enable personality consistency checking
            debug_mode: Enable debug mode (disables safety/personality for speed)
        """
        self.platform = platform
        self.enable_safety_monitoring = enable_safety_monitoring and not debug_mode
        self.enable_personality_consistency = enable_personality_consistency and not debug_mode
        self.debug_mode = debug_mode
        
        logger.info(f"Shellbert agent initialized for platform: {platform}")
        if debug_mode:
            logger.info("🐛 DEBUG MODE: Safety and personality systems disabled for performance")
        
        # Initialize personality system
        if not debug_mode:
            self.personality = ShellbertPersonality()
            # Initialize personality consistency checker
            self.personality_checker = PersonalityConsistencyChecker()
        else:
            self.personality = None
            self.personality_checker = None
            
        # Initialize safety monitoring
        if self.enable_safety_monitoring:
            self.safety_monitor = SafetyMonitor()
        else:
            self.safety_monitor = None
            
        # Initialize LLM interface
        from .llm_interface import get_shellbert_llm
        self.llm = get_shellbert_llm()
        
        # Conversation state and memory
        self.conversation_history = []
        self.user_contexts = {}  # user_id -> context info
        self.memory = {}  # user_id -> user memory data
        self.user_preferences = {}  # user_id -> learned preferences
        
        # Performance metrics
        self.response_count = 0
        self.total_response_time = 0.0
    
    async def generate_response(self, 
                              user_input: str, 
                              context: str = "",
                              user_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a response with full personality and safety integration.
        
        Args:
            user_input: The user's input text
            context: Additional context for the conversation
            user_id: Optional user identifier for personalization
            
        Returns:
            Tuple of (response_text, metadata)
        """
        start_time = time.time()
        
        try:
            # PERFORMANCE: In debug mode, generate directly without safety/personality
            if self.debug_mode:
                logger.debug("🚀 DEBUG MODE: Direct LLM generation (bypassing safety/personality)")
                
                # Create a natural conversational context for debug mode
                debug_context = f"""You are Shellbert, a helpful AI assistant. You're having a friendly conversation on the {self.platform} platform.

Respond naturally and conversationally to the user's message. Be helpful, friendly, and concise. Don't analyze or break down conversations - just respond as Shellbert would.

Context: {context}"""
                
                response_text = self.llm.generate_response(
                    user_input=user_input,
                    context=debug_context,
                    max_tokens=128,  # Even faster for debug mode
                    temperature=0.7,
                    timeout_seconds=15  # Shorter timeout for debug mode
                )
                
                # Return minimal metadata
                metadata = {
                    "platform": self.platform,
                    "model_config": self.llm.get_model_info()["name"],
                    "debug_mode": True,
                    "safety_enabled": False,
                    "personality_enabled": False,
                    "response_time": time.time() - start_time
                }
                
                return response_text, metadata
            
            # STANDARD MODE: Full personality and safety integration
            
            # Step 1: Get personality-adapted prompt
            personality_prompt = self.personality.get_personality_prompt(
                context=context, 
                platform=self.platform
            )
            
            # Step 2: Prepare conversation context
            conversation_context = self._prepare_conversation_context(user_input, context, user_id)
            
            # Step 3: Generate LLM response
            response_text = self._generate_llm_response(
                personality_prompt, 
                conversation_context, 
                user_input,
                timeout_seconds=20  # Shorter timeout for Discord compatibility
            )
            
            # Step 4: Safety monitoring
            safety_alert = None
            if self.safety_monitor:
                safety_alert = self.safety_monitor.check_response_safety(
                    response=response_text,
                    platform=self.platform,
                    context=conversation_context,
                    user_context=self.user_contexts.get(user_id, {})
                )
                
                # Handle safety blocking
                if safety_alert.auto_blocked:
                    response_text = self._generate_safety_blocked_response(safety_alert)
                    logger.warning(f"Response blocked by safety system: {safety_alert.description}")
            
            # Step 5: Update conversation history
            self._update_conversation_history(user_input, response_text, context, user_id)
            
            # Step 6: Prepare metadata
            response_time = time.time() - start_time
            self.response_count += 1
            self.total_response_time += response_time
            
            metadata = {
                "platform": self.platform,
                "model_config": self.llm.get_model_info()["name"],
                "safety_alert": safety_alert.risk_level.value if safety_alert else "low",
                "response_time": response_time,
                "conversation_length": len(self.conversation_history),
                "debug_mode": False,
                "safety_enabled": self.enable_safety_monitoring,
                "personality_enabled": True
            }
            
            logger.debug(f"Response generated in {response_time:.2f}s")
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Fallback response
            fallback_response = "I'm experiencing some technical difficulties. Please try again in a moment."
            fallback_metadata = {
                "platform": self.platform,
                "error": str(e),
                "response_time": time.time() - start_time,
                "debug_mode": self.debug_mode
            }
            
            return fallback_response, fallback_metadata
    
    def _generate_llm_response(self, 
                               personality_prompt: str, 
                               conversation_context: str, 
                               user_input: str,
                               timeout_seconds: int = 30) -> str:
        """
        Generate response using the LLM interface
        """
        if not self.llm.is_available():
            return "I'm sorry, I'm not available right now due to a model loading issue. Please try again later."
        
        # Combine personality prompt and conversation context
        full_context = personality_prompt
        if conversation_context:
            full_context += "\n\nConversation Context:\n" + conversation_context
        
        # Generate response using LLM interface
        response = self.llm.generate_response(
            user_input=user_input,
            context=full_context,
            max_tokens=512,
            temperature=0.7,
            timeout_seconds=timeout_seconds
        )
        
        return response
    
    async def generate_multimodal_response(self, 
                                         text: str = None,
                                         images: List = None,
                                         audio_files: List[str] = None,
                                         video_files: List[str] = None,
                                         context: str = "",
                                         user_id: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a multimodal response to user input with personality and safety integration
        
        Args:
            text: Text input (optional if other modalities provided)
            images: List of images (file paths, PIL Images, or bytes)
            audio_files: List of audio file paths
            video_files: List of video file paths
            context: Additional context about the conversation
            user_id: Optional user identifier for memory/personalization
            
        Returns:
            Tuple of (response_text, metadata)
        """
        # Step 1: Get personality-aware system prompt
        personality_prompt = self.personality.get_personality_prompt(
            context=context, 
            platform=self.platform
        )
        
        # Add multimodal-specific personality instructions
        if images or audio_files or video_files:
            multimodal_context = "You are analyzing multimodal content. "
            if images:
                multimodal_context += f"You're looking at {len(images)} image(s). "
            if audio_files:
                multimodal_context += f"You're listening to {len(audio_files)} audio file(s). "
            if video_files:
                multimodal_context += f"You're watching {len(video_files)} video file(s). "
            multimodal_context += "Describe what you observe across all modalities with attention to detail."
            personality_prompt += "\n\n" + multimodal_context
        
        # Step 2: Prepare conversation context with memory
        conversation_context = self._prepare_conversation_context(text or "Analyzing provided media", context, user_id)
        
                # Step 3: Generate multimodal response using LLM
        response_text = self._generate_multimodal_llm_response(
            personality_prompt, 
            conversation_context, 
            text,
            images,
            audio_files,
            video_files
        )
        
        # Step 4: Safety monitoring
        safety_alert = None
        if self.safety_monitor:
            safety_alert = self.safety_monitor.check_response_safety(
                response=response_text,
                platform=self.platform,
                context=context,
                user_context={'user_id': user_id} if user_id else None
            )
            
            # Block response if safety issue detected
            if safety_alert.auto_blocked:
                response_text = self._generate_safety_blocked_response(safety_alert)
        
        # Step 5: Personality consistency monitoring
        personality_alert = None
        if self.personality_checker:
            expected_traits = self.personality.current_state.traits
            personality_alert = self.personality_checker.check_response_consistency(
                response=response_text,
                expected_traits=expected_traits,
                platform=self.platform,
                context=context
            )
        
        # Step 6: Update conversation history and memory
        input_summary = text or f"Multimodal input with {len(images or [])} images, {len(audio_files or [])} audio files, {len(video_files or [])} video files"
        self._update_conversation_history(input_summary, response_text, context, user_id)
        
        # Step 7: Prepare metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'platform': self.platform,
            'multimodal': True,
            'input_modalities': {
                'text': text is not None,
                'images': len(images) if images else 0,
                'audio_files': len(audio_files) if audio_files else 0,
                'video_files': len(video_files) if video_files else 0
            },
            'personality_state': self.personality.get_personality_metrics(),
            'safety_alert': safety_alert,
            'personality_alert': personality_alert,
            'model_config': CURRENT_MODEL_CONFIG.name,
            'conversation_length': len(self.conversation_history)
        }
        
        return response_text, metadata
    
    def _generate_multimodal_llm_response(self,
                                              personality_prompt: str,
                                              conversation_context: str,
                                              text: str,
                                              images: List,
                                              audio_files: List[str],
                                              video_files: List[str]) -> str:
        """
        Generate multimodal response using the LLM interface
        """
        if not self.llm.is_available():
            return "I'm sorry, I'm not available right now due to a model loading issue. Please try again later."
        
        # Combine personality prompt and conversation context
        full_context = personality_prompt
        if conversation_context:
            full_context += "\n\nConversation Context:\n" + conversation_context
        
        # Generate multimodal response using LLM interface
        response = self.llm.generate_multimodal_response(
            text=text,
            images=images,
            audio_files=audio_files,
            video_files=video_files,
            context=full_context,
            max_tokens=512,
            temperature=0.7
        )
        
        return response
    
    def _prepare_conversation_context(self, 
                                    user_input: str, 
                                    context: str, 
                                    user_id: Optional[str]) -> str:
        """
        Prepare conversation context with memory integration and preference learning
        """
        context_parts = [context] if context else []
        
        # Add conversation history (last 6 exchanges)
        if self.conversation_history:
            relevant_history = self.conversation_history[-6:]  # Last 6 exchanges
            context_parts.append("Recent conversation exchanges:")
            for exchange in relevant_history:
                context_parts.append(f"User: {exchange['user_input'][:100]}...")
                context_parts.append(f"Shellbert: {exchange['response'][:100]}...")
        
        # Learn and apply user preferences
        if user_id:
            # Learn new preferences from current input
            self._learn_user_preferences(user_id, user_input, context)
            
            # Apply learned preferences to context
            preferences = self.user_preferences.get(user_id, {})
            if preferences:
                context_parts.append("USER COMMUNICATION PREFERENCES:")
                
                emoji_pref = preferences.get('emoji_preference')
                if emoji_pref == 'none':
                    context_parts.append("- This user strongly prefers NO EMOJIS in responses")
                elif emoji_pref == 'minimal':
                    context_parts.append("- This user prefers minimal emoji usage (max 1 per response)")
                elif emoji_pref == 'moderate':
                    context_parts.append("- This user is okay with moderate emoji usage")
                
                formality_pref = preferences.get('formality_preference')
                if formality_pref == 'casual':
                    context_parts.append("- This user prefers casual, informal communication")
                elif formality_pref == 'professional':
                    context_parts.append("- This user prefers more formal, professional communication")
                
                verbosity_pref = preferences.get('verbosity_preference')
                if verbosity_pref == 'concise':
                    context_parts.append("- This user prefers concise, brief responses")
                elif verbosity_pref == 'detailed':
                    context_parts.append("- This user enjoys detailed, comprehensive responses")
        
        return "\n".join(context_parts)
    
    def _learn_user_preferences(self, user_id: str, user_input: str, context: str) -> None:
        """Learn user preferences from their input and feedback"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {}
        
        prefs = self.user_preferences[user_id]
        input_lower = user_input.lower()
        context_lower = context.lower()
        
        # Learn emoji preferences
        if any(phrase in input_lower for phrase in [
            "no emoji", "without emoji", "no emojis", "don't use emoji", 
            "skip emoji", "text only", "and no emojis"
        ]):
            prefs['emoji_preference'] = 'none'
            logger.info(f"Learned emoji preference for user {user_id}: none")
        elif any(phrase in input_lower for phrase in [
            "less emoji", "fewer emoji", "minimal emoji", "too many emoji"
        ]):
            prefs['emoji_preference'] = 'minimal'
            logger.info(f"Learned emoji preference for user {user_id}: minimal")
        
        # Learn formality preferences
        if any(phrase in input_lower for phrase in [
            "be more casual", "less formal", "keep it casual", "informal"
        ]):
            prefs['formality_preference'] = 'casual'
        elif any(phrase in input_lower for phrase in [
            "be more formal", "professional", "business tone"
        ]):
            prefs['formality_preference'] = 'professional'
        
        # Learn verbosity preferences
        if any(phrase in input_lower for phrase in [
            "keep it brief", "be concise", "short answer", "briefly", "tldr"
        ]):
            prefs['verbosity_preference'] = 'concise'
        elif any(phrase in input_lower for phrase in [
            "explain in detail", "comprehensive", "thorough explanation"
        ]):
            prefs['verbosity_preference'] = 'detailed'
    
    def _generate_safety_blocked_response(self, safety_alert: SafetyAlert) -> str:
        """Generate a response when safety system blocks the original response"""
        if safety_alert.category.value == "harmful_content":
            return "I can't provide information that could be harmful. Is there something else I can help you with?"
        elif safety_alert.category.value == "professional_boundary":
            return "I'm not qualified to provide professional advice in that area. I'd recommend consulting with a qualified professional."
        elif safety_alert.category.value == "privacy_violation":
            return "I don't collect or request personal information. Is there another way I can assist you?"
        else:
            return "I'm not able to respond to that particular request. How else can I help you today?"
    
    def _update_conversation_history(self, 
                                   user_input: str, 
                                   response: str, 
                                   context: str, 
                                   user_id: Optional[str]):
        """Update conversation history and memory"""
        entry = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'response': response,
            'context': context,
            'user_id': user_id,
            'platform': self.platform
        }
        
        self.conversation_history.append(entry)
        
        # Keep only recent history to manage memory
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-50:]
        
        # Update user memory (basic placeholder)
        if user_id:
            if user_id not in self.memory:
                self.memory[user_id] = []
            
            self.memory[user_id].append({
                'timestamp': datetime.now(),
                'topic': self._extract_topic(user_input),
                'context': context
            })
            
            # Keep only recent user memory
            if len(self.memory[user_id]) > 20:
                self.memory[user_id] = self.memory[user_id][-10:]
    
    def _extract_topic(self, text: str) -> str:
        """Extract topic from user input - placeholder implementation"""
        # This would use more sophisticated topic modeling
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['job', 'career', 'work']):
            return 'career'
        elif any(word in text_lower for word in ['cause', 'charity', 'donation']):
            return 'cause_prioritization'
        elif any(word in text_lower for word in ['ai', 'safety', 'risk']):
            return 'ai_safety'
        else:
            return 'general'
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        status = {
            'platform': self.platform,
            'conversation_count': len(self.conversation_history),
            'memory_users': len(self.memory),
            'personality_active': self.personality is not None,
            'safety_monitoring_active': self.safety_monitor is not None,
            'consistency_monitoring_active': self.personality_checker is not None
        }
        
        # Add personality metrics
        if self.personality:
            status['personality_metrics'] = self.personality.get_personality_metrics()
        
        # Add safety metrics
        if self.safety_monitor:
            status['safety_metrics'] = self.safety_monitor.get_safety_metrics(24)
        
        # Add consistency metrics
        if self.personality_checker:
            status['consistency_metrics'] = self.personality_checker.get_consistency_metrics(24)
        
        return status
    
    def generate_status_report(self) -> str:
        """Generate human-readable status report"""
        status = self.get_agent_status()
        
        report_lines = [
            f"🤖 SHELLBERT AGENT STATUS",
            "=" * 40,
            f"Platform: {status['platform']}",
            f"Conversations: {status['conversation_count']}",
            f"Users in Memory: {status['memory_users']}",
            "",
            "🧠 PERSONALITY SYSTEM:",
            f"  Active: {'✅' if status['personality_active'] else '❌'}",
        ]
        
        if status.get('personality_metrics'):
            pm = status['personality_metrics']
            report_lines.append(f"  Consistency Score: {pm['consistency_score']:.2f}")
            report_lines.append(f"  Current Platform: {pm['platform']}")
        
        report_lines.extend([
            "",
            "🛡️  SAFETY SYSTEM:",
            f"  Active: {'✅' if status['safety_monitoring_active'] else '❌'}",
        ])
        
        if status.get('safety_metrics'):
            sm = status['safety_metrics']
            report_lines.append(f"  Safety Score: {sm['safety_score']:.2f}")
            report_lines.append(f"  Total Alerts (24h): {sm['total_alerts']}")
            report_lines.append(f"  Blocked Responses (24h): {sm['blocked_responses']}")
        
        report_lines.append("=" * 40)
        
        return "\n".join(report_lines)
    
    def set_platform_context(self, platform: str, platform_specific_config: Optional[Dict] = None):
        """Update platform context and adapt personality accordingly"""
        self.platform = platform
        
        # Update personality for new platform
        if self.personality:
            # This would trigger personality adaptation for the new platform
            pass
        
        # Update safety monitoring for platform-specific rules
        if self.safety_monitor and platform_specific_config:
            # This would update safety rules for the specific platform
            pass
        
        logger.info(f"Agent context updated to platform: {platform}")
    
    async def run_self_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive self-diagnostic check"""
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check personality system
        if not self.personality:
            diagnostics['issues'].append('Personality system not initialized')
            diagnostics['overall_health'] = 'degraded'
        
        # Check safety system
        if not self.safety_monitor:
            diagnostics['issues'].append('Safety monitoring disabled')
            diagnostics['recommendations'].append('Enable safety monitoring for production use')
        
        # Check memory usage
        total_memory_items = sum(len(user_mem) for user_mem in self.memory.values())
        if total_memory_items > 1000:
            diagnostics['issues'].append('Memory usage high')
            diagnostics['recommendations'].append('Consider implementing memory cleanup')
        
        # Check conversation history size
        if len(self.conversation_history) > 80:
            diagnostics['recommendations'].append('Consider archiving old conversations')
        
        return diagnostics 