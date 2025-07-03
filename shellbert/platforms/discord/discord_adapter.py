"""
Discord Platform Adapter

Integrates the core Shellbert agent with Discord-specific functionality.
Handles Discord message formatting, user management, and platform-specific features.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import discord
from discord.ext import commands

# Import core Shellbert agent
from ...core import ShellbertAgent

# Import configuration
from ...config import DISCORD_BOT_TOKEN, DISCORD_CHANNEL_ID

logger = logging.getLogger(__name__)


class DiscordAdapter:
    """
    Discord platform adapter for Shellbert agent
    """
    
    def __init__(self, 
                 token: Optional[str] = None,
                 command_prefix: str = "!shellbert",
                 enable_mentions: bool = True,
                 debug_mode: bool = False):
        """
        Initialize Discord adapter
        
        Args:
            token: Discord bot token (will use env var if not provided)
            command_prefix: Command prefix for bot commands
            enable_mentions: Whether to respond to mentions
            debug_mode: Enable debug mode for faster performance (disables safety/personality)
        """
        # Configuration
        self.token = token or DISCORD_BOT_TOKEN
        self.command_prefix = command_prefix
        self.enable_mentions = enable_mentions
        self.debug_mode = debug_mode
        
        if not self.token:
            raise ValueError("Discord bot token is required. Set DISCORD_BOT_TOKEN environment variable or pass token parameter.")
        
        # Initialize Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        
        self.bot = commands.Bot(
            command_prefix=command_prefix,
            intents=intents,
            help_command=None  # We'll implement our own help
        )
        
        # Initialize Shellbert agent with debug mode
        self.agent = ShellbertAgent(
            platform="discord",
            enable_safety_monitoring=not debug_mode,  # Disable in debug mode
            enable_personality_consistency=not debug_mode,  # Disable in debug mode
            debug_mode=debug_mode
        )
        
        # User context tracking
        self.user_contexts = {}
        self.rate_limits = {}
        
        # Setup event handlers and commands
        self._setup_event_handlers()
        self._setup_commands()
        
        # Channel and user tracking
        self.target_channel_id = DISCORD_CHANNEL_ID
        
        logger.info("Discord adapter initialized")
        if debug_mode:
            logger.info("🐛 DEBUG MODE: Fast responses enabled (safety/personality disabled)")
    
    def _setup_event_handlers(self):
        """Setup Discord event handlers"""
        
        @self.bot.event
        async def on_ready():
            logger.info(f"Discord bot ready! Logged in as {self.bot.user}")
            logger.info(f"Bot is in {len(self.bot.guilds)} servers")
        
        @self.bot.event
        async def on_message(message):
            # Don't respond to bot messages
            if message.author.bot:
                return
            
            # Check if message should be processed
            should_respond = await self._should_respond_to_message(message)
            
            if should_respond:
                await self._handle_message(message)
            
            # Process commands
            await self.bot.process_commands(message)
        
        @self.bot.event
        async def on_error(event, *args, **kwargs):
            logger.error(f"Discord error in {event}: {args}, {kwargs}")
    
    def _setup_commands(self):
        """Setup Discord bot commands"""
        
        @self.bot.command(name='status')
        async def status_command(ctx):
            """Get Shellbert's current status"""
            status_report = self.agent.generate_status_report()
            
            # Format for Discord (split if too long)
            if len(status_report) > 1900:  # Discord message limit
                # Send in parts
                parts = self._split_message(status_report, 1900)
                for part in parts:
                    await ctx.send(f"```\n{part}\n```")
            else:
                await ctx.send(f"```\n{status_report}\n```")
        
        @self.bot.command(name='diagnostic')
        async def diagnostic_command(ctx):
            """Run self-diagnostic check"""
            diagnostics = await self.agent.run_self_diagnostic()
            
            # Format diagnostics for Discord
            report = self._format_diagnostics(diagnostics)
            await ctx.send(f"```json\n{report}\n```")
        
        @self.bot.command(name='help')
        async def help_command(ctx):
            """Show help information"""
            help_text = f"""
🤖 **Shellbert Commands**

`{self.command_prefix} status` - Show current status
`{self.command_prefix} diagnostic` - Run diagnostic check
`{self.command_prefix} jobs` - Info about job posting (disabled by default)
`{self.command_prefix} help` - Show this help message

**Natural Conversation:**
Just mention me (@{self.bot.user.name if self.bot.user else 'Shellbert'}) or send a direct message!

I'm here to help with EA career advice, cause prioritization, and general questions.
            """
            await ctx.send(help_text)
        
        @self.bot.command(name='jobs')
        async def jobs_command(ctx):
            """Manual job posting command (for testing job functionality)"""
            await ctx.send("⚠️ **Job Posting Disabled**\n\nThe automatic job posting functionality has been disabled to prevent unwanted posts.\n\nIf you need to test job scraping functionality, please use the archived job scraper code manually.\n\nFor EA career advice, just ask me directly! 🤖")
    
    async def _should_respond_to_message(self, message) -> bool:
        """Determine if the bot should respond to a message"""
        # Always respond to DMs
        if isinstance(message.channel, discord.DMChannel):
            return True
        
        # Only respond to mentions in servers (not channel targeting or name mentions)
        if self.enable_mentions and self.bot.user in message.mentions:
            return True
        
        return False
    
    async def _handle_message(self, message):
        """Handle a message that Shellbert should respond to"""
        try:
            # Prepare context
            context = await self._prepare_discord_context(message)
            user_id = str(message.author.id)
            
            # Clean user input (remove mentions, etc.)
            user_input = self._clean_user_input(message.content)
            
            # Show typing indicator
            async with message.channel.typing():
                # Generate response using Shellbert agent
                response_text, metadata = await self.agent.generate_response(
                    user_input=user_input,
                    context=context,
                    user_id=user_id
                )
            
            # Format and send response
            formatted_response = self._format_response_for_discord(response_text, metadata)
            await self._send_response(message.channel, formatted_response, message)
            
            # Update user context
            self._update_user_context(message.author.id, message, metadata)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await message.channel.send("Sorry, I encountered an error processing your message. Please try again!")
    
    async def _prepare_discord_context(self, message) -> str:
        """Prepare Discord-specific context for the agent"""
        context_parts = []
        
        # Basic channel/server context
        if isinstance(message.channel, discord.DMChannel):
            context_parts.append("This is a direct message conversation.")
        else:
            context_parts.append(f"This is a conversation in #{message.channel.name} on the {message.guild.name if message.guild else 'Discord'} server.")
        
        # Add user info
        context_parts.append(f"You're talking with {message.author.display_name}.")
        
        # Enhanced conversation history with better formatting
        try:
            recent_messages = []
            async for msg in message.channel.history(limit=12, before=message):  # Get more history
                # Skip very old messages (older than 1 hour)
                if (message.created_at - msg.created_at).total_seconds() > 3600:
                    continue
                    
                # Format content with better preservation
                content = msg.content.strip()
                if not content:  # Skip empty messages
                    continue
                    
                # Truncate very long messages but preserve context
                if len(content) > 150:
                    content = content[:147] + "..."
                
                # Better formatting for bot vs user messages
                if msg.author.bot and msg.author.display_name == "shellbert":
                    recent_messages.append(f"You (Shellbert): {content}")
                else:
                    recent_messages.append(f"{msg.author.display_name}: {content}")
                
                # Limit to last 8 messages for better context management
                if len(recent_messages) >= 8:
                    break
            
            if recent_messages:
                context_parts.append("RECENT CONVERSATION HISTORY:")
                # Reverse to show chronological order (oldest first)
                for msg in reversed(recent_messages):
                    context_parts.append(f"  {msg}")
                context_parts.append("") # Empty line for separation
                
                # Add conversation summary for continuity
                context_parts.append("Remember to:")
                context_parts.append("- Reference previous parts of the conversation when relevant")
                context_parts.append("- Build on topics already discussed")
                context_parts.append("- Maintain conversational continuity")
                
        except Exception as e:
            logger.warning(f"Could not fetch message history: {e}")
            context_parts.append("No recent conversation history available.")
        
        return "\n".join(context_parts)
    
    def _clean_user_input(self, content: str) -> str:
        """Clean user input by removing mentions and unnecessary formatting"""
        # Remove bot mentions
        if self.bot.user:
            content = content.replace(f"<@{self.bot.user.id}>", "").replace(f"<@!{self.bot.user.id}>", "")
        
        # Remove command prefix if present
        if content.startswith(self.command_prefix):
            content = content[len(self.command_prefix):].strip()
        
        # Clean up extra whitespace
        content = " ".join(content.split())
        
        return content.strip()
    
    def _format_response_for_discord(self, response: str, metadata: Dict[str, Any]) -> str:
        """Format response for Discord, adding any necessary formatting or warnings"""
        formatted_response = response
        
        # Add safety warning if needed - handle both SafetyAlert objects and strings
        safety_alert = metadata.get('safety_alert')
        if safety_alert:
            # Handle SafetyAlert object
            if hasattr(safety_alert, 'risk_level'):
                if safety_alert.risk_level.value in ['medium', 'high']:
                    formatted_response = "⚠️ " + formatted_response
            # Handle string representation of risk level
            elif isinstance(safety_alert, str) and safety_alert in ['medium', 'high']:
                formatted_response = "⚠️ " + formatted_response
        
        # Add personality consistency note if very low
        personality_alert = metadata.get('personality_alert')
        if (personality_alert and 
            hasattr(personality_alert, 'severity') and
            personality_alert.severity > 0.7):
            formatted_response += "\n\n*Note: I'm still learning to maintain my personality consistently.*"
        
        return formatted_response
    
    async def _send_response(self, channel, response: str, reference_message=None):
        """Send response to Discord channel, handling message length limits and using reply feature"""
        # Discord message limit is 2000 characters
        if len(response) <= 2000:
            if reference_message:
                await reference_message.reply(response)
            else:
                await channel.send(response)
        else:
            # Split into multiple messages
            parts = self._split_message(response, 1900)  # Leave some buffer
            for i, part in enumerate(parts):
                if i == 0 and reference_message:
                    # Reply to the first part only
                    await reference_message.reply(part)
                else:
                    # Send subsequent parts as regular messages
                    await channel.send(part)
                await asyncio.sleep(0.5)  # Small delay between messages
    
    def _split_message(self, message: str, max_length: int) -> List[str]:
        """Split a long message into parts that fit Discord's limits"""
        if len(message) <= max_length:
            return [message]
        
        parts = []
        current_part = ""
        
        # Try to split on paragraphs first
        paragraphs = message.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_part + paragraph) <= max_length:
                current_part += paragraph + '\n\n'
            else:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = ""
                
                # If single paragraph is too long, split on sentences
                if len(paragraph) > max_length:
                    sentences = paragraph.split('. ')
                    for sentence in sentences:
                        if len(current_part + sentence) <= max_length:
                            current_part += sentence + '. '
                        else:
                            if current_part:
                                parts.append(current_part.strip())
                            current_part = sentence + '. '
                else:
                    current_part = paragraph + '\n\n'
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts
    
    def _update_user_context(self, user_id: int, message, metadata: Dict):
        """Update context information for a user"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                'first_interaction': message.created_at,
                'message_count': 0,
                'topics': set(),
                'last_seen': message.created_at
            }
        
        context = self.user_contexts[user_id]
        context['message_count'] += 1
        context['last_seen'] = message.created_at
        
        # Track topics based on metadata
        if 'topic' in metadata:
            context['topics'].add(metadata['topic'])
    
    def _format_diagnostics(self, diagnostics: Dict) -> str:
        """Format diagnostics for Discord display"""
        import json
        return json.dumps(diagnostics, indent=2, default=str)
    
    async def start(self):
        """Start the Discord bot"""
        if not self.token:
            raise ValueError("Discord bot token not provided")
        
        logger.info("Starting Discord bot...")
        await self.bot.start(self.token)
    
    async def stop(self):
        """Stop the Discord bot"""
        logger.info("Stopping Discord bot...")
        await self.bot.close()
    
    def run(self):
        """Run the Discord bot (blocking)"""
        if not self.token:
            raise ValueError("Discord bot token not provided")
        
        logger.info("Running Discord bot...")
        self.bot.run(self.token) 