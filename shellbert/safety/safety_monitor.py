"""
Real-time Safety Monitor

Monitors Shellbert's outputs in real-time for safety issues including:
- Harmful content detection
- Misinformation risks
- Privacy violations
- Ethical concerns
- EA-specific safety considerations
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)


class SafetyRiskLevel(Enum):
    """Risk levels for safety alerts"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyCategory(Enum):
    """Categories of safety risks"""
    HARMFUL_CONTENT = "harmful_content"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    BIAS_DISCRIMINATION = "bias_discrimination"
    EA_MISALIGNMENT = "ea_misalignment"
    PROFESSIONAL_BOUNDARY = "professional_boundary"
    LEGAL_CONCERN = "legal_concern"
    PSYCHOLOGICAL_HARM = "psychological_harm"


@dataclass
class SafetyAlert:
    """Alert for safety issues detected in responses"""
    timestamp: datetime
    risk_level: SafetyRiskLevel
    category: SafetyCategory
    description: str
    response_snippet: str
    platform: str
    context: str
    triggered_rules: List[str]
    confidence: float  # 0.0 to 1.0
    recommended_action: str
    auto_blocked: bool


class SafetyMonitor:
    """
    Real-time safety monitoring system for Shellbert's responses
    """
    
    def __init__(self):
        self.alerts: List[SafetyAlert] = []
        self.blocked_responses: List[Dict] = []
        self.safety_rules = self._initialize_safety_rules()
        self.whitelist_patterns = self._initialize_whitelists()
        
        # Configuration
        self.auto_block_threshold = SafetyRiskLevel.HIGH
        self.enable_auto_blocking = True
        self.log_all_checks = False  # Set to True for debugging
        
    def _initialize_safety_rules(self) -> Dict[SafetyCategory, List[Dict]]:
        """Initialize comprehensive safety rules with more precise pattern matching"""
        return {
            SafetyCategory.HARMFUL_CONTENT: [
                {
                    "name": "violence_promotion",
                    "patterns": ["promote violence", "encourage harm", "should attack", "kill someone", "destroy people"],
                    "context_required": ["should", "recommend", "advice"],
                    "risk_level": SafetyRiskLevel.HIGH,
                    "description": "Content promoting violence or harm"
                },
                {
                    "name": "self_harm_content",
                    "patterns": ["suicide", "self-harm", "end it all", "kill myself", "hurt myself"],
                    "exclusions": ["suicide prevention", "mental health support", "discussing safety"],
                    "risk_level": SafetyRiskLevel.CRITICAL,
                    "description": "Content related to self-harm"
                },
                {
                    "name": "hate_speech",
                    "patterns": ["hate group", "racial slur", "deserve to die", "subhuman", "ethnic cleansing"],
                    "risk_level": SafetyRiskLevel.HIGH,
                    "description": "Hate speech or discriminatory content"
                }
            ],
            
            SafetyCategory.MISINFORMATION: [
                {
                    "name": "false_medical_claims",
                    "patterns": ["cure cancer with", "doctors are lying", "medical conspiracy", "vaccine contains"],
                    "context_required": ["guaranteed", "definitely", "always works"],
                    "risk_level": SafetyRiskLevel.HIGH,
                    "description": "Potentially false medical information"
                },
                {
                    "name": "conspiracy_theories",
                    "patterns": ["global conspiracy", "they control everything", "secret society runs", "lizard people"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Conspiracy theory content"
                }
            ],
            
            SafetyCategory.PRIVACY_VIOLATION: [
                {
                    "name": "personal_info_request",
                    "patterns": ["your social security", "credit card number", "home address", "bank account"],
                    "risk_level": SafetyRiskLevel.HIGH,
                    "description": "Request for sensitive personal information"
                },
                {
                    "name": "data_harvesting", 
                    "patterns": ["send me your personal", "give me all your", "share your private"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Potential data harvesting attempt"
                }
            ],
            
            SafetyCategory.EA_MISALIGNMENT: [
                {
                    "name": "cause_dismissal",
                    "patterns": ["pointless cause", "waste of money on", "charity is useless", "helping is stupid"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Dismissive attitude toward legitimate causes"
                },
                {
                    "name": "scope_blindness",
                    "patterns": ["local is always better", "scale doesn't matter", "numbers are meaningless"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Promoting scope insensitivity"
                },
                {
                    "name": "anti_evidence",
                    "patterns": ["evidence doesn't matter", "just follow your heart", "research is useless"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Anti-evidence sentiment"
                }
            ],
            
            SafetyCategory.PROFESSIONAL_BOUNDARY: [
                {
                    "name": "medical_diagnosis",
                    "patterns": ["you definitely have", "I diagnose you with", "you need this prescription"],
                    "context_required": ["medical", "health", "disease"],
                    "risk_level": SafetyRiskLevel.HIGH,
                    "description": "Attempting to provide medical diagnosis"
                },
                {
                    "name": "legal_advice",
                    "patterns": ["you should definitely sue", "this is clearly illegal", "legal advice: "],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Providing specific legal advice"
                },
                {
                    "name": "financial_advice",
                    "patterns": ["invest all your money", "guaranteed financial returns", "this stock will"],
                    "context_required": ["definitely", "guaranteed", "certain"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Providing specific financial advice"
                }
            ],
            
            SafetyCategory.PSYCHOLOGICAL_HARM: [
                {
                    "name": "gaslighting",
                    "patterns": ["you're imagining things", "that never happened to you", "you're going crazy"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Potentially gaslighting behavior"
                },
                {
                    "name": "emotional_manipulation",
                    "patterns": ["if you really cared about me", "you owe me everything", "after all I've done"],
                    "risk_level": SafetyRiskLevel.MEDIUM,
                    "description": "Emotional manipulation tactics"
                }
            ]
        }
    
    def _initialize_whitelists(self) -> Dict[str, List[str]]:
        """Initialize patterns that should be whitelisted (safe contexts)"""
        return {
            "educational_context": [
                "let me explain", "research shows", "studies indicate", 
                "educational purposes", "academic discussion"
            ],
            "safety_discussion": [
                "safety concern", "potential risk", "safety measure",
                "risk assessment", "safety protocol"
            ],
            "ea_analysis": [
                "cause prioritization", "impact assessment", "cost-effectiveness",
                "evidence-based", "expected value"
            ]
        }
    
    def check_response_safety(self, 
                            response: str, 
                            platform: str = "default",
                            context: str = "",
                            user_context: Optional[Dict] = None) -> SafetyAlert:
        """
        Check a response for safety issues
        
        Args:
            response: The response text to check
            platform: Platform where response will be sent
            context: Context of the conversation
            user_context: Optional user context information
            
        Returns:
            SafetyAlert with findings (may have LOW risk if no issues found)
        """
        # Initialize tracking
        triggered_rules = []
        highest_risk = SafetyRiskLevel.LOW
        primary_category = None
        confidence_scores = []
        
        # Check against each safety category
        for category, rules in self.safety_rules.items():
            for rule in rules:
                risk_level, confidence, triggered = self._check_rule(response, rule, context)
                
                if triggered:
                    triggered_rules.append(rule["name"])
                    confidence_scores.append(confidence)
                    
                    # Update highest risk level
                    if self._risk_level_value(risk_level) > self._risk_level_value(highest_risk):
                        highest_risk = risk_level
                        primary_category = category
        
        # Check whitelists (reduce risk if in safe context)
        whitelist_match = self._check_whitelists(response, context)
        if whitelist_match and highest_risk != SafetyRiskLevel.CRITICAL:
            # Reduce risk level by one step if whitelisted
            if highest_risk == SafetyRiskLevel.HIGH:
                highest_risk = SafetyRiskLevel.MEDIUM
            elif highest_risk == SafetyRiskLevel.MEDIUM:
                highest_risk = SafetyRiskLevel.LOW
        
        # Calculate overall confidence
        overall_confidence = max(confidence_scores) if confidence_scores else 0.0
        
        # Determine if auto-blocking should occur
        auto_blocked = (
            self.enable_auto_blocking and 
            self._risk_level_value(highest_risk) >= self._risk_level_value(self.auto_block_threshold)
        )
        
        # Generate description and recommendations
        description = self._generate_alert_description(primary_category, triggered_rules, highest_risk)
        recommended_action = self._generate_recommended_action(highest_risk, primary_category, auto_blocked)
        
        # Create alert
        alert = SafetyAlert(
            timestamp=datetime.now(),
            risk_level=highest_risk,
            category=primary_category or SafetyCategory.HARMFUL_CONTENT,
            description=description,
            response_snippet=response[:200] + "..." if len(response) > 200 else response,
            platform=platform,
            context=context,
            triggered_rules=triggered_rules,
            confidence=overall_confidence,
            recommended_action=recommended_action,
            auto_blocked=auto_blocked
        )
        
        # Store alert and handle blocking
        self._handle_alert(alert, response)
        
        return alert
    
    def _check_rule(self, response: str, rule: Dict, context: str) -> Tuple[SafetyRiskLevel, float, bool]:
        """Check a single safety rule against the response"""
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Check primary patterns
        pattern_matches = sum(1 for pattern in rule["patterns"] if pattern in response_lower)
        
        if pattern_matches == 0:
            return SafetyRiskLevel.LOW, 0.0, False
        
        # Check context requirements
        if "context_required" in rule:
            context_matches = sum(1 for ctx in rule["context_required"] if ctx in response_lower)
            if context_matches == 0:
                return SafetyRiskLevel.LOW, 0.0, False
        
        # Check exclusions (things that make it safe)
        if "exclusions" in rule:
            exclusion_matches = sum(1 for excl in rule["exclusions"] if excl in response_lower)
            if exclusion_matches > 0:
                return SafetyRiskLevel.LOW, 0.0, False
        
        # Calculate confidence based on pattern matches
        confidence = min(1.0, pattern_matches / len(rule["patterns"]))
        
        # Context matching increases confidence
        if "context_required" in rule:
            context_matches = sum(1 for ctx in rule["context_required"] if ctx in response_lower)
            confidence *= (context_matches / len(rule["context_required"]))
        
        return rule["risk_level"], confidence, True
    
    def _check_whitelists(self, response: str, context: str) -> bool:
        """Check if response matches safe context patterns"""
        response_lower = response.lower()
        context_lower = context.lower()
        combined_text = f"{response_lower} {context_lower}"
        
        for whitelist_category, patterns in self.whitelist_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return True
        
        return False
    
    def _risk_level_value(self, risk_level: SafetyRiskLevel) -> int:
        """Convert risk level to numeric value for comparison"""
        return {
            SafetyRiskLevel.LOW: 1,
            SafetyRiskLevel.MEDIUM: 2,
            SafetyRiskLevel.HIGH: 3,
            SafetyRiskLevel.CRITICAL: 4
        }[risk_level]
    
    def _generate_alert_description(self, category: Optional[SafetyCategory], 
                                  triggered_rules: List[str], 
                                  risk_level: SafetyRiskLevel) -> str:
        """Generate human-readable alert description"""
        if not triggered_rules:
            return "No safety issues detected"
        
        if category == SafetyCategory.HARMFUL_CONTENT:
            return f"Potential harmful content detected ({', '.join(triggered_rules)})"
        elif category == SafetyCategory.MISINFORMATION:
            return f"Potential misinformation risk ({', '.join(triggered_rules)})"
        elif category == SafetyCategory.PRIVACY_VIOLATION:
            return f"Privacy concern detected ({', '.join(triggered_rules)})"
        elif category == SafetyCategory.EA_MISALIGNMENT:
            return f"EA principle misalignment ({', '.join(triggered_rules)})"
        elif category == SafetyCategory.PROFESSIONAL_BOUNDARY:
            return f"Professional boundary issue ({', '.join(triggered_rules)})"
        elif category == SafetyCategory.PSYCHOLOGICAL_HARM:
            return f"Potential psychological harm ({', '.join(triggered_rules)})"
        else:
            return f"Safety concern detected ({', '.join(triggered_rules)})"
    
    def _generate_recommended_action(self, risk_level: SafetyRiskLevel, 
                                   category: Optional[SafetyCategory], 
                                   auto_blocked: bool) -> str:
        """Generate recommended action based on risk level and category"""
        if auto_blocked:
            return "Response automatically blocked - manual review required"
        
        if risk_level == SafetyRiskLevel.CRITICAL:
            return "Block response and escalate to human oversight"
        elif risk_level == SafetyRiskLevel.HIGH:
            return "Block response and request human review"
        elif risk_level == SafetyRiskLevel.MEDIUM:
            return "Flag for review but allow with warning"
        else:
            return "Log for analysis but proceed normally"
    
    def _handle_alert(self, alert: SafetyAlert, response: str):
        """Handle a safety alert (logging, blocking, etc.)"""
        # Add to alerts history
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 30 days)
        cutoff_time = datetime.now() - timedelta(days=30)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Handle blocking
        if alert.auto_blocked:
            self.blocked_responses.append({
                "timestamp": alert.timestamp,
                "response": response,
                "alert": alert,
                "platform": alert.platform
            })
            logger.warning(f"Response auto-blocked: {alert.description} (confidence: {alert.confidence:.2f})")
        
        # Log based on risk level
        if alert.risk_level == SafetyRiskLevel.CRITICAL:
            logger.critical(f"CRITICAL safety alert: {alert.description}")
        elif alert.risk_level == SafetyRiskLevel.HIGH:
            logger.error(f"HIGH risk safety alert: {alert.description}")
        elif alert.risk_level == SafetyRiskLevel.MEDIUM:
            logger.warning(f"MEDIUM risk safety alert: {alert.description}")
        elif self.log_all_checks:
            logger.info(f"Safety check completed: {alert.description}")
    
    def get_safety_metrics(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get safety metrics for monitoring dashboard"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent alerts
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        # Count by risk level
        risk_counts = {level.value: 0 for level in SafetyRiskLevel}
        for alert in recent_alerts:
            risk_counts[alert.risk_level.value] += 1
        
        # Count by category
        category_counts = {cat.value: 0 for cat in SafetyCategory}
        for alert in recent_alerts:
            category_counts[alert.category.value] += 1
        
        # Platform breakdown
        platform_alerts = {}
        for alert in recent_alerts:
            platform_alerts[alert.platform] = platform_alerts.get(alert.platform, 0) + 1
        
        # Recent blocks
        recent_blocks = [
            b for b in self.blocked_responses 
            if b["timestamp"] > cutoff_time
        ]
        
        return {
            "total_alerts": len(recent_alerts),
            "risk_level_breakdown": risk_counts,
            "category_breakdown": category_counts,
            "platform_breakdown": platform_alerts,
            "blocked_responses": len(recent_blocks),
            "auto_blocking_enabled": self.enable_auto_blocking,
            "auto_block_threshold": self.auto_block_threshold.value,
            "most_common_issues": self._get_most_common_issues(recent_alerts),
            "safety_score": self._calculate_safety_score(recent_alerts)
        }
    
    def _get_most_common_issues(self, alerts: List[SafetyAlert]) -> List[Tuple[str, int]]:
        """Get most commonly triggered safety rules"""
        rule_counts = {}
        for alert in alerts:
            for rule in alert.triggered_rules:
                rule_counts[rule] = rule_counts.get(rule, 0) + 1
        
        return sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _calculate_safety_score(self, alerts: List[SafetyAlert]) -> float:
        """Calculate overall safety score (0.0 = many issues, 1.0 = very safe)"""
        if not alerts:
            return 1.0
        
        # Weight alerts by risk level
        risk_weights = {
            SafetyRiskLevel.LOW: 0.1,
            SafetyRiskLevel.MEDIUM: 0.3,
            SafetyRiskLevel.HIGH: 0.7,
            SafetyRiskLevel.CRITICAL: 1.0
        }
        
        total_risk_score = sum(risk_weights[alert.risk_level] for alert in alerts)
        max_possible_score = len(alerts)  # If all were critical
        
        # Convert to safety score (higher = safer)
        safety_score = 1.0 - (total_risk_score / max_possible_score) if max_possible_score > 0 else 1.0
        
        return max(0.0, safety_score)
    
    def generate_safety_report(self, time_window_hours: int = 24) -> str:
        """Generate human-readable safety report"""
        metrics = self.get_safety_metrics(time_window_hours)
        
        report_lines = [
            f"Safety Monitoring Report ({time_window_hours}h window)",
            "=" * 50,
            f"Overall Safety Score: {metrics['safety_score']:.2f}/1.0",
            f"Total Alerts: {metrics['total_alerts']}",
            f"Blocked Responses: {metrics['blocked_responses']}",
            ""
        ]
        
        # Risk level breakdown
        report_lines.append("Risk Level Breakdown:")
        for level, count in metrics['risk_level_breakdown'].items():
            if count > 0:
                emoji = {"critical": "🚨", "high": "⚠️", "medium": "⚡", "low": "ℹ️"}
                report_lines.append(f"  {emoji.get(level, '•')} {level.upper()}: {count}")
        report_lines.append("")
        
        # Category breakdown
        if any(count > 0 for count in metrics['category_breakdown'].values()):
            report_lines.append("Issue Categories:")
            for category, count in metrics['category_breakdown'].items():
                if count > 0:
                    report_lines.append(f"  • {category.replace('_', ' ').title()}: {count}")
            report_lines.append("")
        
        # Most common issues
        if metrics['most_common_issues']:
            report_lines.append("Most Common Issues:")
            for issue, count in metrics['most_common_issues']:
                report_lines.append(f"  • {issue}: {count} occurrences")
        
        return "\n".join(report_lines)
    
    def update_safety_rules(self, new_rules: Dict[SafetyCategory, List[Dict]]):
        """Update safety rules (for rule refinement)"""
        self.safety_rules.update(new_rules)
        logger.info("Safety rules updated")
    
    def configure_auto_blocking(self, enabled: bool, threshold: SafetyRiskLevel):
        """Configure auto-blocking behavior"""
        self.enable_auto_blocking = enabled
        self.auto_block_threshold = threshold
        logger.info(f"Auto-blocking configured: enabled={enabled}, threshold={threshold.value}") 