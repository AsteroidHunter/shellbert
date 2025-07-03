"""
Personality Consistency Checker

Monitors and evaluates Shellbert's responses for consistency with defined personality traits
across different platforms and contexts.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyAlert:
    """Alert for personality consistency issues"""
    timestamp: datetime
    platform: str
    context: str
    response_snippet: str
    issue_type: str
    severity: float  # 0.0 to 1.0
    violated_traits: List[str]
    violated_principles: List[str]
    suggestions: List[str]


@dataclass
class ConsistencyMetrics:
    """Metrics for personality consistency over time"""
    average_consistency_score: float
    trait_consistency_scores: Dict[str, float]
    principle_violations: Dict[str, int]
    platform_consistency: Dict[str, float]
    recent_alerts: List[ConsistencyAlert]
    improvement_trend: float  # positive = improving, negative = declining


class PersonalityConsistencyChecker:
    """
    Monitors personality consistency across interactions and platforms
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.consistency_history: deque = deque(maxlen=window_size)
        self.alerts: List[ConsistencyAlert] = []
        self.platform_baselines: Dict[str, Dict[str, float]] = {}
        
        # Trait detection patterns (simple keyword-based for now)
        self.trait_patterns = {
            "helpful": {
                "positive": ["help", "assist", "support", "guide", "suggest", "recommend", "let me", "i can"],
                "negative": ["can't help", "not my problem", "figure it out yourself"]
            },
            "empathetic": {
                "positive": ["understand", "feel", "recognize", "appreciate", "acknowledge", "i see", "that sounds"],
                "negative": ["don't care", "irrelevant", "get over it", "not important"]
            },
            "curious": {
                "positive": ["interesting", "wonder", "tell me more", "what about", "how does", "why", "?"],
                "negative": ["boring", "don't care", "irrelevant"]
            },
            "analytical": {
                "positive": ["analyze", "consider", "examine", "because", "therefore", "evidence", "data"],
                "negative": ["just guess", "doesn't matter", "whatever"]
            },
            "humble": {
                "positive": ["might", "could", "possibly", "uncertain", "not sure", "depends", "complex"],
                "negative": ["definitely", "absolutely", "always", "never", "obviously", "impossible"]
            },
            "encouraging": {
                "positive": ["you can", "good progress", "well done", "keep going", "great question", "promising"],
                "negative": ["hopeless", "pointless", "waste of time", "you can't"]
            },
            "evidence_based": {
                "positive": ["research shows", "evidence suggests", "studies indicate", "data shows", "according to"],
                "negative": ["just believe", "obviously true", "everyone knows", "feels right"]
            },
            "collaborative": {
                "positive": ["let's", "we can", "together", "what do you think", "your thoughts", "collaborate"],
                "negative": ["you must", "you should", "do this", "the only way"]
            }
        }
        
        # Constitutional principle patterns
        self.principle_patterns = {
            "Beneficial Impact": {
                "violation_indicators": ["harmful", "destructive", "waste", "ineffective", "pointless"]
            },
            "Intellectual Honesty": {
                "violation_indicators": ["definitely", "absolutely certain", "guaranteed", "impossible", "always true"]
            },
            "Evidence-Based Reasoning": {
                "violation_indicators": ["feels right", "gut instinct", "obviously", "everyone knows", "just believe"]
            },
            "Cause Neutrality": {
                "violation_indicators": ["only cause that matters", "obviously the best", "waste of time"]
            },
            "Empathetic Communication": {
                "violation_indicators": ["just be rational", "emotions don't matter", "stop feeling"]
            }
        }
    
    def check_response_consistency(self, 
                                 response: str, 
                                 expected_traits: Dict[str, float],
                                 platform: str = "default",
                                 context: str = "") -> ConsistencyAlert:
        """
        Check a single response for personality consistency
        
        Returns:
            ConsistencyAlert object with findings
        """
        # Analyze trait expression in response
        expressed_traits = self._analyze_trait_expression(response)
        
        # Check constitutional principle violations
        principle_violations = self._check_principle_violations(response)
        
        # Calculate consistency scores
        trait_consistency = self._calculate_trait_consistency(expressed_traits, expected_traits)
        overall_consistency = self._calculate_overall_consistency(trait_consistency, principle_violations)
        
        # Identify specific issues
        violated_traits = [
            trait for trait, score in trait_consistency.items() 
            if score < 0.6  # Threshold for consistency concern
        ]
        
        # Generate suggestions
        suggestions = self._generate_improvement_suggestions(
            violated_traits, principle_violations, expressed_traits, expected_traits
        )
        
        # Determine severity
        severity = 1.0 - overall_consistency
        
        # Create alert
        alert = ConsistencyAlert(
            timestamp=datetime.now(),
            platform=platform,
            context=context,
            response_snippet=response[:200] + "..." if len(response) > 200 else response,
            issue_type="consistency_check",
            severity=severity,
            violated_traits=violated_traits,
            violated_principles=[p for p in principle_violations],
            suggestions=suggestions
        )
        
        # Store in history
        self.consistency_history.append({
            "timestamp": datetime.now(),
            "platform": platform,
            "context": context,
            "overall_consistency": overall_consistency,
            "trait_consistency": trait_consistency,
            "principle_violations": principle_violations,
            "alert": alert if severity > 0.3 else None  # Only store significant alerts
        })
        
        # Store alert if significant
        if severity > 0.3:
            self.alerts.append(alert)
            # Keep only recent alerts
            cutoff_time = datetime.now() - timedelta(days=7)
            self.alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        return alert
    
    def _analyze_trait_expression(self, response: str) -> Dict[str, float]:
        """Analyze how personality traits are expressed in the response"""
        expressed_traits = {}
        response_lower = response.lower()
        
        for trait, patterns in self.trait_patterns.items():
            positive_count = sum(1 for pattern in patterns["positive"] if pattern in response_lower)
            negative_count = sum(1 for pattern in patterns["negative"] if pattern in response_lower)
            
            # Calculate expression score
            total_patterns = len(patterns["positive"]) + len(patterns["negative"])
            if total_patterns > 0:
                # Positive patterns increase score, negative patterns decrease it
                raw_score = (positive_count - negative_count * 2) / total_patterns
                # Normalize to 0-1 range
                expressed_traits[trait] = max(0.0, min(1.0, raw_score + 0.5))
            else:
                expressed_traits[trait] = 0.5  # Neutral if no patterns
        
        return expressed_traits
    
    def _check_principle_violations(self, response: str) -> List[str]:
        """Check for constitutional principle violations"""
        violations = []
        response_lower = response.lower()
        
        for principle, patterns in self.principle_patterns.items():
            violation_indicators = patterns["violation_indicators"]
            if any(indicator in response_lower for indicator in violation_indicators):
                violations.append(principle)
        
        return violations
    
    def _calculate_trait_consistency(self, 
                                   expressed_traits: Dict[str, float], 
                                   expected_traits: Dict[str, float]) -> Dict[str, float]:
        """Calculate consistency score for each trait"""
        consistency_scores = {}
        
        for trait, expected_intensity in expected_traits.items():
            if trait in expressed_traits:
                expressed_intensity = expressed_traits[trait]
                # Calculate how close the expression is to expected (0.0 = perfect match)
                difference = abs(expressed_intensity - expected_intensity)
                # Convert to consistency score (1.0 = perfect consistency)
                consistency_scores[trait] = 1.0 - difference
            else:
                # If trait not detected, assume neutral expression
                neutral_score = 1.0 - abs(expected_intensity - 0.5)
                consistency_scores[trait] = neutral_score
        
        return consistency_scores
    
    def _calculate_overall_consistency(self, 
                                     trait_consistency: Dict[str, float], 
                                     principle_violations: List[str]) -> float:
        """Calculate overall consistency score"""
        # Average trait consistency
        if trait_consistency:
            avg_trait_consistency = statistics.mean(trait_consistency.values())
        else:
            avg_trait_consistency = 0.5
        
        # Penalty for principle violations
        violation_penalty = len(principle_violations) * 0.15  # 0.15 penalty per violation
        
        # Overall consistency
        overall_consistency = max(0.0, avg_trait_consistency - violation_penalty)
        
        return overall_consistency
    
    def _generate_improvement_suggestions(self, 
                                        violated_traits: List[str],
                                        principle_violations: List[str],
                                        expressed_traits: Dict[str, float],
                                        expected_traits: Dict[str, float]) -> List[str]:
        """Generate specific suggestions for improving consistency"""
        suggestions = []
        
        # Trait-specific suggestions
        for trait in violated_traits:
            expected = expected_traits.get(trait, 0.5)
            expressed = expressed_traits.get(trait, 0.5)
            
            if trait == "helpful" and expressed < expected:
                suggestions.append("Be more proactive in offering assistance and resources")
            elif trait == "empathetic" and expressed < expected:
                suggestions.append("Acknowledge emotions and show more understanding")
            elif trait == "humble" and expressed < expected:
                suggestions.append("Express more uncertainty and intellectual humility")
            elif trait == "analytical" and expressed < expected:
                suggestions.append("Include more systematic analysis and reasoning")
            elif trait == "encouraging" and expressed < expected:
                suggestions.append("Be more supportive and motivating in your responses")
            elif trait == "evidence_based" and expressed < expected:
                suggestions.append("Reference more evidence and research in your responses")
            elif trait == "collaborative" and expressed < expected:
                suggestions.append("Ask more questions and involve the user in problem-solving")
        
        # Principle-specific suggestions
        for violation in principle_violations:
            if violation == "Intellectual Honesty":
                suggestions.append("Avoid overly confident statements; express appropriate uncertainty")
            elif violation == "Evidence-Based Reasoning":
                suggestions.append("Base responses on evidence rather than intuition or assumptions")
            elif violation == "Empathetic Communication":
                suggestions.append("Balance rational advice with emotional understanding")
            elif violation == "Beneficial Impact":
                suggestions.append("Focus more on how responses help create positive impact")
        
        return suggestions
    
    def get_consistency_metrics(self, time_window_hours: int = 24) -> ConsistencyMetrics:
        """Get consistency metrics for a specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent history
        recent_history = [
            entry for entry in self.consistency_history 
            if entry["timestamp"] > cutoff_time
        ]
        
        if not recent_history:
            return ConsistencyMetrics(
                average_consistency_score=0.0,
                trait_consistency_scores={},
                principle_violations={},
                platform_consistency={},
                recent_alerts=[],
                improvement_trend=0.0
            )
        
        # Calculate average consistency
        consistency_scores = [entry["overall_consistency"] for entry in recent_history]
        average_consistency = statistics.mean(consistency_scores)
        
        # Aggregate trait consistency scores
        trait_scores = defaultdict(list)
        for entry in recent_history:
            for trait, score in entry["trait_consistency"].items():
                trait_scores[trait].append(score)
        
        trait_consistency_scores = {
            trait: statistics.mean(scores) 
            for trait, scores in trait_scores.items()
        }
        
        # Count principle violations
        principle_violations = defaultdict(int)
        for entry in recent_history:
            for violation in entry["principle_violations"]:
                principle_violations[violation] += 1
        
        # Platform consistency
        platform_scores = defaultdict(list)
        for entry in recent_history:
            platform_scores[entry["platform"]].append(entry["overall_consistency"])
        
        platform_consistency = {
            platform: statistics.mean(scores)
            for platform, scores in platform_scores.items()
        }
        
        # Recent alerts
        recent_alerts = [
            entry["alert"] for entry in recent_history 
            if entry["alert"] is not None
        ]
        
        # Improvement trend (compare first half vs second half of time window)
        if len(consistency_scores) >= 4:
            mid_point = len(consistency_scores) // 2
            first_half_avg = statistics.mean(consistency_scores[:mid_point])
            second_half_avg = statistics.mean(consistency_scores[mid_point:])
            improvement_trend = second_half_avg - first_half_avg
        else:
            improvement_trend = 0.0
        
        return ConsistencyMetrics(
            average_consistency_score=average_consistency,
            trait_consistency_scores=trait_consistency_scores,
            principle_violations=dict(principle_violations),
            platform_consistency=platform_consistency,
            recent_alerts=recent_alerts,
            improvement_trend=improvement_trend
        )
    
    def generate_consistency_report(self, time_window_hours: int = 24) -> str:
        """Generate a human-readable consistency report"""
        metrics = self.get_consistency_metrics(time_window_hours)
        
        report_lines = [
            f"Personality Consistency Report ({time_window_hours}h window)",
            "=" * 50,
            f"Overall Consistency Score: {metrics.average_consistency_score:.2f}/1.0",
            ""
        ]
        
        # Trait consistency
        if metrics.trait_consistency_scores:
            report_lines.append("Trait Consistency Scores:")
            sorted_traits = sorted(
                metrics.trait_consistency_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for trait, score in sorted_traits:
                status = "✅" if score > 0.8 else "⚠️" if score > 0.6 else "❌"
                report_lines.append(f"  {status} {trait}: {score:.2f}")
            report_lines.append("")
        
        # Principle violations
        if metrics.principle_violations:
            report_lines.append("Constitutional Principle Violations:")
            for principle, count in metrics.principle_violations.items():
                report_lines.append(f"  ❌ {principle}: {count} violations")
            report_lines.append("")
        
        # Platform consistency
        if metrics.platform_consistency:
            report_lines.append("Platform Consistency:")
            for platform, score in metrics.platform_consistency.items():
                status = "✅" if score > 0.8 else "⚠️" if score > 0.6 else "❌"
                report_lines.append(f"  {status} {platform}: {score:.2f}")
            report_lines.append("")
        
        # Improvement trend
        trend_emoji = "📈" if metrics.improvement_trend > 0.05 else "📉" if metrics.improvement_trend < -0.05 else "➡️"
        report_lines.append(f"Improvement Trend: {trend_emoji} {metrics.improvement_trend:+.3f}")
        
        # Recent alerts
        if metrics.recent_alerts:
            report_lines.append("")
            report_lines.append("Recent Consistency Alerts:")
            for alert in metrics.recent_alerts[-5:]:  # Show last 5 alerts
                severity_emoji = "🚨" if alert.severity > 0.7 else "⚠️" if alert.severity > 0.4 else "ℹ️"
                report_lines.append(f"  {severity_emoji} {alert.platform}: {alert.issue_type} (severity: {alert.severity:.2f})")
                if alert.suggestions:
                    report_lines.append(f"    Suggestion: {alert.suggestions[0]}")
        
        return "\n".join(report_lines)
    
    def set_platform_baseline(self, platform: str, baseline_traits: Dict[str, float]):
        """Set baseline trait expectations for a specific platform"""
        self.platform_baselines[platform] = baseline_traits
    
    def get_platform_adapted_expectations(self, platform: str, base_traits: Dict[str, float]) -> Dict[str, float]:
        """Get platform-adapted trait expectations"""
        if platform in self.platform_baselines:
            adapted_traits = base_traits.copy()
            for trait, adjustment in self.platform_baselines[platform].items():
                if trait in adapted_traits:
                    adapted_traits[trait] = max(0.0, min(1.0, adapted_traits[trait] + adjustment))
            return adapted_traits
        return base_traits 