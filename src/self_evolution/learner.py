"""
Performance Learner for Self-Evolution.

Learns from feedback to improve agent performance
over time.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LearningRecord:
    """A record of learning from feedback."""
    timestamp: datetime
    event_type: str  # query, search, dream, reasoning
    input_data: dict
    output_data: dict
    feedback: float  # -1 to 1
    improvement_suggestion: Optional[str] = None


@dataclass
class LearnedPattern:
    """A pattern learned from experience."""
    pattern_id: str
    event_type: str
    pattern: str  # Description of the pattern
    frequency: int
    success_rate: float
    last_observed: datetime
    confidence: float = 0.5


class PerformanceLearner:
    """
    Learns from performance feedback to improve agent behavior.
    
    Learning mechanisms:
    1. Pattern recognition from successful actions
    2. Feedback integration (positive/negative)
    3. Strategy adjustment
    4. Temporal patterns (time-of-day effects)
    
    This enables the agent to improve itself based on
    what works and what doesn't.
    """
    
    def __init__(
        self,
        learning_window: int = 100,
        min_samples: int = 10,
        adaptation_rate: float = 0.1,
    ):
        """
        Initialize Performance Learner.
        
        Args:
            learning_window: Number of records to keep for learning
            min_samples: Minimum samples before adapting
            adaptation_rate: How fast to adapt (0-1)
        """
        self.learning_window = learning_window
        self.min_samples = min_samples
        self.adaptation_rate = adaptation_rate
        
        # Learning data
        self._records: deque[LearningRecord] = deque(maxlen=learning_window)
        self._patterns: dict[str, LearnedPattern] = {}
        
        # Learned parameters
        self._learned_weights: dict[str, float] = {}
        self._learned_thresholds: dict[str, float] = {}
        
        # Statistics
        self._stats = {
            "records_processed": 0,
            "patterns_learned": 0,
            "adaptations_made": 0,
        }
        
        logger.info("PerformanceLearner initialized")
    
    def record(
        self,
        event_type: str,
        input_data: dict,
        output_data: dict,
        feedback: float,
        suggestion: Optional[str] = None,
    ) -> None:
        """
        Record an event and its outcome for learning.
        
        Args:
            event_type: Type of event (query, search, dream, reasoning)
            input_data: Input parameters
            output_data: Output/results
            feedback: Feedback score (-1 to 1)
            suggestion: Optional improvement suggestion
        """
        record = LearningRecord(
            timestamp=datetime.now(),
            event_type=event_type,
            input_data=input_data,
            output_data=output_data,
            feedback=feedback,
            improvement_suggestion=suggestion,
        )
        
        self._records.append(record)
        self._stats["records_processed"] += 1
        
        # Update patterns
        self._update_patterns(record)
        
        # Update learned parameters
        if len(self._records) >= self.min_samples:
            self._adapt_parameters()
    
    def _update_patterns(self, record: LearningRecord) -> None:
        """Update learned patterns based on the record."""
        # Extract simple pattern from input
        pattern_key = f"{record.event_type}:{self._extract_pattern_key(record.input_data)}"
        
        if pattern_key in self._patterns:
            pattern = self._patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_observed = record.timestamp
            
            # Update success rate
            if record.feedback > 0:
                pattern.success_rate = (
                    pattern.success_rate * 0.9 + 0.1
                )
            else:
                pattern.success_rate = (
                    pattern.success_rate * 0.9 - 0.1
                )
            
            # Update confidence
            pattern.confidence = min(1.0, pattern.frequency / 20)
        else:
            # Create new pattern
            self._patterns[pattern_key] = LearnedPattern(
                pattern_id=pattern_key,
                event_type=record.event_type,
                pattern=pattern_key,
                frequency=1,
                success_rate=0.5 if record.feedback == 0 else (0.8 if record.feedback > 0 else 0.2),
                last_observed=record.timestamp,
                confidence=0.1,
            )
            
            self._stats["patterns_learned"] = len(self._patterns)
    
    def _extract_pattern_key(self, data: dict) -> str:
        """Extract a simple pattern key from data."""
        # Sort keys for consistency
        return ",".join(sorted(data.keys()))
    
    def _adapt_parameters(self) -> None:
        """Adapt learned parameters based on accumulated experience."""
        recent = list(self._records)[-self.min_samples:]
        
        # Calculate average feedback by event type
        feedback_by_type: dict[str, list[float]] = {}
        for record in recent:
            if record.event_type not in feedback_by_type:
                feedback_by_type[record.event_type] = []
            feedback_by_type[record.event_type].append(record.feedback)
        
        # Update weights
        for event_type, feedbacks in feedback_by_type.items():
            if feedbacks:
                avg_feedback = sum(feedbacks) / len(feedbacks)
                current_weight = self._learned_weights.get(event_type, 0.5)
                
                # Adaptive update
                new_weight = current_weight + self.adaptation_rate * avg_feedback
                new_weight = max(0, min(1, new_weight))
                
                self._learned_weights[event_type] = new_weight
                self._stats["adaptations_made"] += 1
        
        # Update thresholds based on successful outputs
        success_threshold = self._calculate_success_threshold()
        self._learned_thresholds["success"] = success_threshold
    
    def _calculate_success_threshold(self) -> float:
        """Calculate threshold for considering an action successful."""
        recent = list(self._records)[-self.min_samples:]
        
        positive = sum(1 for r in recent if r.feedback > 0)
        return 0.5 if positive > len(recent) / 2 else 0.3
    
    def get_optimal_parameter(
        self,
        event_type: str,
        parameter: str,
        default: float = 0.5,
    ) -> float:
        """
        Get the optimal value for a parameter based on learning.
        
        Args:
            event_type: Type of event
            parameter: Parameter name
            default: Default value if not learned
            
        Returns:
            Optimal parameter value
        """
        key = f"{event_type}:{parameter}"
        return self._learned_weights.get(key, default)
    
    def get_success_prediction(
        self,
        event_type: str,
        input_data: dict,
    ) -> float:
        """
        Predict the likely success of an action.
        
        Args:
            event_type: Type of event
            input_data: Input parameters
            
        Returns:
            Predicted success probability (0-1)
        """
        pattern_key = f"{event_type}:{self._extract_pattern_key(input_data)}"
        
        if pattern_key in self._patterns:
            pattern = self._patterns[pattern_key]
            # Combine pattern success rate with confidence
            return pattern.success_rate * pattern.confidence + 0.5 * (1 - pattern.confidence)
        
        # No pattern, use base rate
        if self._learned_weights.get(event_type):
            return self._learned_weights[event_type]
        
        return 0.5  # Neutral
    
    def suggest_improvements(self, event_type: str) -> list[str]:
        """
        Suggest improvements based on learned patterns.
        
        Args:
            event_type: Type of event
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Get patterns for this event type
        patterns = [
            p for p in self._patterns.values()
            if p.event_type == event_type and p.success_rate > 0.7
        ]
        
        for pattern in patterns[:3]:
            suggestions.append(
                f"Pattern '{pattern.pattern}' succeeded {pattern.success_rate:.0%} of the time"
            )
        
        # Find failed patterns to avoid
        failed_patterns = [
            p for p in self._patterns.values()
            if p.event_type == event_type and p.success_rate < 0.3
        ]
        
        for pattern in failed_patterns[:2]:
            suggestions.append(
                f"Pattern '{pattern.pattern}' had low success rate ({pattern.success_rate:.0%})"
            )
        
        return suggestions
    
    def get_temporal_patterns(self) -> dict[str, Any]:
        """Analyze temporal patterns in performance."""
        if len(self._records) < 10:
            return {}
        
        # Group by hour of day
        hourly: dict[int, list[float]] = {}
        
        for record in self._records:
            hour = record.timestamp.hour
            if hour not in hourly:
                hourly[hour] = []
            hourly[hour].append(record.feedback)
        
        # Calculate average by hour
        temporal_patterns = {}
        for hour, feedbacks in hourly.items():
            temporal_patterns[f"hour_{hour}"] = {
                "avg_feedback": sum(feedbacks) / len(feedbacks),
                "sample_count": len(feedbacks),
            }
        
        return temporal_patterns
    
    def reset(self) -> None:
        """Reset all learned data."""
        self._records.clear()
        self._patterns.clear()
        self._learned_weights.clear()
        self._learned_thresholds.clear()
        
        self._stats = {
            "records_processed": 0,
            "patterns_learned": 0,
            "adaptations_made": 0,
        }
        
        logger.info("PerformanceLearner reset")
    
    @property
    def statistics(self) -> dict:
        """Get learner statistics."""
        return {
            **self._stats,
            "records_stored": len(self._records),
            "patterns_tracked": len(self._patterns),
            "weights_learned": len(self._learned_weights),
            "success_prediction": (
                sum(self._learned_weights.values()) / len(self._learned_weights)
                if self._learned_weights else 0.5
            ),
        }
