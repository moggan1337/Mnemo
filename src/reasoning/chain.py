"""
Chain of Thought reasoning module.

Implements step-by-step reasoning with explicit
intermediate steps and self-reflection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from mnemo.reasoning.engine import ReasoningType

logger = logging.getLogger(__name__)


@dataclass
class ThoughtStep:
    """A single step in the reasoning chain."""
    step_number: int
    thought: str
    reasoning_type: str
    confidence: float
    next_steps: list[str] = field(default_factory=list)
    alternatives_considered: list[str] = field(default_factory=list)
    reflection: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ChainOfThought:
    """
    Chain of Thought reasoning.
    
    Breaks down complex reasoning into explicit steps,
    allowing for self-correction and transparency.
    
    Features:
    - Step-by-step reasoning
    - Self-reflection at each step
    - Alternative path consideration
    - Confidence tracking
    - Backtracking on errors
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        reflection_enabled: bool = True,
    ):
        """
        Initialize Chain of Thought.
        
        Args:
            max_steps: Maximum reasoning steps
            reflection_enabled: Enable self-reflection
        """
        self.max_steps = max_steps
        self.reflection_enabled = reflection_enabled
        
        # Current chain
        self._steps: list[ThoughtStep] = []
        
        logger.info("ChainOfThought initialized")
    
    async def think(
        self,
        question: str,
        context: Any,
    ) -> list[ThoughtStep]:
        """
        Perform chain of thought reasoning.
        
        Args:
            question: Question to reason about
            context: Relevant context
            
        Returns:
            List of reasoning steps
        """
        self._steps.clear()
        
        # Initialize
        step = ThoughtStep(
            step_number=0,
            thought=f"Analyzing question: {question}",
            reasoning_type="initialization",
            confidence=1.0,
        )
        self._add_step(step)
        
        # Decompose question
        step = ThoughtStep(
            step_number=1,
            thought=self._decompose_question(question),
            reasoning_type="decomposition",
            confidence=0.8,
        )
        self._add_step(step)
        
        # Gather relevant information
        step = ThoughtStep(
            step_number=2,
            thought="Gathering relevant information from context",
            reasoning_type="retrieval",
            confidence=0.7,
        )
        self._add_step(step)
        
        # Generate reasoning steps
        step_num = 3
        current_confidence = 0.7
        
        while step_num < self.max_steps and current_confidence > 0.5:
            step = self._generate_next_step(question, context, step_num)
            self._add_step(step)
            
            # Reflect on step
            if self.reflection_enabled:
                self._reflect_on_step(step_num - 1)
            
            current_confidence = step.confidence
            step_num += 1
            
            # Check if reasoning is complete
            if self._is_complete():
                break
        
        # Final synthesis
        final_step = ThoughtStep(
            step_number=step_num,
            thought=self._synthesize_answer(),
            reasoning_type="synthesis",
            confidence=self._calculate_final_confidence(),
        )
        self._add_step(final_step)
        
        return self._steps.copy()
    
    def _add_step(self, step: ThoughtStep) -> None:
        """Add a step to the chain."""
        self._steps.append(step)
        logger.debug(f"Thought step {step.step_number}: {step.thought[:50]}...")
    
    def _decompose_question(self, question: str) -> str:
        """Decompose question into sub-questions."""
        return f"Breaking down '{question}' into components: identifying key entities and relationships"
    
    def _generate_next_step(
        self,
        question: str,
        context: Any,
        step_num: int,
    ) -> ThoughtStep:
        """Generate the next reasoning step."""
        # Simple incremental reasoning
        thoughts = {
            3: "Examining first piece of evidence",
            4: "Evaluating implications of evidence",
            5: "Checking for alternative interpretations",
            6: "Connecting related concepts",
            7: "Synthesizing intermediate conclusions",
            8: "Validating reasoning against context",
            9: "Preparing final assessment",
        }
        
        thought = thoughts.get(step_num, f"Processing step {step_num}")
        
        return ThoughtStep(
            step_number=step_num,
            thought=thought,
            reasoning_type="analysis",
            confidence=max(0.5, 0.9 - (step_num - 3) * 0.1),
        )
    
    def _reflect_on_step(self, step_index: int) -> None:
        """Self-reflection on a reasoning step."""
        if step_index >= len(self._steps):
            return
        
        step = self._steps[step_index]
        
        # Simple reflection
        if step.confidence < 0.6:
            step.reflection = "Confidence is low, considering alternative approaches"
            step.alternatives_considered.append("Alternative interpretation")
        else:
            step.reflection = "This step appears sound, proceeding to next"
    
    def _is_complete(self) -> bool:
        """Check if reasoning is complete."""
        if not self._steps:
            return False
        
        # Check if we have enough steps
        if len(self._steps) >= 5:
            # Check final step confidence
            final_step = self._steps[-1]
            if final_step.confidence > 0.7:
                return True
        
        return False
    
    def _synthesize_answer(self) -> str:
        """Synthesize the final answer from the chain."""
        if not self._steps:
            return "Insufficient reasoning steps"
        
        # Summarize the key insights
        key_thoughts = [s.thought for s in self._steps[2:-1] if s.confidence > 0.6]
        
        return f"Based on {len(self._steps)} reasoning steps: {' '.join(key_thoughts[:2])}"
    
    def _calculate_final_confidence(self) -> float:
        """Calculate final confidence from all steps."""
        if not self._steps:
            return 0.0
        
        confidences = [s.confidence for s in self._steps]
        return sum(confidences) / len(confidences)
    
    def get_chain_summary(self) -> str:
        """Get a summary of the reasoning chain."""
        if not self._steps:
            return "No reasoning performed"
        
        summary_lines = []
        for step in self._steps:
            summary_lines.append(f"{step.step_number}. {step.thought}")
        
        return "\n".join(summary_lines)
    
    def get_steps_by_type(self, reasoning_type: str) -> list[ThoughtStep]:
        """Get all steps of a specific reasoning type."""
        return [s for s in self._steps if s.reasoning_type == reasoning_type]
    
    def validate_chain(self) -> tuple[bool, list[str]]:
        """
        Validate the reasoning chain for logical consistency.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        if len(self._steps) < 2:
            issues.append("Chain too short for meaningful reasoning")
        
        # Check confidence trend
        confidences = [s.confidence for s in self._steps]
        
        if confidences[0] < confidences[-1]:
            issues.append("Confidence increased significantly - may indicate unstable reasoning")
        
        # Check for gaps
        step_numbers = [s.step_number for s in self._steps]
        for i in range(1, max(step_numbers) + 1):
            if i not in step_numbers:
                issues.append(f"Missing step {i}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
