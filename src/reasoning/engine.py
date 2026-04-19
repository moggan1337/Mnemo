"""
Reasoning Engine for Mnemo.

Provides logical inference capabilities including
deductive, inductive, and abductive reasoning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning supported."""
    DEDUCTIVE = "deductive"   # General to specific
    INDUCTIVE = "inductive"   # Specific to general
    ABDUCTIVE = "abductive"   # Best explanation
    ANALOGICAL = "analogical"  # Similarity-based
    CAUSAL = "causal"         # Cause and effect


@dataclass
class Inference:
    """A single inference step."""
    premise: str
    conclusion: str
    rule: str  # The inference rule used
    confidence: float  # 0 to 1
    chain_id: str  # Groups related inferences
    
    def to_dict(self) -> dict:
        return {
            "premise": self.premise,
            "conclusion": self.conclusion,
            "rule": self.rule,
            "confidence": self.confidence,
            "chain_id": self.chain_id,
        }


@dataclass
class ReasoningResult:
    """Result of a reasoning operation."""
    reasoning_type: ReasoningType
    question: str
    answer: str
    confidence: float
    inferences: list[Inference]
    chain_of_thought: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    alternative_answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "reasoning_type": self.reasoning_type.value,
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "inferences": [i.to_dict() for i in self.inferences],
            "chain_of_thought": self.chain_of_thought,
            "supporting_evidence": self.supporting_evidence,
            "alternative_answers": self.alternative_answers,
            "metadata": self.metadata,
        }


class ReasoningEngine:
    """
    Reasoning Engine for Mnemo.
    
    Provides multiple reasoning modes:
    - Deductive: Deriving specific conclusions from general rules
    - Inductive: Drawing general conclusions from specific observations
    - Abductive: Finding the best explanation for observations
    - Analogical: Reasoning based on similarities
    - Causal: Understanding cause and effect
    
    Usage:
        engine = ReasoningEngine()
        result = await engine.reason(
            question="What will happen?",
            context=knowledge_base,
            reasoning_type=ReasoningType.DEDUCTIVE
        )
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        confidence_threshold: float = 0.6,
        enable_chain_of_thought: bool = True,
    ):
        """
        Initialize Reasoning Engine.
        
        Args:
            max_depth: Maximum reasoning depth
            confidence_threshold: Minimum confidence to accept
            enable_chain_of_thought: Enable step-by-step reasoning
        """
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold
        self.enable_chain_of_thought = enable_chain_of_thought
        
        # Reasoning chains
        self._active_chains: dict[str, list[Inference]] = {}
        
        logger.info("ReasoningEngine initialized")
    
    async def reason(
        self,
        question: str,
        context: Any,
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
    ) -> ReasoningResult:
        """
        Perform reasoning on a question.
        
        Args:
            question: The question to reason about
            context: Relevant knowledge/context
            reasoning_type: Type of reasoning to use
            
        Returns:
            ReasoningResult with answer and reasoning trace
        """
        chain_id = self._generate_chain_id()
        
        if reasoning_type == ReasoningType.DEDUCTIVE:
            return await self._deductive_reason(question, context, chain_id)
        elif reasoning_type == ReasoningType.INDUCTIVE:
            return await self._inductive_reason(question, context, chain_id)
        elif reasoning_type == ReasoningType.ABDUCTIVE:
            return await self._abductive_reason(question, context, chain_id)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            return await self._analogical_reason(question, context, chain_id)
        elif reasoning_type == ReasoningType.CAUSAL:
            return await self._causal_reason(question, context, chain_id)
        else:
            return await self._deductive_reason(question, context, chain_id)
    
    async def _deductive_reason(
        self,
        question: str,
        context: Any,
        chain_id: str,
    ) -> ReasoningResult:
        """Deductive reasoning: if A implies B, and A is true, then B is true."""
        chain = []
        chain_of_thought = [
            f"Starting deductive reasoning for: {question}",
        ]
        
        # Extract premises from context
        premises = self._extract_premises(context)
        
        if not premises:
            return ReasoningResult(
                reasoning_type=ReasoningType.DEDUCTIVE,
                question=question,
                answer="Insufficient information for deduction.",
                confidence=0.2,
                inferences=[],
                chain_of_thought=chain_of_thought,
            )
        
        # Apply modus ponens
        for premise in premises[:5]:
            inference = Inference(
                premise=premise,
                conclusion=f"Therefore: {question}",
                rule="modus_ponens",
                confidence=0.7,
                chain_id=chain_id,
            )
            chain.append(inference)
            chain_of_thought.append(f"Based on premise: {premise}")
        
        answer = self._generate_deductive_answer(question, premises)
        confidence = self._calculate_deductive_confidence(chain)
        
        return ReasoningResult(
            reasoning_type=ReasoningType.DEDUCTIVE,
            question=question,
            answer=answer,
            confidence=confidence,
            inferences=chain,
            chain_of_thought=chain_of_thought,
            supporting_evidence=premises,
        )
    
    async def _inductive_reason(
        self,
        question: str,
        context: Any,
        chain_id: str,
    ) -> ReasoningResult:
        """Inductive reasoning: generalizing from specific cases."""
        chain = []
        chain_of_thought = [
            f"Starting inductive reasoning for: {question}",
        ]
        
        # Extract observations
        observations = self._extract_observations(context)
        
        if len(observations) < 2:
            return ReasoningResult(
                reasoning_type=ReasoningType.INDUCTIVE,
                question=question,
                answer="Need more observations for induction.",
                confidence=0.3,
                inferences=[],
            )
        
        # Find patterns
        pattern = self._find_pattern(observations)
        
        inference = Inference(
            premise=f"Observations: {observations}",
            conclusion=f"Pattern: {pattern}",
            rule="pattern_recognition",
            confidence=0.6,
            chain_id=chain_id,
        )
        chain.append(inference)
        
        # Generate generalization
        generalization = f"Based on {len(observations)} observations, {pattern}"
        
        return ReasoningResult(
            reasoning_type=ReasoningType.INDUCTIVE,
            question=question,
            answer=generalization,
            confidence=min(0.9, len(observations) * 0.15 + 0.4),
            inferences=chain,
            chain_of_thought=chain_of_thought + [pattern],
            supporting_evidence=observations,
        )
    
    async def _abductive_reason(
        self,
        question: str,
        context: Any,
        chain_id: str,
    ) -> ReasoningResult:
        """Abductive reasoning: finding the best explanation."""
        chain = []
        chain_of_thought = [
            f"Starting abductive reasoning (inference to best explanation) for: {question}",
        ]
        
        # Extract observations
        observations = self._extract_observations(context)
        
        # Generate possible explanations
        explanations = []
        for i, obs in enumerate(observations[:5]):
            explanation = f"Observation {i+1} could be explained by: {self._generate_explanation(obs)}"
            explanations.append(explanation)
            chain_of_thought.append(explanation)
        
        # Select best explanation
        best_explanation = explanations[0] if explanations else "No clear explanation found."
        
        return ReasoningResult(
            reasoning_type=ReasoningType.ABDUCTIVE,
            question=question,
            answer=f"Best explanation: {best_explanation}",
            confidence=0.5,
            inferences=chain,
            chain_of_thought=chain_of_thought,
            alternative_answers=explanations[1:],
        )
    
    async def _analogical_reason(
        self,
        question: str,
        context: Any,
        chain_id: str,
    ) -> ReasoningResult:
        """Analogical reasoning: using similarities to draw conclusions."""
        chain =_thought = [f"Starting analogical reasoning for: {question}"]
        
        # Find analogous cases
        analogies = self._find_analogies(question, context)
        
        if analogies:
            best_analogy = analogies[0]
            answer = f"Similar to {best_analogy['source']}: {best_analogy['conclusion']}"
            confidence = best_analogy.get("confidence", 0.6)
        else:
            answer = "No suitable analogy found."
            confidence = 0.3
        
        return ReasoningResult(
            reasoning_type=ReasoningType.ANALOGICAL,
            question=question,
            answer=answer,
            confidence=confidence,
            inferences=[],
            chain_of_thought=chain_of_thought,
            supporting_evidence=[a["source"] for a in analogies] if analogies else [],
        )
    
    async def _causal_reason(
        self,
        question: str,
        context: Any,
        chain_id: str,
    ) -> ReasoningResult:
        """Causal reasoning: understanding cause and effect."""
        chain_of_thought = [f"Starting causal reasoning for: {question}"]
        
        # Extract causal relationships
        causes = self._extract_causes(context)
        effects = self._extract_effects(context)
        
        if causes and effects:
            causal_chain = self._build_causal_chain(causes, effects)
            answer = f"Based on causal analysis: {causal_chain}"
            confidence = 0.7
        else:
            answer = "Causal relationships unclear from available information."
            confidence = 0.3
        
        return ReasoningResult(
            reasoning_type=ReasoningType.CAUSAL,
            question=question,
            answer=answer,
            confidence=confidence,
            inferences=[],
            chain_of_thought=chain_of_thought,
        )
    
    # ==================== Helper Methods ====================
    
    def _extract_premises(self, context: Any) -> list[str]:
        """Extract premises from context."""
        if isinstance(context, list):
            return [str(c) for c in context]
        elif isinstance(context, str):
            # Simple sentence splitting
            return [s.strip() for s in context.split('.') if s.strip()]
        return [str(context)]
    
    def _extract_observations(self, context: Any) -> list[str]:
        """Extract observations from context."""
        return self._extract_premises(context)
    
    def _generate_explanation(self, observation: str) -> str:
        """Generate a potential explanation for an observation."""
        return f"the underlying mechanism related to '{observation[:30]}...'"
    
    def _find_pattern(self, observations: list[str]) -> str:
        """Find patterns in observations."""
        if len(observations) < 2:
            return "insufficient data"
        
        # Simple pattern: common words
        all_words = []
        for obs in observations:
            all_words.extend(obs.lower().split())
        
        word_counts: dict[str, int] = {}
        for word in all_words:
            if len(word) > 4:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            most_common = max(word_counts.items(), key=lambda x: x[1])
            if most_common[1] >= 2:
                return f"'{most_common[0]}' appears repeatedly"
        
        return "various factors contribute"
    
    def _generate_deductive_answer(
        self,
        question: str,
        premises: list[str],
    ) -> str:
        """Generate a deductive answer."""
        if premises:
            return f"Based on the premises, regarding '{question}': {premises[0]}"
        return f"Cannot determine answer to '{question}' from available information."
    
    def _calculate_deductive_confidence(self, chain: list[Inference]) -> float:
        """Calculate confidence for deductive reasoning."""
        if not chain:
            return 0.0
        
        avg_confidence = sum(i.confidence for i in chain) / len(chain)
        return min(1.0, avg_confidence * 1.2)  # Boost for valid chain
    
    def _find_analogies(self, question: str, context: Any) -> list[dict]:
        """Find analogous cases."""
        # Simple implementation - would use embeddings in production
        return []
    
    def _extract_causes(self, context: Any) -> list[str]:
        """Extract causes from context."""
        causes = []
        
        if isinstance(context, str):
            # Look for causal indicators
            import re
            cause_patterns = [
                r'(\w+)\s+causes\s+(\w+)',
                r'(\w+)\s+leads\s+to\s+(\w+)',
                r'(\w+)\s+results\s+in\s+(\w+)',
            ]
            
            for pattern in cause_patterns:
                matches = re.findall(pattern, context.lower())
                for match in matches:
                    causes.append(f"{match[0]} -> {match[1]}")
        
        return causes
    
    def _extract_effects(self, context: Any) -> list[str]:
        """Extract effects from context."""
        # Similar to causes but looking at the outcome
        return self._extract_causes(context)
    
    def _build_causal_chain(
        self,
        causes: list[str],
        effects: list[str],
    ) -> str:
        """Build a causal chain description."""
        if causes and effects:
            return f"{causes[0]}, which results in {effects[0]}"
        return "Causal chain unclear"
    
    def _generate_chain_id(self) -> str:
        """Generate a unique chain ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    @property
    def statistics(self) -> dict:
        """Get reasoning engine statistics."""
        return {
            "active_chains": len(self._active_chains),
            "max_depth": self.max_depth,
            "confidence_threshold": self.confidence_threshold,
        }
