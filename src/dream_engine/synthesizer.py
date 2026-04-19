"""
Insight Synthesizer - Generates novel insights.

Creates new insights by combining abstractions and
connections in creative ways.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from mnemo.dream_engine.condenser import Abstraction
from mnemo.dream_engine.integrator import Connection

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """
    A generated insight from the synthesis process.
    
    Insights are novel observations or conclusions generated
    by combining existing knowledge in new ways.
    """
    id: str
    content: str  # The insight text
    insight_type: str  # discovery, hypothesis, pattern, analogy
    confidence: float  # 0 to 1
    novelty: float  # 0 to 1, how novel this insight is
    supporting_evidence: list[str] = field(default_factory=list)  # Source IDs
    generated_from: list[str] = field(default_factory=list)  # Abstraction/Connection IDs
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def quality(self) -> float:
        """Overall quality score (confidence * novelty)."""
        return self.confidence * self.newness
    
    @property
    def newness(self) -> float:
        """Alias for novelty."""
        return self.novelty
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "insight_type": self.insight_type,
            "confidence": self.confidence,
            "novelty": self.novelty,
            "supporting_evidence": self.supporting_evidence,
            "generated_from": self.generated_from,
            "metadata": self.metadata,
        }


class InsightSynthesizer:
    """
    Generates novel insights from abstractions and connections.
    
    The synthesis process:
    1. Identify patterns across abstractions
    2. Find analogies between different domains
    3. Generate hypotheses from connections
    4. Create novel combinations
    5. Evaluate and rank insights
    
    This mimics the creative problem-solving that occurs
    during human dreaming, combining disparate ideas
    into new understanding.
    """
    
    def __init__(
        self,
        novelty_threshold: float = 0.6,
        max_insights: int = 10,
        creativity_factor: float = 0.2,
    ):
        """
        Initialize the synthesizer.
        
        Args:
            novelty_threshold: Minimum novelty for an insight
            max_insights: Maximum insights to generate
            creativity_factor: How creative/random the synthesis is (0-1)
        """
        self.novelty_threshold = novelty_threshold
        self.max_insights = max_insights
        self.creativity_factor = creativity_factor
        
        # Track generated insights to avoid repetition
        self._generated_content: set[str] = set()
        
        logger.info("InsightSynthesizer initialized")
    
    async def synthesize(
        self,
        abstractions: list[Abstraction],
        connections: list[Connection],
    ) -> list[Insight]:
        """
        Synthesize insights from abstractions and connections.
        
        Args:
            abstractions: Condensed abstractions
            connections: Discovered connections
            
        Returns:
            List of generated insights
        """
        insights = []
        
        # Generate different types of insights
        pattern_insights = await self._find_patterns(abstractions, connections)
        insights.extend(pattern_insights)
        
        analogy_insights = await self._find_analogies(abstractions)
        insights.extend(analogy_insights)
        
        hypothesis_insights = await self._generate_hypotheses(
            abstractions, connections
        )
        insights.extend(hypothesis_insights)
        
        # Filter by novelty and confidence
        filtered = self._filter_insights(insights)
        
        # Sort by quality and limit
        filtered.sort(key=lambda i: i.quality, reverse=True)
        result = filtered[:self.max_insights]
        
        # Update generated content tracker
        for insight in result:
            self._generated_content.add(insight.content.lower())
        
        logger.info(f"Synthesis complete: {len(result)} insights generated")
        
        return result
    
    async def _find_patterns(
        self,
        abstractions: list[Abstraction],
        connections: list[Connection],
    ) -> list[Insight]:
        """Find recurring patterns across abstractions."""
        insights = []
        
        if len(abstractions) < 3:
            return insights
        
        # Find common entities across abstractions
        all_entities: dict[str, int] = {}
        for absraction in abstractions:
            for entity in absraction.entities:
                all_entities[entity] = all_entities.get(entity, 0) + 1
        
        # Find entities that appear in multiple abstractions
        recurring_entities = {
            entity: count
            for entity, count in all_entities.items()
            if count >= 2
        }
        
        for entity, count in recurring_entities.items():
            insight = Insight(
                id=self._generate_id(),
                content=f"The concept '{entity}' appears across {count} different contexts, suggesting a fundamental relationship.",
                insight_type="pattern",
                confidence=min(1.0, count / len(abstractions)),
                novelty=0.7,
                supporting_evidence=[entity],
            )
            insights.append(insight)
        
        # Find common concepts
        all_concepts: dict[str, int] = {}
        for absraction in abstractions:
            for concept in absraction.concepts:
                all_concepts[concept] = all_concepts.get(concept, 0) + 1
        
        recurring_concepts = {
            concept: count
            for concept, count in all_concepts.items()
            if count >= 2
        }
        
        for concept, count in recurring_concepts.items():
            if count >= 2:
                insight = Insight(
                    id=self._generate_id(),
                    content=f"The concept of '{concept}' is mentioned {count} times, indicating a central theme.",
                    insight_type="pattern",
                    confidence=min(1.0, count / len(abstractions)),
                    novelty=0.6,
                    supporting_evidence=[concept],
                )
                insights.append(insight)
        
        return insights
    
    async def _find_analogies(
        self,
        abstractions: list[Abstraction],
    ) -> list[Insight]:
        """Find analogies between different abstraction groups."""
        insights = []
        
        if len(abstractions) < 2:
            return insights
        
        # Look for abstractions from different source groups
        # that share similar concepts but different entities
        
        for i, abs1 in enumerate(abstractions):
            for abs2 in abstractions[i+1:]:
                # Check for shared concepts
                shared_concepts = set(abs1.concepts) & set(abs2.concepts)
                
                if shared_concepts and abs1.entities != abs2.entities:
                    # Found a potential analogy
                    concept = list(shared_concepts)[0]
                    
                    # Check for creative combination
                    if random.random() < self.creativity_factor + 0.3:
                        insight = Insight(
                            id=self._generate_id(),
                            content=f"The relationship between {abs1.entities[0] if abs1.entities else 'X'} and "
                                   f"{abs2.entities[0] if abs2.entities else 'Y'} may parallel each other "
                                   f"through the shared concept of '{concept}'.",
                            insight_type="analogy",
                            confidence=0.6,
                            novelty=0.7,
                            supporting_evidence=abs1.entities + abs2.entities,
                        )
                        insights.append(insight)
        
        return insights
    
    async def _generate_hypotheses(
        self,
        abstractions: list[Abstraction],
        connections: list[Connection],
    ) -> list[Insight]:
        """Generate hypotheses from abstraction patterns."""
        insights = []
        
        if not connections:
            return insights
        
        # Analyze connection patterns
        strong_connections = [c for c in connections if c.strength > 0.7]
        
        for conn in strong_connections[:3]:
            # Find associated abstractions
            source_abs = self._find_abstraction_by_id(conn.source_id, abstractions)
            target_abs = self._find_abstraction_by_id(conn.target_id, abstractions)
            
            if source_abs and target_abs:
                # Generate hypothesis
                if source_abs.concepts and target_abs.concepts:
                    hypothesis = Insight(
                        id=self._generate_id(),
                        content=f"If '{source_abs.concepts[0]}' leads to '{target_abs.concepts[0]}', "
                               f"then this pattern may apply to other {source_abs.entities[0] if source_abs.entities else 'similar'} contexts.",
                        insight_type="hypothesis",
                        confidence=conn.strength * 0.8,
                        novelty=0.6,
                        generated_from=[conn.id],
                    )
                    insights.append(hypothesis)
        
        # Generate "what if" questions from unexpected connections
        unexpected = [c for c in connections if c.strength < 0.5]
        
        for conn in unexpected[:2]:
            source_abs = self._find_abstraction_by_id(conn.source_id, abstractions)
            target_abs = self._find_abstraction_by_id(conn.target_id, abstractions)
            
            if source_abs and target_abs:
                insight = Insight(
                    id=self._generate_id(),
                    content=f"What if the unexpected connection between '{source_abs.summary[:50]}' "
                           f"and '{target_abs.summary[:50]}' reveals a hidden pattern?",
                    insight_type="hypothesis",
                    confidence=0.4,
                    novelty=0.9,  # Unexpected connections are novel
                    generated_from=[conn.id],
                )
                insights.append(insight)
        
        return insights
    
    def _find_abstraction_by_id(
        self,
        ab_id: str,
        abstractions: list[Abstraction],
    ) -> Optional[Abstraction]:
        """Find an abstraction by ID."""
        for absraction in abstractions:
            if absraction.id == ab_id:
                return absraction
        return None
    
    def _filter_insights(self, insights: list[Insight]) -> list[Insight]:
        """Filter insights based on quality criteria."""
        filtered = []
        
        for insight in insights:
            # Check novelty threshold
            if insight.novelty < self.novelty_threshold:
                continue
            
            # Check for duplicates
            content_lower = insight.content.lower()
            if content_lower in self._generated_content:
                continue
            
            # Check minimum confidence
            if insight.confidence < 0.3:
                continue
            
            filtered.append(insight)
        
        return filtered
    
    def _generate_id(self) -> str:
        """Generate a unique ID for an insight."""
        import uuid
        return str(uuid.uuid4())[:12]
    
    def evaluate_insight(
        self,
        insight: Insight,
        existing_insights: list[Insight],
    ) -> float:
        """
        Evaluate the novelty of an insight compared to existing ones.
        
        Returns a novelty score from 0 to 1.
        """
        if not existing_insights:
            return 1.0
        
        # Check for similar content
        max_similarity = 0.0
        insight_words = set(insight.content.lower().split())
        
        for existing in existing_insights:
            existing_words = set(existing.content.lower().split())
            
            if not insight_words or not existing_words:
                continue
            
            # Jaccard similarity
            intersection = len(insight_words & existing_words)
            union = len(insight_words | existing_words)
            similarity = intersection / union if union > 0 else 0.0
            
            max_similarity = max(max_similarity, similarity)
        
        # Novelty is inverse of max similarity
        return 1.0 - max_similarity
