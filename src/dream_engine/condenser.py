"""
Dream Condenser - Memory Compression and Abstraction.

Condenses memories into abstract representations while
preserving key semantic content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Abstraction:
    """
    An abstract representation of condensed memory content.
    
    Represents the core semantic content of one or more memories
    in a compressed form.
    """
    id: str
    summary: str  # Short summary
    key_points: list[str]  # Core insights
    entities: list[str]  # Named entities
    concepts: list[str]  # Key concepts
    sentiment_score: float = 0.0  # -1 to 1
    importance: float = 0.5  # 0 to 1
    source_memory_ids: list[str] = field(default_factory=list)
    abstraction_level: int = 1  # Higher = more abstract
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "summary": self.summary,
            "key_points": self.key_points,
            "entities": self.entities,
            "concepts": self.concepts,
            "sentiment_score": self.sentiment_score,
            "importance": self.importance,
            "source_memory_ids": self.source_memory_ids,
            "abstraction_level": self.abstraction_level,
        }


class DreamCondenser:
    """
    Condenses memories into abstract representations.
    
    The condensation process:
    1. Extract key entities and concepts
    2. Identify core semantic content
    3. Generate concise summaries
    4. Assign importance scores
    5. Create abstraction hierarchy
    
    This mimics how humans consolidate memories during sleep,
    distilling experiences into their essential meaning.
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.3,
        max_abstraction_level: int = 3,
        importance_threshold: float = 0.3,
    ):
        """
        Initialize the condenser.
        
        Args:
            compression_ratio: Target compression (0.3 = compress to 30%)
            max_abstraction_level: Maximum abstraction depth
            importance_threshold: Minimum importance to preserve
        """
        self.compression_ratio = compression_ratio
        self.max_abstraction_level = max_abstraction_level
        self.importance_threshold = importance_threshold
        
        logger.info(f"DreamCondenser initialized (ratio={compression_ratio})")
    
    async def condense(
        self,
        memories: list,
        target_count: Optional[int] = None,
    ) -> list[Abstraction]:
        """
        Condense memories into abstract representations.
        
        Args:
            memories: List of Memory objects to condense
            target_count: Target number of abstractions
            
        Returns:
            List of Abstraction objects
        """
        if not memories:
            return []
        
        # Calculate target count based on compression ratio
        if target_count is None:
            target_count = max(
                1,
                int(len(memories) * self.compression_ratio)
            )
        
        abstractions = []
        
        # Group similar memories
        groups = self._group_memories(memories)
        
        # Process each group
        for group in groups:
            abstraction = await self._condense_group(group)
            if abstraction.importance >= self.importance_threshold:
                abstractions.append(abstraction)
        
        # Sort by importance and limit
        abstractions.sort(key=lambda a: a.importance, reverse=True)
        
        # Hierarchical abstraction if too many
        if len(abstractions) > target_count:
            abstractions = await self._create_hierarchy(
                abstractions,
                target_count
            )
        
        logger.info(
            f"Condensed {len(memories)} memories into {len(abstractions)} abstractions"
        )
        
        return abstractions
    
    async def _condense_group(self, memory_group: list) -> Abstraction:
        """Condense a group of related memories into one abstraction."""
        import uuid
        
        # Combine content from all memories in group
        combined_content = []
        all_entities = []
        all_concepts = []
        
        for memory in memory_group:
            content = str(memory.content)
            combined_content.append(content)
            
            # Extract entities and concepts
            entities = self._extract_entities(content)
            concepts = self._extract_concepts(content)
            
            all_entities.extend(entities)
            all_concepts.extend(concepts)
        
        # Generate summary
        summary = self._generate_summary(combined_content)
        
        # Extract key points
        key_points = self._extract_key_points(combined_content)
        
        # Deduplicate entities and concepts
        unique_entities = list(set(all_entities))[:20]
        unique_concepts = list(set(all_concepts))[:15]
        
        # Calculate importance
        importance = self._calculate_importance(
            summary, key_points, len(memory_group)
        )
        
        # Create abstraction
        abstraction = Abstraction(
            id=str(uuid.uuid4())[:12],
            summary=summary,
            key_points=key_points,
            entities=unique_entities,
            concepts=unique_concepts,
            importance=importance,
            source_memory_ids=[m.id for m in memory_group],
        )
        
        return abstraction
    
    def _group_memories(
        self,
        memories: list,
    ) -> list[list]:
        """Group similar memories together."""
        if len(memories) <= 3:
            return [[m] for m in memories]
        
        groups = []
        used = set()
        
        for i, memory in enumerate(memories):
            if i in used:
                continue
            
            group = [memory]
            used.add(i)
            
            content_i = str(memory.content).lower()
            
            # Find similar memories
            for j, other in enumerate(memories[i+1:], i+1):
                if j in used:
                    continue
                
                content_j = str(other.content).lower()
                
                # Simple similarity check
                if self._calculate_similarity(content_i, content_j) > 0.5:
                    group.append(other)
                    used.add(j)
                    
                    if len(group) >= 5:  # Limit group size
                        break
            
            groups.append(group)
        
        return groups
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        # Word-based Jaccard similarity
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        entities = []
        
        # Simple capitalized phrase extraction
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.extend(capitalized)
        
        # Common entity patterns
        patterns = [
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',  # Titles
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',  # CamelCase
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # Deduplicate and return
        return list(set(entities))[:20]
    
    def _extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text."""
        concepts = []
        
        # Technical terms ( CamelCase, hyphenated)
        tech_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b', text)
        concepts.extend(tech_terms)
        
        # Common concept patterns
        concept_words = {
            'algorithm', 'model', 'system', 'process', 'method',
            'approach', 'technique', 'framework', 'architecture',
            'network', 'learning', 'training', 'data', 'feature',
            'prediction', 'classification', 'optimization', 'analysis',
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in concept_words:
                concepts.append(word)
        
        return list(set(concepts))[:15]
    
    def _generate_summary(self, contents: list[str]) -> str:
        """Generate a summary from content list."""
        if not contents:
            return ""
        
        # Take first few items and combine
        sample = contents[:3]
        combined = " ".join(sample)
        
        # Truncate to reasonable length
        if len(combined) > 200:
            # Try to end at sentence boundary
            truncated = combined[:200]
            last_period = truncated.rfind('.')
            if last_period > 100:
                truncated = truncated[:last_period + 1]
            else:
                truncated += "..."
            return truncated
        
        return combined
    
    def _extract_key_points(self, contents: list[str]) -> list[str]:
        """Extract key points from content."""
        key_points = []
        
        # Extract sentences that might be important
        for content in contents[:5]:  # Limit processing
            # Split into sentences
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Keep sentences that:
                # - Are not too short
                # - Contain important words
                # - Are unique
                
                if len(sentence) < 20:
                    continue
                
                important_words = {
                    'important', 'key', 'significant', 'main',
                    'primary', 'essential', 'critical', 'fundamental',
                    'discovered', 'found', 'showed', 'demonstrated',
                }
                
                words = sentence.lower().split()
                
                if any(word in important_words for word in words):
                    if sentence not in key_points:
                        key_points.append(sentence)
        
        # Limit to top points
        return key_points[:5]
    
    def _calculate_importance(
        self,
        summary: str,
        key_points: list[str],
        memory_count: int,
    ) -> float:
        """Calculate importance score for an abstraction."""
        importance = 0.5
        
        # More source memories = higher importance (up to a point)
        if memory_count >= 5:
            importance += 0.15
        elif memory_count >= 3:
            importance += 0.1
        elif memory_count >= 2:
            importance += 0.05
        
        # More key points = higher importance
        if len(key_points) >= 4:
            importance += 0.15
        elif len(key_points) >= 2:
            importance += 0.1
        elif len(key_points) >= 1:
            importance += 0.05
        
        # Longer summary = more content preserved
        if len(summary) > 100:
            importance += 0.1
        
        return min(1.0, importance)
    
    async def _create_hierarchy(
        self,
        abstractions: list[Abstraction],
        target_count: int,
    ) -> list[Abstraction]:
        """Create a hierarchy of abstractions at different levels."""
        if len(abstractions) <= target_count:
            return abstractions
        
        # Keep the most important ones
        result = abstractions[:target_count]
        
        # Mark which were collapsed
        collapsed_ids = [a.id for a in abstractions[target_count:]]
        
        # Add metadata about what was collapsed
        for ab in result:
            ab.source_memory_ids.extend(collapsed_ids)
        
        return result
    
    def reabstract(
        self,
        abstraction: Abstraction,
        level: int,
    ) -> Abstraction:
        """
        Create a higher-level abstraction from an existing one.
        
        Args:
            abstraction: Source abstraction
            level: Target abstraction level (higher = more abstract)
            
        Returns:
            New, more abstract Abstraction
        """
        import uuid
        
        # Generate more concise summary
        new_summary = self._abstract_summary(abstraction.summary, level)
        
        # Select fewer key points
        num_points = max(1, len(abstraction.key_points) // level)
        new_key_points = abstraction.key_points[:num_points]
        
        # Keep subset of entities/concepts
        num_entities = max(3, len(abstraction.entities) // level)
        num_concepts = max(2, len(abstraction.concepts) // level)
        
        return Abstraction(
            id=str(uuid.uuid4())[:12],
            summary=new_summary,
            key_points=new_key_points,
            entities=abstraction.entities[:num_entities],
            concepts=abstraction.concepts[:num_concepts],
            importance=abstraction.importance,
            source_memory_ids=abstraction.source_memory_ids,
            abstraction_level=level,
        )
    
    def _abstract_summary(self, summary: str, level: int) -> str:
        """Create a more abstract summary."""
        # Progressive truncation
        lengths = {1: 150, 2: 100, 3: 50}
        target_len = lengths.get(level, 50)
        
        if len(summary) <= target_len:
            return summary
        
        # Truncate and try to end at word boundary
        truncated = summary[:target_len]
        last_space = truncated.rfind(' ')
        
        if last_space > target_len * 0.7:
            truncated = truncated[:last_space]
        
        return truncated + "..."
