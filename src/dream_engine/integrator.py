"""
Memory Integrator - Links abstractions with existing knowledge.

Discovers connections between new abstractions and
existing knowledge structures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from mnemo.dream_engine.condenser import Abstraction

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """
    A discovered connection between abstractions or entities.
    
    Represents a meaningful relationship found during
    the integration phase of dreaming.
    """
    id: str
    source_id: str  # Entity or abstraction ID
    target_id: str  # Entity or abstraction ID
    connection_type: str  # type of connection found
    strength: float  # 0 to 1
    description: str = ""
    bidirectional: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "connection_type": self.connection_type,
            "strength": self.strength,
            "description": self.description,
            "bidirectional": self.bidirectional,
            "metadata": self.metadata,
        }


class MemoryIntegrator:
    """
    Integrates new abstractions with existing knowledge.
    
    The integration phase:
    1. Compare abstractions with existing knowledge
    2. Find semantic similarities
    3. Discover unexpected connections
    4. Strengthen related concepts
    5. Build the knowledge network
    
    This process helps create a coherent understanding
    by linking new information with what already exists.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.6,
        max_connections_per_abstraction: int = 10,
    ):
        """
        Initialize the integrator.
        
        Args:
            similarity_threshold: Minimum similarity for connection
            max_connections_per_abstraction: Maximum connections to create
        """
        self.similarity_threshold = similarity_threshold
        self.max_connections_per_abstraction = max_connections_per_abstraction
        
        # Existing abstractions (would be loaded from knowledge graph)
        self._existing_abstractions: list[Abstraction] = []
        
        logger.info("MemoryIntegrator initialized")
    
    async def integrate(
        self,
        abstractions: list[Abstraction],
        existing_knowledge: Optional[list[Abstraction]] = None,
    ) -> list[Connection]:
        """
        Integrate abstractions by discovering connections.
        
        Args:
            abstractions: New abstractions to integrate
            existing_knowledge: Existing abstractions to connect to
            
        Returns:
            List of discovered connections
        """
        if existing_knowledge:
            self._existing_abstractions = existing_knowledge
        
        connections = []
        
        # Find connections within new abstractions
        internal = await self._find_internal_connections(abstractions)
        connections.extend(internal)
        
        # Find connections to existing knowledge
        if self._existing_abstractions:
            external = await self._find_external_connections(
                abstractions,
                self._existing_abstractions
            )
            connections.extend(external)
        
        # Filter and rank connections
        connections = self._filter_connections(connections)
        
        logger.info(
            f"Integration complete: {len(connections)} connections discovered"
        )
        
        return connections
    
    async def _find_internal_connections(
        self,
        abstractions: list[Abstraction],
    ) -> list[Connection]:
        """Find connections between abstractions in the same batch."""
        connections = []
        
        for i, abs1 in enumerate(abstractions):
            for abs2 in abstractions[i+1:]:
                connection = self._compare_abstractions(abs1, abs2)
                if connection and connection.strength >= self.similarity_threshold:
                    connections.append(connection)
        
        return connections
    
    async def _find_external_connections(
        self,
        new_abstractions: list[Abstraction],
        existing_abstractions: list[Abstraction],
    ) -> list[Connection]:
        """Find connections between new and existing abstractions."""
        connections = []
        
        for new_abs in new_abstractions:
            for existing_abs in existing_abstractions:
                connection = self._compare_abstractions(new_abs, existing_abs)
                if connection and connection.strength >= self.similarity_threshold:
                    connections.append(connection)
        
        return connections
    
    def _compare_abstractions(
        self,
        abs1: Abstraction,
        abs2: Abstraction,
    ) -> Optional[Connection]:
        """Compare two abstractions and return a connection if found."""
        import uuid
        
        # Calculate various similarity scores
        entity_sim = self._set_similarity(abs1.entities, abs2.entities)
        concept_sim = self._set_similarity(abs1.concepts, abs2.concepts)
        semantic_sim = self._semantic_similarity(
            abs1.summary, abs2.summary
        )
        
        # Weighted combination
        overall_similarity = (
            entity_sim * 0.3 +
            concept_sim * 0.4 +
            semantic_sim * 0.3
        )
        
        if overall_similarity < self.similarity_threshold:
            return None
        
        # Determine connection type
        connection_type = self._determine_connection_type(
            entity_sim, concept_sim, semantic_sim
        )
        
        # Generate description
        description = self._generate_connection_description(
            abs1, abs2, connection_type, overall_similarity
        )
        
        return Connection(
            id=str(uuid.uuid4())[:16],
            source_id=abs1.id,
            target_id=abs2.id,
            connection_type=connection_type,
            strength=overall_similarity,
            description=description,
            bidirectional=(connection_type in ["similar_to", "related_to"]),
            metadata={
                "entity_similarity": entity_sim,
                "concept_similarity": concept_sim,
                "semantic_similarity": semantic_sim,
            }
        )
    
    def _set_similarity(self, set1: list, set2: list) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        
        s1, s2 = set(set1), set(set2)
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        
        return intersection / union if union > 0 else 0.0
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        # Simple word overlap for now
        # In production, would use embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        }
        
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_connection_type(
        self,
        entity_sim: float,
        concept_sim: float,
        semantic_sim: float,
    ) -> str:
        """Determine the type of connection."""
        if entity_sim > 0.5:
            return "shares_entities"
        elif concept_sim > 0.5:
            return "related_concepts"
        elif semantic_sim > 0.5:
            return "similar_context"
        elif concept_sim > 0.3:
            return "partially_related"
        else:
            return "weakly_connected"
    
    def _generate_connection_description(
        self,
        abs1: Abstraction,
        abs2: Abstraction,
        connection_type: str,
        strength: float,
    ) -> str:
        """Generate a human-readable description of the connection."""
        descriptions = {
            "shares_entities": f"Both discuss: {', '.join(abs1.entities[:3])}",
            "related_concepts": f"Both involve: {', '.join(abs1.concepts[:3])}",
            "similar_context": "Appear in similar contexts",
            "partially_related": "Share some conceptual overlap",
            "weakly_connected": "Distantly related",
        }
        
        base = descriptions.get(connection_type, "Connected")
        
        # Add strength indicator
        if strength > 0.8:
            strength_desc = "Strongly"
        elif strength > 0.6:
            strength_desc = "Moderately"
        else:
            strength_desc = "Slightly"
        
        return f"{strength_desc} {base}"
    
    def _filter_connections(
        self,
        connections: list[Connection],
    ) -> list[Connection]:
        """Filter and rank connections."""
        # Remove duplicates (same source-target pair)
        seen_pairs = set()
        unique = []
        
        for conn in connections:
            pair = tuple(sorted([conn.source_id, conn.target_id]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique.append(conn)
        
        # Limit per abstraction
        by_source: dict[str, list[Connection]] = {}
        for conn in unique:
            if conn.source_id not in by_source:
                by_source[conn.source_id] = []
            
            if len(by_source[conn.source_id]) < self.max_connections_per_abstraction:
                by_source[conn.source_id].append(conn)
        
        # Flatten and sort by strength
        result = []
        for conns in by_source.values():
            result.extend(conns)
        
        result.sort(key=lambda c: c.strength, reverse=True)
        
        return result
    
    def strengthen_connection(
        self,
        connection: Connection,
        factor: float = 0.1,
    ) -> Connection:
        """Strengthen an existing connection."""
        connection.strength = min(1.0, connection.strength + factor)
        return connection
    
    def weaken_connection(
        self,
        connection: Connection,
        factor: float = 0.1,
    ) -> Connection:
        """Weaken an existing connection."""
        connection.strength = max(0.0, connection.strength - factor)
        
        # Remove if below threshold
        if connection.strength < self.similarity_threshold * 0.5:
            return None
        
        return connection
