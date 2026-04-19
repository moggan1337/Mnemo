"""
Query Engine for the Knowledge Graph.

Provides advanced querying capabilities including
inference, path finding, and subgraph extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from mnemo.knowledge_graph.graph import KnowledgeGraph, Entity, Relation, EntityType, RelationType

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries supported by the engine."""
    SIMPLE = "simple"           # Direct lookup
    TRAVERSAL = "traversal"     # Graph traversal
    INFERENCE = "inference"     # Inference-based
    AGGREGATION = "aggregation" # Aggregation queries
    PATTERN = "pattern"         # Pattern matching


@dataclass
class QueryResult:
    """Result of a query operation."""
    query_type: QueryType
    entities: list[Entity]
    relations: list[Relation]
    paths: list[list[Entity]]  # For traversal queries
    statistics: dict
    execution_time: float
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query_type": self.query_type.value,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "paths": [[e.to_dict() for e in path] for path in self.paths],
            "statistics": self.statistics,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
        }


class QueryEngine:
    """
    Query engine for the knowledge graph.
    
    Supports:
    - Simple lookups
    - Graph traversal (BFS, DFS)
    - Path finding
    - Inference (transitive closure)
    - Subgraph extraction
    - Pattern matching
    """
    
    def __init__(self, graph: KnowledgeGraph):
        """
        Initialize query engine.
        
        Args:
            graph: KnowledgeGraph to query
        """
        self.graph = graph
    
    # ==================== Simple Queries ====================
    
    def find_entity(
        self,
        name: Optional[str] = None,
        entity_id: Optional[str] = None,
        entity_type: Optional[EntityType] = None,
    ) -> Optional[Entity]:
        """Find a single entity by name or ID."""
        if entity_id:
            return self.graph.get_entity(entity_id)
        
        if name:
            return self.graph.get_entity_by_name(name, entity_type)
        
        return None
    
    def find_entities(
        self,
        query: str,
        entity_types: Optional[list[EntityType]] = None,
        limit: int = 20,
    ) -> list[Entity]:
        """Find entities matching a query."""
        types = [EntityType(t) for t in entity_types] if entity_types else None
        return self.graph.search(query, types, limit)
    
    def get_entity_relations(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> list[Relation]:
        """Get all relations for an entity."""
        return self.graph.get_relations_for_entity(entity_id, relation_type)
    
    # ==================== Traversal Queries ====================
    
    def traverse(
        self,
        start_id: str,
        direction: str = "both",  # outgoing, incoming, both
        max_depth: int = 2,
        relation_types: Optional[list[RelationType]] = None,
    ) -> list[tuple[Entity, Relation, str]]:
        """
        Traverse the graph from a starting entity.
        
        Args:
            start_id: Starting entity ID
            direction: Traversal direction
            max_depth: Maximum traversal depth
            relation_types: Filter by relation types
            
        Returns:
            List of (entity, relation, direction) tuples
        """
        import time
        start = time.time()
        
        results = []
        visited = {start_id}
        frontier = [(start_id, 0, "start")]
        
        while frontier:
            current_id, depth, direction_info = frontier.pop(0)
            
            if depth >= max_depth:
                continue
            
            relations = self.graph.get_relations_for_entity(current_id)
            
            for relation in relations:
                # Filter by type
                if relation_types and relation.relation_type not in relation_types:
                    continue
                
                # Determine target
                if relation.source_id == current_id:
                    neighbor_id = relation.target_id
                    rel_direction = "outgoing"
                else:
                    neighbor_id = relation.source_id
                    rel_direction = "incoming"
                
                # Apply direction filter
                if direction == "outgoing" and rel_direction == "incoming":
                    continue
                if direction == "incoming" and rel_direction == "outgoing":
                    continue
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    
                    if neighbor := self.graph.get_entity(neighbor_id):
                        results.append((neighbor, relation, rel_direction))
                        
                        if depth + 1 < max_depth:
                            frontier.append((neighbor_id, depth + 1, rel_direction))
        
        return results
    
    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5,
    ) -> Optional[list[Entity]]:
        """Find shortest path between two entities."""
        paths = self.graph.find_paths(source_id, target_id, max_length)
        
        if paths:
            return min(paths, key=len)
        
        return None
    
    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5,
    ) -> list[list[Entity]]:
        """Find all paths between two entities."""
        return self.graph.find_paths(source_id, target_id, max_length)
    
    # ==================== Inference Queries ====================
    
    def infer_related(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> list[tuple[Entity, float]]:
        """
        Infer related entities through transitive closure.
        
        Args:
            entity_id: Starting entity
            depth: Inference depth
            
        Returns:
            List of (entity, confidence) tuples
        """
        related = {}
        
        def dfs(current_id: str, current_depth: int, confidence: float):
            if current_depth > depth:
                return
            
            for neighbor, relation in self.graph.get_neighbors(current_id):
                if neighbor.id == entity_id:
                    continue
                
                # Calculate confidence based on path
                relation_confidence = relation.confidence * relation.weight
                new_confidence = confidence * relation_confidence
                
                # Update if higher confidence
                if neighbor.id not in related or related[neighbor.id] < new_confidence:
                    related[neighbor.id] = new_confidence
                    dfs(neighbor.id, current_depth + 1, new_confidence)
        
        # Start with direct neighbors
        for neighbor, relation in self.graph.get_neighbors(entity_id):
            confidence = relation.confidence * relation.weight
            related[neighbor.id] = confidence
            dfs(neighbor.id, 1, confidence)
        
        # Convert to entities
        results = []
        for eid, conf in sorted(related.items(), key=lambda x: x[1], reverse=True):
            if entity := self.graph.get_entity(eid):
                results.append((entity, conf))
        
        return results
    
    def infer_types(
        self,
        entity_id: str,
    ) -> list[EntityType]:
        """
        Infer possible entity types based on relations.
        
        Args:
            entity_id: Entity to analyze
            
        Returns:
            List of possible entity types with confidence
        """
        if entity := self.graph.get_entity(entity_id):
            return [entity.entity_type]
        
        return []
    
    def reason_about(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Attempt to reason about a question using the graph.
        
        This is a simple implementation that:
        1. Finds relevant entities
        2. Explores their relationships
        3. Generates a response based on findings
        """
        # Find relevant entities
        entities = self.find_entities(question, limit=10)
        
        if not entities:
            return {
                "answer": "I don't have enough information to answer this question.",
                "confidence": 0.1,
                "supporting_entities": [],
                "reasoning": "No relevant entities found in knowledge graph.",
            }
        
        # Collect information
        all_relations = []
        all_neighbors = []
        
        for entity in entities[:3]:
            all_relations.extend(self.get_entity_relations(entity.id))
            all_neighbors.extend(self.traverse(entity.id, max_depth=2))
        
        # Generate answer
        if all_relations:
            relation_summary = self._summarize_relations(all_relations)
            neighbor_summary = self._summarize_neighbors(all_neighbors)
            
            answer = f"Based on the knowledge graph: {relation_summary}"
            
            if neighbor_summary:
                answer += f" Additionally: {neighbor_summary}"
            
            return {
                "answer": answer,
                "confidence": 0.7,
                "supporting_entities": [e.to_dict() for e in entities[:3]],
                "supporting_relations": [r.to_dict() for r in all_relations[:5]],
            }
        
        return {
            "answer": f"I found {len(entities)} related entities but no direct connections.",
            "confidence": 0.3,
            "supporting_entities": [e.to_dict() for e in entities[:3]],
        }
    
    # ==================== Aggregation Queries ====================
    
    def count_entities(
        self,
        entity_type: Optional[EntityType] = None,
    ) -> int:
        """Count entities, optionally filtered by type."""
        if entity_type:
            return len(self.graph._type_index.get(entity_type, set()))
        return len(self.graph._entities)
    
    def count_relations(
        self,
        relation_type: Optional[RelationType] = None,
    ) -> int:
        """Count relations, optionally filtered by type."""
        if relation_type:
            return len(self.graph._relation_type_index.get(relation_type, set()))
        return len(self.graph._relations)
    
    def get_most_connected(
        self,
        entity_type: Optional[EntityType] = None,
        limit: int = 10,
    ) -> list[tuple[Entity, int]]:
        """Get most connected entities."""
        entities = []
        
        for entity in self.graph._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            entities.append((entity, entity.degree))
        
        entities.sort(key=lambda x: x[1], reverse=True)
        return entities[:limit]
    
    def get_statistics(self) -> dict:
        """Get graph statistics."""
        return {
            "total_entities": len(self.graph._entities),
            "total_relations": len(self.graph._relations),
            "entity_types": {
                et.value: len(eids)
                for et, eids in self.graph._type_index.items()
            },
            "relation_types": {
                rt.value: len(rids)
                for rt, rids in self.graph._relation_type_index.items()
            },
            "avg_connections": (
                len(self.graph._relations) / max(1, len(self.graph._entities))
            ),
            "most_connected": [
                {"name": e.name, "degree": e.degree}
                for e, _ in self.get_most_connected(limit=5)
            ],
        }
    
    # ==================== Subgraph Extraction ====================
    
    def extract_subgraph(
        self,
        center_id: str,
        radius: int = 2,
    ) -> dict[str, Any]:
        """
        Extract a subgraph around a central entity.
        
        Args:
            center_id: Central entity ID
            radius: Extraction radius
            
        Returns:
            Dictionary with entities and relations
        """
        entity_ids = {center_id}
        relation_ids = set()
        
        # BFS to collect entities within radius
        frontier = [center_id]
        for _ in range(radius):
            next_frontier = []
            
            for eid in frontier:
                # Get relations
                for rel in self.graph.get_relations_for_entity(eid):
                    relation_ids.add(rel.id)
                    entity_ids.add(rel.source_id)
                    entity_ids.add(rel.target_id)
                
                # Add neighbors to next frontier
                for neighbor, _ in self.graph.get_neighbors(eid):
                    if neighbor.id not in entity_ids:
                        next_frontier.append(neighbor.id)
            
            frontier = next_frontier
        
        # Extract entities and relations
        entities = [
            self.graph._entities[eid].to_dict()
            for eid in entity_ids
            if eid in self.graph._entities
        ]
        
        relations = [
            self.graph._relations[rid].to_dict()
            for rid in relation_ids
            if rid in self.graph._relations
        ]
        
        return {
            "center": center_id,
            "radius": radius,
            "entities": entities,
            "relations": relations,
            "entity_count": len(entities),
            "relation_count": len(relations),
        }
    
    # ==================== Utility ====================
    
    def _summarize_relations(self, relations: list[Relation]) -> str:
        """Generate a summary of relations."""
        if not relations:
            return "No direct relationships found."
        
        type_counts = {}
        for rel in relations:
            rt = rel.relation_type.value
            type_counts[rt] = type_counts.get(rt, 0) + 1
        
        parts = []
        for rt, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            parts.append(f"{count} {rt} relationships")
        
        return ", ".join(parts[:3])
    
    def _summarize_neighbors(self, neighbors: list) -> str:
        """Generate a summary of neighboring entities."""
        if not neighbors:
            return ""
        
        entity_types = {}
        for entity, _, _ in neighbors:
            et = entity.entity_type.value
            entity_types[et] = entity_types.get(et, 0) + 1
        
        if entity_types:
            return f"Connected to {len(neighbors)} related entities including {list(entity_types.keys())[0]}s."
        
        return ""
