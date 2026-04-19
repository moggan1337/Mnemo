"""
Knowledge Graph implementation for Mnemo.

Provides entity extraction, relationship mapping, and graph operations
for building a structured knowledge base.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
import uuid

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    CONCEPT = "concept"           # Abstract concepts
    PERSON = "person"             # People
    ORGANIZATION = "organization" # Organizations/companies
    LOCATION = "location"        # Places
    EVENT = "event"              # Events
    DOCUMENT = "document"        # Documents/papers
    TECHNOLOGY = "technology"     # Technologies/tools
    THEORY = "theory"            # Theories/models
    METHOD = "method"           # Methods/algorithms
    TERM = "term"                # Key terms
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Types of relations between entities."""
    IS_A = "is_a"                 # Categorization (X is a Y)
    PART_OF = "part_of"           # Containment (X is part of Y)
    RELATED_TO = "related_to"      # General relatedness
    CAUSES = "causes"             # Causation
    ENABLES = "enables"           # Enables/facilitates
    CONTRADICTS = "contradicts"   # Contradiction
    SUPPORTS = "supports"         # Evidence/support
    CITES = "cites"              # Citation
    DEVELOPS = "develops"         # Building upon
    USES = "uses"                # Usage relationship
    SIMILAR_TO = "similar_to"     # Similarity
    PRECEDES = "precedes"        # Temporal ordering
    DEFINES = "defines"          # Definition
    APPLIED_TO = "applied_to"    # Application


@dataclass
class Entity:
    """
    An entity node in the knowledge graph.
    
    Entities represent concepts, people, organizations, or any
    distinct thing that can be uniquely identified.
    """
    id: str
    name: str
    entity_type: EntityType
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    confidence: float = 1.0  # Extraction confidence
    source: Optional[str] = None  # Source document/page
    
    # Embedding (for similarity search)
    embedding: Optional[list[float]] = None
    
    # Statistics
    in_degree: int = 0
    out_degree: int = 0
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID from name and type."""
        content = f"{self.name}:{self.entity_type.value}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    @property
    def degree(self) -> int:
        """Total connections."""
        return self.in_degree + self.out_degree
    
    def update(self, **kwargs) -> None:
        """Update entity properties."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = time.time()
    
    def add_property(self, key: str, value: Any) -> None:
        """Add or update a property."""
        self.properties[key] = value
        self.updated_at = time.time()
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "properties": self.properties,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "source": self.source,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Create from dictionary."""
        entity_type = EntityType(data.get("entity_type", "unknown"))
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=entity_type,
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            confidence=data.get("confidence", 1.0),
            source=data.get("source"),
            in_degree=data.get("in_degree", 0),
            out_degree=data.get("out_degree", 0),
        )


@dataclass
class Relation:
    """
    A relation edge in the knowledge graph.
    
    Relations connect entities and represent the relationships
    between them.
    """
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    
    # Properties
    properties: dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    confidence: float = 1.0
    bidirectional: bool = False
    weight: float = 1.0  # Relation strength
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        content = f"{self.source_id}:{self.relation_type.value}:{self.target_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "created_at": self.created_at,
            "confidence": self.confidence,
            "bidirectional": self.bidirectional,
            "weight": self.weight,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Relation":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data.get("relation_type", "related_to")),
            properties=data.get("properties", {}),
            created_at=data.get("created_at", time.time()),
            confidence=data.get("confidence", 1.0),
            bidirectional=data.get("bidirectional", False),
            weight=data.get("weight", 1.0),
        )


class KnowledgeGraph:
    """
    Knowledge Graph for storing and querying entities and relations.
    
    Architecture:
    ```
    ┌──────────────────────────────────────────────────────┐
    │                  Knowledge Graph                     │
    │  ┌─────────────┐           ┌─────────────┐         │
    │  │   Entities   │──────────│   Relations │         │
    │  │  (Nodes)     │           │   (Edges)   │         │
    │  └─────────────┘           └─────────────┘         │
    │        │                          │                 │
    │        ▼                          ▼                 │
    │  ┌─────────────┐           ┌─────────────┐         │
    │  │ Embeddings  │           │   Indexes   │         │
    │  │  (Vectors)  │           │   (Search)  │         │
    │  └─────────────┘           └─────────────┘         │
    └──────────────────────────────────────────────────────┘
    ```
    
    Features:
    - Entity extraction and storage
    - Relation mapping
    - Similarity search using embeddings
    - Path finding between entities
    - Community detection
    - Inference capabilities
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_embeddings: bool = True,
    ):
        """
        Initialize knowledge graph.
        
        Args:
            storage_path: Path for persistent storage
            embedding_model: Model for generating embeddings
            enable_embeddings: Enable semantic embeddings
        """
        self.storage_path = storage_path or Path.home() / ".mnemo" / "knowledge"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = embedding_model
        self.enable_embeddings = enable_embeddings
        
        # In-memory graph structure
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        
        # Indexes for fast lookup
        self._entity_index: dict[str, list[str]] = {}  # name -> entity_ids
        self._type_index: dict[EntityType, set[str]] = {
            et: set() for et in EntityType
        }
        self._relation_type_index: dict[RelationType, set[str]] = {
            rt: set() for rt in RelationType
        }
        
        # Adjacency lists
        self._outgoing: dict[str, set[str]] = {}  # entity_id -> relation_ids
        self._incoming: dict[str, set[str]] = {}  # entity_id -> relation_ids
        
        # Embedding index (if enabled)
        self._embedding_index: Optional[Any] = None
        self._initialize_embedding_index()
        
        # Statistics
        self._stats = {
            "entities_added": 0,
            "relations_added": 0,
            "queries_executed": 0,
            "inferences_made": 0,
        }
        
        logger.info("KnowledgeGraph initialized")
    
    def _initialize_embedding_index(self) -> None:
        """Initialize embedding index for similarity search."""
        if not self.enable_embeddings:
            return
        
        try:
            # Lazy import for optional dependency
            import chromadb
            from chromadb.config import Settings
            
            client = chromadb.Client(settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ))
            
            self._embedding_index = client.create_collection(
                name="entity_embeddings",
                metadata={"description": "Entity embeddings for similarity search"}
            )
            
            logger.info("Embedding index initialized")
            
        except ImportError:
            logger.warning("chromadb not available, embeddings disabled")
            self.enable_embeddings = False
        except Exception as e:
            logger.warning(f"Failed to initialize embedding index: {e}")
            self.enable_embeddings = False
    
    # ==================== Entity Operations ====================
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        description: str = "",
        properties: Optional[dict] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
    ) -> Entity:
        """
        Add an entity to the graph.
        
        Args:
            name: Entity name
            entity_type: Type of entity
            description: Entity description
            properties: Additional properties
            source: Source document
            confidence: Extraction confidence
            
        Returns:
            Created Entity
        """
        # Check if entity already exists
        existing = self.get_entity_by_name(name, entity_type)
        if existing:
            # Update existing
            existing.update(
                description=description or existing.description,
                source=source,
            )
            for k, v in (properties or {}).items():
                existing.add_property(k, v)
            return existing
        
        # Create new entity
        entity = Entity(
            id=str(uuid.uuid4())[:12],
            name=name,
            entity_type=entity_type,
            description=description,
            properties=properties or {},
            source=source,
            confidence=confidence,
        )
        
        # Store
        self._entities[entity.id] = entity
        self._stats["entities_added"] += 1
        
        # Update indexes
        self._index_entity(entity)
        
        # Add embedding
        if self.enable_embeddings:
            self._add_entity_embedding(entity)
        
        logger.debug(f"Added entity: {entity.name} ({entity.entity_type.value})")
        return entity
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entity_by_name(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
    ) -> Optional[Entity]:
        """Get entity by name."""
        normalized = name.lower().strip()
        
        for entity in self._entities.values():
            if entity.name.lower() == normalized:
                if entity_type is None or entity.entity_type == entity_type:
                    return entity
        
        return None
    
    def get_entities_by_type(
        self,
        entity_type: EntityType,
        limit: int = 100,
    ) -> list[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self._type_index.get(entity_type, set())
        return [
            self._entities[eid]
            for eid in list(entity_ids)[:limit]
            if eid in self._entities
        ]
    
    def update_entity(self, entity_id: str, **kwargs) -> bool:
        """Update an entity's properties."""
        if entity := self._entities.get(entity_id):
            entity.update(**kwargs)
            return True
        return False
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and its relations."""
        if entity_id not in self._entities:
            return False
        
        # Remove relations
        relations_to_remove = self.get_relations_for_entity(entity_id)
        for relation in relations_to_remove:
            self._remove_relation(relation.id)
        
        # Remove from indexes
        self._unindex_entity(entity_id)
        
        # Remove embedding
        if self.enable_embeddings and self._embedding_index:
            try:
                self._embedding_index.delete(ids=[entity_id])
            except Exception:
                pass
        
        # Remove entity
        del self._entities[entity_id]
        
        return True
    
    # ==================== Relation Operations ====================
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        properties: Optional[dict] = None,
        bidirectional: bool = False,
        confidence: float = 1.0,
        weight: float = 1.0,
    ) -> Optional[Relation]:
        """
        Add a relation between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relation
            properties: Relation properties
            bidirectional: Create reverse relation too
            confidence: Relation confidence
            weight: Relation strength
            
        Returns:
            Created Relation or None if entities don't exist
        """
        # Verify entities exist
        if source_id not in self._entities or target_id not in self._entities:
            logger.warning(f"Cannot create relation: entity not found")
            return None
        
        # Create relation
        relation = Relation(
            id=str(uuid.uuid4())[:16],
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            properties=properties or {},
            bidirectional=bidirectional,
            confidence=confidence,
            weight=weight,
        )
        
        # Store
        self._relations[relation.id] = relation
        self._stats["relations_added"] += 1
        
        # Update adjacency lists
        if source_id not in self._outgoing:
            self._outgoing[source_id] = set()
        self._outgoing[source_id].add(relation.id)
        
        if target_id not in self._incoming:
            self._incoming[target_id] = set()
        self._incoming[target_id].add(relation.id)
        
        # Update entity degrees
        self._entities[source_id].out_degree += 1
        self._entities[target_id].in_degree += 1
        
        # Update indexes
        self._relation_type_index[relation_type].add(relation.id)
        
        # Create reverse relation if bidirectional
        if bidirectional:
            reverse_type = self._get_reverse_relation_type(relation_type)
            self.add_relation(
                target_id,
                source_id,
                reverse_type,
                properties=properties,
                bidirectional=False,
                confidence=confidence * 0.9,
                weight=weight,
            )
        
        logger.debug(
            f"Added relation: {source_id} --[{relation_type.value}]--> {target_id}"
        )
        
        return relation
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self._relations.get(relation_id)
    
    def get_relations_for_entity(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
    ) -> list[Relation]:
        """Get all relations for an entity."""
        relations = []
        
        # Outgoing relations
        for rel_id in self._outgoing.get(entity_id, set()):
            if rel := self._relations.get(rel_id):
                if relation_type is None or rel.relation_type == relation_type:
                    relations.append(rel)
        
        # Incoming relations
        for rel_id in self._incoming.get(entity_id, set()):
            if rel := self._relations.get(rel_id):
                if relation_type is None or rel.relation_type == relation_type:
                    relations.append(rel)
        
        return relations
    
    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        max_depth: int = 1,
    ) -> list[tuple[Entity, Relation]]:
        """Get neighboring entities."""
        neighbors = []
        visited = {entity_id}
        frontier = [(entity_id, 0)]
        
        while frontier:
            current_id, depth = frontier.pop(0)
            
            if depth >= max_depth:
                continue
            
            for relation in self.get_relations_for_entity(current_id, relation_type):
                # Determine neighbor
                if relation.source_id == current_id:
                    neighbor_id = relation.target_id
                else:
                    neighbor_id = relation.source_id
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    
                    if neighbor := self._entities.get(neighbor_id):
                        neighbors.append((neighbor, relation))
                        frontier.append((neighbor_id, depth + 1))
        
        return neighbors
    
    def _remove_relation(self, relation_id: str) -> None:
        """Remove a relation."""
        if relation := self._relations.get(relation_id):
            # Update adjacency
            self._outgoing.get(relation.source_id, set()).discard(relation_id)
            self._incoming.get(relation.target_id, set()).discard(relation_id)
            
            # Update entity degrees
            if relation.source_id in self._entities:
                self._entities[relation.source_id].out_degree = max(
                    0, self._entities[relation.source_id].out_degree - 1
                )
            if relation.target_id in self._entities:
                self._entities[relation.target_id].in_degree = max(
                    0, self._entities[relation.target_id].in_degree - 1
                )
            
            # Update type index
            self._relation_type_index[relation.relation_type].discard(relation_id)
            
            # Remove
            del self._relations[relation_id]
    
    # ==================== Search and Query ====================
    
    def search(
        self,
        query: str,
        entity_types: Optional[list[EntityType]] = None,
        limit: int = 20,
    ) -> list[Entity]:
        """
        Search for entities by name or description.
        
        Args:
            query: Search query
            entity_types: Filter by entity types
            limit: Maximum results
            
        Returns:
            List of matching entities
        """
        self._stats["queries_executed"] += 1
        
        query_lower = query.lower()
        results = []
        
        for entity in self._entities.values():
            # Type filter
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # Name match
            if query_lower in entity.name.lower():
                results.append((entity, 1.0))
                continue
            
            # Description match
            if query_lower in entity.description.lower():
                results.append((entity, 0.7))
                continue
            
            # Property match
            for prop_value in entity.properties.values():
                if query_lower in str(prop_value).lower():
                    results.append((entity, 0.5))
                    break
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [e for e, _ in results[:limit]]
    
    def find_similar(
        self,
        entity_id: str,
        limit: int = 10,
    ) -> list[tuple[Entity, float]]:
        """Find similar entities using embeddings."""
        if not self.enable_embeddings or not self._embedding_index:
            return []
        
        try:
            # Get query embedding
            if entity := self._entities.get(entity_id):
                if entity.embedding:
                    results = self._embedding_index.query(
                        query_embeddings=[entity.embedding],
                        n_results=limit + 1,
                    )
                    
                    similar = []
                    for i, (eid, distance) in enumerate(zip(
                        results["ids"][0],
                        results["distances"][0]
                    )):
                        if eid != entity_id and (e := self._entities.get(eid)):
                            similarity = 1 - distance  # Convert distance to similarity
                            similar.append((e, similarity))
                    
                    return similar
        
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
        
        return []
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5,
    ) -> list[list[Entity]]:
        """
        Find paths between two entities.
        
        Args:
            source_id: Start entity ID
            target_id: End entity ID
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of entities)
        """
        paths = []
        
        def dfs(
            current_id: str,
            target_id: str,
            path: list[str],
            visited: set[str],
            length: int,
        ):
            if length > max_length:
                return
            
            if current_id == target_id:
                paths.append([self._entities[mid] for mid in path])
                return
            
            for neighbor, _ in self.get_neighbors(current_id):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    path.append(neighbor.id)
                    dfs(neighbor.id, target_id, path, visited, length + 1)
                    path.pop()
                    visited.remove(neighbor.id)
        
        visited = {source_id}
        dfs(source_id, target_id, [source_id], visited, 0)
        
        return paths
    
    # ==================== Memory Integration ====================
    
    async def add_memory(self, memory) -> dict[str, Any]:
        """
        Add a memory to the knowledge graph.
        
        Extracts entities and relations from memory content.
        """
        content = str(memory.content)
        
        # Simple entity extraction
        entities = self._extract_entities(content)
        
        updates = {
            "entities_added": 0,
            "relations_added": 0,
        }
        
        # Add entities
        for entity_data in entities:
            entity = self.add_entity(
                name=entity_data["name"],
                entity_type=entity_data["type"],
                description=entity_data.get("description", ""),
                source=memory.id,
                confidence=entity_data.get("confidence", 0.8),
            )
            updates["entities_added"] += 1
            
            # Create entity mentions in properties
            entity.add_property("memory_id", memory.id)
        
        return updates
    
    def _extract_entities(self, text: str) -> list[dict]:
        """Extract entities from text (simple implementation)."""
        entities = []
        
        # Simple capitalized phrase extraction
        import re
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        seen = set()
        for phrase in capitalized:
            if len(phrase) > 2 and phrase not in seen:
                seen.add(phrase)
                entities.append({
                    "name": phrase,
                    "type": EntityType.CONCEPT,
                    "confidence": 0.7,
                })
        
        return entities[:20]  # Limit extraction
    
    # ==================== Index Management ====================
    
    def _index_entity(self, entity: Entity) -> None:
        """Add entity to indexes."""
        # Name index
        name_lower = entity.name.lower()
        if name_lower not in self._entity_index:
            self._entity_index[name_lower] = []
        self._entity_index[name_lower].append(entity.id)
        
        # Type index
        self._type_index[entity.entity_type].add(entity.id)
    
    def _unindex_entity(self, entity_id: str) -> None:
        """Remove entity from indexes."""
        if entity := self._entities.get(entity_id):
            # Name index
            name_lower = entity.name.lower()
            if name_lower in self._entity_index:
                self._entity_index[name_lower] = [
                    eid for eid in self._entity_index[name_lower]
                    if eid != entity_id
                ]
            
            # Type index
            self._type_index[entity.entity_type].discard(entity_id)
    
    async def _add_entity_embedding(self, entity: Entity) -> None:
        """Add embedding for an entity."""
        if not self.enable_embeddings or not self._embedding_index:
            return
        
        try:
            # Lazy import
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(self.embedding_model)
            text = f"{entity.name} {entity.description}"
            embedding = model.encode(text).tolist()
            
            entity.embedding = embedding
            
            self._embedding_index.add(
                ids=[entity.id],
                embeddings=[embedding],
                documents=[text],
            )
            
        except Exception as e:
            logger.warning(f"Failed to add embedding: {e}")
    
    # ==================== Utility ====================
    
    def _get_reverse_relation_type(self, rt: RelationType) -> RelationType:
        """Get reverse relation type."""
        reverse_map = {
            RelationType.IS_A: RelationType.IS_A,
            RelationType.PART_OF: RelationType.CONTAINS,
            RelationType.CAUSES: RelationType.CAUSED_BY,
            RelationType.ENABLES: RelationType.ENABLED_BY,
            RelationType.SUPPORTS: RelationType.SUPPORTED_BY,
            RelationType.CITES: RelationType.CITED_BY,
            RelationType.DEVELOPS: RelationType.DEVELOPED_BY,
            RelationType.USES: RelationType.USED_BY,
            RelationType.PRECEDES: RelationType.FOLLOWS,
            RelationType.APPLIED_TO: RelationType.APPLIES,
            RelationType.RELATED_TO: RelationType.RELATED_TO,
            RelationType.SIMILAR_TO: RelationType.SIMILAR_TO,
            RelationType.CONTRADICTS: RelationType.CONTRADICTED_BY,
            RelationType.DEFINES: RelationType.DEFINED_BY,
        }
        
        return reverse_map.get(rt, RelationType.RELATED_TO)
    
    @property
    def statistics(self) -> dict:
        """Get graph statistics."""
        return {
            **self._stats,
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
            "entities_by_type": {
                et.value: len(eids)
                for et, eids in self._type_index.items()
            },
        }
    
    def export(self) -> dict:
        """Export graph data."""
        return {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations.values()],
            "statistics": self.statistics,
        }
    
    def clear(self) -> None:
        """Clear all graph data."""
        self._entities.clear()
        self._relations.clear()
        self._entity_index.clear()
        self._type_index = {et: set() for et in EntityType}
        self._relation_type_index = {rt: set() for rt in RelationType}
        self._outgoing.clear()
        self._incoming.clear()
        
        if self._embedding_index:
            try:
                self._embedding_index.delete(where={"id": {"$ne": ""}})
            except Exception:
                pass
