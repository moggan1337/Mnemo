"""
Graph Storage implementations.

Provides persistent storage for the knowledge graph using
SQLite or other backends.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional
import sqlite3

from mnemo.knowledge_graph.graph import Entity, Relation, EntityType, RelationType

logger = logging.getLogger(__name__)


class GraphStorage:
    """
    Persistent storage for the knowledge graph.
    
    Supports SQLite backend with optional JSON export.
    
    Schema:
    - entities: id, name, type, description, properties, metadata
    - relations: id, source_id, target_id, type, properties, metadata
    """
    
    def __init__(
        self,
        storage_path: Path,
        backend: str = "sqlite",
    ):
        """
        Initialize storage.
        
        Args:
            storage_path: Path for storage files
            backend: Storage backend ("sqlite", "json", "memory")
        """
        self.storage_path = storage_path
        self.backend = backend
        
        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "json":
            self._json_file = storage_path.with_suffix(".json")
        elif backend == "memory":
            pass  # No initialization needed
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        logger.info(f"GraphStorage initialized with {backend} backend")
    
    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        db_path = self.storage_path.with_suffix(".db")
        
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        cursor = self._conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                description TEXT DEFAULT '',
                properties TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                in_degree INTEGER DEFAULT 0,
                out_degree INTEGER DEFAULT 0
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                bidirectional INTEGER DEFAULT 0,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id)
        """)
        
        self._conn.commit()
    
    # ==================== Entity Operations ====================
    
    def save_entity(self, entity: Entity) -> bool:
        """Save an entity to storage."""
        if self.backend == "sqlite":
            return self._save_entity_sqlite(entity)
        elif self.backend == "json":
            return self._save_entity_json(entity)
        return False
    
    def _save_entity_sqlite(self, entity: Entity) -> bool:
        """Save entity to SQLite."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO entities
                (id, name, entity_type, description, properties, created_at, 
                 updated_at, confidence, source, in_degree, out_degree)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity.id,
                entity.name,
                entity.entity_type.value,
                entity.description,
                json.dumps(entity.properties),
                entity.created_at,
                entity.updated_at,
                entity.confidence,
                entity.source,
                entity.in_degree,
                entity.out_degree,
            ))
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save entity: {e}")
            return False
    
    def _save_entity_json(self, entity: Entity) -> bool:
        """Save entity to JSON file."""
        try:
            data = self._load_json()
            data.setdefault("entities", {})[entity.id] = entity.to_dict()
            self._save_json(data)
            return True
        except Exception as e:
            logger.error(f"Failed to save entity to JSON: {e}")
            return False
    
    def load_entity(self, entity_id: str) -> Optional[Entity]:
        """Load an entity from storage."""
        if self.backend == "sqlite":
            return self._load_entity_sqlite(entity_id)
        elif self.backend == "json":
            return self._load_entity_json(entity_id)
        return None
    
    def _load_entity_sqlite(self, entity_id: str) -> Optional[Entity]:
        """Load entity from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=EntityType(row["entity_type"]),
            description=row["description"],
            properties=json.loads(row["properties"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            confidence=row["confidence"],
            source=row["source"],
            in_degree=row["in_degree"],
            out_degree=row["out_degree"],
        )
    
    def _load_entity_json(self, entity_id: str) -> Optional[Entity]:
        """Load entity from JSON."""
        data = self._load_json()
        entity_data = data.get("entities", {}).get(entity_id)
        
        if entity_data:
            return Entity.from_dict(entity_data)
        return None
    
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity from storage."""
        if self.backend == "sqlite":
            try:
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
                self._conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to delete entity: {e}")
                return False
        elif self.backend == "json":
            data = self._load_json()
            if entity_id in data.get("entities", {}):
                del data["entities"][entity_id]
                self._save_json(data)
                return True
        return False
    
    def load_all_entities(self) -> list[Entity]:
        """Load all entities from storage."""
        if self.backend == "sqlite":
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM entities")
            
            entities = []
            for row in cursor.fetchall():
                entities.append(Entity(
                    id=row["id"],
                    name=row["name"],
                    entity_type=EntityType(row["entity_type"]),
                    description=row["description"],
                    properties=json.loads(row["properties"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    confidence=row["confidence"],
                    source=row["source"],
                    in_degree=row["in_degree"],
                    out_degree=row["out_degree"],
                ))
            return entities
        elif self.backend == "json":
            data = self._load_json()
            return [
                Entity.from_dict(ed) 
                for ed in data.get("entities", {}).values()
            ]
        return []
    
    # ==================== Relation Operations ====================
    
    def save_relation(self, relation: Relation) -> bool:
        """Save a relation to storage."""
        if self.backend == "sqlite":
            return self._save_relation_sqlite(relation)
        elif self.backend == "json":
            return self._save_relation_json(relation)
        return False
    
    def _save_relation_sqlite(self, relation: Relation) -> bool:
        """Save relation to SQLite."""
        try:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO relations
                (id, source_id, target_id, relation_type, properties, created_at,
                 confidence, bidirectional, weight)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                relation.id,
                relation.source_id,
                relation.target_id,
                relation.relation_type.value,
                json.dumps(relation.properties),
                relation.created_at,
                relation.confidence,
                1 if relation.bidirectional else 0,
                relation.weight,
            ))
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save relation: {e}")
            return False
    
    def _save_relation_json(self, relation: Relation) -> bool:
        """Save relation to JSON file."""
        try:
            data = self._load_json()
            data.setdefault("relations", {})[relation.id] = relation.to_dict()
            self._save_json(data)
            return True
        except Exception as e:
            logger.error(f"Failed to save relation to JSON: {e}")
            return False
    
    def load_relation(self, relation_id: str) -> Optional[Relation]:
        """Load a relation from storage."""
        if self.backend == "sqlite":
            return self._load_relation_sqlite(relation_id)
        elif self.backend == "json":
            return self._load_relation_json(relation_id)
        return None
    
    def _load_relation_sqlite(self, relation_id: str) -> Optional[Relation]:
        """Load relation from SQLite."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM relations WHERE id = ?", (relation_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return Relation(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            relation_type=RelationType(row["relation_type"]),
            properties=json.loads(row["properties"]),
            created_at=row["created_at"],
            confidence=row["confidence"],
            bidirectional=bool(row["bidirectional"]),
            weight=row["weight"],
        )
    
    def _load_relation_json(self, relation_id: str) -> Optional[Relation]:
        """Load relation from JSON."""
        data = self._load_json()
        rel_data = data.get("relations", {}).get(relation_id)
        
        if rel_data:
            return Relation.from_dict(rel_data)
        return None
    
    def load_all_relations(self) -> list[Relation]:
        """Load all relations from storage."""
        if self.backend == "sqlite":
            cursor = self._conn.cursor()
            cursor.execute("SELECT * FROM relations")
            
            relations = []
            for row in cursor.fetchall():
                relations.append(Relation(
                    id=row["id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relation_type=RelationType(row["relation_type"]),
                    properties=json.loads(row["properties"]),
                    created_at=row["created_at"],
                    confidence=row["confidence"],
                    bidirectional=bool(row["bidirectional"]),
                    weight=row["weight"],
                ))
            return relations
        elif self.backend == "json":
            data = self._load_json()
            return [
                Relation.from_dict(rd) 
                for rd in data.get("relations", {}).values()
            ]
        return []
    
    def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation from storage."""
        if self.backend == "sqlite":
            try:
                cursor = self._conn.cursor()
                cursor.execute("DELETE FROM relations WHERE id = ?", (relation_id,))
                self._conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to delete relation: {e}")
                return False
        elif self.backend == "json":
            data = self._load_json()
            if relation_id in data.get("relations", {}):
                del data["relations"][relation_id]
                self._save_json(data)
                return True
        return False
    
    # ==================== JSON Helpers ====================
    
    def _load_json(self) -> dict:
        """Load data from JSON file."""
        if self._json_file.exists():
            with open(self._json_file) as f:
                return json.load(f)
        return {"entities": {}, "relations": {}}
    
    def _save_json(self, data: dict) -> None:
        """Save data to JSON file."""
        with open(self._json_file, "w") as f:
            json.dump(data, f, indent=2)
    
    # ==================== Utility ====================
    
    def close(self) -> None:
        """Close storage connections."""
        if self.backend == "sqlite" and hasattr(self, "_conn"):
            self._conn.close()
    
    def clear(self) -> None:
        """Clear all stored data."""
        if self.backend == "sqlite":
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM relations")
            cursor.execute("DELETE FROM entities")
            self._conn.commit()
        elif self.backend == "json":
            self._save_json({"entities": {}, "relations": {}})
    
    def export(self) -> dict:
        """Export all data."""
        if self.backend == "json":
            return self._load_json()
        
        return {
            "entities": [e.to_dict() for e in self.load_all_entities()],
            "relations": [r.to_dict() for r in self.load_all_relations()],
        }
    
    def import_data(self, data: dict) -> bool:
        """Import data from dictionary."""
        try:
            for entity_data in data.get("entities", []):
                entity = Entity.from_dict(entity_data)
                self.save_entity(entity)
            
            for relation_data in data.get("relations", []):
                relation = Relation.from_dict(relation_data)
                self.save_relation(relation)
            
            return True
        except Exception as e:
            logger.error(f"Failed to import data: {e}")
            return False
