"""
Working Memory module for Mnemo.

Implements a multi-layered working memory system that bridges
perception, short-term storage, and long-term consolidation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from collections import deque
import heapq


class MemoryType(Enum):
    """Types of memories in the working memory system."""
    PERCEPT = "percept"           # Raw sensory/input data
    EPISODE = "episode"           # Specific events or experiences
    SEMANTIC = "semantic"         # Factual knowledge
    PROCEDURAL = "procedural"     # Skills and methods
    EMOTIONAL = "emotional"       # Affective memories
    DREAM = "dream"               # Processed dream content


class MemoryPriority(Enum):
    """Priority levels for memory importance."""
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NEGLECTED = 0


@dataclass
class Memory:
    """
    Core memory unit in Mnemo's cognitive architecture.
    
    Each memory stores content along with metadata for importance
    scoring, decay, and consolidation decisions.
    """
    id: str
    content: Any
    memory_type: MemoryType
    priority: MemoryPriority = MemoryPriority.MEDIUM
    importance: float = 0.5  # 0.0 to 1.0
    activation: float = 0.5   # Current activation level
    decay_rate: float = 0.01  # How fast importance decays
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    consolidation_level: float = 0.0  # 0.0 = fragile, 1.0 = consolidated
    associations: list[str] = field(default_factory=list)  # Related memory IDs
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def access(self) -> None:
        """Update access statistics when memory is retrieved."""
        self.last_accessed = time.time()
        self.access_count += 1
        self.activation = min(1.0, self.activation + 0.1)
    
    def update_importance(self, delta: float) -> None:
        """Adjust importance score with bounds checking."""
        self.importance = max(0.0, min(1.0, self.importance + delta))
    
    def decay(self) -> None:
        """Apply time-based decay to memory importance."""
        time_elapsed = time.time() - self.last_accessed
        decay_factor = self.decay_rate * time_elapsed / 3600  # Per hour
        self.importance = max(0.0, self.importance - decay_factor)
        self.activation = max(0.0, self.activation - decay_factor * 0.5)
    
    def consolidate(self, strength: float = 0.1) -> None:
        """Move memory toward long-term storage."""
        self.consolidation_level = min(1.0, self.consolidation_level + strength)
    
    def to_dict(self) -> dict:
        """Serialize memory to dictionary."""
        return {
            "id": self.id,
            "content": str(self.content),
            "memory_type": self.memory_type.value,
            "priority": self.priority.name,
            "importance": self.importance,
            "activation": self.activation,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "consolidation_level": self.consolidation_level,
            "associations": self.associations,
            "tags": list(self.tags),
            "metadata": self.metadata,
        }


class WorkingMemory:
    """
    Multi-layered working memory with priority queues and decay.
    
    Architecture:
    - Primary buffer: Recently active memories (fast access)
    - Priority queue: High-importance items awaiting consolidation
    - Long-term buffer: Memories being transferred to knowledge graph
    - Dream buffer: Memories queued for dream processing
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        forgetting_rate: float = 0.05,
        consolidation_threshold: float = 0.8,
    ):
        self.max_size = max_size
        self.forgetting_rate = forgetting_rate
        self.consolidation_threshold = consolidation_threshold
        
        # Storage layers
        self._primary: dict[str, Memory] = {}
        self._priority_queue: list[tuple[float, str]] = []  # (priority_score, memory_id)
        self._recent: deque[str] = deque(maxlen=100)  # Recently accessed IDs
        
        # Statistics
        self._stats = {
            "total_accesses": 0,
            "consolidations": 0,
            "forgotten": 0,
            "dreams_processed": 0,
        }
    
    @property
    def size(self) -> int:
        """Current number of memories."""
        return len(self._primary)
    
    @property
    def statistics(self) -> dict:
        """Return memory statistics."""
        return {
            **self._stats,
            "current_size": self.size,
            "queue_size": len(self._priority_queue),
        }
    
    def add(
        self,
        content: Any,
        memory_type: MemoryType,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        importance: Optional[float] = None,
        tags: Optional[set[str]] = None,
        metadata: Optional[dict] = None,
    ) -> Memory:
        """Add a new memory to working memory."""
        import uuid
        
        memory_id = str(uuid.uuid4())
        
        # Calculate initial importance
        if importance is None:
            importance = self._calculate_importance(content, memory_type, priority)
        
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            priority=priority,
            importance=importance,
            tags=tags or set(),
            metadata=metadata or {},
        )
        
        # Store in primary buffer
        self._primary[memory_id] = memory
        self._update_priority_queue(memory)
        self._recent.append(memory_id)
        
        # Evict if over capacity
        if len(self._primary) > self.max_size:
            self._evict_low_priority()
        
        return memory
    
    def get(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        if memory := self._primary.get(memory_id):
            memory.access()
            self._stats["total_accesses"] += 1
            return memory
        return None
    
    def search(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        min_importance: float = 0.0,
        tags: Optional[set[str]] = None,
        limit: int = 50,
    ) -> list[Memory]:
        """Search memories by various criteria."""
        results = []
        
        for memory in self._primary.values():
            # Filter by type
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Filter by importance
            if memory.importance < min_importance:
                continue
            
            # Filter by tags
            if tags and not tags.intersection(memory.tags):
                continue
            
            # Text search in content
            if query:
                query_lower = query.lower()
                content_str = str(memory.content).lower()
                if query_lower not in content_str:
                    # Also check metadata
                    metadata_str = str(memory.metadata).lower()
                    if query_lower not in metadata_str:
                        continue
            
            results.append(memory)
        
        # Sort by importance and recency
        results.sort(
            key=lambda m: (m.importance, m.last_accessed),
            reverse=True
        )
        
        return results[:limit]
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update memory properties."""
        if memory := self._primary.get(memory_id):
            for key, value in kwargs.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)
            self._update_priority_queue(memory)
            return True
        return False
    
    def associate(self, memory_id1: str, memory_id2: str) -> bool:
        """Create association between two memories."""
        if memory1 := self._primary.get(memory_id1):
            if memory2 := self._primary.get(memory_id2):
                if memory_id2 not in memory1.associations:
                    memory1.associations.append(memory_id2)
                if memory_id1 not in memory2.associations:
                    memory2.associations.append(memory_id1)
                return True
        return False
    
    def consolidate_ready(self) -> list[Memory]:
        """Get memories ready for long-term consolidation."""
        ready = []
        for memory in self._primary.values():
            if memory.consolidation_level >= self.consolidation_threshold:
                ready.append(memory)
        return sorted(ready, key=lambda m: m.importance, reverse=True)
    
    def dream_ready(self, min_memories: int = 10) -> list[Memory]:
        """Get memories queued for dream processing."""
        # Get recent memories that haven't been dreamed yet
        recent_memories = []
        for memory_id in list(self._recent):
            if memory := self._primary.get(memory_id):
                if memory.memory_type != MemoryType.DREAM:
                    recent_memories.append(memory)
        
        if len(recent_memories) >= min_memories:
            return sorted(recent_memories, key=lambda m: m.importance, reverse=True)[:min_memories]
        return []
    
    def mark_dreamed(self, memory_ids: list[str]) -> None:
        """Mark memories as having been processed by dream engine."""
        for memory_id in memory_ids:
            if memory := self._primary.get(memory_id):
                memory.metadata["dreamed"] = True
                memory.metadata["last_dream"] = time.time()
        self._stats["dreams_processed"] += len(memory_ids)
    
    def forget(self, memory_id: str) -> bool:
        """Remove a memory from working memory."""
        if memory_id in self._primary:
            del self._primary[memory_id]
            self._stats["forgotten"] += 1
            return True
        return False
    
    def decay_all(self) -> int:
        """Apply decay to all memories. Returns count of decayed memories."""
        count = 0
        for memory in self._primary.values():
            memory.decay()
            count += 1
            
            # Remove memories that have decayed below threshold
            if memory.importance <= 0.05:
                self.forget(memory.id)
        
        return count
    
    def _calculate_importance(
        self,
        content: Any,
        memory_type: MemoryType,
        priority: MemoryPriority,
    ) -> float:
        """Calculate initial importance score for a new memory."""
        # Base importance from priority
        importance = {
            MemoryPriority.CRITICAL: 0.9,
            MemoryPriority.HIGH: 0.7,
            MemoryPriority.MEDIUM: 0.5,
            MemoryPriority.LOW: 0.3,
            MemoryPriority.NEGLECTED: 0.1,
        }.get(priority, 0.5)
        
        # Type-based adjustment
        type_modifier = {
            MemoryType.PERCEPT: 0.8,
            MemoryType.EPISODE: 0.9,
            MemoryType.SEMANTIC: 1.0,
            MemoryType.PROCEDURAL: 0.95,
            MemoryType.EMOTIONAL: 0.85,
            MemoryType.DREAM: 0.7,
        }.get(memory_type, 1.0)
        
        # Content length modifier (longer content slightly more important)
        content_len = len(str(content))
        length_modifier = min(1.2, 0.8 + (content_len / 10000))
        
        return min(1.0, importance * type_modifier * length_modifier)
    
    def _update_priority_queue(self, memory: Memory) -> None:
        """Update memory's position in priority queue."""
        # Remove existing entry if present
        self._priority_queue = [
            (score, mid) for score, mid in self._priority_queue
            if mid != memory.id
        ]
        
        # Calculate priority score
        score = memory.importance * memory.activation
        
        # Add to queue
        heapq.heappush(self._priority_queue, (-score, memory.id))
    
    def _evict_low_priority(self) -> None:
        """Evict lowest priority memories when over capacity."""
        # Get lowest priority memories
        candidates = sorted(
            self._primary.values(),
            key=lambda m: (m.importance, m.access_count)
        )
        
        # Remove bottom 10%
        evict_count = max(1, len(candidates) // 10)
        for memory in candidates[:evict_count]:
            self.forget(memory.id)
    
    def get_associations(self, memory_id: str, depth: int = 1) -> list[Memory]:
        """Get associated memories up to a certain depth."""
        if memory := self._primary.get(memory_id):
            visited = {memory_id}
            result = [memory]
            frontier = list(memory.associations)
            
            while frontier and depth > 0:
                next_frontier = []
                for assoc_id in frontier:
                    if assoc_id not in visited:
                        if assoc_mem := self._primary.get(assoc_id):
                            visited.add(assoc_id)
                            result.append(assoc_mem)
                            next_frontier.extend(assoc_mem.associations)
                frontier = next_frontier
                depth -= 1
            
            return result
        return []
    
    def export(self) -> list[dict]:
        """Export all memories as dictionaries."""
        return [m.to_dict() for m in self._primary.values()]
    
    def clear(self) -> None:
        """Clear all memories."""
        self._primary.clear()
        self._priority_queue.clear()
        self._recent.clear()
