"""Tests for Mnemo."""

import pytest
import asyncio


class TestWorkingMemory:
    """Tests for WorkingMemory."""
    
    def test_add_memory(self):
        """Test adding memories."""
        from mnemo.core.memory import WorkingMemory, MemoryType, MemoryPriority
        
        memory = WorkingMemory()
        mem = memory.add(
            content="Test content",
            memory_type=MemoryType.PERCEPT,
            priority=MemoryPriority.HIGH,
        )
        
        assert mem.id is not None
        assert mem.content == "Test content"
        assert memory.size == 1
    
    def test_search_memories(self):
        """Test searching memories."""
        from mnemo.core.memory import WorkingMemory, MemoryType
        
        memory = WorkingMemory()
        memory.add("Python is great", MemoryType.SEMANTIC)
        memory.add("JavaScript is also great", MemoryType.SEMANTIC)
        memory.add("C++ is fast", MemoryType.SEMANTIC)
        
        results = memory.search(query="Python")
        assert len(results) >= 1
    
    def test_associations(self):
        """Test memory associations."""
        from mnemo.core.memory import WorkingMemory, MemoryType
        
        memory = WorkingMemory()
        m1 = memory.add("Python", MemoryType.CONCEPT)
        m2 = memory.add("Programming", MemoryType.CONCEPT)
        
        memory.associate(m1.id, m2.id)
        
        associations = memory.get_associations(m1.id)
        assert len(associations) >= 1


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph."""
    
    def test_add_entity(self):
        """Test adding entities."""
        from mnemo.knowledge_graph import KnowledgeGraph
        from mnemo.knowledge_graph.graph import EntityType
        
        graph = KnowledgeGraph()
        entity = graph.add_entity(
            name="Artificial Intelligence",
            entity_type=EntityType.CONCEPT,
            description="The simulation of intelligence",
        )
        
        assert entity.id is not None
        assert entity.name == "Artificial Intelligence"
    
    def test_add_relation(self):
        """Test adding relations."""
        from mnemo.knowledge_graph import KnowledgeGraph
        from mnemo.knowledge_graph.graph import EntityType, RelationType
        
        graph = KnowledgeGraph()
        e1 = graph.add_entity("ML", EntityType.TECHNOLOGY)
        e2 = graph.add_entity("AI", EntityType.CONCEPT)
        
        relation = graph.add_relation(
            e1.id, e2.id, RelationType.IS_A
        )
        
        assert relation is not None
        assert relation.relation_type == RelationType.IS_A
    
    def test_search(self):
        """Test entity search."""
        from mnemo.knowledge_graph import KnowledgeGraph
        from mnemo.knowledge_graph.graph import EntityType
        
        graph = KnowledgeGraph()
        graph.add_entity("Machine Learning", EntityType.CONCEPT)
        graph.add_entity("Deep Learning", EntityType.CONCEPT)
        graph.add_entity("Python", EntityType.TECHNOLOGY)
        
        results = graph.search("learning")
        assert len(results) >= 2


class TestDreamEngine:
    """Tests for DreamEngine."""
    
    def test_condense(self):
        """Test memory condensation."""
        from mnemo.dream_engine.condenser import DreamCondenser
        from mnemo.core.memory import Memory, MemoryType
        
        condenser = DreamCondenser()
        
        # Create mock memories
        class MockMemory:
            def __init__(self, content):
                self.id = "test"
                self.content = content
        
        memories = [
            MockMemory("Python is a programming language"),
            MockMemory("Python is widely used in ML"),
        ]
        
        abstractions = asyncio.run(condenser.condense(memories))
        assert len(abstractions) >= 1


class TestReasoning:
    """Tests for ReasoningEngine."""
    
    def test_deductive_reasoning(self):
        """Test deductive reasoning."""
        from mnemo.reasoning import ReasoningEngine
        from mnemo.reasoning.engine import ReasoningType
        
        engine = ReasoningEngine()
        result = asyncio.run(engine.reason(
            question="What follows from this?",
            context=["All humans are mortal", "Socrates is human"],
            reasoning_type=ReasoningType.DEDUCTIVE,
        ))
        
        assert result.reasoning_type == ReasoningType.DEDUCTIVE
        assert result.confidence > 0


class TestSelfEvolution:
    """Tests for StrategyOptimizer."""
    
    def test_mutation(self):
        """Test strategy mutation."""
        from mnemo.self_evolution.optimizer import SearchStrategy
        
        strategy = SearchStrategy(
            name="test",
            query_patterns=["test"],
            source_preferences=["web"],
            depth_preference="medium",
            max_results=10,
            filters={},
        )
        
        mutated = strategy.mutate(rate=1.0)
        
        assert mutated.name == "test_mutated"
    
    def test_crossover(self):
        """Test strategy crossover."""
        from mnemo.self_evolution.optimizer import SearchStrategy
        
        s1 = SearchStrategy(
            name="strategy1",
            query_patterns=["pattern1"],
            source_preferences=["web"],
            depth_preference="deep",
            max_results=10,
            filters={},
        )
        
        s2 = SearchStrategy(
            name="strategy2",
            query_patterns=["pattern2"],
            source_preferences=["papers"],
            depth_preference="shallow",
            max_results=20,
            filters={},
        )
        
        child = s1.crossover(s2)
        
        assert child.name == "strategy1_x_strategy2"


class TestPerceiver:
    """Tests for Perceiver components."""
    
    def test_source_creation(self):
        """Test source object creation."""
        from mnemo.perceiver.base import Source
        
        source = Source(
            url="https://example.com",
            title="Example",
            content="Test content",
            source_type="web",
        )
        
        assert source.url == "https://example.com"
        assert source.domain == "example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
