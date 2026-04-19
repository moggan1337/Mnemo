"""Example: Knowledge Graph with Mnemo."""

from mnemo.knowledge_graph import KnowledgeGraph, QueryEngine
from mnemo.knowledge_graph.graph import EntityType, RelationType


def main():
    print("=" * 60)
    print("Mnemo Knowledge Graph - Example")
    print("=" * 60)
    
    # Create knowledge graph
    graph = KnowledgeGraph()
    engine = QueryEngine(graph)
    
    # Add entities
    print("\n🏗️ Building knowledge graph...")
    
    transformer = graph.add_entity(
        name="Transformer",
        entity_type=EntityType.TECHNOLOGY,
        description="Neural network architecture using self-attention mechanisms",
        properties={"year": 2017, "paper": "Attention Is All You Need"}
    )
    
    attention = graph.add_entity(
        name="Self-Attention",
        entity_type=EntityType.METHOD,
        description="Mechanism allowing each element to attend to all others",
        properties={"complexity": "O(n²)"}
    )
    
    bert = graph.add_entity(
        name="BERT",
        entity_type=EntityType.TECHNOLOGY,
        description="Bidirectional Encoder Representations from Transformers",
        properties={"year": 2018, "company": "Google"}
    )
    
    gpt = graph.add_entity(
        name="GPT",
        entity_type=EntityType.TECHNOLOGY,
        description="Generative Pre-trained Transformer",
        properties={"year": 2018, "company": "OpenAI"}
    )
    
    nlp = graph.add_entity(
        name="NLP",
        entity_type=EntityType.CONCEPT,
        description="Natural Language Processing"
    )
    
    # Add relations
    print("🔗 Creating relations...")
    
    graph.add_relation(
        attention.id, transformer.id,
        RelationType.ENABLES,
        confidence=0.95
    )
    
    graph.add_relation(
        transformer.id, bert.id,
        RelationType.DEVELOPS,
        confidence=0.90
    )
    
    graph.add_relation(
        transformer.id, gpt.id,
        RelationType.DEVELOPS,
        confidence=0.90
    )
    
    graph.add_relation(
        bert.id, nlp.id,
        RelationType.APPLIED_TO,
        confidence=0.85
    )
    
    graph.add_relation(
        gpt.id, nlp.id,
        RelationType.APPLIED_TO,
        confidence=0.85
    )
    
    # Query the graph
    print("\n" + "-" * 60)
    print("GRAPH STATISTICS:")
    print("-" * 60)
    
    stats = engine.get_statistics()
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Avg connections: {stats['avg_connections']:.2f}")
    
    print("\n  Entities by type:")
    for etype, count in stats.get('entity_types', {}).items():
        print(f"    {etype}: {count}")
    
    # Search for entities
    print("\n" + "-" * 60)
    print("SEARCH RESULTS for 'transformer':")
    print("-" * 60)
    
    results = engine.find_entities("transformer")
    for entity in results:
        print(f"  • {entity.name} ({entity.entity_type.value})")
        print(f"    {entity.description[:60]}...")
    
    # Find paths
    print("\n" + "-" * 60)
    print("PATH FINDING (Attention → NLP):")
    print("-" * 60)
    
    path = engine.find_shortest_path(attention.id, nlp.id)
    if path:
        print(f"  Path: {' → '.join([e.name for e in path])}")
    else:
        print("  No path found")
    
    # Reasoning
    print("\n" + "-" * 60)
    print("REASONING ABOUT 'attention':")
    print("-" * 60)
    
    result = engine.reason_about("How does attention relate to NLP?")
    print(f"  Answer: {result['answer']}")
    print(f"  Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
