# Mnemo - Autonomous Research Agent with Dream Distillation

<p align="center">
  <img src="docs/logo.png" alt="Mnemo Logo" width="200"/>
</p>

<p align="center">
  <strong>🧠 An intelligent research agent that learns while you sleep</strong>
</p>

<p align="center">
  <a href="https://github.com/moggan1337/Mnemo/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg"/>
  </a>
  <a href="https://github.com/moggan1337/Mnemo/releases">
    <img src="https://img.shields.io/badge/Version-0.1.0-green.svg"/>
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.10+-yellow.svg"/>
  </a>
</p>

---

## 🎬 Demo
![Mnemo Demo](demo.gif)

*Autonomous research with dream distillation*

## Screenshots
| Component | Preview |
|-----------|---------|
| Research Dashboard | ![dashboard](screenshots/dashboard.png) |
| Knowledge Graph | ![graph](screenshots/knowledge-graph.png) |
| Dream State | ![dream](screenshots/dream-state.png) |

## Visual Description
Research dashboard shows active queries and source gathering. Knowledge graph displays learned concepts with connections. Dream state shows consolidation process with memory integration.

---


## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [How Dream Distillation Works](#how-dream-distillation-works)
5. [Knowledge Graph Structure](#knowledge-graph-structure)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [Usage Examples](#usage-examples)
9. [Configuration](#configuration)
10. [API Reference](#api-reference)
11. [Development](#development)
12. [Contributing](#contributing)
13. [License](#license)

---

## Overview

**Mnemo** (named after the Greek goddess of memory) is an autonomous research agent that combines advanced perception capabilities with a unique "dream distillation" process. The agent can:

- 🔍 Search the web and academic databases
- 📚 Read and summarize research papers
- 🌐 Crawl documentation sites
- 🧠 Build and query a knowledge graph
- 💭 Reason about complex questions
- 😴 "Dream" to consolidate memories and generate insights
- 🔄 Self-evolve to improve its strategies over time

### The Science Behind Mnemo

Mnemo is inspired by cognitive science research on memory consolidation during sleep. Just as humans consolidate daily experiences into long-term memories during REM sleep, Mnemo processes its accumulated knowledge through periodic "dream cycles" to:

1. Compress redundant information
2. Discover unexpected connections
3. Generate novel insights
4. Strengthen important memories

---

## Key Features

### 1. Perceiver - Multi-Source Information Gathering

The Perceiver module handles all external information gathering:

| Source | Description |
|--------|-------------|
| **Web Search** | DuckDuckGo-powered web search with rate limiting |
| **Paper Reader** | arXiv and Semantic Scholar integration for academic papers |
| **Doc Crawler** | Configurable crawler for documentation sites |

### 2. Dream Engine - Memory Consolidation

```
┌─────────────────────────────────────────────────────────────┐
│                      Dream Cycle                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  ACCUMULATE │───▶│  CONDENSE   │───▶│ INTEGRATE   │     │
│  │  Memories   │    │  (Compress) │    │   (Link)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                             │               │
│                                             ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   STORE     │◀───│  SYNTHESIZE │◀───│  CONNECT    │   │
│  │  Insights   │    │  (Create)   │    │  (Relate)   │   │
│  └─────────────┘    └─────────────┘    └─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3. Knowledge Graph - Structured Memory

- **Entities**: Concepts, people, organizations, technologies, theories
- **Relations**: is_a, part_of, causes, enables, cites, related_to
- **Embeddings**: Semantic similarity search
- **Inference**: Path finding, transitive closure

### 4. Reasoning Engine - Logical Inference

Supports multiple reasoning modes:
- **Deductive**: General rules → Specific conclusions
- **Inductive**: Specific observations → General patterns
- **Abductive**: Observations → Best explanation
- **Analogical**: Similarity-based reasoning
- **Causal**: Cause and effect chains

### 5. Self-Evolution - Continuous Improvement

The agent learns from its own performance:
- **Strategy Optimization**: Genetic algorithms for search strategies
- **Performance Learning**: Feedback integration and pattern recognition
- **Adaptive Parameters**: Automatically adjusting based on success rates

---

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Mnemo Agent                                     │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                           User Interface                               │   │
│  │   CLI (Command Line)  │  Python API  │  Future: Web UI             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                        │
│                                      ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          Core Agent                                    │   │
│  │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │   │
│  │   │  Working   │  │   Dream    │  │  Stats &   │  │  Config    │   │   │
│  │   │  Memory    │  │ Scheduler  │  │  Logging   │  │  Manager   │   │   │
│  │   └────────────┘  └────────────┘  └────────────┘  └────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                        │
│         ┌────────────────────────────┼────────────────────────────┐          │
│         │                            │                            │          │
│         ▼                            ▼                            ▼          │
│  ┌─────────────┐           ┌─────────────┐            ┌─────────────┐   │
│  │  Perceiver  │           │   Dream     │            │   Reasoning  │   │
│  │  (Gather)   │           │   Engine    │            │   (Think)    │   │
│  └─────────────┘           └─────────────┘            └─────────────┘   │
│         │                            │                            │          │
│         └────────────────────────────┼────────────────────────────┘          │
│                                      ▼                                         │
│                        ┌─────────────────────────┐                           │
│                        │    Knowledge Graph      │                           │
│                        │    (Structured Memory)  │                           │
│                        └─────────────────────────┘                           │
│                                      │                                        │
│                                      ▼                                         │
│                        ┌─────────────────────────┐                           │
│                        │   Self-Evolution       │                           │
│                        │   (Learn & Improve)     │                           │
│                        └─────────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interactions

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Perceiver  │────▶│    Memory    │────▶│Knowledge Gr. │
│              │     │              │     │              │
│ • Web Search │     │ • Short-term │     │ • Entities   │
│ • Paper Rd.  │     │ • Priority   │     │ • Relations  │
│ • Doc Crawl  │     │ • Decay      │     │ • Embeddings │
└──────────────┘     └──────────────┘     └──────────────┘
                           │                      │
                           │                      │
                           ▼                      │
                    ┌──────────────┐              │
                    │   Dream     │◀─────────────┘
                    │   Engine    │
                    │              │
                    │ • Condense   │
                    │ • Integrate  │
                    │ • Synthesize │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Reasoning  │
                    │              │
                    │ • Deductive │
                    │ • Inductive │
                    │ • Chain of  │
                    │   Thought   │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │Self-Evolution│
                    │              │
                    │ • Strategies │
                    │ • Feedback  │
                    │ • Adapt     │
                    └──────────────┘
```

---

## How Dream Distillation Works

Dream distillation is Mnemo's unique approach to memory consolidation. Here's how it works:

### The Dream Cycle

```
                    ┌─────────────────┐
                    │   SLEEP CYCLE   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Phase 1:      │ │   Phase 2:      │ │   Phase 3:      │
│   CONDENSATION  │ │   INTEGRATION   │ │   SYNTHESIS     │
│                 │ │                 │ │                 │
│ • Select recent │ │ • Compare new   │ │ • Find patterns │
│   memories     │ │   abstractions  │ │ • Generate     │
│ • Extract key  │ │   with existing │ │   hypotheses   │
│   entities     │ │   knowledge    │ │ • Create novel │
│ • Generate     │ │ • Find hidden  │ │   connections │
│   summaries    │ │   connections  │ │ • Produce     │
│ • Compress to  │ │ • Strengthen   │ │   insights   │
│   30% of orig │ │   weak links   │ │               │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    RESULTS     │
                    │               │
                    │ • Condensed   │
                    │   abstractions│
                    │ • New insights│
                    │ • Enhanced KG │
                    │ • Better      │
                    │   reasoning   │
                    └─────────────────┘
```

### Detailed Process

#### Phase 1: Condensation

1. **Memory Selection**: Select recent memories that haven't been processed
2. **Entity Extraction**: Identify named entities and key concepts
3. **Summarization**: Generate concise summaries (target: 30% of original size)
4. **Abstraction**: Create hierarchical representations at multiple levels

```python
# Example condensation
original_memories = [
    "Transformer models use self-attention mechanisms",
    "BERT is a transformer-based language model",
    "GPT uses transformer decoder architecture",
]

abstraction = Abstraction(
    summary="Transformers are a key architecture in modern NLP",
    entities=["Transformer", "BERT", "GPT"],
    concepts=["self-attention", "NLP", "language model"],
    importance=0.85
)
```

#### Phase 2: Integration

1. **Similarity Analysis**: Compare new abstractions with existing knowledge
2. **Connection Discovery**: Find semantic relationships
3. **Link Strengthening**: Reinforce connections between related concepts
4. **Cross-Referencing**: Build the knowledge graph

```python
# Example integration
connections = [
    Connection(
        source_id="transformer",
        target_id="attention",
        strength=0.9,
        type="related_to"
    ),
    Connection(
        source_id="bert",
        target_id="transformer",
        strength=0.95,
        type="is_a"
    ),
]
```

#### Phase 3: Synthesis

1. **Pattern Recognition**: Find recurring themes across abstractions
2. **Hypothesis Generation**: Create "what if" scenarios
3. **Analogy Making**: Connect concepts from different domains
4. **Insight Production**: Generate novel, actionable insights

```python
# Example synthesis
insights = [
    Insight(
        content="The attention mechanism may have applications beyond NLP",
        type="hypothesis",
        confidence=0.7,
        novelty=0.8
    ),
]
```

### Dream Scheduling

Dreams are triggered based on:
- **Time-based**: Every 60 minutes (configurable)
- **Memory-based**: When ≥10 new memories accumulated
- **Adaptive**: More frequent when insights are being generated

---

## Knowledge Graph Structure

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| `CONCEPT` | Abstract ideas | Intelligence, Learning |
| `PERSON` | Individuals | Geoffrey Hinton, Yann LeCun |
| `ORGANIZATION` | Companies/institutions | Google, OpenAI, MIT |
| `LOCATION` | Places | San Francisco, USA |
| `EVENT` | Happenings | NeurIPS 2023 |
| `DOCUMENT` | Papers, articles | "Attention Is All You Need" |
| `TECHNOLOGY` | Tools, techniques | PyTorch, Kubernetes |
| `THEORY` | Frameworks, models | Connectionism |
| `METHOD` | Algorithms, approaches | Backpropagation |
| `TERM` | Domain vocabulary | Gradient descent |

### Relation Types

| Type | Description | Example |
|------|-------------|---------|
| `is_a` | Categorization | "BERT is a model" |
| `part_of` | Containment | "Attention is part of Transformer" |
| `related_to` | General connection | "NLP related to linguistics" |
| `causes` | Causation | "Learning causes weight updates" |
| `enables` | Facilitation | "GPU enables deep learning" |
| `contradicts` | Opposition | "Batch norm contradicts dropout" |
| `supports` | Evidence | "Evidence supports theory" |
| `cites` | References | "Paper A cites Paper B" |
| `develops` | Extension | "BERT develops transformer" |
| `uses` | Application | "AlphaFold uses transformers" |

### Knowledge Graph Schema

```
┌─────────────────────────────────────────────────────────────┐
│                        entities                              │
├─────────────────────────────────────────────────────────────┤
│ id          │ TEXT PRIMARY KEY                              │
│ name        │ TEXT NOT NULL                                 │
│ entity_type │ TEXT NOT NULL                                 │
│ description │ TEXT                                          │
│ properties  │ JSON                                          │
│ created_at  │ REAL                                          │
│ updated_at  │ REAL                                          │
│ confidence  │ REAL                                          │
│ source      │ TEXT                                          │
│ in_degree   │ INTEGER                                       │
│ out_degree  │ INTEGER                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ 1:N
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        relations                             │
├─────────────────────────────────────────────────────────────┤
│ id             │ TEXT PRIMARY KEY                           │
│ source_id      │ TEXT (FK → entities.id)                    │
│ target_id      │ TEXT (FK → entities.id)                    │
│ relation_type  │ TEXT                                       │
│ properties     │ JSON                                       │
│ created_at     │ REAL                                       │
│ confidence     │ REAL                                       │
│ bidirectional  │ INTEGER                                    │
│ weight         │ REAL                                       │
└─────────────────────────────────────────────────────────────┘
```

### Query Capabilities

```python
# Search entities
results = graph.search("transformer", entity_types=[EntityType.TECHNOLOGY])

# Find paths
paths = graph.find_paths(source_id, target_id, max_length=5)

# Get neighbors
neighbors = graph.get_neighbors(entity_id, max_depth=2)

# Similarity search
similar = graph.find_similar(entity_id, limit=10)
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or poetry

### Install from Source

```bash
# Clone the repository
git clone https://github.com/moggan1337/Mnemo.git
cd Mnemo

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Install Dependencies Only

```bash
pip install requests beautifulsoup4 lxml networkx pydantic tenacity
pip install arxiv sqlalchemy duckduckgo-search sentence-transformers chromadb
```

---

## Quick Start

### Basic Research Query

```python
import asyncio
from mnemo import MnemoAgent

async def main():
    # Create agent
    agent = MnemoAgent()
    
    # Run research
    result = await agent.research(
        query="What are the latest advances in transformer models?",
        depth="deep"
    )
    
    print(result.summary)
    print(f"Confidence: {result.confidence}")

asyncio.run(main())
```

### CLI Usage

```bash
# Research query
mnemo research "What are the latest advances in AGI?" --depth deep

# Web search
mnemo search "machine learning transformers"

# Trigger dream
mnemo dream

# Interactive agent
mnemo agent
```

### Knowledge Graph Operations

```python
from mnemo.knowledge_graph import KnowledgeGraph, QueryEngine
from mnemo.knowledge_graph.graph import EntityType, RelationType

# Create graph
graph = KnowledgeGraph()

# Add entities
ai = graph.add_entity("Artificial Intelligence", EntityType.CONCEPT)
ml = graph.add_entity("Machine Learning", EntityType.CONCEPT)

# Add relation
graph.add_relation(ai.id, ml.id, RelationType.IS_A)

# Query
engine = QueryEngine(graph)
results = engine.find_entities("intelligence")
```

---

## Usage Examples

### Example 1: Deep Research on a Topic

```python
import asyncio
from mnemo import MnemoAgent

async def research_example():
    agent = MnemoAgent()
    
    # Deep research with multiple sources
    result = await agent.research(
        query="What are the key advances in few-shot learning?",
        depth="deep",
        sources=["web", "papers"],
        max_results=30
    )
    
    print(f"Found {len(result.sources)} sources")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nKey Findings:")
    for finding in result.key_findings:
        print(f"  • {finding}")
    
    return result

asyncio.run(research_example())
```

### Example 2: Triggering Dream Cycle

```python
import asyncio
from mnemo import MnemoAgent

async def dream_example():
    agent = MnemoAgent()
    
    # Add some memories first
    agent.memory.add("Transformer attention mechanism", 
                     MemoryType.SEMANTIC, 
                     priority=MemoryPriority.HIGH)
    agent.memory.add("BERT uses bidirectional transformers",
                     MemoryType.SEMANTIC)
    
    # Trigger dream
    result = await agent.dream()
    
    print(f"Processed {result['memories_processed']} memories")
    print(f"Generated {len(result['insights'])} insights")
    
    for insight in result['insights']:
        print(f"  💡 {insight}")

asyncio.run(dream_example())
```

### Example 3: Reasoning About a Question

```python
import asyncio
from mnemo import MnemoAgent
from mnemo.reasoning.engine import ReasoningType

async def reasoning_example():
    agent = MnemoAgent()
    
    # Add some knowledge
    agent.memory.add("All neural networks have weights", MemoryType.SEMANTIC)
    agent.memory.add("Transformers are neural networks", MemoryType.SEMANTIC)
    
    # Reason about it
    result = await agent.reason(
        question="What can we conclude about transformers?",
        method="chain_of_thought"
    )
    
    print("Chain of Thought:")
    for step in result["thoughts"]:
        print(f"  {step}")
    print(f"\nConclusion: {result['conclusion']}")
    print(f"Confidence: {result['confidence']:.2%}")

asyncio.run(reasoning_example())
```

### Example 4: Custom Knowledge Graph Query

```python
from mnemo.knowledge_graph import KnowledgeGraph, QueryEngine
from mnemo.knowledge_graph.graph import EntityType, RelationType

# Create and populate graph
graph = KnowledgeGraph()

# Add entities
transformer = graph.add_entity(
    "Transformer", 
    EntityType.TECHNOLOGY,
    description="Neural network architecture using self-attention"
)
attention = graph.add_entity(
    "Self-Attention", 
    EntityType.METHOD,
    description="Mechanism allowing nodes to attend to other nodes"
)

# Create relation
graph.add_relation(
    attention.id, 
    transformer.id, 
    RelationType.ENABLES,
    confidence=0.95
)

# Query the graph
engine = QueryEngine(graph)

# Find paths
path = engine.find_shortest_path(attention.id, transformer.id)
print(f"Path found: {[e.name for e in path]}")

# Get statistics
stats = engine.get_statistics()
print(f"Total entities: {stats['total_entities']}")
```

### Example 5: Continuous Dream Mode

```python
import asyncio
from mnemo import MnemoAgent

async def continuous_dreaming():
    agent = MnemoAgent()
    
    # Define memory provider
    def memory_provider():
        return agent.memory.dream_ready(min_memories=10)
    
    # Run continuous dreaming
    await agent.dream_engine.dream_continuous(
        memory_provider=memory_provider,
        interval_seconds=300  # Every 5 minutes
    )

# To stop: agent.dream_engine.stop()
```

---

## Configuration

### Configuration File (YAML)

```yaml
# ~/.mnemo/config.yaml
app_name: Mnemo
version: 0.1.0
debug: false
log_level: INFO

perceiver:
  max_concurrent_searches: 5
  search_timeout_seconds: 30
  max_papers_per_query: 20
  
dream_engine:
  dream_interval_minutes: 60
  min_memories_for_dream: 10
  compression_ratio: 0.3
  
knowledge_graph:
  storage_backend: sqlite
  db_path: ~/.mnemo/knowledge.db
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  
reasoning:
  inference_depth: 3
  confidence_threshold: 0.6
  
self_evolution:
  learning_rate: 0.01
  exploration_rate: 0.1
```

### Environment Variables

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export MNEMO_DEBUG=true
export MNEMO_LOG_LEVEL=DEBUG
```

### Programmatic Configuration

```python
from mnemo import MnemoConfig
from mnemo.core.config import PerceiverConfig, DreamEngineConfig

config = MnemoConfig(
    debug=True,
    perceiver=PerceiverConfig(
        max_concurrent_searches=10
    ),
    dream_engine=DreamEngineConfig(
        dream_interval_minutes=30,
        compression_ratio=0.25
    )
)

agent = MnemoAgent(config)
```

---

## API Reference

### MnemoAgent

```python
class MnemoAgent:
    async def research(
        query: str,
        depth: str = "medium",
        sources: list[str] = ["web", "papers"],
        max_results: int = 20
    ) -> ResearchResult
    
    async def dream(
        memories: list = None,
        enable_synthesis: bool = True
    ) -> dict
    
    async def reason(
        question: str,
        context: str = None,
        method: str = "chain_of_thought"
    ) -> dict
    
    async def evolve() -> dict
    
    async def start() -> None
    async def stop() -> None
```

### KnowledgeGraph

```python
class KnowledgeGraph:
    def add_entity(
        name: str,
        entity_type: EntityType,
        description: str = "",
        properties: dict = None
    ) -> Entity
    
    def add_relation(
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        confidence: float = 1.0
    ) -> Relation
    
    def search(query: str, limit: int = 20) -> list[Entity]
    def find_similar(entity_id: str, limit: int = 10) -> list
    def find_paths(source_id: str, target_id: str, max_length: int = 5) -> list
```

### WorkingMemory

```python
class WorkingMemory:
    def add(content: Any, memory_type: MemoryType, ...) -> Memory
    def get(memory_id: str) -> Optional[Memory]
    def search(query: str = None, memory_type: MemoryType = None, ...) -> list
    def associate(memory_id1: str, memory_id2: str) -> bool
    def dream_ready(min_memories: int = 10) -> list
```

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_mnemo.py -v

# Run with coverage
pytest tests/ --cov=mnemo --cov-report=html
```

### Code Style

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking
mypy src/
```

### Project Structure

```
Mnemo/
├── src/
│   └── mnemo/
│       ├── __init__.py
│       ├── cli.py              # Command-line interface
│       ├── core/               # Core agent components
│       │   ├── agent.py        # Main agent
│       │   ├── config.py       # Configuration
│       │   ├── memory.py       # Working memory
│       │   └── scheduler.py    # Dream scheduler
│       ├── perceiver/          # Information gathering
│       │   ├── base.py
│       │   ├── web_search.py
│       │   ├── paper_reader.py
│       │   └── doc_crawler.py
│       ├── knowledge_graph/    # Knowledge storage
│       │   ├── graph.py
│       │   ├── query_engine.py
│       │   └── storage.py
│       ├── dream_engine/       # Memory consolidation
│       │   ├── core.py
│       │   ├── condenser.py
│       │   ├── integrator.py
│       │   └── synthesizer.py
│       ├── reasoning/          # Logical inference
│       │   ├── engine.py
│       │   ├── chain.py
│       │   └── logic.py
│       └── self_evolution/     # Learning
│           ├── optimizer.py
│           └── learner.py
├── tests/
│   └── test_mnemo.py
├── docs/
│   └── logo.png
├── examples/
├── config/
├── README.md
├── LICENSE
└── pyproject.toml
```

---

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Run all tests before submitting

---

## Roadmap

- [ ] **v0.2.0**: Multi-agent collaboration
- [ ] **v0.3.0**: Persistent knowledge graph with Neo4j support
- [ ] **v0.4.0**: Web UI for agent interaction
- [ ] **v0.5.0**: Plugin system for custom perceivers
- [ ] **v1.0.0**: Production-ready release

---

## References

If you use Mnemo in your research, please cite:

```bibtex
@software{Mnemo,
  title = {Mnemo: Autonomous Research Agent with Dream Distillation},
  author = {Moggan},
  version = {0.1.0},
  year = {2024},
  url = {https://github.com/moggan1337/Mnemo}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with 🧠 by <a href="https://github.com/moggan1337">Moggan</a>
</p>
