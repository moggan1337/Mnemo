"""
Configuration management for Mnemo Agent.

Provides centralized configuration with environment variable support,
defaults, and validation for all Mnemo components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PerceiverConfig:
    """Configuration for the Perceiver component."""
    
    max_concurrent_searches: int = 5
    search_timeout_seconds: int = 30
    max_papers_per_query: int = 20
    max_documents_crawl: int = 100
    crawl_depth: int = 3
    user_agent: str = "Mnemo-Research-Agent/0.1"
    cache_ttl_seconds: int = 3600
    retry_attempts: int = 3
    retry_delay_seconds: int = 2
    
    # Web search providers
    search_provider: str = "duckduckgo"  # or "arxiv", "both"
    
    # Paper database APIs
    use_arxiv: bool = True
    use_semantic_scholar: bool = False
    semantic_scholar_api_key: Optional[str] = None
    
    # Content extraction
    extract_tables: bool = True
    extract_figures: bool = False
    min_content_length: int = 100


@dataclass
class DreamEngineConfig:
    """Configuration for the Dream Engine component."""
    
    # Dream cycle timing
    dream_interval_minutes: int = 60
    min_memories_for_dream: int = 10
    max_memories_per_dream: int = 100
    
    # Condensation parameters
    compression_ratio: float = 0.3  # Target compression to 30% of original
    min_importance_threshold: float = 0.5
    abstraction_levels: int = 3
    
    # Memory consolidation
    consolidation_strength: float = 0.8
    forgetting_rate: float = 0.1
    
    # Dream quality
    enable_novel_connections: bool = True
    novelty_threshold: float = 0.6
    creative_blend_ratio: float = 0.2
    
    # Processing
    max_dream_duration_seconds: int = 300
    parallel_dream_workers: int = 2


@dataclass
class KnowledgeGraphConfig:
    """Configuration for the Knowledge Graph component."""
    
    # Storage
    storage_backend: str = "sqlite"  # or "neo4j", "memory"
    db_path: Path = field(default_factory=lambda: Path.home() / ".mnemo" / "knowledge.db")
    
    # Graph parameters
    max_node_degree: int = 100
    entity_similarity_threshold: float = 0.85
    relation_confidence_threshold: float = 0.7
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    use_cache: bool = True
    
    # Graph operations
    enable_transitive_closure: bool = True
    max_path_length: int = 5
    community_detection: bool = True
    
    # Query optimization
    index_enabled: bool = True
    query_timeout_seconds: int = 10


@dataclass
class ReasoningConfig:
    """Configuration for the Reasoning component."""
    
    # Inference engine
    inference_depth: int = 3
    max_hypotheses: int = 10
    confidence_threshold: float = 0.6
    
    # Logical reasoning
    enable_deductive: bool = True
    enable_inductive: bool = True
    enable_abductive: bool = True
    
    # Uncertainty handling
    use_bayesian: bool = True
    use_fuzzy: bool = False
    
    # Chain of thought
    enable_chain_of_thought: bool = True
    max_thought_steps: int = 10
    
    # Context window
    max_context_tokens: int = 4096
    relevance_threshold: float = 0.5


@dataclass
class SelfEvolutionConfig:
    """Configuration for the Self-Evolution component."""
    
    # Learning parameters
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    exploitation_rate: float = 0.9
    
    # Strategy adaptation
    adaptation_interval_minutes: int = 30
    min_samples_for_adaptation: int = 20
    strategy_patience: int = 5
    
    # Search strategy evolution
    mutate_search_queries: bool = True
    crossover_strategies: bool = True
    mutation_probability: float = 0.2
    
    # Performance tracking
    track_all_metrics: bool = True
    performance_window: int = 100
    
    # Feedback
    enable_feedback_loop: bool = True
    feedback_weight: float = 0.3


@dataclass
class MnemoConfig:
    """
    Main configuration class for Mnemo Agent.
    
    Aggregates all component configurations and provides
    application-level settings.
    """
    
    # Application settings
    app_name: str = "Mnemo"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Directories
    data_dir: Path = field(default_factory=lambda: Path.home() / ".mnemo")
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".mnemo" / "cache")
    models_dir: Path = field(default_factory=lambda: Path.home() / ".mnemo" / "models")
    
    # Component configurations
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    dream_engine: DreamEngineConfig = field(default_factory=DreamEngineConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    self_evolution: SelfEvolutionConfig = field(default_factory=SelfEvolutionConfig)
    
    # Working memory
    working_memory_size: int = 1000
    memory_forgetting_rate: float = 0.05
    
    # API settings (optional)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Ensure directories exist and apply environment variable overrides."""
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Apply configuration overrides from environment variables."""
        if openai_key := os.getenv("OPENAI_API_KEY"):
            self.openai_api_key = openai_key
        
        if anthropic_key := os.getenv("ANTHROPIC_API_KEY"):
            self.anthropic_api_key = anthropic_key
        
        if debug := os.getenv("MNEMO_DEBUG"):
            self.debug = debug.lower() in ("true", "1", "yes")
        
        if log_level := os.getenv("MNEMO_LOG_LEVEL"):
            self.log_level = log_level.upper()
    
    @classmethod
    def from_file(cls, path: Path) -> "MnemoConfig":
        """Load configuration from a YAML or JSON file."""
        import json
        import yaml
        
        with open(path) as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "app_name": self.app_name,
            "version": self.version,
            "debug": self.debug,
            "log_level": self.log_level,
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "models_dir": str(self.models_dir),
            "perceiver": self.perceiver.__dict__,
            "dream_engine": self.dream_engine.__dict__,
            "knowledge_graph": {
                k: str(v) if isinstance(v, Path) else v 
                for k, v in self.knowledge_graph.__dict__.items()
            },
            "reasoning": self.reasoning.__dict__,
            "self_evolution": self.self_evolution.__dict__,
        }
    
    def save(self, path: Path):
        """Save configuration to a file."""
        import json
        
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# Global default configuration
default_config = MnemoConfig()
