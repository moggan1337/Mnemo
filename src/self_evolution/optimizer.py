"""
Strategy Optimizer for Self-Evolution.

Optimizes search and reasoning strategies based on
performance feedback.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchStrategy:
    """A search strategy with parameters."""
    name: str
    query_patterns: list[str]
    source_preferences: list[str]  # Order of preferred sources
    depth_preference: str  # shallow, medium, deep
    max_results: int
    filters: dict[str, Any]
    performance_score: float = 0.5
    sample_count: int = 0
    
    def mutate(self, rate: float = 0.2) -> "SearchStrategy":
        """Create a mutated copy of this strategy."""
        import copy
        
        new_strategy = copy.deepcopy(self)
        new_strategy.name = f"{self.name}_mutated"
        
        # Mutate query patterns
        if random.random() < rate:
            # Add, remove, or modify a pattern
            mutation_type = random.choice(["add", "remove", "modify"])
            
            if mutation_type == "add":
                new_strategy.query_patterns.append("new_pattern")
            elif mutation_type == "remove" and new_strategy.query_patterns:
                new_strategy.query_patterns.pop(random.randint(0, len(new_strategy.query_patterns) - 1))
            elif mutation_type == "modify" and new_strategy.query_patterns:
                idx = random.randint(0, len(new_strategy.query_patterns) - 1)
                new_strategy.query_patterns[idx] = f"modified_{new_strategy.query_patterns[idx]}"
        
        # Mutate source preferences
        if random.random() < rate:
            random.shuffle(new_strategy.source_preferences)
        
        # Mutate depth
        if random.random() < rate:
            depths = ["shallow", "medium", "deep"]
            new_strategy.depth_preference = random.choice(depths)
        
        # Mutate max results
        if random.random() < rate:
            new_strategy.max_results = max(5, min(50, self.max_results + random.randint(-5, 5)))
        
        return new_strategy
    
    def crossover(self, other: "SearchStrategy") -> "SearchStrategy":
        """Create offspring by crossing two strategies."""
        import copy
        
        child = copy.deepcopy(self)
        child.name = f"{self.name}_x_{other.name}"
        
        # Mix query patterns
        if random.random() < 0.5:
            child.query_patterns = self.query_patterns[:len(self.query_patterns)//2] + \
                                   other.query_patterns[len(other.query_patterns)//2:]
        
        # Mix source preferences
        if random.random() < 0.5:
            child.source_preferences = self.source_preferences
        
        # Mix depth
        if random.random() < 0.5:
            child.depth_preference = other.depth_preference
        
        return child
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "query_patterns": self.query_patterns,
            "source_preferences": self.source_preferences,
            "depth_preference": self.depth_preference,
            "max_results": self.max_results,
            "filters": self.filters,
            "performance_score": self.performance_score,
            "sample_count": self.sample_count,
        }


@dataclass
class PerformanceRecord:
    """Record of a strategy's performance."""
    strategy_name: str
    query: str
    success: bool
    results_count: int
    quality_score: float
    response_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class StrategyOptimizer:
    """
    Optimizes search strategies through evolutionary algorithms.
    
    The optimization process:
    1. Track performance of current strategies
    2. Generate variations through mutation/crossover
    3. Select best performers
    4. Update strategy parameters
    
    This allows the agent to automatically improve its
    search effectiveness over time.
    """
    
    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.3,
        selection_pressure: float = 0.7,
    ):
        """
        Initialize Strategy Optimizer.
        
        Args:
            population_size: Number of strategies in population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            selection_pressure: Selection pressure (0-1)
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_pressure = selection_pressure
        
        # Strategy population
        self._strategies: list[SearchStrategy] = []
        self._performance_history: list[PerformanceRecord] = []
        
        # Initialize with default strategies
        self._initialize_default_strategies()
        
        logger.info("StrategyOptimizer initialized")
    
    def _initialize_default_strategies(self) -> None:
        """Initialize with default search strategies."""
        defaults = [
            SearchStrategy(
                name="broad_search",
                query_patterns=["{topic}", "what is {topic}"],
                source_preferences=["web", "papers"],
                depth_preference="medium",
                max_results=20,
                filters={},
            ),
            SearchStrategy(
                name="deep_research",
                query_patterns=["{topic} research", "{topic} latest", "{topic} advances"],
                source_preferences=["papers", "docs", "web"],
                depth_preference="deep",
                max_results=30,
                filters={"relevance_threshold": 0.7},
            ),
            SearchStrategy(
                name="quick_scan",
                query_patterns=["{topic} overview", "{topic} summary"],
                source_preferences=["web"],
                depth_preference="shallow",
                max_results=10,
                filters={},
            ),
        ]
        
        self._strategies = defaults
    
    def record_performance(
        self,
        strategy_name: str,
        query: str,
        success: bool,
        results_count: int,
        quality_score: float,
        response_time: float,
    ) -> None:
        """Record the performance of a strategy."""
        record = PerformanceRecord(
            strategy_name=strategy_name,
            query=query,
            success=success,
            results_count=results_count,
            quality_score=quality_score,
            response_time=response_time,
        )
        
        self._performance_history.append(record)
        
        # Update strategy's performance score
        for strategy in self._strategies:
            if strategy.name == strategy_name:
                self._update_strategy_score(strategy)
                break
    
    def _update_strategy_score(self, strategy: SearchStrategy) -> None:
        """Update a strategy's performance score."""
        # Get recent records for this strategy
        records = [
            r for r in self._performance_history[-100:]
            if r.strategy_name == strategy.name
        ]
        
        if not records:
            return
        
        strategy.sample_count = len(records)
        
        # Calculate weighted score
        success_rate = sum(1 for r in records if r.success) / len(records)
        avg_quality = sum(r.quality_score for r in records) / len(records)
        avg_time = sum(r.response_time for r in records) / len(records)
        
        # Normalize time (faster is better)
        time_score = max(0, 1 - (avg_time / 60))  # Assume 60s is bad
        
        # Combine scores
        strategy.performance_score = (
            success_rate * 0.4 +
            avg_quality * 0.4 +
            time_score * 0.2
        )
    
    def select_strategy(self) -> SearchStrategy:
        """Select the best strategy based on performance."""
        if not self._strategies:
            raise ValueError("No strategies available")
        
        # Sort by performance
        sorted_strategies = sorted(
            self._strategies,
            key=lambda s: s.performance_score,
            reverse=True
        )
        
        # Selection with pressure
        # Higher pressure = more likely to pick top performers
        if random.random() < self.selection_pressure:
            # Pick from top performers
            top_n = max(1, len(sorted_strategies) // 3)
            return random.choice(sorted_strategies[:top_n])
        else:
            # Random selection
            return random.choice(self._strategies)
    
    def evolve(self) -> list[SearchStrategy]:
        """
        Evolve the strategy population.
        
        Returns:
            New generation of strategies
        """
        if len(self._strategies) < 2:
            return self._strategies.copy()
        
        # Sort by performance
        sorted_strategies = sorted(
            self._strategies,
            key=lambda s: s.performance_score,
            reverse=True
        )
        
        # Keep top performers
        keep_count = max(2, len(sorted_strategies) // 2)
        new_population = sorted_strategies[:keep_count]
        
        # Generate new strategies
        while len(new_population) < self.population_size:
            parent1 = self.select_strategy()
            parent2 = self.select_strategy()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = parent1.crossover(parent2)
            else:
                child = parent1.mutate(0)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = child.mutate(self.mutation_rate)
            
            new_population.append(child)
        
        self._strategies = new_population
        
        logger.info(
            f"Evolved to {len(new_population)} strategies, "
            f"best score: {new_population[0].performance_score:.3f}"
        )
        
        return new_population.copy()
    
    def get_best_strategy(self) -> Optional[SearchStrategy]:
        """Get the highest performing strategy."""
        if not self._strategies:
            return None
        
        return max(self._strategies, key=lambda s: s.performance_score)
    
    def get_strategy_by_name(self, name: str) -> Optional[SearchStrategy]:
        """Get a strategy by name."""
        for strategy in self._strategies:
            if strategy.name == name:
                return strategy
        return None
    
    def add_strategy(self, strategy: SearchStrategy) -> None:
        """Add a new strategy to the population."""
        self._strategies.append(strategy)
    
    @property
    def statistics(self) -> dict:
        """Get optimizer statistics."""
        if not self._strategies:
            return {"strategies": 0}
        
        scores = [s.performance_score for s in self._strategies]
        
        return {
            "population_size": len(self._strategies),
            "best_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "worst_score": min(scores),
            "total_records": len(self._performance_history),
            "best_strategy": self.get_best_strategy().name if self._strategies else None,
        }
