"""
Dream Engine - Core implementation.

The Dream Engine is responsible for memory consolidation,
insight generation, and the "dreaming" process that helps
the agent process and understand information.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable
from enum import Enum

from mnemo.dream_engine.condenser import DreamCondenser
from mnemo.dream_engine.integrator import MemoryIntegrator
from mnemo.dream_engine.synthesizer import InsightSynthesizer

logger = logging.getLogger(__name__)


class DreamPhase(Enum):
    """Phases of dream processing."""
    CONDENSATION = "condensation"
    INTEGRATION = "integration"
    SYNTHESIS = "synthesis"
    FINALIZATION = "finalization"


@dataclass
class DreamResult:
    """Result of a dream cycle."""
    phase: DreamPhase
    memories_processed: int
    abstractions_created: int
    connections_discovered: int
    insights_generated: int
    duration_seconds: float
    quality_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class DreamEngine:
    """
    Dream Engine for Mnemo - Memory Consolidation and Insight Generation.
    
    The Dream Engine processes memories during "sleep" cycles, performing:
    
    1. **Condensation**: Compress and abstract memories into core concepts
    2. **Integration**: Link new abstractions with existing knowledge
    3. **Synthesis**: Generate novel insights from the combined information
    
    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                      Dream Engine                           │
    │  ┌───────────────┐    ┌───────────────┐    ┌────────────┐ │
    │  │  Condensation │───▶│  Integration  │───▶│  Synthesis │ │
    │  │   (Compress)   │    │    (Link)     │    │  (Create)  │ │
    │  └───────────────┘    └───────────────┘    └────────────┘ │
    │         │                    │                    │         │
    │         ▼                    ▼                    ▼         │
    │  ┌───────────────┐    ┌───────────────┐    ┌────────────┐ │
    │  │  Abstractions │    │  Connections  │    │  Insights  │ │
    │  │   (Summaries) │    │   (Relations)  │    │  (Novel)   │ │
    │  └───────────────┘    └───────────────┘    └────────────┘ │
    └─────────────────────────────────────────────────────────────┘
    ```
    
    Usage:
        engine = DreamEngine(config)
        result = await engine.dream(memories)
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        condenser: Optional[DreamCondenser] = None,
        integrator: Optional[MemoryIntegrator] = None,
        synthesizer: Optional[InsightSynthesizer] = None,
    ):
        """
        Initialize Dream Engine.
        
        Args:
            config: Configuration dictionary
            condenser: Custom condenser (or use default)
            integrator: Custom integrator (or use default)
            synthesizer: Custom synthesizer (or use default)
        """
        self.config = config or {}
        
        # Components
        compression_ratio = self.config.get("compression_ratio", 0.3)
        novelty_threshold = self.config.get("novelty_threshold", 0.6)
        
        self.condenser = condenser or DreamCondenser(
            compression_ratio=compression_ratio
        )
        self.integrator = integrator or MemoryIntegrator()
        self.synthesizer = synthesizer or InsightSynthesizer(
            novelty_threshold=novelty_threshold
        )
        
        # State
        self._is_running = False
        self._current_phase: Optional[DreamPhase] = None
        
        # Statistics
        self._stats = {
            "cycles_completed": 0,
            "memories_processed": 0,
            "insights_generated": 0,
            "connections_discovered": 0,
            "total_duration": 0.0,
        }
        
        # Callbacks
        self._on_phase_start: Optional[Callable] = None
        self._on_insight: Optional[Callable] = None
        self._on_complete: Optional[Callable] = None
        
        logger.info("DreamEngine initialized")
    
    async def dream(
        self,
        memories: list,
        enable_synthesis: bool = True,
        progress_callback: Optional[Callable] = None,
    ) -> DreamResult:
        """
        Execute a dream cycle on the given memories.
        
        This is the main entry point for dream processing.
        
        Args:
            memories: List of Memory objects to process
            enable_synthesis: Whether to generate novel insights
            progress_callback: Optional callback for progress updates
            
        Returns:
            DreamResult with processing statistics
        """
        start_time = time.time()
        self._is_running = True
        
        memories_processed = len(memories)
        abstractions_created = 0
        connections_discovered = 0
        insights_generated = 0
        
        metadata = {}
        
        try:
            # Phase 1: Condensation
            self._current_phase = DreamPhase.CONDENSATION
            if self._on_phase_start:
                self._on_phase_start(DreamPhase.CONDENSATION)
            
            abstractions = await self.condenser.condense(memories)
            abstractions_created = len(abstractions)
            
            if progress_callback:
                progress_callback(0.33, "Condensation complete")
            
            logger.info(
                f"Dream phase 1/3 complete: {abstractions_created} abstractions created"
            )
            
            # Phase 2: Integration
            self._current_phase = DreamPhase.INTEGRATION
            if self._on_phase_start:
                self._on_phase_start(DreamPhase.INTEGRATION)
            
            connections = await self.integrator.integrate(abstractions)
            connections_discovered = len(connections)
            
            if progress_callback:
                progress_callback(0.66, "Integration complete")
            
            logger.info(
                f"Dream phase 2/3 complete: {connections_discovered} connections found"
            )
            
            # Phase 3: Synthesis
            if enable_synthesis:
                self._current_phase = DreamPhase.SYNTHESIS
                if self._on_phase_start:
                    self._on_phase_start(DreamPhase.SYNTHESIS)
                
                insights = await self.synthesizer.synthesize(
                    abstractions, connections
                )
                insights_generated = len(insights)
                
                metadata["insights"] = insights
                
                for insight in insights:
                    if self._on_insight:
                        self._on_insight(insight)
                
                if progress_callback:
                    progress_callback(0.9, "Synthesis complete")
                
                logger.info(
                    f"Dream phase 3/3 complete: {insights_generated} insights generated"
                )
            
            # Finalization
            self._current_phase = DreamPhase.FINALIZATION
            
            # Calculate quality score
            quality_score = self._calculate_quality(
                memories_processed,
                abstractions_created,
                connections_discovered,
                insights_generated,
            )
            
        except Exception as e:
            logger.error(f"Error during dream cycle: {e}")
            quality_score = 0.0
            metadata["error"] = str(e)
        
        finally:
            self._is_running = False
            self._current_phase = None
        
        duration = time.time() - start_time
        
        # Update statistics
        self._update_stats(
            memories_processed,
            connections_discovered,
            insights_generated,
            duration,
        )
        
        result = DreamResult(
            phase=self._current_phase or DreamPhase.FINALIZATION,
            memories_processed=memories_processed,
            abstractions_created=abstractions_created,
            connections_discovered=connections_discovered,
            insights_generated=insights_generated,
            duration_seconds=duration,
            quality_score=quality_score,
            metadata=metadata,
        )
        
        if self._on_complete:
            self._on_complete(result)
        
        return result
    
    async def dream_continuous(
        self,
        memory_provider: Callable[[], list],
        interval_seconds: float = 60,
    ) -> None:
        """
        Run dream cycles continuously.
        
        Args:
            memory_provider: Callable that returns memories to process
            interval_seconds: Time between dream cycles
        """
        logger.info("Starting continuous dream mode")
        
        while self._is_running:
            try:
                memories = memory_provider()
                
                if len(memories) >= 10:  # Minimum threshold
                    await self.dream(memories)
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in continuous dream: {e}")
                await asyncio.sleep(5)  # Brief pause on error
        
        logger.info("Continuous dream mode stopped")
    
    def stop(self) -> None:
        """Stop the dream engine."""
        self._is_running = False
    
    def _calculate_quality(
        self,
        memories: int,
        abstractions: int,
        connections: int,
        insights: int,
    ) -> float:
        """Calculate the quality score of the dream cycle."""
        score = 0.0
        
        # Efficiency of condensation
        if memories > 0:
            condensation_ratio = abstractions / memories
            score += min(0.3, condensation_ratio * 0.3)
        
        # Connection discovery rate
        if abstractions > 1:
            connection_rate = connections / (abstractions * (abstractions - 1) / 2)
            score += min(0.3, connection_rate * 0.3)
        
        # Insight generation
        if connections > 0:
            insight_rate = insights / connections
            score += min(0.2, insight_rate * 0.2)
        
        # Overall activity bonus
        total_activity = memories + abstractions + connections + insights
        if total_activity > 50:
            score += 0.2
        elif total_activity > 20:
            score += 0.1
        
        return min(1.0, score)
    
    def _update_stats(
        self,
        memories: int,
        connections: int,
        insights: int,
        duration: float,
    ) -> None:
        """Update running statistics."""
        self._stats["cycles_completed"] += 1
        self._stats["memories_processed"] += memories
        self._stats["connections_discovered"] += connections
        self._stats["insights_generated"] += insights
        self._stats["total_duration"] += duration
    
    @property
    def statistics(self) -> dict:
        """Get Dream Engine statistics."""
        avg_duration = (
            self._stats["total_duration"] / self._stats["cycles_completed"]
            if self._stats["cycles_completed"] > 0 else 0
        )
        
        return {
            **self._stats,
            "avg_cycle_duration": avg_duration,
            "is_running": self._is_running,
            "current_phase": self._current_phase.value if self._current_phase else None,
        }
    
    @property
    def is_running(self) -> bool:
        """Check if dream engine is running."""
        return self._is_running
