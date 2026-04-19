"""
Dream Scheduler for Mnemo.

Manages the timing and execution of dream cycles,
integrating with the overall agent architecture.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from collections import deque

logger = logging.getLogger(__name__)


class DreamPhase(Enum):
    """Phases of the dream cycle."""
    IDLE = "idle"
    ACCUMULATION = "accumulation"
    CONDENSATION = "condensation"
    INTEGRATION = "integration"
    SYNTHESIS = "synthesis"
    COMPLETE = "complete"


class SleepQuality(Enum):
    """Quality assessment of dream processing."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    NONE = "none"


@dataclass
class DreamCycle:
    """Represents a single dream cycle."""
    id: str
    started_at: float
    completed_at: Optional[float] = None
    phase: DreamPhase = DreamPhase.IDLE
    memories_processed: int = 0
    new_insights: int = 0
    connections_discovered: int = 0
    quality: SleepQuality = SleepQuality.NONE
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamSchedule:
    """Schedule configuration for dream cycles."""
    interval_minutes: int = 60
    min_memories: int = 10
    max_duration_seconds: int = 300
    enable_adaptive: bool = True
    min_interval_minutes: int = 15
    max_interval_minutes: int = 240


class DreamScheduler:
    """
    Manages dream cycle scheduling and execution.
    
    Features:
    - Fixed interval scheduling with adaptive adjustment
    - Memory-based triggering
    - Phase tracking and quality assessment
    - Integration with async/await patterns
    """
    
    def __init__(self, schedule: Optional[DreamSchedule] = None):
        self.schedule = schedule or DreamSchedule()
        
        # State
        self._current_cycle: Optional[DreamCycle] = None
        self._last_cycle_time: float = 0
        self._cycles: deque[DreamCycle] = deque(maxlen=100)
        self._is_running: bool = False
        self._is_paused: bool = False
        
        # Callbacks
        self._on_cycle_start: Optional[Callable] = None
        self._on_cycle_complete: Optional[Callable] = None
        self._on_phase_change: Optional[Callable] = None
        
        # Threading
        self._timer_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_cycles": 0,
            "failed_cycles": 0,
            "avg_duration": 0.0,
            "total_memories_processed": 0,
        }
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running
    
    @property
    def current_phase(self) -> DreamPhase:
        """Get current dream phase."""
        if self._current_cycle:
            return self._current_cycle.phase
        return DreamPhase.IDLE
    
    @property
    def statistics(self) -> dict:
        """Get scheduler statistics."""
        return {
            **self._stats,
            "is_running": self._is_running,
            "is_paused": self._is_paused,
            "cycles_completed": len(self._cycles),
            "time_since_last_cycle": time.time() - self._last_cycle_time,
        }
    
    def start(
        self,
        on_cycle_start: Optional[Callable] = None,
        on_cycle_complete: Optional[Callable] = None,
        on_phase_change: Optional[Callable] = None,
    ) -> None:
        """
        Start the dream scheduler.
        
        Args:
            on_cycle_start: Callback when a cycle starts
            on_cycle_complete: Callback when a cycle completes
            on_phase_change: Callback when phase changes
        """
        with self._lock:
            if self._is_running:
                logger.warning("Dream scheduler already running")
                return
            
            self._is_running = True
            self._is_paused = False
            
            # Register callbacks
            self._on_cycle_start = on_cycle_start
            self._on_cycle_complete = on_cycle_complete
            self._on_phase_change = on_phase_change
            
            # Start timer thread
            self._timer_thread = threading.Thread(
                target=self._timer_loop,
                daemon=True,
                name="DreamScheduler-Timer"
            )
            self._timer_thread.start()
            
            logger.info("Dream scheduler started")
    
    def stop(self) -> None:
        """Stop the dream scheduler."""
        with self._lock:
            self._is_running = False
            if self._timer_thread:
                self._timer_thread.join(timeout=5)
            logger.info("Dream scheduler stopped")
    
    def pause(self) -> None:
        """Pause the scheduler without stopping it."""
        self._is_paused = True
        logger.info("Dream scheduler paused")
    
    def resume(self) -> None:
        """Resume a paused scheduler."""
        self._is_paused = False
        logger.info("Dream scheduler resumed")
    
    def trigger_cycle(self, memory_count: int) -> Optional[DreamCycle]:
        """
        Manually trigger a dream cycle.
        
        Args:
            memory_count: Number of memories available for processing
            
        Returns:
            DreamCycle if triggered, None otherwise
        """
        if not self._is_running or self._is_paused:
            return None
        
        if memory_count < self.schedule.min_memories:
            logger.debug(
                f"Memory threshold not met: {memory_count} < "
                f"{self.schedule.min_memories}"
            )
            return None
        
        return self._create_cycle()
    
    def set_phase(self, phase: DreamPhase) -> None:
        """Update current dream phase."""
        if self._current_cycle:
            old_phase = self._current_cycle.phase
            self._current_cycle.phase = phase
            
            if self._on_phase_change and old_phase != phase:
                self._on_phase_change(self._current_cycle, old_phase, phase)
            
            logger.debug(f"Dream cycle phase: {old_phase.value} -> {phase.value}")
    
    def complete_cycle(
        self,
        memories_processed: int = 0,
        new_insights: int = 0,
        connections: int = 0,
        metadata: Optional[dict] = None,
    ) -> DreamCycle:
        """
        Mark the current dream cycle as complete.
        
        Args:
            memories_processed: Number of memories processed
            new_insights: New insights generated
            connections: New connections discovered
            metadata: Additional metadata
            
        Returns:
            The completed DreamCycle
        """
        if not self._current_cycle:
            raise RuntimeError("No active dream cycle to complete")
        
        cycle = self._current_cycle
        cycle.completed_at = time.time()
        cycle.duration_seconds = cycle.completed_at - cycle.started_at
        cycle.phase = DreamPhase.COMPLETE
        cycle.memories_processed = memories_processed
        cycle.new_insights = new_insights
        cycle.connections_discovered = connections
        cycle.quality = self._assess_quality(cycle)
        cycle.metadata = metadata or {}
        
        # Update statistics
        self._update_stats(cycle)
        self._cycles.append(cycle)
        self._last_cycle_time = time.time()
        self._current_cycle = None
        
        # Trigger callback
        if self._on_cycle_complete:
            self._on_cycle_complete(cycle)
        
        logger.info(
            f"Dream cycle completed: {memories_processed} memories, "
            f"{new_insights} insights, quality={cycle.quality.value}"
        )
        
        return cycle
    
    def get_next_scheduled_time(self) -> datetime:
        """Get the next scheduled dream time."""
        interval = self._calculate_interval()
        return datetime.fromtimestamp(
            self._last_cycle_time + interval * 60
        )
    
    def get_recent_cycles(self, count: int = 10) -> list[DreamCycle]:
        """Get recent dream cycles."""
        return list(self._cycles)[-count:]
    
    def _timer_loop(self) -> None:
        """Main timer loop running in separate thread."""
        while self._is_running:
            try:
                if not self._is_paused:
                    interval = self._calculate_interval()
                    sleep_time = interval * 60 - (time.time() - self._last_cycle_time)
                    
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 60))  # Check every minute
                    else:
                        # Time to trigger - signal will be picked up by main loop
                        logger.debug("Dream cycle timer elapsed")
                else:
                    time.sleep(10)  # Shorter sleep when paused
                    
            except Exception as e:
                logger.error(f"Error in dream scheduler timer: {e}")
                time.sleep(10)
    
    def _calculate_interval(self) -> int:
        """Calculate adaptive interval based on recent performance."""
        if not self.schedule.enable_adaptive:
            return self.schedule.interval_minutes
        
        recent = self.get_recent_cycles(5)
        if not recent:
            return self.schedule.interval_minutes
        
        # Calculate average quality
        avg_quality = sum(
            self._quality_score(c.quality) for c in recent
        ) / len(recent)
        
        # Adjust interval based on quality
        if avg_quality >= 0.8:
            # Excellent quality - can wait longer
            return min(
                self.schedule.max_interval_minutes,
                int(self.schedule.interval_minutes * 1.5)
            )
        elif avg_quality >= 0.5:
            return self.schedule.interval_minutes
        else:
            # Poor quality - dream more frequently
            return max(
                self.schedule.min_interval_minutes,
                int(self.schedule.interval_minutes * 0.5)
            )
    
    def _create_cycle(self) -> DreamCycle:
        """Create a new dream cycle."""
        import uuid
        
        cycle = DreamCycle(
            id=str(uuid.uuid4()),
            started_at=time.time(),
            phase=DreamPhase.ACCUMULATION,
        )
        
        self._current_cycle = cycle
        self._stats["total_cycles"] += 1
        
        if self._on_cycle_start:
            self._on_cycle_start(cycle)
        
        logger.info(f"Dream cycle started: {cycle.id}")
        return cycle
    
    def _update_stats(self, cycle: DreamCycle) -> None:
        """Update running statistics."""
        self._stats["total_memories_processed"] += cycle.memories_processed
        
        # Calculate rolling average duration
        total_duration = (
            self._stats["avg_duration"] * (self._stats["total_cycles"] - 1) +
            cycle.duration_seconds
        )
        self._stats["avg_duration"] = total_duration / self._stats["total_cycles"]
    
    def _assess_quality(self, cycle: DreamCycle) -> SleepQuality:
        """Assess the quality of a completed dream cycle."""
        # Quality based on outcomes
        score = 0.0
        
        # More memories processed = better
        if cycle.memories_processed >= 50:
            score += 0.3
        elif cycle.memories_processed >= 20:
            score += 0.2
        elif cycle.memories_processed >= 10:
            score += 0.1
        
        # Insights generated
        if cycle.new_insights >= 5:
            score += 0.3
        elif cycle.new_insights >= 2:
            score += 0.2
        elif cycle.new_insights >= 1:
            score += 0.1
        
        # Connections discovered
        if cycle.connections_discovered >= 10:
            score += 0.2
        elif cycle.connections_discovered >= 5:
            score += 0.15
        elif cycle.connections_discovered >= 1:
            score += 0.1
        
        # Duration efficiency (not too fast, not too slow)
        ideal_duration = 120  # 2 minutes
        duration_ratio = cycle.duration_seconds / ideal_duration
        if 0.5 <= duration_ratio <= 2.0:
            score += 0.2
        
        # Map score to quality
        if score >= 0.8:
            return SleepQuality.EXCELLENT
        elif score >= 0.6:
            return SleepQuality.GOOD
        elif score >= 0.4:
            return SleepQuality.FAIR
        else:
            return SleepQuality.POOR
    
    @staticmethod
    def _quality_score(quality: SleepQuality) -> float:
        """Convert quality enum to numeric score."""
        return {
            SleepQuality.EXCELLENT: 1.0,
            SleepQuality.GOOD: 0.75,
            SleepQuality.FAIR: 0.5,
            SleepQuality.POOR: 0.25,
            SleepQuality.NONE: 0.0,
        }.get(quality, 0.0)


class AsyncDreamScheduler(DreamScheduler):
    """Async-compatible dream scheduler."""
    
    async def start_async(
        self,
        on_cycle_start: Optional[Callable] = None,
        on_cycle_complete: Optional[Callable] = None,
        on_phase_change: Optional[Callable] = None,
    ) -> None:
        """Start scheduler with async callbacks."""
        self.start(on_cycle_start, on_cycle_complete, on_phase_change)
    
    async def wait_for_next_cycle(self, timeout: Optional[float] = None) -> DreamCycle:
        """Wait for the next dream cycle to complete."""
        start_time = time.time()
        
        while self._is_running:
            if self._current_cycle and self._current_cycle.phase == DreamPhase.COMPLETE:
                return self._current_cycle
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timed out waiting for dream cycle")
            
            await asyncio.sleep(0.1)
        
        raise RuntimeError("Scheduler stopped while waiting")
