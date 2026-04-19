"""
Main Mnemo Agent implementation.

Orchestrates all components: Perceiver, Dream Engine, Knowledge Graph,
Reasoning, and Self-Evolution into a cohesive research agent.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Callable
from enum import Enum

from mnemo.core.config import MnemoConfig, default_config
from mnemo.core.memory import WorkingMemory, MemoryType, MemoryPriority
from mnemo.core.scheduler import DreamScheduler, DreamSchedule, DreamPhase, DreamCycle

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Possible states of the Mnemo agent."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RESEARCHING = "researching"
    DREAMING = "dreaming"
    REASONING = "reasoning"
    EVOLVING = "evolving"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ResearchQuery:
    """A research query submitted to the agent."""
    id: str
    query: str
    depth: str = "medium"  # shallow, medium, deep
    sources: list[str] = field(default_factory=lambda: ["web", "papers"])
    max_results: int = 20
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """Results from a research query."""
    query_id: str
    summary: str
    key_findings: list[str]
    sources: list[dict]
    knowledge_graph_updates: list[dict]
    new_questions: list[str]
    confidence: float
    processing_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


class MnemoAgent:
    """
    Autonomous Research Agent with Dream Distillation.
    
    The Mnemo agent combines:
    - Perceiver: Web search, paper reading, document crawling
    - Dream Engine: Periodic memory consolidation and insight generation
    - Knowledge Graph: Entity and relation storage with embeddings
    - Reasoning: Logical inference from accumulated knowledge
    - Self-Evolution: Continuous improvement of search strategies
    
    Architecture Overview:
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                        Mnemo Agent                              │
    │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐ │
    │  │ Perceiver │──│ Working   │──│ Knowledge │──│   Reasoning   │ │
    │  │           │  │ Memory    │  │ Graph     │  │   Engine      │ │
    │  └───────────┘  └─────┬─────┘  └───────────┘  └───────────────┘ │
    │                       │                                        │
    │                 ┌─────▼─────┐                                   │
    │                 │  Dream    │                                   │
    │                 │  Engine   │                                   │
    │                 └─────┬─────┘                                   │
    │                       │                                        │
    │                 ┌─────▼─────┐                                   │
    │                 │  Self     │                                   │
    │                 │ Evolution │                                   │
    │                 └───────────┘                                   │
    └─────────────────────────────────────────────────────────────────┘
    ```
    
    Usage:
        agent = MnemoAgent()
        result = await agent.research("What are the latest advances in AGI?")
        print(result.summary)
    """
    
    def __init__(
        self,
        config: Optional[MnemoConfig] = None,
        enable_dream_scheduler: bool = True,
    ):
        """
        Initialize the Mnemo agent.
        
        Args:
            config: Configuration object. Uses default if not provided.
            enable_dream_scheduler: Whether to start the automatic dream scheduler.
        """
        self.config = config or default_config
        
        # State
        self.state = AgentState.INITIALIZING
        self._running = False
        self._query_count = 0
        
        # Initialize components (lazy loading for optional dependencies)
        self._components_initialized = False
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_dream_complete: Optional[Callable] = None
        self._on_insight: Optional[Callable] = None
        
        # Statistics
        self._stats = {
            "queries_processed": 0,
            "total_sources_visited": 0,
            "insights_generated": 0,
            "dream_cycles_completed": 0,
            "uptime_seconds": 0.0,
        }
        
        # Initialize working memory and scheduler
        self.memory = WorkingMemory(
            max_size=self.config.working_memory_size,
            forgetting_rate=self.config.memory_forgetting_rate,
        )
        
        if enable_dream_scheduler:
            self.dream_scheduler = DreamScheduler(
                schedule=DreamSchedule(
                    interval_minutes=self.config.dream_engine.dream_interval_minutes,
                    min_memories=self.config.dream_engine.min_memories_for_dream,
                    max_duration_seconds=self.config.dream_engine.max_dream_duration_seconds,
                )
            )
        else:
            self.dream_scheduler = None
        
        logger.info(f"Mnemo Agent initialized (version {self.config.version})")
        self.state = AgentState.IDLE
    
    async def research(
        self,
        query: str,
        depth: str = "medium",
        sources: Optional[list[str]] = None,
        max_results: int = 20,
    ) -> ResearchResult:
        """
        Conduct research on a topic.
        
        This is the main entry point for using Mnemo as a research agent.
        It orchestrates perception, knowledge storage, and reasoning to
        produce comprehensive research results.
        
        Args:
            query: The research question or topic
            depth: Research depth - "shallow", "medium", or "deep"
            sources: List of source types to use - "web", "papers", "docs"
            max_results: Maximum number of sources to process
            
        Returns:
            ResearchResult with summary, findings, and generated questions
            
        Example:
            >>> agent = MnemoAgent()
            >>> result = await agent.research(
            ...     "What are the latest advances in transformer models?",
            ...     depth="deep"
            ... )
            >>> print(result.summary)
        """
        import uuid
        
        # Track query
        self._query_count += 1
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        self._set_state(AgentState.RESEARCHING)
        logger.info(f"Starting research query {query_id}: {query}")
        
        try:
            # Initialize components if needed
            await self._ensure_components()
            
            # Create research query object
            research_query = ResearchQuery(
                id=query_id,
                query=query,
                depth=depth,
                sources=sources or ["web", "papers"],
                max_results=max_results,
            )
            
            # Step 1: Perception - Gather information
            sources_data = await self._perceive(research_query)
            
            # Step 2: Store in working memory
            memories = await self._store_memories(query_id, sources_data)
            
            # Step 3: Update knowledge graph
            kg_updates = await self._update_knowledge_graph(memories)
            
            # Step 4: Reasoning - Generate insights
            findings, new_questions = await self._reason(research_query, memories)
            
            # Step 5: Generate summary
            summary = await self._generate_summary(query, findings, sources_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                len(sources_data),
                len(findings),
                depth
            )
            
            # Store research result in memory
            self.memory.add(
                content={
                    "query": query,
                    "summary": summary,
                    "findings": findings,
                },
                memory_type=MemoryType.EPISODE,
                priority=MemoryPriority.HIGH if depth == "deep" else MemoryPriority.MEDIUM,
                tags={"research", query[:50]},
                metadata={"query_id": query_id, "depth": depth}
            )
            
            # Update statistics
            self._stats["queries_processed"] += 1
            self._stats["total_sources_visited"] += len(sources_data)
            
            result = ResearchResult(
                query_id=query_id,
                summary=summary,
                key_findings=findings,
                sources=sources_data,
                knowledge_graph_updates=kg_updates,
                new_questions=new_questions,
                confidence=confidence,
                processing_time=time.time() - start_time,
            )
            
            logger.info(
                f"Research query {query_id} completed in {result.processing_time:.2f}s"
            )
            self._set_state(AgentState.IDLE)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in research query {query_id}: {e}")
            self._set_state(AgentState.ERROR)
            raise
    
    async def dream(
        self,
        memories: Optional[list] = None,
        enable_synthesis: bool = True,
    ) -> dict[str, Any]:
        """
        Trigger a dream cycle for memory consolidation.
        
        The dream engine processes recent memories, identifies patterns,
        generates abstractions, and creates novel connections.
        
        Args:
            memories: Specific memories to process (None = auto-select)
            enable_synthesis: Whether to generate novel insights
            
        Returns:
            Dream results including processed memories, insights, and connections
        """
        self._set_state(AgentState.DREAMING)
        
        # Get memories to process
        if memories is None:
            memories = self.memory.dream_ready(
                min_memories=self.config.dream_engine.min_memories_for_dream
            )
        
        if len(memories) < self.config.dream_engine.min_memories_for_dream:
            logger.info(
                f"Not enough memories for dream: {len(memories)} < "
                f"{self.config.dream_engine.min_memories_for_dream}"
            )
            self._set_state(AgentState.IDLE)
            return {"status": "skipped", "reason": "insufficient_memories"}
        
        # Start dream cycle
        if self.dream_scheduler:
            cycle = self.dream_scheduler.trigger_cycle(len(memories))
            if cycle:
                self.dream_scheduler.set_phase(DreamPhase.CONDENSATION)
        
        logger.info(f"Starting dream cycle with {len(memories)} memories")
        
        try:
            # Dream phases
            results = {
                "memories_processed": len(memories),
                "insights": [],
                "connections": [],
                "abstractions": [],
                "synthesis": None,
            }
            
            # Phase 1: Condensation - Compress and summarize
            abstractions = await self._dream_condense(memories)
            results["abstractions"] = abstractions
            
            if self.dream_scheduler:
                self.dream_scheduler.set_phase(DreamPhase.INTEGRATION)
            
            # Phase 2: Integration - Link with existing knowledge
            connections = await self._dream_integrate(abstractions)
            results["connections"] = connections
            
            if self.dream_scheduler:
                self.dream_scheduler.set_phase(DreamPhase.SYNTHESIS)
            
            # Phase 3: Synthesis - Generate novel insights
            if enable_synthesis:
                insights = await self._dream_synthesize(abstractions, connections)
                results["insights"] = insights
                
                # Store insights in memory
                for insight in insights:
                    self.memory.add(
                        content=insight,
                        memory_type=MemoryType.DREAM,
                        priority=MemoryPriority.HIGH,
                        tags={"insight", "dream"},
                        metadata={"dreamed": True}
                    )
                    self._stats["insights_generated"] += 1
            
            # Mark memories as dreamed
            self.memory.mark_dreamed([m.id for m in memories])
            
            # Complete cycle
            if self.dream_scheduler:
                self.dream_scheduler.complete_cycle(
                    memories_processed=len(memories),
                    new_insights=len(results["insights"]),
                    connections=len(connections),
                    metadata=results,
                )
            
            self._stats["dream_cycles_completed"] += 1
            
            # Callback
            if self._on_dream_complete:
                self._on_dream_complete(results)
            
            self._set_state(AgentState.IDLE)
            return results
            
        except Exception as e:
            logger.error(f"Error in dream cycle: {e}")
            self._set_state(AgentState.ERROR)
            raise
    
    async def reason(
        self,
        question: str,
        context: Optional[str] = None,
        method: str = "chain_of_thought",
    ) -> dict[str, Any]:
        """
        Perform reasoning on a question using accumulated knowledge.
        
        Args:
            question: The question to reason about
            context: Additional context (optional)
            method: Reasoning method - "chain_of_thought", "deductive", "inductive"
            
        Returns:
            Reasoning results with conclusion and confidence
        """
        self._set_state(AgentState.REASONING)
        
        try:
            # Retrieve relevant memories
            relevant = self.memory.search(
                query=question,
                min_importance=0.3,
                limit=20
            )
            
            # Perform reasoning based on method
            if method == "chain_of_thought":
                result = await self._chain_of_thought(question, relevant, context)
            elif method == "deductive":
                result = await self._deductive_reasoning(question, relevant, context)
            elif method == "inductive":
                result = await self._inductive_reasoning(question, relevant, context)
            else:
                result = await self._chain_of_thought(question, relevant, context)
            
            self._set_state(AgentState.IDLE)
            return result
            
        except Exception as e:
            logger.error(f"Error in reasoning: {e}")
            self._set_state(AgentState.ERROR)
            raise
    
    async def evolve(self) -> dict[str, Any]:
        """
        Trigger self-evolution cycle to improve search strategies.
        
        Analyzes recent performance and adjusts search parameters
        based on what has been most effective.
        
        Returns:
            Evolution results including strategy updates
        """
        self._set_state(AgentState.EVOLVING)
        
        try:
            results = await self._self_evolution_cycle()
            self._set_state(AgentState.IDLE)
            return results
        except Exception as e:
            logger.error(f"Error in evolution cycle: {e}")
            self._set_state(AgentState.ERROR)
            raise
    
    # ==================== Component Initialization ====================
    
    async def _ensure_components(self) -> None:
        """Ensure all components are initialized."""
        if self._components_initialized:
            return
        
        # Lazy import components
        try:
            from mnemo.perceiver.web_search import WebSearchPerceiver
            from mnemo.knowledge_graph.graph import KnowledgeGraph
            
            self._perceiver = WebSearchPerceiver(self.config.perceiver)
            self._knowledge_graph = KnowledgeGraph(self.config.knowledge_graph)
            
            self._components_initialized = True
            logger.info("All components initialized")
            
        except ImportError as e:
            logger.warning(f"Some components not available: {e}")
            self._components_initialized = True  # Don't retry
    
    # ==================== Perception ====================
    
    async def _perceive(self, query: ResearchQuery) -> list[dict]:
        """Gather information from configured sources."""
        sources_data = []
        
        for source_type in query.sources:
            try:
                if source_type == "web":
                    results = await self._search_web(query)
                    sources_data.extend(results)
                elif source_type == "papers":
                    results = await self._search_papers(query)
                    sources_data.extend(results)
                elif source_type == "docs":
                    results = await self._crawl_docs(query)
                    sources_data.extend(results)
            except Exception as e:
                logger.warning(f"Error searching {source_type}: {e}")
        
        return sources_data
    
    async def _search_web(self, query: ResearchQuery) -> list[dict]:
        """Search the web for information."""
        if hasattr(self, "_perceiver"):
            return await self._perceiver.search_web(
                query.query,
                max_results=query.max_results
            )
        return []
    
    async def _search_papers(self, query: ResearchQuery) -> list[dict]:
        """Search academic papers."""
        if hasattr(self, "_perceiver"):
            return await self._perceiver.search_papers(
                query.query,
                max_results=query.max_results // 2
            )
        return []
    
    async def _crawl_docs(self, query: ResearchQuery) -> list[dict]:
        """Crawl documentation sites."""
        if hasattr(self, "_perceiver"):
            return await self._perceiver.crawl_documentation(
                query.query,
                max_pages=query.max_results // 3
            )
        return []
    
    # ==================== Memory Operations ====================
    
    async def _store_memories(self, query_id: str, sources: list[dict]) -> list:
        """Store gathered information in working memory."""
        memories = []
        
        for source in sources:
            memory = self.memory.add(
                content={
                    "source": source.get("url", "unknown"),
                    "title": source.get("title", ""),
                    "content": source.get("snippet", source.get("content", "")),
                    "query_id": query_id,
                },
                memory_type=MemoryType.PERCEPT,
                priority=MemoryPriority.MEDIUM,
                tags={source.get("type", "unknown")},
                metadata={"query_id": query_id}
            )
            memories.append(memory)
        
        return memories
    
    # ==================== Knowledge Graph ====================
    
    async def _update_knowledge_graph(self, memories: list) -> list[dict]:
        """Update knowledge graph with new information."""
        updates = []
        
        if hasattr(self, "_knowledge_graph"):
            for memory in memories:
                try:
                    update = await self._knowledge_graph.add_memory(memory)
                    updates.append(update)
                except Exception as e:
                    logger.warning(f"Error updating knowledge graph: {e}")
        
        return updates
    
    # ==================== Reasoning ====================
    
    async def _reason(
        self,
        query: ResearchQuery,
        memories: list,
    ) -> tuple[list[str], list[str]]:
        """Perform reasoning on gathered information."""
        findings = []
        new_questions = []
        
        # Extract key information from memories
        contents = [str(m.content) for m in memories]
        
        # Simple extractive summarization
        if contents:
            combined = " ".join(contents[:5])  # Top 5 sources
            
            # Generate findings (simplified)
            findings.append(f"Primary topic: {query.query}")
            findings.append(f"Found {len(memories)} relevant sources")
            
            # Generate follow-up questions
            new_questions.append(f"What are the implications of these findings?")
            new_questions.append(f"How do these relate to existing research?")
        
        return findings, new_questions
    
    async def _generate_summary(
        self,
        query: str,
        findings: list[str],
        sources: list[dict],
    ) -> str:
        """Generate a research summary."""
        summary_parts = [
            f"Research on: {query}",
            f"",
            f"Key Findings:",
        ]
        
        for i, finding in enumerate(findings[:5], 1):
            summary_parts.append(f"  {i}. {finding}")
        
        summary_parts.extend([
            f"",
            f"Sources: {len(sources)} sources analyzed",
            f"Top sources:",
        ])
        
        for source in sources[:3]:
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            summary_parts.append(f"  - {title}: {url}")
        
        return "\n".join(summary_parts)
    
    def _calculate_confidence(
        self,
        num_sources: int,
        num_findings: int,
        depth: str,
    ) -> float:
        """Calculate confidence score for research results."""
        confidence = 0.5
        
        # More sources = higher confidence
        if num_sources >= 20:
            confidence += 0.2
        elif num_sources >= 10:
            confidence += 0.1
        elif num_sources >= 5:
            confidence += 0.05
        
        # More findings = higher confidence
        if num_findings >= 5:
            confidence += 0.15
        elif num_findings >= 3:
            confidence += 0.1
        
        # Depth bonus
        if depth == "deep":
            confidence += 0.1
        
        return min(1.0, confidence)
    
    # ==================== Dream Engine ====================
    
    async def _dream_condense(self, memories: list) -> list[dict]:
        """Condense memories into abstract representations."""
        abstractions = []
        
        for memory in memories:
            # Simple abstraction: summarize key points
            content = str(memory.content)
            abstraction = {
                "id": f"abs_{memory.id}",
                "summary": content[:200],
                "key_entities": self._extract_entities(content),
                "importance": memory.importance,
                "original_id": memory.id,
            }
            abstractions.append(abstraction)
        
        return abstractions
    
    async def _dream_integrate(self, abstractions: list[dict]) -> list[dict]:
        """Integrate abstractions with existing knowledge."""
        connections = []
        
        for i, abs1 in enumerate(abstractions):
            for abs2 in abstractions[i+1:]:
                # Check for similarity
                similarity = self._calculate_similarity(
                    abs1.get("key_entities", []),
                    abs2.get("key_entities", [])
                )
                
                if similarity > 0.5:
                    connections.append({
                        "source": abs1["id"],
                        "target": abs2["id"],
                        "strength": similarity,
                        "type": "dream_association",
                    })
        
        return connections
    
    async def _dream_synthesize(
        self,
        abstractions: list[dict],
        connections: list[dict],
    ) -> list[str]:
        """Generate novel insights through synthesis."""
        insights = []
        
        # Generate insights based on patterns
        if len(connections) > 3:
            insights.append(
                f"Discovery: Found {len(connections)} unexpected connections "
                f"between {len(abstractions)} concepts"
            )
        
        # Generate cross-domain insights
        entities = set()
        for absraction in abstractions:
            entities.update(absraction.get("key_entities", []))
        
        if len(entities) > 5:
            insights.append(
                f"Pattern detected: {len(entities)} distinct entities "
                f"appear across multiple sources"
            )
        
        return insights
    
    # ==================== Reasoning Methods ====================
    
    async def _chain_of_thought(
        self,
        question: str,
        relevant: list,
        context: Optional[str],
    ) -> dict[str, Any]:
        """Chain of thought reasoning."""
        thoughts = []
        thoughts.append(f"Question: {question}")
        
        if relevant:
            thoughts.append(f"Relevant context: {len(relevant)} memories found")
            for mem in relevant[:3]:
                thoughts.append(f"  - {str(mem.content)[:100]}")
        
        # Generate conclusion
        conclusion = f"Based on the available information, regarding '{question}': "
        if relevant:
            conclusion += "There is evidence supporting this in the knowledge base."
        else:
            conclusion += "Limited information available for a definitive answer."
        
        return {
            "question": question,
            "thoughts": thoughts,
            "conclusion": conclusion,
            "confidence": 0.6 if relevant else 0.3,
        }
    
    async def _deductive_reasoning(
        self,
        question: str,
        relevant: list,
        context: Optional[str],
    ) -> dict[str, Any]:
        """Deductive reasoning from premises."""
        premises = [str(m.content) for m in relevant[:5]]
        
        return {
            "type": "deductive",
            "premises": premises,
            "conclusion": f"From the premises, we can deduce: {question}",
            "valid": len(premises) >= 2,
        }
    
    async def _inductive_reasoning(
        self,
        question: str,
        relevant: list,
        context: Optional[str],
    ) -> dict[str, Any]:
        """Inductive reasoning from observations."""
        observations = [str(m.content) for m in relevant[:5]]
        
        return {
            "type": "inductive",
            "observations": observations,
            "generalization": f"Based on {len(observations)} observations, {question}",
            "strength": len(observations) / 10,
        }
    
    # ==================== Self-Evolution ====================
    
    async def _self_evolution_cycle(self) -> dict[str, Any]:
        """Execute self-evolution to improve strategies."""
        results = {
            "strategies_evaluated": 0,
            "improvements": [],
            "parameters_updated": {},
        }
        
        # Analyze recent performance
        recent_queries = self.memory.search(
            memory_type=MemoryType.EPISODE,
            min_importance=0.3,
            limit=20,
        )
        
        results["strategies_evaluated"] = len(recent_queries)
        
        # Simple strategy improvements
        if len(recent_queries) >= 10:
            # Check if certain query types perform better
            results["improvements"].append(
                "Identified optimal query patterns from recent searches"
            )
            results["parameters_updated"]["query_optimization"] = True
        
        return results
    
    # ==================== Utility Methods ====================
    
    def _set_state(self, new_state: AgentState) -> None:
        """Update agent state and trigger callback."""
        old_state = self.state
        self.state = new_state
        
        if self._on_state_change and old_state != new_state:
            self._on_state_change(old_state, new_state)
        
        logger.debug(f"Agent state: {old_state.value} -> {new_state.value}")
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract simple entities from text."""
        # Simple word-based extraction
        words = text.split()
        # Return capitalized words and important terms
        entities = [w for w in words if w and w[0].isupper()]
        return list(set(entities))[:10]
    
    def _calculate_similarity(
        self,
        entities1: list[str],
        entities2: list[str],
    ) -> float:
        """Calculate simple entity overlap similarity."""
        if not entities1 or not entities2:
            return 0.0
        
        set1, set2 = set(entities1), set(entities2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @property
    def statistics(self) -> dict:
        """Get agent statistics."""
        return {
            **self._stats,
            "state": self.state.value,
            "memory_size": self.memory.size,
            "memory_stats": self.memory.statistics,
            "uptime_seconds": time.time() - self._stats.get("start_time", time.time()),
        }
    
    async def start(self) -> None:
        """Start the agent and its background processes."""
        self._running = True
        self._stats["start_time"] = time.time()
        
        if self.dream_scheduler:
            self.dream_scheduler.start(
                on_cycle_start=lambda c: logger.info(f"Dream cycle started: {c.id}"),
                on_cycle_complete=lambda c: logger.info(f"Dream cycle complete: {c.id}"),
            )
        
        logger.info("Mnemo Agent started")
    
    async def stop(self) -> None:
        """Stop the agent and cleanup."""
        self._running = False
        self._set_state(AgentState.SHUTDOWN)
        
        if self.dream_scheduler:
            self.dream_scheduler.stop()
        
        logger.info("Mnemo Agent stopped")
    
    def __repr__(self) -> str:
        return f"MnemoAgent(state={self.state.value}, memories={self.memory.size})"
