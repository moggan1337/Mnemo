#!/usr/bin/env python3
"""
Mnemo CLI - Command Line Interface.

Usage:
    mnemo research "your research question"
    mnemo search "search query"
    mnemo dream
    mnemo graph
    mnemo agent
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path

from mnemo import MnemoAgent, MnemoConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


async def cmd_research(args) -> None:
    """Run a research query."""
    config = MnemoConfig(debug=args.verbose)
    agent = MnemoAgent(config, enable_dream_scheduler=False)
    
    print(f"🔬 Researching: {args.query}")
    print("-" * 50)
    
    result = await agent.research(
        query=args.query,
        depth=args.depth,
        sources=args.sources.split(",") if args.sources else ["web"],
        max_results=args.max_results,
    )
    
    print(f"\n📊 Confidence: {result.confidence:.2f}")
    print(f"⏱️  Processing time: {result.processing_time:.2f}s")
    print(f"\n📄 Summary:\n{result.summary}")
    
    if result.new_questions:
        print(f"\n❓ Follow-up questions:")
        for q in result.new_questions:
            print(f"   - {q}")
    
    if args.stats:
        print(f"\n📈 Statistics:")
        for key, value in agent.statistics.items():
            print(f"   {key}: {value}")


async def cmd_search(args) -> None:
    """Search for information."""
    from mnemo.perceiver import WebSearchPerceiver
    
    config = MnemoConfig()
    perceiver = WebSearchPerceiver()
    
    print(f"🔍 Searching: {args.query}")
    print("-" * 50)
    
    result = await perceiver.search(args.query, max_results=args.max_results)
    
    print(f"Found {len(result.results)} results in {result.execution_time:.2f}s")
    print(f"Provider: {result.provider}")
    print()
    
    for i, source in enumerate(result.results, 1):
        print(f"{i}. {source.title}")
        print(f"   URL: {source.url}")
        print(f"   Content: {source.content[:200]}...")
        print()


async def cmd_dream(args) -> None:
    """Trigger a dream cycle."""
    config = MnemoConfig()
    agent = MnemoAgent(config)
    
    print("😴 Initiating dream cycle...")
    print("-" * 50)
    
    result = await agent.dream()
    
    if result.get("status") == "skipped":
        print(f"⚠️  Dream skipped: {result.get('reason')}")
        return
    
    print(f"✅ Dream cycle complete!")
    print(f"   Memories processed: {result.get('memories_processed', 0)}")
    print(f"   Insights generated: {len(result.get('insights', []))}")
    print(f"   Connections found: {len(result.get('connections', []))}")
    
    if result.get("insights"):
        print(f"\n💡 Insights:")
        for insight in result["insights"][:5]:
            print(f"   - {insight}")


async def cmd_graph(args) -> None:
    """Query the knowledge graph."""
    from mnemo.knowledge_graph import KnowledgeGraph, QueryEngine
    from mnemo.knowledge_graph.graph import EntityType
    
    config = MnemoConfig()
    graph = KnowledgeGraph()
    engine = QueryEngine(graph)
    
    if args.add:
        print(f"➕ Adding entity: {args.add}")
        entity = graph.add_entity(
            name=args.add,
            entity_type=EntityType.CONCEPT,
            description=args.description or "",
        )
        print(f"   Created: {entity.id}")
    
    elif args.query:
        print(f"🔎 Searching: {args.query}")
        results = engine.find_entities(args.query)
        
        for entity in results[:10]:
            print(f"\n   {entity.name} ({entity.entity_type.value})")
            print(f"   ID: {entity.id}")
            print(f"   Description: {entity.description[:100]}...")
    
    elif args.stats:
        stats = engine.get_statistics()
        print("📊 Knowledge Graph Statistics:")
        print(f"   Total entities: {stats['total_entities']}")
        print(f"   Total relations: {stats['total_relations']}")
        
        print("\n   Entities by type:")
        for et, count in stats.get("entity_types", {}).items():
            print(f"      {et}: {count}")


async def cmd_agent(args) -> None:
    """Run the agent interactively."""
    config = MnemoConfig(debug=args.verbose)
    agent = MnemoAgent(config)
    
    print("🚀 Mnemo Agent started")
    print("Type 'help' for commands, 'quit' to exit")
    print()
    
    await agent.start()
    
    try:
        while True:
            query = input("mnemo> ").strip()
            
            if query.lower() in ("quit", "exit", "q"):
                break
            
            if query.lower() == "help":
                print("Commands:")
                print("  research <query> - Conduct research")
                print("  dream           - Trigger dream cycle")
                print("  stats           - Show statistics")
                print("  help            - Show this help")
                print("  quit            - Exit")
                continue
            
            if query.lower() == "stats":
                for key, value in agent.statistics.items():
                    print(f"  {key}: {value}")
                continue
            
            if query.lower().startswith("research "):
                q = query[9:]
                print(f"Researching: {q}...")
                result = await agent.research(q)
                print(result.summary)
                continue
            
            if query.lower() == "dream":
                result = await agent.dream()
                print(f"Dream complete: {len(result.get('insights', []))} insights")
                continue
            
            if query:
                print("Unknown command. Type 'help' for commands.")
    
    finally:
        await agent.stop()
        print("👋 Agent stopped")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mnemo - Autonomous Research Agent with Dream Distillation"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Conduct research")
    research_parser.add_argument("query", help="Research query")
    research_parser.add_argument("--depth", choices=["shallow", "medium", "deep"], 
                                 default="medium", help="Research depth")
    research_parser.add_argument("--sources", default="web", help="Comma-separated sources")
    research_parser.add_argument("--max-results", type=int, default=20, help="Max results")
    research_parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the web")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--max-results", type=int, default=10, help="Max results")
    
    # Dream command
    subparsers.add_parser("dream", help="Trigger dream cycle")
    
    # Graph command
    graph_parser = subparsers.add_parser("graph", help="Knowledge graph operations")
    graph_parser.add_argument("--add", metavar="NAME", help="Add entity")
    graph_parser.add_argument("--query", metavar="TEXT", help="Search entities")
    graph_parser.add_argument("--description", help="Entity description")
    graph_parser.add_argument("--stats", action="store_true", help="Show statistics")
    
    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run interactive agent")
    agent_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    if args.command == "research":
        asyncio.run(cmd_research(args))
    elif args.command == "search":
        asyncio.run(cmd_search(args))
    elif args.command == "dream":
        asyncio.run(cmd_dream(args))
    elif args.command == "graph":
        asyncio.run(cmd_graph(args))
    elif args.command == "agent":
        asyncio.run(cmd_agent(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
