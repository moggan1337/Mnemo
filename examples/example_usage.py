"""Example: Basic research with Mnemo."""

import asyncio
from mnemo import MnemoAgent


async def main():
    print("=" * 60)
    print("Mnemo Research Agent - Example")
    print("=" * 60)
    
    # Create agent
    agent = MnemoAgent()
    
    # Run research
    print("\n🔬 Conducting research on transformer models...")
    result = await agent.research(
        query="What are the latest advances in transformer models?",
        depth="medium",
        sources=["web"],
        max_results=10
    )
    
    print(f"\n✅ Research completed!")
    print(f"   Confidence: {result.confidence:.2%}")
    print(f"   Sources: {len(result.sources)}")
    print(f"   Time: {result.processing_time:.2f}s")
    
    print("\n" + "-" * 60)
    print("SUMMARY:")
    print("-" * 60)
    print(result.summary)
    
    # Show statistics
    print("\n" + "-" * 60)
    print("AGENT STATISTICS:")
    print("-" * 60)
    stats = agent.statistics
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
