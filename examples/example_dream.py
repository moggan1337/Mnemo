"""Example: Dream Engine with Mnemo."""

import asyncio
from mnemo import MnemoAgent
from mnemo.core.memory import MemoryType, MemoryPriority


async def main():
    print("=" * 60)
    print("Mnemo Dream Engine - Example")
    print("=" * 60)
    
    agent = MnemoAgent()
    
    # Add memories to dream about
    print("\n📚 Adding memories...")
    
    memories = [
        ("Neural networks learn representations", MemoryPriority.HIGH),
        ("Deep learning enables feature extraction", MemoryPriority.HIGH),
        ("Transformers use self-attention", MemoryPriority.HIGH),
        ("BERT applies transformers bidirectionally", MemoryPriority.MEDIUM),
        ("GPT uses autoregressive modeling", MemoryPriority.MEDIUM),
        ("Attention mechanisms are computationally expensive", MemoryPriority.LOW),
        ("Efficient transformers aim to reduce complexity", MemoryPriority.LOW),
        ("Flash Attention is an optimized implementation", MemoryPriority.LOW),
        ("Sparse attention reduces quadratic complexity", MemoryPriority.LOW),
        ("Linear attention approximates softmax", MemoryPriority.LOW),
    ]
    
    for content, priority in memories:
        agent.memory.add(
            content=content,
            memory_type=MemoryType.SEMANTIC,
            priority=priority
        )
    
    print(f"   Added {len(memories)} memories")
    print(f"   Memory size: {agent.memory.size}")
    
    # Trigger dream
    print("\n😴 Starting dream cycle...")
    result = await agent.dream()
    
    print(f"\n✅ Dream completed!")
    print(f"   Memories processed: {result.get('memories_processed', 0)}")
    print(f"   Connections found: {len(result.get('connections', []))}")
    print(f"   Insights generated: {len(result.get('insights', []))}")
    
    if result.get('insights'):
        print("\n💡 Generated Insights:")
        for i, insight in enumerate(result['insights'], 1):
            print(f"   {i}. {insight}")
    
    if result.get('abstractions'):
        print(f"\n📝 Abstractions created: {len(result['abstractions'])}")


if __name__ == "__main__":
    asyncio.run(main())
