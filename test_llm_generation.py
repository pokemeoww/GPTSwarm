#!/usr/bin/env python3
"""
Test script for HuggingFace LLM generation.
Tests the LLM with fixed crosswords prompts to debug output issues.

Usage:
    python3 -m poetry run python3 test_llm_generation.py --model "hf:Qwen/Qwen2.5-0.5B-Instruct"
"""

import asyncio
import argparse
from swarm.llm.llm_registry import LLMRegistry
from swarm.llm.format import Message


# Fixed crosswords prompt from the error log
CROSSWORDS_PROMPT = """Let's play a 5 x 5 mini crossword, where each word should have exactly 5 letters.

<current query>
Current Board:
_____
_____
_____
_____
_____

Unfilled:
h2. _____ -- An engine

v2. _____ -- An Indian antelope

h5. _____ -- To mock; to sneer

h3. _____ -- Pretentious; flowery

v3. _____ -- To intend; to plan; to devise; a nettle; to guess

v5. _____ -- Desiccator; more dry

v1. _____ -- To heap

v4. _____ -- A nozzle

h1. _____ -- An agendum; something to be done

h4. _____ -- A salon; a hall

Filled:

Changed:


Given the current status, list all possible answers for unfilled or changed words, and your confidence levels (certain/high/medium/low), using the format "h1. apple (medium)". Use "certain" cautiously and only when you are 100% sure this is the correct word. You can list more then one possible answer for each word.
"""


SIMPLE_PROMPT = "What is 2+2? Answer with just the number."


async def test_llm_generation(model_name: str, test_simple: bool = True):
    """Test LLM generation with various prompts."""
    
    print("=" * 80)
    print("Testing HuggingFace LLM Generation")
    print("=" * 80)
    print(f"Model: {model_name}")
    print()
    
    # Initialize LLM
    print("Loading model...")
    llm = LLMRegistry.get(model_name)
    print(f"✓ Model loaded successfully")
    print()
    
    # Test 1: Simple prompt (if enabled)
    if test_simple:
        print("=" * 80)
        print("Test 1: Simple Prompt")
        print("=" * 80)
        print(f"Prompt: {SIMPLE_PROMPT}")
        print()
        
        try:
            response = await llm.agen(
                [Message(role="user", content=SIMPLE_PROMPT)],
                temperature=0.0,
                max_tokens=50
            )
            print(f"✓ Response: {response}")
            print(f"✓ Response length: {len(response)} characters")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
    
    # Test 2: Crosswords prompt
    print("=" * 80)
    print("Test 2: Crosswords Prompt")
    print("=" * 80)
    print(f"Prompt length: {len(CROSSWORDS_PROMPT)} characters")
    print(f"First 200 chars: {CROSSWORDS_PROMPT[:200]}...")
    print()
    
    print("Generating with temperature=0.0, max_tokens=512...")
    try:
        response = await llm.agen(
            [Message(role="user", content=CROSSWORDS_PROMPT)],
            temperature=0.0,
            max_tokens=512
        )
        print(f"✓ Response received")
        print(f"✓ Response length: {len(response)} characters")
        print()
        print("Response content:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        print()
        
        # Check if output is all exclamation marks
        if response.strip() and all(c == '!' for c in response.strip()):
            print("⚠️  WARNING: Output is all exclamation marks!")
            print("⚠️  This indicates the model is too small or having issues.")
        elif not response.strip():
            print("⚠️  WARNING: Output is empty!")
        else:
            print("✓ Output looks reasonable")
            
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    print()
    
    # Test 3: Different temperature
    print("=" * 80)
    print("Test 3: Same Prompt with temperature=0.7")
    print("=" * 80)
    
    try:
        response = await llm.agen(
            [Message(role="user", content=CROSSWORDS_PROMPT)],
            temperature=0.7,
            max_tokens=512
        )
        print(f"✓ Response length: {len(response)} characters")
        print()
        print("Response content:")
        print("-" * 80)
        print(response[:500] + ("..." if len(response) > 500 else ""))
        print("-" * 80)
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    # Test 4: Very short max_tokens
    print("=" * 80)
    print("Test 4: Short Generation (max_tokens=50)")
    print("=" * 80)
    
    try:
        response = await llm.agen(
            [Message(role="user", content=CROSSWORDS_PROMPT)],
            temperature=0.0,
            max_tokens=50
        )
        print(f"✓ Response: {response}")
    except Exception as e:
        print(f"✗ Error: {e}")
    print()
    
    print("=" * 80)
    print("Testing Complete")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test HuggingFace LLM generation")
    parser.add_argument(
        "--model",
        type=str,
        default="hf:Qwen/Qwen2.5-0.5B-Instruct",
        help="Model name (e.g., hf:Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--skip-simple",
        action="store_true",
        help="Skip simple prompt test"
    )
    
    args = parser.parse_args()
    
    # Run async test
    asyncio.run(test_llm_generation(args.model, test_simple=not args.skip_simple))


if __name__ == "__main__":
    main()
