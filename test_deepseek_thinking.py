#!/usr/bin/env python3
"""
Comprehensive tests for DeepSeek V3.2 thinking mode implementation.

Tests:
1. Basic thinking mode - Verify reasoning_content is returned
2. Multi-turn conversation - Verify reasoning_content is handled correctly
3. Interleaved thinking with tools - Verify thinking → tool → thinking pattern
4. Error handling - Verify proper error handling

Run with: python test_deepseek_thinking.py
"""

import json
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_provider import get_deepseek_thinking_client, DeepSeekThinkingClient


pytestmark = [pytest.mark.integration, pytest.mark.live_llm]

# Test configuration
THINKING_BUDGET = 2048  # 2K tokens for reasoning during tests


def test_1_basic_thinking_mode():
    """Test 1: Basic thinking mode - Verify reasoning_content is returned."""
    print("\n" + "="*60)
    print("TEST 1: Basic Thinking Mode")
    print("="*60)

    client = get_deepseek_thinking_client()

    messages = [
        {"role": "user", "content": "Which is larger: 9.11 or 9.8? Think step by step."}
    ]

    print(f"Input: {messages[0]['content']}")

    result = client.invoke_thinking(
        messages=messages,
        thinking_budget=THINKING_BUDGET,
        max_tokens=500,
    )

    assert "error" not in result, result.get("error")

    print(f"\n📊 Token Usage:")
    if result["usage"]:
        print(f"  - Prompt tokens: {result['usage'].prompt_tokens}")
        print(f"  - Completion tokens: {result['usage'].completion_tokens}")
        print(f"  - Total tokens: {result['usage'].total_tokens}")

    print(f"\n🧠 Reasoning Content ({len(result['reasoning_content'])} chars):")
    print("-" * 40)
    print(result["reasoning_content"][:500])
    if len(result["reasoning_content"]) > 500:
        print(f"... (truncated, total {len(result['reasoning_content'])} chars)")

    print(f"\n💡 Final Answer ({len(result['content'])} chars):")
    print("-" * 40)
    print(result["content"])

    # Verify
    has_reasoning = len(result["reasoning_content"]) > 0
    has_answer = len(result["content"]) > 0
    has_tool_calls = result["tool_calls"] is None or len(result["tool_calls"]) == 0

    success = has_reasoning and has_answer and has_tool_calls

    print(f"\n✓ Test 1 Result: {'PASS' if success else 'FAIL'}")
    print(f"  - Has reasoning_content: {has_reasoning}")
    print(f"  - Has content: {has_answer}")
    print(f"  - No tool calls (expected): {has_tool_calls}")

    assert success


def test_2_multi_turn_conversation():
    """Test 2: Multi-turn conversation - Verify reasoning_content handling."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Turn Conversation with Thinking Mode")
    print("="*60)

    client = get_deepseek_thinking_client()

    # Turn 1
    print("\n--- Turn 1 ---")
    messages = [
        {"role": "user", "content": "What is 15 * 23?"}
    ]

    result1 = client.invoke_thinking(
        messages=messages,
        thinking_budget=THINKING_BUDGET,
        max_tokens=200,
    )

    print(f"Question: {messages[0]['content']}")
    print(f"Reasoning: {result1['reasoning_content'][:100]}...")
    print(f"Answer: {result1['content']}")

    assert "error" not in result1, result1.get("error")

    # Prepare for Turn 2 - IMPORTANT: Only pass content, not reasoning_content
    messages.append({
        "role": "assistant",
        "content": result1["content"],  # Only content, not reasoning_content
    })
    messages.append({
        "role": "user",
        "content": "Now what is 15 * 24?"
    })

    # Turn 2
    print("\n--- Turn 2 ---")
    print(f"Question: {messages[-1]['content']}")

    result2 = client.invoke_thinking(
        messages=messages,
        thinking_budget=THINKING_BUDGET,
        max_tokens=200,
    )

    print(f"Reasoning: {result2['reasoning_content'][:100]}...")
    print(f"Answer: {result2['content']}")

    # Verify
    has_reasoning_2 = len(result2["reasoning_content"]) > 0
    has_answer_2 = len(result2["content"]) > 0
    answer_correct = "360" in result2["content"]

    print(f"\n✓ Test 2 Result: {'PASS' if has_reasoning_2 and has_answer_2 and answer_correct else 'FAIL'}")
    print(f"  - Has reasoning_content: {has_reasoning_2}")
    print(f"  - Has content: {has_answer_2}")
    print(f"  - Answer is correct (360): {answer_correct}")

    assert has_reasoning_2 and has_answer_2 and answer_correct


def test_3_thinking_with_tools():
    """Test 3: Interleaved thinking with tool calls."""
    print("\n" + "="*60)
    print("TEST 3: Interleaved Thinking with Tool Calls")
    print("="*60)

    client = get_deepseek_thinking_client()

    # Define test tools
    def get_current_date() -> str:
        """Get the current date."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

    def get_weather(location: str, date: str) -> str:
        """Get mock weather information."""
        # Mock weather data
        weather_data = {
            "Hangzhou": "Cloudy, 7-13°C",
            "Beijing": "Sunny, 15-22°C",
            "Shanghai": "Rainy, 18-25°C",
        }
        return weather_data.get(location, "Unknown location")

    # Test with tools
    result = client.invoke_thinking_with_tools(
        user_message="What's the weather in Hangzhou today?",
        tools=[get_current_date, get_weather],
        max_turns=5,
        thinking_budget=THINKING_BUDGET,
        system_prompt="You are a helpful assistant. Always call the date tool first.",
    )

    print(f"Question: What's the weather in Hangzhou today?")
    print(f"\n📊 Full Trace:")

    for i, step in enumerate(result["full_trace"]):
        print(f"\n--- Step {i+1} ---")
        print(f"Reasoning: {step['reasoning'][:80]}..." if step['reasoning'] else "Reasoning: (empty)")
        if step['content']:
            print(f"Content: {step['content']}")
        if step['tool_calls']:
            print(f"Tool Calls: {len(step['tool_calls'])} function(s)")
            for tc in step['tool_calls']:
                print(f"  - {tc.function.name}({tc.function.arguments})")

    print(f"\n💡 Final Answer:")
    print(result["final_answer"])

    # Verify
    has_trace = len(result["full_trace"]) > 0
    has_final_answer = len(result["final_answer"]) > 0
    no_error = result["error"] is None
    used_date_tool = any("get_current_date" in str(step.get("tool_calls", "")) for step in result["full_trace"])
    used_weather_tool = any("get_weather" in str(step.get("tool_calls", "")) for step in result["full_trace"])

    print(f"\n✓ Test 3 Result: {'PASS' if has_trace and has_final_answer and no_error and used_date_tool and used_weather_tool else 'FAIL'}")
    print(f"  - Has trace: {has_trace}")
    print(f"  - Has final answer: {has_final_answer}")
    print(f"  - No errors: {no_error}")
    print(f"  - Used date tool: {used_date_tool}")
    print(f"  - Used weather tool: {used_weather_tool}")

    assert has_trace and has_final_answer and no_error and used_date_tool and used_weather_tool


def test_4_streaming_thinking_mode():
    """Test 4: Streaming mode with thinking."""
    print("\n" + "="*60)
    print("TEST 4: Streaming Thinking Mode")
    print("="*60)

    client = get_deepseek_thinking_client()

    messages = [
        {"role": "user", "content": "Count from 1 to 5 step by step."}
    ]

    print(f"Input: {messages[0]['content']}")

    result = client.invoke_thinking(
        messages=messages,
        thinking_budget=THINKING_BUDGET,
        max_tokens=300,
        stream=True,
    )

    assert "error" not in result, result.get("error")

    print(f"\n📊 Token Usage:")
    if result["usage"]:
        print(f"  - Prompt tokens: {result['usage'].prompt_tokens}")
        print(f"  - Completion tokens: {result['usage'].completion_tokens}")
        print(f"  - Total tokens: {result['usage'].total_tokens}")

    print(f"\n🧠 Reasoning Content ({len(result['reasoning_content'])} chars):")
    print(result["reasoning_content"])

    print(f"\n💡 Final Answer:")
    print(result["content"])

    # Verify streaming still returns both reasoning_content and content
    has_reasoning = len(result["reasoning_content"]) > 0
    has_answer = len(result["content"]) > 0
    contains_counting = any(str(i) in result["content"] for i in range(1, 6))

    print(f"\n✓ Test 4 Result: {'PASS' if has_reasoning and has_answer and contains_counting else 'FAIL'}")
    print(f"  - Has reasoning_content: {has_reasoning}")
    print(f"  - Has content: {has_answer}")
    print(f"  - Contains counting (1-5): {contains_counting}")

    assert has_reasoning and has_answer and contains_counting


def test_5_clear_reasoning_from_messages():
    """Test 5: Verify reasoning_content is properly cleared."""
    print("\n" + "="*60)
    print("TEST 5: Clear reasoning_content from Messages")
    print("="*60)

    client = get_deepseek_thinking_client()

    # Create messages with reasoning_content
    messages_with_reasoning = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "Answer 1", "reasoning_content": "Thinking for question 1"},
        {"role": "user", "content": "Second question"},
    ]

    print("Original messages:")
    for msg in messages_with_reasoning:
        print(f"  - {msg['role']}: content='{msg.get('content', '')}', "
              f"reasoning_content='{msg.get('reasoning_content', 'N/A')}'")

    # Clear reasoning_content
    cleaned = client.clear_reasoning_from_messages(messages_with_reasoning)

    print("\nCleaned messages:")
    for msg in cleaned:
        has_reasoning = "reasoning_content" in msg
        print(f"  - {msg['role']}: content='{msg.get('content', '')}', "
              f"has_reasoning={has_reasoning}")

    # Verify no reasoning_content remains
    has_any_reasoning = any("reasoning_content" in str(msg) for msg in cleaned)
    all_have_content = all("content" in msg for msg in cleaned if msg["role"] in ["user", "assistant"])

    print(f"\n✓ Test 5 Result: {'PASS' if not has_any_reasoning and all_have_content else 'FAIL'}")
    print(f"  - No reasoning_content remains: {not has_any_reasoning}")
    print(f"  - All messages have content: {all_have_content}")

    assert not has_any_reasoning and all_have_content


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("DeepSeek V3.2 Thinking Mode - Comprehensive Test Suite")
    print("="*70)
    print(f"Using SiliconFlow API with thinking_budget={THINKING_BUDGET}")
    print("="*70)

    tests = [
        ("Basic Thinking Mode", test_1_basic_thinking_mode),
        ("Multi-Turn Conversation", test_2_multi_turn_conversation),
        ("Interleaved Thinking with Tools", test_3_thinking_with_tools),
        ("Streaming Thinking Mode", test_4_streaming_thinking_mode),
        ("Clear reasoning_content", test_5_clear_reasoning_from_messages),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"\n❌ {name} raised exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 All tests passed! DeepSeek thinking mode is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    sys.exit(run_all_tests())
