"""
Quick test for LLM utils - minimal test cases
Use this for a quick verification in notebooks or interactive sessions.
"""

from utils.llm_utils import get_llm, call_llm

# Quick test 1: Initialize and count tokens
print("Test 1: Token Counting")
llm = get_llm()
test_text = "Hello world"
token_count = llm.count_tokens(test_text)
print(f"  Text: '{test_text}' -> {token_count} tokens")
print("  ✓ Passed\n")

# Quick test 2: Simple generation
print("Test 2: Simple Generation")
response = call_llm(
    prompt="What is 2+2? Answer with just the number.",
    max_new_tokens=10,
    temperature=0.1,
)
print(f"  Response: '{response}'")
print("  ✓ Passed\n")

# Quick test 3: Instruction following
print("Test 3: Instruction Following")
response = llm.generate(
    prompt="List three primary colors separated by commas:",
    max_new_tokens=20,
    temperature=0.3,
)
print(f"  Response: '{response}'")
print("  ✓ Passed\n")

print("All quick tests passed! ✅")
