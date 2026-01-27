"""
Simple test cases for utils/llm_utils.py
Run this to verify the LLM utilities work correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.llm_utils import LocalLLM, get_llm, call_llm


def test_token_counting():
    """Test token counting functionality"""
    print("\n" + "="*60)
    print("TEST 1: Token Counting")
    print("="*60)

    llm = get_llm()
    test_text = "Hello, this is a test sentence."
    token_count = llm.count_tokens(test_text)

    print(f"Input text: '{test_text}'")
    print(f"Token count: {token_count}")
    assert token_count > 0, "Token count should be greater than 0"
    print("✓ Token counting works correctly!")


def test_simple_generation():
    """Test simple text generation"""
    print("\n" + "="*60)
    print("TEST 2: Simple Text Generation")
    print("="*60)

    llm = get_llm()
    prompt = "What is 2+2? Answer in one sentence."

    print(f"Prompt: '{prompt}'")
    print("Generating response...")

    response = llm.generate(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.7,
    )

    print(f"Response: '{response}'")
    assert len(response) > 0, "Response should not be empty"
    print("✓ Text generation works correctly!")


def test_instruction_formatting():
    """Test that instruction formatting works for Mistral"""
    print("\n" + "="*60)
    print("TEST 3: Instruction Formatting (Mistral)")
    print("="*60)

    llm = get_llm()
    prompt = "List three colors: red, blue, and"

    print(f"Prompt: '{prompt}'")
    print("Generating with chat template...")

    response = llm.generate(
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.3,  # Lower temperature for more deterministic output
        use_chat_template=True,
    )

    print(f"Response: '{response}'")
    assert len(response) > 0, "Response should not be empty"
    print("✓ Instruction formatting works correctly!")


def test_call_with_retry():
    """Test retry logic"""
    print("\n" + "="*60)
    print("TEST 4: Retry Logic")
    print("="*60)

    llm = get_llm()
    prompt = "Say 'Hello' in one word."

    print(f"Prompt: '{prompt}'")
    print("Calling with retry...")

    response = llm.call_with_retry(
        prompt=prompt,
        max_retries=3,
        max_new_tokens=20,
        temperature=0.5,
    )

    print(f"Response: '{response}'")
    assert len(response) > 0, "Response should not be empty"
    print("✓ Retry logic works correctly!")


def test_convenience_function():
    """Test the convenience call_llm function"""
    print("\n" + "="*60)
    print("TEST 5: Convenience Function (call_llm)")
    print("="*60)

    prompt = "What is the capital of France? Answer in one word."

    print(f"Prompt: '{prompt}'")
    print("Using call_llm convenience function...")

    response = call_llm(
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.3,
        max_retries=2,
    )

    print(f"Response: '{response}'")
    assert len(response) > 0, "Response should not be empty"
    print("✓ Convenience function works correctly!")


def test_medical_prompt():
    """Test with a medical-related prompt (similar to your use case)"""
    print("\n" + "="*60)
    print("TEST 6: Medical Prompt (Similar to Your Use Case)")
    print("="*60)

    llm = get_llm()
    prompt = """Extract the following information from this medical note:

Note: Patient has history of diabetes and hypertension. Allergies: penicillin. Medications: metformin, lisinopril.

Output format:
ALLERGIES: [list of allergies]
MEDICATIONS: [list of medications]"""

    print("Prompt: Medical note extraction")
    print("Generating response...")

    response = llm.generate(
        prompt=prompt,
        max_new_tokens=100,
        temperature=0.5,
    )

    print(f"Response:\n{response}")
    assert len(response) > 0, "Response should not be empty"
    print("✓ Medical prompt works correctly!")


def test_multiple_calls():
    """Test that multiple calls work without reinitializing"""
    print("\n" + "="*60)
    print("TEST 7: Multiple Calls (Reusing Model)")
    print("="*60)

    llm = get_llm()  # Should reuse existing instance

    prompts = [
        "Name one fruit:",
        "Name one vegetable:",
        "Name one animal:",
    ]

    print("Making multiple calls...")
    for i, prompt in enumerate(prompts, 1):
        response = llm.generate(
            prompt=prompt,
            max_new_tokens=10,
            temperature=0.5,
        )
        print(f"  {i}. Prompt: '{prompt}' -> Response: '{response.strip()}'")

    print("✓ Multiple calls work correctly!")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("STARTING LLM UTILS TESTS")
    print("="*60)
    print("\nNote: This will download the model on first run (if not cached).")
    print("Make sure you have GPU enabled in Colab or sufficient RAM for CPU mode.\n")

    try:
        test_token_counting()
        test_simple_generation()
        test_instruction_formatting()
        test_call_with_retry()
        test_convenience_function()
        test_medical_prompt()
        test_multiple_calls()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)

    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()

    if success:
        print("\n✅ All tests completed successfully!")
        print("\nYou can now use the LLM utilities in your notebook.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
