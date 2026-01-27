"""
LLM utilities for calling local Hugging Face models.
Supports Mistral-7B-Instruct-v0.2 and other instruction-following models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Optional


class LocalLLM:
    """
    Local LLM wrapper for Hugging Face models.
    Designed to work with Mistral-7B-Instruct-v0.2 and similar instruction models.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the local LLM.

        Args:
            model_name: Hugging Face model identifier
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            load_in_4bit: Use 4-bit quantization (saves memory)
            load_in_8bit: Use 8-bit quantization (saves memory)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad_token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model loading kwargs
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
        }

        # Add quantization if requested
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            model_kwargs["quantization_config"] = quantization_config

        # Load model
        use_device_map = False
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            use_device_map = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Check if model was actually loaded with device_map (fallback check)
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            use_device_map = True

        # Create pipeline for text generation
        # If using device_map="auto", don't pass device to pipeline
        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model,
            "tokenizer": self.tokenizer,
        }

        # Only add device if not using device_map
        # When device_map is used, accelerate handles device placement
        if not use_device_map:
            pipeline_kwargs["device"] = 0 if self.device == "cuda" else -1

        # Use dtype instead of torch_dtype (newer API, but still supported)
        # dtype is optional and can be omitted if model already has correct dtype
        if self.device == "cuda":
            pipeline_kwargs["dtype"] = torch.float16

        self.generator = pipeline(**pipeline_kwargs)

        print("Model loaded successfully!")

    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        return len(self.tokenizer.encode(text))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        use_chat_template: bool = True,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, higher = more random)
            do_sample: Whether to use sampling
            use_chat_template: Whether to use the model's chat template (for instruction models)

        Returns:
            Generated text string
        """
        try:
            # Use chat template for instruction models like Mistral
            if use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback for models without chat template
                formatted_prompt = prompt

            # Generate response
            outputs = self.generator(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
            )

            # Extract generated text
            generated_text = outputs[0]['generated_text'].strip()

            # Clean up chat template artifacts if present
            if "</s>" in generated_text:
                generated_text = generated_text.split("</s>")[0].strip()

            return generated_text

        except Exception as e:
            print(f"Error in generate: {e}")
            return ""

    def call_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Call the LLM with retry logic.

        Args:
            prompt: Input prompt
            max_retries: Maximum number of retry attempts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text, or empty string if all retries fail
        """
        for attempt in range(max_retries):
            try:
                result = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                if result:  # If we got a response, return it
                    return result
            except Exception as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return ""
        return ""


# Global LLM instance (lazy loading)
_llm_instance = None


def get_llm(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    force_reload: bool = False,
    **kwargs
) -> LocalLLM:
    """
    Get or create the global LLM instance.

    Args:
        model_name: Model name to load
        force_reload: Force reload even if instance exists
        **kwargs: Additional arguments for LocalLLM initialization

    Returns:
        LocalLLM instance
    """
    global _llm_instance

    if _llm_instance is None or force_reload:
        _llm_instance = LocalLLM(model_name=model_name, **kwargs)

    return _llm_instance


def call_llm(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    max_retries: int = 3,
    model_name: Optional[str] = None,
) -> str:
    """
    Convenience function to call the LLM.

    Args:
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        model_name: Optional model name (uses default if not provided)

    Returns:
        Generated text
    """
    if model_name:
        llm = get_llm(model_name=model_name, force_reload=True)
    else:
        llm = get_llm()

    return llm.call_with_retry(
        prompt,
        max_retries=max_retries,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
