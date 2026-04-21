#!/usr/bin/env python3
"""
Prompt Generator
Generate test prompts based on input_token_num
Supports multiple generation methods and configurations
"""

import random
import string
import json
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Union
from infinibench.common.prompt_data import get_template_names, get_topic_names
import logging

from infinibench.common.prompt_data import (
    PRESET_TEMPLATES,
    PRESET_TOPICS,
    DEFAULT_TEMPLATE_NAME,
    DEFAULT_TOPIC_NAME,
    DEFAULT_CHARS_PER_TOKEN,
    get_template,
    get_topics,
)

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Prompt Generator Class"""

    def __init__(
        self,
        method: str = "template",  # template, random, file, fixed
        template_name: str = "ai_qa",
        topic_name: str = "ai_ml",
        fixed_prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        tokenizer=None,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    ):
        """Initialize prompt generator"""
        self.method = method
        self.template_name = template_name
        self.topic_name = topic_name
        self.fixed_prompt = fixed_prompt
        self.prompt_file = prompt_file
        self.tokenizer = tokenizer
        self.chars_per_token = chars_per_token

        # Load templates and topics from common/prompt_data.py
        self.templates = get_template(template_name)
        self.topics = get_topics(topic_name)

        # Load prompts from file (if needed)
        self.file_prompts = []
        if method == "file" and prompt_file and Path(prompt_file).exists():
            self._load_prompts_from_file()

    def _load_prompts_from_file(self):
        """Load prompts from file"""
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()

                if self.prompt_file.endswith(".json"):
                    data = json.loads(content)
                    # Use the common parsing function
                    self.file_prompts = _parse_json_prompts(data)
                else:
                    # Text file: one prompt per line
                    self.file_prompts = [
                        line.strip() for line in content.split("\n") if line.strip()
                    ]

            logger.info(
                f"Loaded {len(self.file_prompts)} prompts from {self.prompt_file}"
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {self.prompt_file}: {e}")
            self.file_prompts = []
        except Exception as e:
            logger.error(f"Failed to load prompts from file: {e}")
            self.file_prompts = []

    def generate(self, token_num: int, prompt_id: int = 0) -> str:
        """Generate single prompt (compatible with original interface)"""
        return self.generate_prompt(token_num, prompt_id)

    def generate_prompt(self, token_num: int, prompt_id: int = 0) -> str:
        """Generate single prompt"""
        if self.method == "fixed" and self.fixed_prompt:
            # Use fixed prompt
            return self._adjust_length(self.fixed_prompt, token_num)

        elif self.method == "file" and self.file_prompts:
            # Select prompt from file
            prompt_idx = prompt_id % len(self.file_prompts)
            base_prompt = self.file_prompts[prompt_idx]
            return self._adjust_length(base_prompt, token_num)

        elif self.method == "random":
            # Random generation
            return self._generate_random_prompt(token_num)

        else:  # Default to template method
            # Select template and topic
            template_idx = prompt_id % len(self.templates)
            topic_idx = prompt_id % len(self.topics)

            template = self.templates[template_idx]
            topic = self.topics[topic_idx]

            # Generate base prompt
            base_prompt = template.format(topic=topic)

            # Adjust length
            return self._adjust_length(base_prompt, token_num)

    def _adjust_length(self, prompt: str, token_num: int) -> str:
        """Adjust prompt length to specified token count"""
        if self.tokenizer:
            # Use tokenizer for precise control
            try:
                tokens = self.tokenizer.encode(prompt)

                if len(tokens) >= token_num:
                    # Truncate
                    tokens = tokens[:token_num]
                    return self.tokenizer.decode(tokens, skip_special_tokens=True)
                else:
                    # Repeat until desired length is reached
                    repeat_count = (token_num + len(tokens) - 1) // len(tokens)
                    repeated_tokens = tokens * repeat_count
                    repeated_tokens = repeated_tokens[:token_num]
                    return self.tokenizer.decode(
                        repeated_tokens, skip_special_tokens=True
                    )
            except Exception as e:
                logger.warning(
                    f"Tokenizer length adjustment failed: {e}, using fallback"
                )
                # Fallback to character-level adjustment
                return self._adjust_length_fallback(prompt, token_num)
        else:
            # Fallback method
            return self._adjust_length_fallback(prompt, token_num)

    def _adjust_string_to_length(self, text: str, target_chars: int) -> str:
        """Generic method: adjust string to target character length"""
        if not text:
            # Return space padding or empty string
            return " " * target_chars if target_chars > 0 else ""

        current_length = len(text)

        if current_length >= target_chars:
            # Truncate to target length
            return text[:target_chars]
        else:
            # Calculate repeat count
            repeat_count = (target_chars + current_length - 1) // current_length
            repeated_text = text * repeat_count
            return repeated_text[:target_chars]

    def _adjust_length_fallback(self, prompt: str, token_num: int) -> str:
        """Fallback method: character-level length adjustment"""
        estimated_chars = token_num * self.chars_per_token
        return self._adjust_string_to_length(prompt, estimated_chars)

    def _generate_random_prompt(self, token_num: int) -> str:
        """Generate random prompt"""
        return _generate_random_prompt_common(
            token_num=token_num,
            tokenizer=self.tokenizer,
            vocab_size=getattr(self, "_vocab_size", 32000),
            exclude_special_tokens=getattr(self, "_exclude_special_tokens", True),
            chars_per_token=self.chars_per_token,
        )

    def _generate_random_text(self, token_num: int) -> str:
        """Generate random text"""
        total_chars = token_num * self.chars_per_token

        # Use letters, digits, and common punctuation
        chars = string.ascii_letters + string.digits + " .,!?;:\n-"

        # Generate base random text
        base_length = min(100, total_chars)
        random_text = "".join(random.choices(chars, k=base_length))

        # Use generic method to adjust to correct length
        return self._adjust_string_to_length(random_text, total_chars)

    def generate_prompts(self, num_prompts: int, token_num: int) -> List[str]:
        """Generate multiple prompts"""
        prompts = []
        for i in range(num_prompts):
            prompt = self.generate_prompt(token_num, i)

            # Add unique identifier
            unique_suffix = f" [Request {i+1}:{self._generate_unique_suffix()}]"

            # Ensure total length doesn't exceed token limit after adding suffix
            if self.tokenizer:
                try:
                    base_tokens = self.tokenizer.encode(prompt)
                    suffix_tokens = self.tokenizer.encode(unique_suffix)

                    if len(base_tokens) + len(suffix_tokens) <= token_num:
                        prompt += unique_suffix
                    else:
                        # Adjust base prompt length to accommodate suffix
                        adjusted_token_num = token_num - len(suffix_tokens)
                        if adjusted_token_num > 0:
                            prompt = (
                                self.generate_prompt(adjusted_token_num, i)
                                + unique_suffix
                            )
                except Exception as e:
                    logger.warning(f"Tokenizer suffix adjustment failed: {e}")
                    # Simple append without precise control
                    prompt += unique_suffix
            else:
                # Simple estimation
                suffix_chars = len(unique_suffix)
                base_chars = token_num * self.chars_per_token - suffix_chars
                if base_chars > 0:
                    prompt = prompt[:base_chars] + unique_suffix

            prompts.append(prompt)

        logger.info(f"Generated {len(prompts)} prompts using method: {self.method}")
        if prompts:
            logger.debug(f"First prompt preview: {prompts[0][:100]}...")

        return prompts

    def _generate_unique_suffix(self) -> str:
        """Generate unique identifier suffix"""
        return "".join(random.choices(string.ascii_letters + string.digits, k=6))


# ==================== Original Functions ====================


def create_prompt_generator(
    tokenizer=None, method: str = "random", **kwargs
) -> Callable[[int], str]:
    """
    Create prompt generator function

    New recommended usage: Create PromptGenerator class instance directly
    """
    logger.warning(
        "Using deprecated create_prompt_generator function. "
        "Consider using PromptGenerator class directly."
    )

    if method == "random":
        return _create_random_prompt_generator(tokenizer, **kwargs)
    elif method == "template":
        return _create_template_prompt_generator(**kwargs)
    else:
        raise ValueError(f"Unknown prompt generation method: {method}")


def _generate_random_prompt_common(
    token_num: int,
    tokenizer=None,
    vocab_size: int = 32000,
    exclude_special_tokens: bool = True,
    chars_per_token: int = 4,
) -> str:
    """
    Common random prompt generation function
    Shared by both new and legacy interfaces
    """
    if tokenizer:
        # Use tokenizer for precise control
        try:
            if exclude_special_tokens and hasattr(tokenizer, "special_tokens_map"):
                # Get special tokens
                special_tokens = tokenizer.special_tokens_map.values()
                special_token_ids = set(tokenizer.convert_tokens_to_ids(special_tokens))

                # Generate non-special tokens
                valid_token_ids = []
                for token_id in range(vocab_size):
                    if token_id not in special_token_ids:
                        valid_token_ids.append(token_id)

                if valid_token_ids:
                    token_ids = random.choices(valid_token_ids, k=token_num)
                else:
                    token_ids = random.choices(list(range(vocab_size)), k=token_num)
            else:
                # Simple random token ID selection
                token_ids = random.choices(list(range(vocab_size)), k=token_num)

            # Decode to text
            prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
            return prompt

        except Exception as e:
            # If tokenizer-based method fails, fall back to random text
            logger.warning(
                f"Tokenizer-based prompt generation failed: {e}. "
                f"Falling back to random text generation."
            )
            # Continue with character-level generation below

    # Character-level generation (fallback)
    total_chars = token_num * chars_per_token
    chars = string.ascii_letters + string.digits + " .,!?;:\n"
    random_text = "".join(random.choices(chars, k=total_chars))
    return random_text


def _create_random_prompt_generator(
    tokenizer=None,
    vocab_size: int = 32000,
    exclude_special_tokens: bool = True,
    **kwargs,
) -> Callable[[int], str]:
    """Create random token generator (compatible with original interface)"""

    # Extract parameters from kwargs
    chars_per_token = kwargs.get("chars_per_token", 4)

    def generate_random_prompt(token_num: int) -> str:
        """Generate a random prompt containing the specified number of tokens"""
        return _generate_random_prompt_common(
            token_num=token_num,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            exclude_special_tokens=exclude_special_tokens,
            chars_per_token=chars_per_token,
        )

    return generate_random_prompt


def _create_template_prompt_generator(**kwargs) -> Callable[[int], str]:
    """Create template prompt generator (compatible with original interface)"""
    templates = kwargs.get(
        "templates",
        [
            "Explain the concept of artificial intelligence in simple terms. ",
            "What are the main applications of machine learning in today's world? ",
            "Describe the history and development of deep learning. ",
            "Compare and contrast natural language processing with computer vision. ",
        ],
    )

    def generate_template_prompt(token_num: int) -> str:
        """Generate prompt using template"""
        # Select a template
        template = random.choice(templates)

        # Repeat template until desired length is reached
        estimated_template_tokens = 15
        repeat_count = max(1, token_num // estimated_template_tokens)

        prompt = template * repeat_count

        # Truncate to approximate length
        estimated_chars = token_num * 4  # Assume average 4 characters per token
        if len(prompt) > estimated_chars:
            prompt = prompt[:estimated_chars]

        return prompt

    return generate_template_prompt


# ==================== Helper Functions ====================


def create_prompt_generator_from_config(
    config: Dict[str, Any], tokenizer=None
) -> PromptGenerator:
    """Create PromptGenerator instance from configuration"""
    # Extract configuration parameters
    prompt_config = config.get("prompt_config", {})

    return PromptGenerator(
        method=prompt_config.get("method", "template"),
        template_name=prompt_config.get("template_name", DEFAULT_TEMPLATE_NAME),
        topic_name=prompt_config.get("topic_name", DEFAULT_TOPIC_NAME),
        fixed_prompt=prompt_config.get("fixed_prompt"),
        prompt_file=prompt_config.get("prompt_file"),
        tokenizer=tokenizer,
        chars_per_token=prompt_config.get("chars_per_token", DEFAULT_CHARS_PER_TOKEN),
    )


def _parse_json_prompts(data: Any) -> List[str]:
    """Parse prompt list from JSON data"""
    if isinstance(data, list):
        # Directly use each element in the list
        return [str(item) for item in data]
    elif isinstance(data, dict):
        # Simplified: only process string values in the dictionary
        prompts = []
        for value in data.values():
            if isinstance(value, str):
                prompts.append(value)
            elif isinstance(value, list):
                prompts.extend([str(v) for v in value if isinstance(v, str)])
        return prompts
    elif isinstance(data, str):
        # If JSON is a single string, use it directly
        return [data]
    else:
        # Other types are not supported
        logger.warning(f"Unsupported JSON data type for prompts: {type(data).__name__}")
        return []


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompt list from file (generic function)"""
    if not Path(file_path).exists():
        logger.error(f"Prompt file not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

            if file_path.endswith(".json"):
                data = json.loads(content)
                # Use the common parsing function
                return _parse_json_prompts(data)
            else:
                # Text file: one prompt per line
                return [line.strip() for line in content.split("\n") if line.strip()]

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Failed to load prompts from file {file_path}: {e}")

    return []
