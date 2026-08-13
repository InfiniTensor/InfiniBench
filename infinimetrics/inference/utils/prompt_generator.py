# utils/prompt_generator.py (Extended Version)
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
import logging

logger = logging.getLogger(__name__)

# Preset prompt template library
PRESET_TEMPLATES = {
    "ai_qa": [
        "Explain the concept of {topic} in simple terms.",
        "What are the main applications of {topic} in today's world?",
        "Describe the history and development of {topic}.",
        "Compare and contrast {topic} with similar technologies.",
        "What are the ethical considerations surrounding {topic}?",
        "How does {topic} impact our daily lives?",
        "What are the future trends in {topic}?",
        "What are the key challenges in {topic} research?"
    ],
    
    "general_qa": [
        "Tell me about {topic}.",
        "What is {topic}?",
        "Can you explain {topic}?",
        "I need information about {topic}.",
        "Please provide details about {topic}.",
        "Help me understand {topic}."
    ],
    
    "technical": [
        "Discuss the technical implementation of {topic}.",
        "What are the algorithms used in {topic}?",
        "Explain the architecture of {topic} systems.",
        "What are the performance considerations for {topic}?",
        "Describe the scalability challenges in {topic}."
    ]
}

# Preset topic library
PRESET_TOPICS = {
    "ai_ml": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "neural networks",
        "transformers",
        "large language models",
        "generative AI"
    ],
    
    "tech": [
        "cloud computing",
        "blockchain technology",
        "quantum computing",
        "Internet of Things",
        "edge computing",
        "distributed systems",
        "cybersecurity",
        "databases",
        "software engineering"
    ],
    
    "science": [
        "climate change",
        "genetic engineering",
        "space exploration",
        "renewable energy",
        "quantum physics",
        "biotechnology",
        "nanotechnology"
    ]
}


class PromptGenerator:
    """Prompt Generator Class (New)"""
    
    def __init__(
        self,
        method: str = "template",  # template, random, file, fixed
        template_name: str = "ai_qa",
        topic_name: str = "ai_ml",
        fixed_prompt: Optional[str] = None,
        prompt_file: Optional[str] = None,
        tokenizer = None,
        chars_per_token: int = 4
    ):
        """
        Initialize prompt generator
        """
        self.method = method
        self.template_name = template_name
        self.topic_name = topic_name
        self.fixed_prompt = fixed_prompt
        self.prompt_file = prompt_file
        self.tokenizer = tokenizer
        self.chars_per_token = chars_per_token
        
        # Load templates and topics
        self.templates = PRESET_TEMPLATES.get(template_name, PRESET_TEMPLATES["ai_qa"])
        self.topics = PRESET_TOPICS.get(topic_name, PRESET_TOPICS["ai_ml"])
        
        # Load prompts from file (if needed)
        self.file_prompts = []
        if method == "file" and prompt_file and Path(prompt_file).exists():
            self._load_prompts_from_file()
    
    def _load_prompts_from_file(self):
        """Load prompts from file"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
                if self.prompt_file.endswith('.json'):
                    data = json.loads(content)
                    if isinstance(data, list):
                        self.file_prompts = [str(item) for item in data]
                    elif isinstance(data, dict):
                        # Try to extract all string values
                        for value in data.values():
                            if isinstance(value, str):
                                self.file_prompts.append(value)
                            elif isinstance(value, list):
                                self.file_prompts.extend([str(v) for v in value if isinstance(v, str)])
                else:
                    # Text file, one prompt per line
                    self.file_prompts = [line.strip() for line in content.split('\n') if line.strip()]
                    
            logger.info(f"Loaded {len(self.file_prompts)} prompts from {self.prompt_file}")
            
        except Exception as e:
            logger.error(f"Failed to load prompts from file: {e}")
            self.file_prompts = []
    
    def generate(self, token_num: int, prompt_id: int = 0) -> str:
        """Generate single prompt (compatible with original interface)"""
        return self.generate_prompt(token_num, prompt_id)
    
    def generate_prompt(self, token_num: int, prompt_id: int = 0) -> str:
        """
        Generate single prompt
        
        Args:
            token_num: Required number of tokens
            prompt_id: Prompt ID (for generating different prompts)
            
        Returns:
            Prompt text
        """
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
                    return self.tokenizer.decode(repeated_tokens, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Tokenizer length adjustment failed: {e}, using fallback")
                # Fallback to character-level adjustment
                return self._adjust_length_fallback(prompt, token_num)
        else:
            # Fallback method
            return self._adjust_length_fallback(prompt, token_num)
    
    def _adjust_length_fallback(self, prompt: str, token_num: int) -> str:
        """Fallback method: character-level length adjustment"""
        estimated_chars = token_num * self.chars_per_token
        
        if len(prompt) >= estimated_chars:
            # Truncate
            return prompt[:estimated_chars]
        else:
            # Repeat until desired length is reached
            repeat_count = (estimated_chars + len(prompt) - 1) // len(prompt)
            repeated_prompt = prompt * repeat_count
            return repeated_prompt[:estimated_chars]
    
    def _generate_random_prompt(self, token_num: int) -> str:
        """Generate random prompt"""
        if self.tokenizer:
            # Randomly select token IDs
            try:
                vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 32000
                token_ids = [random.randint(0, vocab_size-1) for _ in range(token_num)]
                return self.tokenizer.decode(token_ids, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Tokenizer random generation failed: {e}, using fallback")
                # Fallback to character-level generation
                return self._generate_random_text(token_num)
        else:
            # Fallback method
            return self._generate_random_text(token_num)
    
    def _generate_random_text(self, token_num: int) -> str:
        """Generate random text (fallback method)"""
        total_chars = token_num * self.chars_per_token
        
        # Use letters, digits, and common punctuation
        chars = string.ascii_letters + string.digits + ' .,!?;:\n-'
        random_text = ''.join(random.choices(chars, k=total_chars))
        
        return random_text
    
    def generate_prompts(self, num_prompts: int, token_num: int) -> List[str]:
        """
        Generate multiple prompts
        
        Args:
            num_prompts: Number of prompts to generate
            token_num: Token count for each prompt
            
        Returns:
            List of prompts
        """
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
                            prompt = self.generate_prompt(adjusted_token_num, i) + unique_suffix
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
        return ''.join(random.choices(string.ascii_letters + string.digits, k=6))


# ==================== Original Functions (Maintaining Compatibility) ====================

def create_prompt_generator(
    tokenizer=None,
    method: str = "random",
    **kwargs
) -> Callable[[int], str]:
    """
    Create prompt generator function (compatible with original interface)
    
    New recommended usage: Create PromptGenerator class instance directly
    
    Args:
        tokenizer: Optional tokenizer for precise token counting
        method: Generation method, "random" or "template"
        **kwargs: Additional parameters passed to specific generator

    Returns:
        Function: (token_num) -> prompt_text
    """
    logger.warning("Using deprecated create_prompt_generator function. "
                   "Consider using PromptGenerator class directly.")
    
    if method == "random":
        return _create_random_prompt_generator(tokenizer, **kwargs)
    elif method == "template":
        return _create_template_prompt_generator(**kwargs)
    else:
        raise ValueError(f"Unknown prompt generation method: {method}")


def _create_random_prompt_generator(
    tokenizer=None,
    vocab_size: int = 32000,
    exclude_special_tokens: bool = True,
    **kwargs
) -> Callable[[int], str]:
    """Create random token generator (compatible with original interface)"""
    
    # Extract parameters from kwargs
    chars_per_token = kwargs.get('chars_per_token', 4)

    def generate_random_prompt(token_num: int) -> str:
        """Generate random prompt with specified token count"""
        nonlocal tokenizer, vocab_size, exclude_special_tokens, chars_per_token

        if tokenizer is not None:
            # Use tokenizer for precise token control
            # Generate random token ID sequence
            try:
                if exclude_special_tokens and hasattr(tokenizer, 'special_tokens_map'):
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
                # If tokenizer method fails, fall back to random text generation
                logger.warning(f"Tokenizer-based prompt generation failed: {e}. "
                               f"Falling back to random text generation.")
                tokenizer = None  # Mark tokenizer as unavailable

        # Method 1: No tokenizer, generate random text (estimate token count)
        total_chars = token_num * chars_per_token

        # Generate random text
        chars = string.ascii_letters + string.digits + ' .,!?;:\n'
        random_text = ''.join(random.choices(chars, k=total_chars))

        return random_text

    return generate_random_prompt


def _create_template_prompt_generator(**kwargs) -> Callable[[int], str]:
    """Create template prompt generator (compatible with original interface)"""
    templates = kwargs.get('templates', [
        "Explain the concept of artificial intelligence in simple terms. ",
        "What are the main applications of machine learning in today's world? ",
        "Describe the history and development of deep learning. ",
        "Compare and contrast natural language processing with computer vision. ",
    ])

    def generate_template_prompt(token_num: int) -> str:
        """Generate prompt using template"""
        # Select a template
        template = random.choice(templates)

        # Repeat template until desired length is reached
        estimated_template_tokens = 15
        repeat_count = max(1, token_num // estimated_template_tokens)

        prompt = template * repeat_count

        # Truncate to approximate length (simple handling)
        estimated_chars = token_num * 4  # Assume average 4 characters per token
        if len(prompt) > estimated_chars:
            prompt = prompt[:estimated_chars]

        return prompt

    return generate_template_prompt


# ==================== New Helper Functions ====================

def create_prompt_generator_from_config(
    config: Dict[str, Any], 
    tokenizer=None
) -> PromptGenerator:
    """
    Create PromptGenerator instance from configuration (recommended new way)
    
    Args:
        config: Configuration dictionary containing prompt-related settings
        tokenizer: Optional tokenizer
        
    Returns:
        PromptGenerator instance
    """
    # Extract configuration parameters
    prompt_config = config.get("prompt_config", {})
    
    return PromptGenerator(
        method=prompt_config.get("method", "template"),
        template_name=prompt_config.get("template_name", "ai_qa"),
        topic_name=prompt_config.get("topic_name", "ai_ml"),
        fixed_prompt=prompt_config.get("fixed_prompt"),
        prompt_file=prompt_config.get("prompt_file"),
        tokenizer=tokenizer,
        chars_per_token=prompt_config.get("chars_per_token", 4)
    )


def load_prompts_from_file(file_path: str) -> List[str]:
    """Load prompt list from file (general function)"""
    if not Path(file_path).exists():
        logger.error(f"Prompt file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
            if file_path.endswith('.json'):
                data = json.loads(content)
                if isinstance(data, list):
                    return [str(item) for item in data]
                elif isinstance(data, dict):
                    # Try to extract all string values
                    prompts = []
                    for value in data.values():
                        if isinstance(value, str):
                            prompts.append(value)
                        elif isinstance(value, list):
                            prompts.extend([str(v) for v in value if isinstance(v, str)])
                    return prompts
            else:
                # Text file, one prompt per line
                return [line.strip() for line in content.split('\n') if line.strip()]
                
    except Exception as e:
        logger.error(f"Failed to load prompts from file {file_path}: {e}")
    
    return []


# ==================== Test Functions ====================

def test_prompt_generator():
    """Test prompt generator"""
    print("Testing prompt generators...")

    # Test original functions (maintain compatibility)
    print("\n1. Testing legacy functions:")
    
    # Test random generator (no tokenizer)
    print("\n   Random generator (no tokenizer):")
    random_gen = create_prompt_generator(method="random", chars_per_token=3)
    prompt1 = random_gen(10)  # Generate prompt with 10 tokens
    print(f"     Generated prompt (10 tokens): {prompt1[:50]}...")

    # Test template generator
    print("\n   Template generator:")
    template_gen = create_prompt_generator(method="template")
    prompt2 = template_gen(20)  # Generate prompt with 20 tokens
    print(f"     Generated prompt (20 tokens): {prompt2[:50]}...")

    # Test new PromptGenerator class
    print("\n2. Testing new PromptGenerator class:")
    
    config = {
        "prompt_config": {
            "method": "template",
            "template_name": "ai_qa",
            "topic_name": "ai_ml",
            "chars_per_token": 4
        }
    }
    
    generator = create_prompt_generator_from_config(config)
    
    # Generate single prompt
    single_prompt = generator.generate(15)
    print(f"   Single prompt (15 tokens): {single_prompt[:50]}...")
    
    # Generate multiple prompts
    prompts = generator.generate_prompts(3, 10)
    print(f"   Generated {len(prompts)} prompts:")
    for i, prompt in enumerate(prompts):
        print(f"     Prompt {i+1}: {prompt[:50]}...")

    print("\nPrompt generators test completed.")


if __name__ == "__main__":
    test_prompt_generator()