#!/usr/bin/env python3
"""
Random Token Generation Utility
Used for performance testing to generate random token sequences
"""
import random
import logging
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenGeneratorConfig:
    """Configuration for token generator"""
    exclude_special_tokens: bool = True
    min_token_id: int = 0
    max_token_id: Optional[int] = None
    custom_excluded_ids: Set[int] = None
    
    def __post_init__(self):
        if self.custom_excluded_ids is None:
            self.custom_excluded_ids = set()

class TokenGenerator:
    """Random token generator"""
    
    def __init__(self, tokenizer, config: Optional[TokenGeneratorConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or TokenGeneratorConfig()
        
        # Initialize token information
        self._init_token_info()
    
    def _init_token_info(self):
        """Initialize token information"""
        # Get vocabulary size
        if hasattr(self.tokenizer, 'vocab_size'):
            self.vocab_size = self.tokenizer.vocab_size
        elif hasattr(self.tokenizer, 'get_vocab_size'):
            self.vocab_size = self.tokenizer.get_vocab_size()
        else:
            # Try to get via len
            try:
                self.vocab_size = len(self.tokenizer)
            except:
                self.vocab_size = 32000  # Default value
                logger.warning(f"Cannot determine vocab_size, using default: {self.vocab_size}")
        
        # Set maximum token ID
        if self.config.max_token_id is None:
            self.config.max_token_id = self.vocab_size - 1
        
        # Get special token IDs
        self.special_token_ids = self._get_special_token_ids()
        
        # Calculate valid token range
        self.valid_token_ids = self._get_valid_token_ids()
        
        logger.info(f"TokenGenerator initialized: vocab_size={self.vocab_size}, "
                   f"valid_tokens={len(self.valid_token_ids)}, "
                   f"special_tokens={len(self.special_token_ids)}")
    
    def _get_special_token_ids(self) -> Set[int]:
        """Get set of special token IDs"""
        special_ids = set()
        
        # Get special tokens from tokenizer
        special_tokens_map = getattr(self.tokenizer, 'special_tokens_map', {})
        
        # Process special token mappings
        for key, token in special_tokens_map.items():
            token_id = self._convert_to_token_id(token)
            if token_id is not None:
                special_ids.add(token_id)
        
        # Check common special token attributes
        common_special_attrs = [
            'bos_token', 'eos_token', 'pad_token', 'unk_token',
            'sep_token', 'cls_token', 'mask_token'
        ]
        
        for attr_name in common_special_attrs:
            token = getattr(self.tokenizer, attr_name, None)
            if token is not None:
                token_id = self._convert_to_token_id(token)
                if token_id is not None:
                    special_ids.add(token_id)
        
        # Add custom excluded tokens
        special_ids.update(self.config.custom_excluded_ids)
        
        return special_ids
    
    def _convert_to_token_id(self, token) -> Optional[int]:
        """Convert token to ID"""
        if isinstance(token, int):
            return token
        elif isinstance(token, str):
            try:
                return self.tokenizer.convert_tokens_to_ids(token)
            except:
                return None
        elif hasattr(token, 'content'):  # Handle special token objects
            try:
                return self.tokenizer.convert_tokens_to_ids(token.content)
            except:
                return None
        return None
    
    def _get_valid_token_ids(self) -> List[int]:
        """Get list of valid token IDs (excluding special tokens)"""
        if not self.config.exclude_special_tokens:
            # If not excluding special tokens, return all tokens
            return list(range(self.config.min_token_id, self.config.max_token_id + 1))
        
        # Exclude special tokens
        all_ids = set(range(self.config.min_token_id, self.config.max_token_id + 1))
        valid_ids = sorted(list(all_ids - self.special_token_ids))
        
        if not valid_ids:
            logger.warning("No valid tokens after exclusion, using all tokens")
            valid_ids = list(range(self.config.min_token_id, self.config.max_token_id + 1))
        
        return valid_ids
    
    def generate_tokens(self, num_tokens: int) -> List[int]:
        """
        Generate random token sequence
        
        Args:
            num_tokens: Number of tokens to generate
        
        Returns:
            List of token IDs
        """
        if not self.valid_token_ids:
            raise ValueError("No valid tokens available for generation")
        
        tokens = random.choices(self.valid_token_ids, k=num_tokens)
        
        logger.debug(f"Generated {num_tokens} random tokens from {len(self.valid_token_ids)} valid tokens")
        return tokens
    
    def generate_token_batch(self, batch_size: int, tokens_per_sample: int) -> List[List[int]]:
        """
        Generate batch of token sequences
        
        Args:
            batch_size: Batch size
            tokens_per_sample: Number of tokens per sample
        
        Returns:
            List of token ID batches
        """
        batch = []
        for i in range(batch_size):
            tokens = self.generate_tokens(tokens_per_sample)
            batch.append(tokens)
        
        logger.info(f"Generated {batch_size} samples, {tokens_per_sample} tokens each")
        return batch
    
    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs to text"""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def batch_to_text(self, batch_tokens: List[List[int]]) -> List[str]:
        """Batch convert token IDs to text"""
        texts = []
        for tokens in batch_tokens:
            text = self.tokens_to_text(tokens)
            texts.append(text)
        return texts
    
    def get_token_info(self) -> Dict[str, Any]:
        """Get token information statistics"""
        return {
            "vocab_size": self.vocab_size,
            "valid_token_count": len(self.valid_token_ids),
            "special_token_count": len(self.special_token_ids),
            "min_token_id": min(self.valid_token_ids) if self.valid_token_ids else 0,
            "max_token_id": max(self.valid_token_ids) if self.valid_token_ids else 0,
            "exclude_special": self.config.exclude_special_tokens
        }

def create_token_generator(tokenizer, **kwargs):
    """
    Convenience function to create token generator
    
    Args:
        tokenizer: Tokenizer object
        **kwargs: Parameters passed to TokenGeneratorConfig
    
    Returns:
        TokenGenerator instance
    """
    config = TokenGeneratorConfig(**kwargs)
    return TokenGenerator(tokenizer, config)

# Unit tests
if __name__ == "__main__":
    # Mock a simple tokenizer for testing
    class MockTokenizer:
        vocab_size = 50000
        special_tokens_map = {
            "bos_token": 0,
            "eos_token": 1,
            "pad_token": 2,
            "unk_token": 3
        }
        
        def convert_tokens_to_ids(self, token):
            if isinstance(token, str):
                # Simple mapping
                return hash(token) % 100
            return token
        
        def decode(self, tokens, skip_special_tokens=True):
            return f"Decoded text with {len(tokens)} tokens"
    
    # Test token generator
    tokenizer = MockTokenizer()
    generator = TokenGenerator(tokenizer)
    
    print("Token Info:", generator.get_token_info())
    
    # Generate some tokens
    tokens = generator.generate_tokens(10)
    print(f"Generated tokens: {tokens}")
    print(f"Decoded: {generator.tokens_to_text(tokens)}")
    
    # Generate batch
    batch = generator.generate_token_batch(3, 5)
    print(f"Generated batch: {batch}")