from typing import Optional
from class_registry import ClassRegistry

from swarm.llm.llm import LLM
from swarm.utils.log import logger


class LLMRegistry:
    registry = ClassRegistry()
    _cache = {}  # Cache for model instances (singleton pattern)

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None:
            model_name = "gpt-4-1106-preview"

        # Check cache first
        if model_name in cls._cache:
            logger.info(f"Reusing cached LLM instance: {model_name}")
            return cls._cache[model_name]

        # Create new instance
        logger.info(f"Creating new LLM instance: {model_name}")
        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif model_name.startswith('hf:'):
            # HuggingFace model: format is "hf:model_name"
            # e.g., "hf:meta-llama/Llama-2-7b-chat-hf"
            hf_model_name = model_name[3:]  # Remove "hf:" prefix
            model = cls.registry.get('HuggingFaceChat', hf_model_name)
        else: # any version of GPTChat like "gpt-4-1106-preview"
            model = cls.registry.get('GPTChat', model_name)

        # Cache the instance
        cls._cache[model_name] = model
        return model
