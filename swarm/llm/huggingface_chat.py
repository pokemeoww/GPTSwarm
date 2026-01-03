import asyncio
import os
from typing import List, Union, Optional
from dotenv import load_dotenv
import torch
#from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoTokenizer, AutoModelForCausalLM

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry
from icl_retrieval.utils.device_utils import get_device


load_dotenv()


@LLMRegistry.register('HuggingFaceChat')
class HuggingFaceChat(LLM):
    """
    HuggingFace LLM wrapper that uses transformers library for local model inference.
    Supports chat models with apply_chat_template.
    """

    def __init__(
        self, 
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize HuggingFace chat model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-chat-hf")
            device: Device to load model on. If None, auto-detects (MPS/CUDA/CPU)
            torch_dtype: Data type for model weights. If None, uses float16
            load_in_8bit: Whether to load model in 8-bit quantization
            load_in_4bit: Whether to load model in 4-bit quantization
        """
        self.model_name = model_name
        
        # Auto-detect device if not specified
        if device is None:
            # TODO: temp
            device = "cuda"
        self.device = device
        
        # Set default dtype
        if torch_dtype is None:
            # Use float16 for GPU, float32 for CPU
            if 'cuda' in device or 'mps' in device:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        
        logger.info(f"Loading HuggingFace model: {model_name}")
        logger.info(f"Device: {device}, dtype: {torch_dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        model_kwargs = {
            "torch_dtype": torch_dtype,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Always move to device manually (no accelerate needed)
        if not load_in_8bit and not load_in_4bit:
            self.model = self.model.to(device)
        
        self.model.eval()
        logger.info(f"Model loaded successfully on {device}")

    def _messages_to_prompt(self, messages: List[Message]) -> str:
        """
        Convert Message objects to prompt string using chat template.
        
        Args:
            messages: List of Message objects
            
        Returns:
            Formatted prompt string
        """
        # Convert Message objects to dict format for apply_chat_template
        chat_messages = []
        for msg in messages:
            chat_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Use apply_chat_template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                logger.warning(f"apply_chat_template failed: {e}, falling back to simple format")
        
        # Fallback: simple concatenation
        prompt = ""
        for msg in chat_messages:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
        
        prompt += "Assistant: "
        return prompt

    def _generate(
        self,
        messages: List[Message],
        max_tokens: int,
        temperature: float,
        num_comps: int = 1,
    ) -> Union[List[str], str]:
        """
        Internal generation method.
        
        Args:
            messages: List of Message objects
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_comps: Number of completions to generate
            
        Returns:
            Generated text(s)
        """
        print("In generate function")
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # Adjust based on model's max length
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        print("[passed]======= line170 ======= ")

        # 检查关键 token ID 是否有效
        vocab_size = self.model.config.vocab_size
        print(f"Model vocab_size: {vocab_size}")

        # 检查 pad_token_id 和 eos_token_id 是否小于 vocab_size
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        print(f"pad_token_id: {pad_token_id}, eos_token_id: {eos_token_id}")

        assert pad_token_id is None or pad_token_id < vocab_size, f"pad_token_id {pad_token_id} >= vocab_size {vocab_size}"
        assert eos_token_id is None or eos_token_id < vocab_size, f"eos_token_id {eos_token_id} >= vocab_size {vocab_size}"

        # 在 generate 调用前检查输入
        print("Inputs keys:", inputs.keys())
        print("Input IDs dtype:", inputs['input_ids'].dtype)
        print("Input IDs min/max:", inputs['input_ids'].min(), inputs['input_ids'].max())

        # 确保最大索引不越界
        if inputs['input_ids'].numel() > 0:
            assert inputs['input_ids'].max() < vocab_size, "Input contains token ID larger than vocab_size!"
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                num_return_sequences=num_comps,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        print("[passed]======= line181 ======= ")
        
        # Decode outputs
        # Use attention_mask to get actual prompt length (excluding padding)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            # 确保是 Python int，不是 torch.Tensor
            prompt_lens = attention_mask.sum(dim=1).cpu().numpy().tolist()  # 转换为 Python list
        else:
            prompt_lens = [inputs['input_ids'].size(1)] * num_comps
        
        print("[passed]======= line191 ======= ")

        responses = []
        for i in range(num_comps):
            # Slice from actual prompt end, not from padded length
            # 🔧 修复2：添加边界检查
            start_idx = int(prompt_lens[i])  # 确保是整数
            seq_len = outputs.size(1)  # 序列总长度
            
            if start_idx >= seq_len:
                # 如果没有生成任何新 token
                response = ""
            else:
                # 🔧 修复3：使用正确的切片语法
                gen_ids = outputs[i, start_idx:]
                response = self.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True
                )
            responses.append(response.strip())

        print("[passed]======= line212 ======= ")
        if num_comps == 1:
            return responses[0]
        return responses

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        Async generation method.
        
        Args:
            messages: List of Message objects or single string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_comps: Number of completions
            
        Returns:
            Generated text(s)
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate,
            messages,
            max_tokens,
            temperature,
            num_comps
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """
        Synchronous generation method.
        
        Args:
            messages: List of Message objects or single string
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_comps: Number of completions
            
        Returns:
            Generated text(s)
        """
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return self._generate(messages, max_tokens, temperature, num_comps)
