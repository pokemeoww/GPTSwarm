import asyncio
import os
from dataclasses import asdict
from typing import List, Union, Optional
from dotenv import load_dotenv
import random
import async_timeout
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import time
from typing import Dict, Any

from swarm.utils.log import logger
from swarm.llm.format import Message
from swarm.llm.price import cost_count
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry


LM_STUDIO_URL = "http://localhost:1234/v1"


load_dotenv()
if os.getenv("USE_QWEN_API", "false").lower() == "true":
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    if DASHSCOPE_API_KEY is None:
        raise ValueError("DASHSCOPE_API_KEY is not set in environment variables.")
    print("[GPTChat] Using Qwen model with Dashscope API key.")

OPENAI_API_KEYS=[os.getenv(f"OPENAI_API_KEY")]
for i in range(10):
    if os.getenv(f"OPENAI_API_KEY{i}"):
        OPENAI_API_KEYS.append(os.getenv(f"OPENAI_API_KEY{i}"))


def gpt_chat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return ''

    api_kwargs: Dict[str, Any]
    if model == "lmstudio":
        api_kwargs = dict(base_url=LM_STUDIO_URL)
    
    # TODO: for QWEN Support
    elif "qwen" in model.lower() and os.getenv("USE_QWEN_API", "false").lower() == "true":
        print("[DEBUG][QWEN] Using Qwen model with Dashscope API key.")
        api_kwargs = dict(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
    else:
        api_key = random.sample(OPENAI_API_KEYS, 1)[0]
        api_kwargs = dict(api_key=api_key)

    if os.getenv("USE_QWEN_API", "false").lower() == "true":
        client = OpenAI(**api_kwargs)
    else:
        #TODO: use vllm to host local model
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="",
        )

    formated_messages = [asdict(message) for message in messages]
    
    response = client.chat.completions.create(model=model,
    messages=formated_messages,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    n=num_comps)
    
    if num_comps == 1:
        cost_count(response, model)
        return response.choices[0].message.content

    cost_count(response, model)

    return [choice.message.content for choice in response.choices]


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def gpt_achat(
    model: str,
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
    return_cost=False,
) -> Union[List[str], str]:
    if messages[0].content == '$skip$':
        return '' 

    extra_body = {}

    api_kwargs: Dict[str, Any]
    if model == "lmstudio":
        api_kwargs = dict(base_url=LM_STUDIO_URL)

    # TODO: for QWEN/DS Support
    elif os.getenv("USE_QWEN_API", "false").lower() == "true":
        print("[DEBUG][QWEN] Using Qwen/DS model with Dashscope API key.")
        api_kwargs = dict(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        # Whether output reasoning
        if os.getenv("QWEN_API_OUTPUT_REASONING", "false").lower() == "true":
            extra_body = {"enable_thinking": True, "max_tokens_thinking": 50}
    else:
        api_key = random.sample(OPENAI_API_KEYS, 1)[0]
        api_kwargs = dict(api_key=api_key)

    #aclient = AsyncOpenAI(**api_kwargs)

    if os.getenv("USE_QWEN_API", "false").lower() == "true":
        print("[DEBUG][QWEN] Creating AsyncOpenAI client for Qwen model.")
        aclient = AsyncOpenAI(**api_kwargs)
    else:
        #TODO: use vllm to host local model
        aclient = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="",
    )

    formated_messages = [asdict(message) for message in messages]
    try:
        async with async_timeout.timeout(1000):
            response = await aclient.chat.completions.create(model=model,
            messages=formated_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            n=num_comps,
            extra_body=extra_body
            )
    except asyncio.TimeoutError:
        print('Timeout')
        raise TimeoutError("GPT Timeout")

    has_reasoning = os.getenv("QWEN_API_OUTPUT_REASONING", "false").lower() == "true"
    if num_comps == 1:
        thinking = ""
        if has_reasoning:
            thinking = response.choices[0].message.reasoning_content
        cost_count(response, model)
        return response.choices[0].message.content
    
    cost_count(response, model)

    thinking = []

    if has_reasoning:
        for choice in response.choices:
            message = choice.message
            if hasattr(message, 'reasoning_content') and message.reasoning_content is not None:
                thinking.append(message.reasoning_content)
            else:
                thinking.append("")

    return [choice.message.content for choice in response.choices]


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return await gpt_achat(self.model_name,
                               messages,
                               max_tokens,
                               temperature,
                               num_comps)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return gpt_chat(self.model_name,
                        messages, 
                        max_tokens,
                        temperature,
                        num_comps)
