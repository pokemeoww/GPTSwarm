import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-84f41bcd6c1a4474a4c27deaa28e705d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-plus", #qwen3-max-preview
    messages=[{"role": "user", "content": "What is 123 + 456?"}],
    max_tokens=200,
    extra_body={"enable_thinking": True, "max_tokens_thinking": 150}
)

print("Raw output:", response)

reasoning = response.choices[0].message.reasoning_content

print("reasoning content is: ", reasoning)