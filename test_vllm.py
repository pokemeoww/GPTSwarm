from openai import OpenAI

'''
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --dtype auto \
'''

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="",
)

completion = client.chat.completions.create(
    model="/root/autodl-tmp/qwen_4b_model/qwen/Qwen3-4B-Instruct-2507",
    messages=[
        {"role": "user", "content": "1+1="},
    ],
)

print(completion.choices[0].message)