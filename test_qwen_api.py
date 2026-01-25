#!/usr/bin/env python3
"""
Simple test script for Qwen Max reasoning output
"""

import os
import re
from openai import OpenAI
from datetime import datetime

# Configuration
LLM_API_KEY = "sk-84f41bcd6c1a4474a4c27deaa28e705d"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-plus"

def test_math_problem_with_raw_output():
    """Test the exact math problem from your example"""
    print("\n" + "="*80)
    print("🧪 TEST: Math Problem with Raw Output")
    print("="*80)
    
    # Exact problem from your example
    math_problem = """Find the characteristic of the ring 2Z.
    
Options: 0, 2, 3, 5, 10, 12, 20, 30, 50, 100
    
Please show your reasoning and then give the final answer."""
    
    print(f"📚 PROBLEM:\n{math_problem}")
    print("-"*80)
    
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=30.0
    )
    
    try:
        # Make the API call
        print("📡 Calling Qwen Max API...")
        completion = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a math expert. Show your step-by-step reasoning, then provide the final answer."},
                {"role": "user", "content": math_problem}
            ],
            max_tokens=1000,
            temperature=0.1,
            extra_body={"enable_thinking": True}
        )
        
        content = completion.choices[0].message.content
        print("raw output is: ", content)
        thinking_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        final_reply_match = re.search(r'</think>\s*(.*)', content, re.DOTALL)

        reasoning_content = thinking_match.group(1).strip() if thinking_match else "未找到思考过程。"
        content = final_reply_match.group(1).strip() if final_reply_match else content.strip()

        print("\n🧠 REASONING:\n", reasoning_content)
        print("\n✅ FINAL ANSWER:\n", content)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Qwen Max Reasoning Test")
    print(f"Model: {LLM_MODEL_NAME}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not LLM_API_KEY or LLM_API_KEY == "sk-your-api-key-here":
        print("\n❌ Please set your API key:")
        print("   Method 1: export DASHSCOPE_API_KEY=your-api-key")
        print("   Method 2: Edit the LLM_API_KEY variable in this script")
        print("\n💡 You can get API key from: https://dashscope.console.aliyun.com/")
        return
    
    test_math_problem_with_raw_output()

if __name__ == "__main__":
    main()