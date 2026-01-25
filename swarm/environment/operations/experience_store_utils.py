# /root/sijia/GPTSwarm/swarm/environment/operations/experience_store_utils.py
import json
import os
from datetime import datetime

def store_experience(input_data: str, output_data: str, is_solved: str, 
                     reasoning: str = "",
                     file_path: str = ""):
    """
    存储输入输出到JSONL文件（追加模式）
    
    Args:
        file_path: 文件路径
        input_data: 输入
        output_data: 输出
    """
    # 创建数据
    data = {
        "input": input_data,
        "output": output_data,
        "is_solved": is_solved,
    }
    if reasoning:
        data["reasoning"] = reasoning
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 追加写入
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"已保存到: {file_path}")