"""
randomly split a dataset into train/dev/test sets
according to given proportions
[0.5, 0.2, 0.3]
"""

#!/usr/bin/env python3
"""
将JSONL文件分割为train/dev/test集
用法: python random_split.py input.jsonl [--ratios 0.5 0.2 0.3] [--seed 42]
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

def split_jsonl(
    input_file: str,
    ratios: Tuple[float, float, float] = (0.5, 0.2, 0.3),
    seed: int = 42,
    output_prefix: str = None
) -> Tuple[List, List, List]:
    """
    分割JSONL文件
    
    Args:
        input_file: 输入JSONL文件路径
        ratios: (train_ratio, dev_ratio, test_ratio)
        seed: 随机种子
        output_prefix: 输出文件前缀
        
    Returns:
        (train_data, dev_data, test_data)
    """
    # 验证比例
    assert len(ratios) == 3, "需要3个比例值"
    assert abs(sum(ratios) - 1.0) < 1e-9, f"比例之和应为1.0，当前为{sum(ratios)}"
    
    # 读取数据
    print(f"读取文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = [json.loads(line.strip()) for line in lines if line.strip()]
    print(f"总样本数: {len(data)}")
    
    # 设置随机种子
    random.seed(seed)
    random.shuffle(data)
    
    # 计算分割点
    n_total = len(data)
    train_end = int(n_total * ratios[0])
    dev_end = train_end + int(n_total * ratios[1])
    
    # 分割数据
    train_data = data[:train_end]
    dev_data = data[train_end:dev_end]
    test_data = data[dev_end:]
    
    # 输出统计信息
    print(f"训练集: {len(train_data)} 样本 ({len(train_data)/n_total:.1%})")
    print(f"开发集: {len(dev_data)} 样本 ({len(dev_data)/n_total:.1%})")
    print(f"测试集: {len(test_data)} 样本 ({len(test_data)/n_total:.1%})")
    
    # 保存文件
    if output_prefix is None:
        input_path = Path(input_file)
        output_prefix = input_path.stem
    
    output_files = {
        'train': f"{output_prefix}_train.jsonl",
        'dev': f"{output_prefix}_dev.jsonl", 
        'test': f"{output_prefix}_test.jsonl"
    }
    
    for name, file_path in output_files.items():
        data_to_save = train_data if name == 'train' else dev_data if name == 'dev' else test_data
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_to_save:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"保存 {name} 集: {file_path} ({len(data_to_save)} 样本)")
    
    return train_data, dev_data, test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分割JSONL文件")
    parser.add_argument("input_file", help="输入JSONL文件路径")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.5, 0.2, 0.3],
                       help="train/dev/test比例，默认 [0.5, 0.2, 0.3]")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认42")
    parser.add_argument("--output", type=str, help="输出文件前缀")
    
    args = parser.parse_args()
    
    split_jsonl(
        input_file=args.input_file,
        ratios=tuple(args.ratios),
        seed=args.seed,
        output_prefix=args.output
    )