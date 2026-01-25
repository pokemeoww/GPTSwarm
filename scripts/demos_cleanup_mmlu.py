#!/usr/bin/env python3
"""
处理经验文件脚本（基于input+output去重）
"""

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
import argparse

def calculate_combo_hash(input_data: Any, output_data: Any) -> str:
    """计算input+output的组合哈希值"""
    combo = {
        "input": input_data,
        "output": output_data
    }
    data_str = json.dumps(combo, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()

def process_experience_files(
    input_files: List[str],
    output_file: str = "list_demos_code_writing.json"
) -> Dict[str, Any]:
    """
    处理经验文件（基于input+output去重）
    """
    all_experiences = []
    seen_combo_hashes: Set[str] = set()  # 存储input+output的哈希
    stats = {
        "total_read": 0,
        "solved_count": 0,
        "duplicate_count": 0,
        "unique_solved": 0
    }
    
    print("=" * 60)
    print("处理经验文件（基于input+output去重）")
    print("=" * 60)
    
    for file_path in input_files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"⚠ 文件不存在: {file_path}")
            continue
        
        print(f"\n处理文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    stats["total_read"] += 1
                    
                    try:
                        exp = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"  ⚠ 第{line_num}行JSON解析失败: {e}")
                        continue
                    
                    # 检查必需字段
                    if "input" not in exp or "output" not in exp:
                        print(f"  ⚠ 第{line_num}行缺少input或output字段")
                        continue
                    
                    # 检查是否解决
                    is_solved = exp.get("is_solved", False)
                    if isinstance(is_solved, str):
                        is_solved = is_solved.lower() in ['true', '1', 'yes', 't']
                    
                    if not is_solved:
                        continue
                    
                    stats["solved_count"] += 1

                    # TODO: output: answer + reasoning 合并
                    exp['output'] = "ANSWER: " + exp["output"] + "\nREASONING: " + exp['reasoning']
                    
                    # 基于input+output去重
                    combo_hash = calculate_combo_hash(exp["input"], exp["output"])
                    if combo_hash in seen_combo_hashes:
                        stats["duplicate_count"] += 1
                        continue
                    
                    seen_combo_hashes.add(combo_hash)
                    
                    # 提取input和output
                    processed_exp = {
                        "input": exp["input"],
                        "output": exp["output"]
                    }
                    
                    # 可选：添加元数据
                    metadata = {}
                    for key in ["timestamp", "metadata", "model", "task_type"]:
                        if key in exp:
                            metadata[key] = exp[key]
                    
                    if metadata:
                        processed_exp["metadata"] = metadata
                    
                    all_experiences.append(processed_exp)
                    stats["unique_solved"] += 1
                    
        except Exception as e:
            print(f"❌ 处理文件失败 {file_path}: {e}")
    
    # 保存结果
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for exp in all_experiences:
            f.write(json.dumps(exp, ensure_ascii=False) + '\n')
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"总读取记录: {stats['total_read']}")
    print(f"已解决记录: {stats['solved_count']}")
    print(f"重复记录: {stats['duplicate_count']}")
    print(f"唯一已解决记录: {stats['unique_solved']}")
    print(f"输出文件: {output_path}")
    
    if stats['unique_solved'] > 0:
        print(f"样本示例（前2个）:")
        for i, exp in enumerate(all_experiences[:2]):
            input_str = str(exp['input'])[:80] + "..." if len(str(exp['input'])) > 80 else str(exp['input'])
            output_str = str(exp['output'])[:80] + "..." if len(str(exp['output'])) > 80 else str(exp['output'])
            print(f"  [{i+1}] 输入: {input_str}")
            print(f"      输出: {output_str}")
            combo_hash = calculate_combo_hash(exp['input'], exp['output'])
            print(f"      哈希: {combo_hash[:16]}...")
            print()
    
    return stats

def main():
    list_experience_files = [
            "/root/sijia/GPTSwarm/data/experiences/2026-01-08_13-33-27_mmlu_pro_demo_train_14B/direct_answer_demos.jsonl",
        ]

    output_path = "data/experiences/demos_jan_8_mmlu_pro_demo_train_14B/list_demos_mmlu_pro.json"
    # 处理文件
    process_experience_files(list_experience_files, output_path)

if __name__ == "__main__":
    main()