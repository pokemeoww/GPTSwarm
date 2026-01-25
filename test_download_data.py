#!/usr/bin/env python3
"""
最简单的 MMLU-Pro 下载脚本
"""

import requests
import json
import pandas as pd
from pathlib import Path
import sys
import time

def download_with_progress(url, save_path):
    """带进度条的下载"""
    print(f"下载: {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\r进度: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({percent:.1f}%)")
                        sys.stdout.flush()
        
        print(f"\n✅ 已保存: {save_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

def main():
    # 数据集保存目录
    save_dir = Path("/root/sijia/GPTSwarm/datasets/MMLU_PRO")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"保存到: {save_dir}")
    
    # MMLU-Pro 数据文件
    files = {
        "test": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet",
        "validation": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/validation-00000-of-00001.parquet",
        "train": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/train-00000-of-00001.parquet"
    }
    
    # 下载文件
    for name, url in files.items():
        file_path = save_dir / f"{name}.parquet"
        
        if file_path.exists():
            print(f"文件已存在: {file_path}")
            continue
            
        success = download_with_progress(url, file_path)
        if not success:
            print(f"跳过 {name}...")
    
    # 转换格式（可选）
    print("\n" + "="*50)
    print("转换文件格式...")
    
    for file_path in save_dir.glob("*.parquet"):
        jsonl_path = file_path.with_suffix(".jsonl")
        
        if jsonl_path.exists():
            print(f"JSONL文件已存在: {jsonl_path}")
            continue
        
        try:
            print(f"读取: {file_path.name}")
            df = pd.read_parquet(file_path)
            
            print(f"转换为JSONL: {jsonl_path.name}")
            df.to_json(
                jsonl_path,
                orient="records",
                lines=True,
                force_ascii=False
            )
            
            print(f"  样本数: {len(df)}")
            print(f"  列: {list(df.columns)}")
            
        except Exception as e:
            print(f"❌ 转换失败 {file_path}: {e}")
    
    # 生成信息文件
    info = {
        "name": "MMLU-Pro",
        "description": "Massive Multitask Language Understanding - Pro version",
        "source": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
        "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {}
    }
    
    for file_path in save_dir.glob("*"):
        if file_path.is_file():
            info["files"][file_path.name] = {
                "size_bytes": file_path.stat().st_size,
                "size_mb": file_path.stat().st_size / 1024 / 1024
            }
    
    with open(save_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("🎯 下载完成！")
    print(f"目录: {save_dir}")
    for name, stats in info["files"].items():
        print(f"  - {name}: {stats['size_mb']:.1f} MB")

if __name__ == "__main__":
    main()