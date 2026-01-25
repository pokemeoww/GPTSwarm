#!/usr/bin/env python3
from modelscope import snapshot_download
import os
import sys

def download_model():
    # 模型名称
    model_id = 'Qwen/Qwen2.5-3B-Instruct'
    
    # 保存路径
    save_dir = '/root/autodl-tmp/'
    
    # 创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始下载模型: {model_id}")
    print(f"保存到: {save_dir}")
    print("=" * 50)
    
    try:
        # 下载模型
        model_dir = snapshot_download(
            model_id=model_id,
            cache_dir=save_dir,  # 保存到指定目录
            revision='master',    # 使用主分支
            ignore_file_pattern=[],  # 下载所有文件
            local_files_only=False,
        )
        
        print(f"✅ 下载完成！")
        print(f"模型路径: {model_dir}")
        
        # 统计文件信息
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
                file_count += 1
                if file.endswith(('.bin', '.safetensors', '.pth', '.pt')):
                    print(f"  - {file}: {size/1024/1024:.2f} MB")
        
        print(f"\n📊 统计信息:")
        print(f"文件数量: {file_count}")
        print(f"总大小: {total_size/1024/1024/1024:.2f} GB")
        
        return model_dir
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()