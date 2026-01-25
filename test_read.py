import pandas as pd

# # 替换为您的实际文件路径
# file_path = "/root/sijia/GPTSwarm/datasets/MMLU_PRO/data/validation-00000-of-00001.parquet"

# # 读取 Parquet 文件
# df = pd.read_parquet(file_path, engine='pyarrow')

# # 查看数据框的基本信息
# print("数据集形状（行数, 列数）:", df.shape)
# print("\n前几行数据：")
# print(df.head())
# print("\n列名：")
# print(df.columns.tolist())

import pandas as pd
import os
from pathlib import Path

def parquet_to_csv(parquet_path, csv_path=None):
    """
    将单个Parquet文件转换为CSV
    
    参数:
        parquet_path: Parquet文件路径
        csv_path: CSV输出路径（可选，默认同目录同名）
    """
    # 如果未指定输出路径，则在同目录生成同名CSV
    if csv_path is None:
        csv_path = str(Path(parquet_path).with_suffix('.csv'))
    
    # 读取Parquet文件
    print(f"正在读取: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # 保存为CSV
    df.to_csv(csv_path, index=False)
    print(f"已保存: {csv_path}")
    print(f"转换完成！共 {len(df)} 行, {len(df.columns)} 列")

def batch_convert(input_dir, output_dir=None, pattern="*.parquet"):
    """
    批量转换目录中的所有Parquet文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录（可选，默认同目录）
        pattern: 文件匹配模式
    """
    if output_dir is None:
        output_dir = input_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有Parquet文件
    input_path = Path(input_dir)
    parquet_files = list(input_path.glob(pattern))
    
    if not parquet_files:
        print(f"在 {input_dir} 中未找到 {pattern} 文件")
        return
    
    print(f"找到 {len(parquet_files)} 个Parquet文件")
    
    for i, parquet_file in enumerate(parquet_files, 1):
        # 生成输出路径
        csv_filename = parquet_file.with_suffix('.csv').name
        csv_path = Path(output_dir) / csv_filename
        
        print(f"[{i}/{len(parquet_files)}] 转换: {parquet_file.name}")
        parquet_to_csv(str(parquet_file), str(csv_path))
    
    print("批量转换完成！")

# 使用示例
if __name__ == "__main__":
    # 示例1：转换单个文件
    #parquet_to_csv("data.parquet", "data.csv")
    
    # 示例2：批量转换目录
    batch_convert("/root/sijia/GPTSwarm/datasets/MMLU_PRO/data", "/root/sijia/GPTSwarm/datasets/MMLU_PRO/data")
    
    # 示例3：转换特定模式的文件
   # batch_convert("./data", pattern="*.parquet.gzip")