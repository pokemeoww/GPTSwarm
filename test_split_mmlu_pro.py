import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_parquet(
    "/root/sijia/GPTSwarm/datasets/MMLU_PRO/data/mmlu_pro_final_train.parquet"
)

# 按照 'category' 字段进行分层抽样，分出 15% 做调参
# 剩下的 85% 作为你论文最终汇报的 Test 成绩
train, demo_collect_train = train_test_split(
    df, 
    test_size=0.4, 
    stratify=df['category'], 
    random_state=42
)

# # 再从剩下的 85% 中分出 30% 作为验证集（Val），70% 作为训练集（Train）
# # train是为了 collect final demo；val这里是为了训练reranker
# final_train_df, final_val_df = train_test_split(
#     train_df, 
#     test_size=0.3, 
#     stratify=train_df['category'], 
#     random_state=42
# )

train.to_parquet("/root/sijia/GPTSwarm/datasets/MMLU_PRO/data/mmlu_pro_final_evaluator_train.parquet", index=False)
demo_collect_train.to_parquet("/root/sijia/GPTSwarm/datasets/MMLU_PRO/data/mmlu_pro_demo_collection_train.parquet", index=False)  

