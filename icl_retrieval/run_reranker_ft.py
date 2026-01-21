import copy
import math
import os
import json
import argparse
import pickle
import random

import sys

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import get_scheduler, AutoModel, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append("./")

from icl_retrieval.reranker import RerankerModel
from icl_retrieval.utils.encode_data import encode_corpus
from icl_retrieval.utils.retrievers import FaissRetriever
from icl_retrieval.utils.device_utils import get_device

from icl_retrieval.utils.io_utils import load_jsonl, dump_jsonl, load_json, dump_json

"""
poetry run python3 /root/sijia/GPTSwarm/icl_retrieval/run_reranker_ft.py
rm -r experiments/run_1
"""

# 收集训练数据的金标准：
def collect_data_for_training(list_demo_data, demo_retriever=None,
                              mode=None, output_dir=None, max_length=512):

    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["model_name"],
    )
    tokenizer.padding_side = "left"
    gpt_model = AutoModelForCausalLM.from_pretrained(
        os.environ["model_name"],
        # TODO: check dtype
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16
    ).to(device)
    gpt_model.eval()

    list_queries = []
    list_outputs = []
    list_sample_demos = []
    list_demo_llm_scores = []

    print("[DEBUG] Length of demo data: ", len(list_demo_data))
    for idx, samp in tqdm(enumerate(list_demo_data)):
        print("[DEBUG] processing: ", idx)
        query = samp["input"]
        list_queries.append(query)
        output = samp["output"]
        list_outputs.append(output)

        output_vector = demo_retriever.embed_model.encode(
            [output]
        )
        output_vector = torch.tensor(output_vector)

        demos = demo_retriever.search_once(
            query,
            top_k=20,
            # top_k=int(os.environ["TOP_K"])
        )["query2query"]


        # TODO: add a filtering step!
        # 过滤掉与当前query相同的demo
        filtered_demos = []
        for demo in demos:
            demo_input = demo["sample"]["input"]
            # 如果demo的input与当前query不同，才保留
            if query not in demo_input and demo_input != query:
                filtered_demos.append(demo)
            
            # 如果已经收集到足够的demos，就停止
            if len(filtered_demos) >= 15:
                break        
        
        demos = filtered_demos

        if len(demos) == 0:
            print(f"[WARN] No demos left after filtering at idx={idx}. Using unfiltered top-1 fallback.")
            demos = demo_retriever.search_once(query, top_k=1)["query2query"]
            if len(demos) == 0:
                print(f"[WARN] Retriever returned 0 demos at idx={idx}. Skipping.")
                continue
        
        print("DEBUG: Number of demo left is: ", len(demos))

        # TODO: end of filtering step

        list_sample_demos.append(demos)

        # 对每个demo，采用大模型条件生成，得到 评分
        demo_llm_scores = []
        list_llm_messages = []
        list_input_ids = []
        list_attention_masks = []

        messages_tmp = [
            {"role": "assistant", "content": output},
        ]
        message_content_tmp = tokenizer.apply_chat_template(messages_tmp, tokenize=True)
        len_output = len(message_content_tmp)

        for demo_i in demos:

            # 填充prompt
            input_i = demo_i["sample"]["input"]
            output_i = demo_i["sample"]["output"]
            demo_str_i = f"Input: {input_i}\nOutput: {output_i}\n"

            # 填充user prompt
            user_prompt = f"Demonstrations: \n {demo_str_i}\n\n" + f"Query: {query}"

            messages_2 = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": output},
            ]
            tmp2 = tokenizer.apply_chat_template(messages_2, tokenize=True)
            len_whole = len(tmp2)
            if len_whole > 1024 * 2:
                print("len_whole: ", len_whole)
            message_content_2 = tokenizer.apply_chat_template(messages_2, tokenize=False)
            model_inputs_2 = tokenizer(
                message_content_2,
                padding="max_length",
                truncation=True,
                max_length=1024 * 2 + 512,
                return_tensors='pt',
            )
            input_ids_2 = model_inputs_2["input_ids"]
            attention_mask_2 = model_inputs_2["attention_mask"]
            # print("input_ids_2: ", input_ids_2[: , -25: ])
            # print("input_ids_2: ", input_ids_2.shape)

            list_input_ids.append(
                input_ids_2
            )
            list_attention_masks.append(attention_mask_2)

        batch_input_ids = torch.cat(
            list_input_ids,
            dim=0
        )
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_masks = torch.cat(
            list_attention_masks,
            dim=0
        )
        batch_attention_masks = batch_attention_masks.to(device)

        s_bsz = 4
        for idx in range(batch_input_ids.shape[0] // s_bsz):
            with torch.no_grad():
                outputs_2 = gpt_model(
                    input_ids=batch_input_ids[idx * s_bsz: (idx + 1) * s_bsz ],
                    attention_mask=batch_attention_masks[idx * s_bsz: (idx + 1) * s_bsz ],
                )
                # print(outputs_2)
                logits_2 = outputs_2.logits

            for d_idx in range(logits_2.shape[0]):
                answer_ids = batch_input_ids[idx * s_bsz + d_idx, - len_output: ].reshape(-1, 1)
                answer_logits = logits_2[d_idx, - len_output: , :].reshape(-1, logits_2.shape[-1])
                # print("answer_ids: ", answer_ids.shape)
                # print("answer_logits: ", answer_logits.shape)
                answer_probs = torch.softmax(answer_logits, dim=1)
                answer_probs = torch.gather(
                    answer_probs,
                    1,
                    answer_ids
                )
                # print("answer_probs: ", answer_probs.shape)
                score_ = torch.sum(answer_probs).to(torch.float32).cpu().numpy().tolist()  # score 越高是越好
                # print("score_: ", score_)
                demo_llm_scores.append(score_)

        # print("demo_llm_scores: ", demo_llm_scores)
        list_demo_llm_scores.append(demo_llm_scores)

        torch.cuda.empty_cache()

        if len(list_demo_llm_scores) % 25 == 0 or idx == len(list_demo_data) - 1:
            pickle.dump(
                list_queries,
                open(os.path.join(output_dir, f"list_queries_{mode}.pkl"), "wb")
            )
            pickle.dump(
                list_outputs,
                open(os.path.join(output_dir, f"list_outputs_{mode}.pkl"), "wb")
            )
            pickle.dump(
                list_sample_demos,
                open(os.path.join(output_dir, f"list_sample_demos_{mode}.pkl"), "wb")
            )
            pickle.dump(
                list_demo_llm_scores,
                open(os.path.join(output_dir, f"list_demo_llm_scores_{mode}.pkl"), "wb")
            )

            del outputs_2, logits_2, answer_probs
            torch.cuda.empty_cache()

    return list_queries, list_outputs, list_sample_demos, \
           list_demo_llm_scores


def reranker_loss_1(ranks_gt, scores_pred, gap=1):
    # 采用hinge的格式

    total_loss = None
    count = 0
    bsz = len(ranks_gt)
    acc = 0
    var = 0
    for i in range(bsz):
        for j in range(bsz):
            r_i = ranks_gt[i]
            r_j = ranks_gt[j]
            if r_j < r_i:
                continue

            coeff = max(0.0, r_j - r_i)

            sp_i = scores_pred[i]
            sp_j = scores_pred[j]
            log_value = - torch.log(1 + sp_j - sp_i)

            if total_loss is None:
                total_loss = coeff ** 1.2 * log_value
            else:
                total_loss += coeff ** 1.2 * log_value

            if sp_j >= sp_i:
                acc += 1

            count += 1

    if count == 0:
        loss = None
        acc = 0.0
        var_ = 0.0
    else:
        loss = total_loss / count
        acc = acc / count

        # 计算 variance
        var_ = torch.std(scores_pred, dim=0, keepdim=False)
        # print("score var_: ", var_)
        # var_ = torch.clamp(var_, max=0.2)
        #
        # loss = loss - 0.5 * var_
        if random.uniform(0, 1) < 0.01:
            print("score var_: ", var_)

    return loss, acc, var_


def run(args):
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 demo数据
    list_demo_data = []
    for root, dirs, files in os.walk(args.demo_data_path):
        for file in files:
            if file.endswith(".json") and file.startswith("list_demos_"):
                print(file)
                print(os.path.join(root, file))
                list_demo_data += load_jsonl(
                    os.path.join(root, file)
                )
    
    # Debug mode: use only 300 samples for quick testing
    print(f"Total samples loaded: {len(list_demo_data)}")

    if args.debug:
        random.shuffle(list_demo_data)
        list_demo_data = list_demo_data[:300]
        print(f"[DEBUG MODE] Using {len(list_demo_data)} samples for training")
    else:
        print(f"Using all {len(list_demo_data)} samples for training")

    # 进行数据拆分
    if args.debug or not os.path.exists(os.path.join(args.output_dir, "list_demo_data_train.json")):

        random.shuffle(list_demo_data)
        list_demo_data_train = list_demo_data[: int(len(list_demo_data) * 0.9)]
        list_demo_data_test = list_demo_data[int(len(list_demo_data) * 0.9):]
        dump_jsonl(
            list_demo_data_train,
            os.path.join(args.output_dir, "list_demo_data_train.json")
        )
        dump_jsonl(
            list_demo_data_test,
            os.path.join(args.output_dir, "list_demo_data_test.json")
        )

        # 编码和npy数据
        encode_corpus(
            os.path.join(args.output_dir, "list_demo_data_train.json"),
            model_name_or_path=args.embed_model_path,
            to_path=os.path.join(args.output_dir, "demo_data_train.npy"),
            batch_size=64
        )
        encode_corpus(
            os.path.join(args.output_dir, "list_demo_data_test.json"),
            model_name_or_path=args.embed_model_path,
            to_path=os.path.join(args.output_dir, "demo_data_test.npy"),
            batch_size=64
        )

    else:
        list_demo_data_train = load_jsonl(
            os.path.join(args.output_dir, "list_demo_data_train.json")
        )
        list_demo_data_test = load_jsonl(
            os.path.join(args.output_dir, "list_demo_data_test.json")
        )
    

    print("Train samples: ", len(list_demo_data_train))
    print("Test samples: ", len(list_demo_data_test))


    # 初始化 faiss retriever
    device = get_device()
    embed_model = SentenceTransformer(
        args.embed_model_path
    ).to(device)
    embed_model.eval()
    demo_retriever = FaissRetriever(
        os.path.join(args.output_dir, "list_demo_data_train.json"),
        os.path.join(args.output_dir, "demo_data_train.npy"),
        embed_model=embed_model
    )

    # reranker
    demo_reranker = RerankerModel(
        bert_model_path=args.bert_model_path
    )

    # optimizer
    for n, p in demo_reranker.named_parameters():
        print(n, p.requires_grad)
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in demo_reranker.named_parameters() if not any(nd in n for nd in no_decay)
                       ],
            "weight_decay": 1e-4,
        },
        {
            "params": [p for n, p in demo_reranker.named_parameters() if any(nd in n for nd in no_decay)
                       ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=len(list_demo_data_train) // args.gradient_accumulation_steps * args.num_epochs
    )

    # 准备训练所需数据
    if args.debug or not os.path.exists(os.path.join(args.output_dir, "list_queries_train.pkl")):
        print("Collecting data for training...")
        print("Training set:", len(list_demo_data_train))
        print("Test set:", len(list_demo_data_test))

        list_queries_train, list_outputs_train, \
        list_sample_demos_train, list_demo_llm_scores_train = collect_data_for_training(
            list_demo_data_train, demo_retriever=demo_retriever,
            mode="train", output_dir=args.output_dir
        )
        list_queries_test, list_outputs_test, \
        list_sample_demos_test, list_demo_llm_scores_test = collect_data_for_training(
            list_demo_data_test, demo_retriever=demo_retriever,
            mode="test", output_dir=args.output_dir
        )
    else:
        list_queries_train = pickle.load(
            open(os.path.join(args.output_dir, "list_queries_train.pkl"), "rb")
        )
        list_outputs_train = pickle.load(
            open(os.path.join(args.output_dir, "list_outputs_train.pkl"), "rb")
        )
        list_sample_demos_train = pickle.load(
            open(os.path.join(args.output_dir, "list_sample_demos_train.pkl"), "rb")
        )
        list_demo_llm_scores_train = pickle.load(
            open(os.path.join(args.output_dir, "list_demo_llm_scores_train.pkl"), "rb")
        )
        list_queries_test = pickle.load(
            open(os.path.join(args.output_dir, "list_queries_test.pkl"), "rb")
        )
        list_outputs_test = pickle.load(
            open(os.path.join(args.output_dir, "list_outputs_test.pkl"), "rb")
        )
        list_sample_demos_test = pickle.load(
            open(os.path.join(args.output_dir, "list_sample_demos_test.pkl"), "rb")
        )
        list_demo_llm_scores_test = pickle.load(
            open(os.path.join(args.output_dir, "list_demo_llm_scores_test.pkl"), "rb")
        )

    # 训练步骤
    # training
    steps = 0
    completed_steps = 0
    best_test_acc = 0
    patience = 0
    for epoch_idx in tqdm(range(args.num_epochs)):
        print("==== currently in epoch: ", epoch_idx)

        for samp_idx, (query, output, demos, demo_llm_scores) in tqdm(enumerate(zip(
                list_queries_train,
                list_outputs_train,
                list_sample_demos_train,
                list_demo_llm_scores_train,
        ))):

            # prepare input for reranker
            pairs = []
            for demo_ in demos:
                pairs.append(
                    [demo_["sample"]["input"], query]
                )
            sentence_a = [pair[0] for pair in pairs]
            sentence_b = [pair[1] for pair in pairs]

            # 批量编码句子对
            inputs = demo_reranker.tokenizer(
                sentence_a,
                sentence_b,
                max_length=512,  # 最大长度（按需调整）
                padding='longest',  # 填充到最大长度
                truncation=True,  # 截断超过长度的部分
                return_tensors='pt',  # 返回PyTorch张量
                return_token_type_ids=True  # 需要token_type_ids
            )
            inputs = {k_: v_.to(demo_reranker.device) for k_, v_ in inputs.items()}

            # reranker scores
            demo_reranker.train()
            reranker_scores = demo_reranker(
                **inputs
            )
            # print("reranker_scores: ", reranker_scores)

            # 计算损失
            # demo_scores, demo_ranking = torch.sort(
            #     torch.tensor(demo_llm_scores), descending=False
            # )
            # print(demo_llm_scores, demo_ranking)
            loss_, acc_, var_ = reranker_loss_1(
                demo_llm_scores,
                reranker_scores
            )
            if loss_ is None:
                continue
            # print("demo_llm_scores: ", demo_llm_scores)
            # print("reranker_scores: ", reranker_scores)
            if random.uniform(0, 1) < 0.002:
                print("loss: ", loss_)

            loss_.backward()
            steps += 1
            # for n, p in demo_reranker.named_parameters():
            #     print(n, p.requires_grad, p.grad)

            # for n, p in demo_reranker.named_parameters():
            #     print(n, p.grad)

            if steps % args.gradient_accumulation_steps == 0:
                # print("steps: ", steps)
                torch.nn.utils.clip_grad_norm_(
                    demo_reranker.parameters(),
                    2.0
                )
                completed_steps += 1
                # print("completed_steps: ", completed_steps)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if completed_steps % 25 == 0 or completed_steps == 1:
                    acc_test = 0
                    var_test = 0
                    loss_test = 0
                    count_test = 0
                    for samp_idx, (query, output, demos, demo_llm_scores) in tqdm(enumerate(zip(
                            list_queries_test,
                            list_outputs_test,
                            list_sample_demos_test,
                            list_demo_llm_scores_test
                    ))):
                        # prepare input for reranker
                        pairs = []
                        for demo_ in demos:
                            pairs.append(
                                [demo_["sample"]["input"], query]
                            )
                        sentence_a = [pair[0] for pair in pairs]
                        sentence_b = [pair[1] for pair in pairs]
                        # 批量编码句子对
                        inputs = demo_reranker.tokenizer(
                            sentence_a,
                            sentence_b,
                            max_length=512,  # 最大长度（按需调整）
                            padding='longest',  # 填充到最大长度
                            truncation=True,  # 截断超过长度的部分
                            return_tensors='pt',  # 返回PyTorch张量
                            return_token_type_ids=True  # 需要token_type_ids
                        )
                        inputs = {k_: v_.to(demo_reranker.device) for k_, v_ in inputs.items()}

                        demo_reranker.eval()
                        with torch.no_grad():
                            reranker_scores = demo_reranker(
                                **inputs
                            )

                            # 计算损失
                            loss_, acc_, var_ = reranker_loss_1(
                                demo_llm_scores,
                                reranker_scores
                            )

                        count_test += 1
                        acc_test += acc_
                        var_test += var_
                        if loss_ is not None:
                            loss_test += loss_

                        if count_test >= 50:
                            break

                    acc_test = acc_test / count_test
                    var_test = var_test / count_test
                    loss_test = loss_test / count_test

                    if acc_test > best_test_acc:
                        best_test_acc = acc_test
                        patience = 0

                        print(f"completed_steps： {completed_steps}; acc: {acc_test}")
                        print(f"completed_steps： {completed_steps}; loss: {loss_test}")
                        print(f"completed_steps： {completed_steps}; var: {var_test}")

                        # 保存模型=
                        demo_reranker.model.save_pretrained(
                            os.path.join(
                                args.output_dir,
                                f"demo_reranker_{completed_steps}"
                            )
                        )
                        demo_reranker.tokenizer.save_pretrained(
                            os.path.join(
                                args.output_dir,
                                f"demo_reranker_{completed_steps}"
                            )
                        )

                    else:
                        patience += 1

                    print(f"patience： ", patience)

                    if patience >= 100:
                        break

        if patience >= 100:
            break

    print("best_test_acc: ", best_test_acc)
    # TODO: 比pretrain的效果好一点可以


def parse_args():
    args = argparse.ArgumentParser()

    # task
    # 路径
    args.add_argument('--output_dir', type=str, default="experiments/run_crosswords_gptmini/")
    args.add_argument('--demo_data_path', type=str, default="/root/sijia/GPTSwarm/data/crosswords_demos/gpt5_mini_Jan_19_dev")
    args.add_argument('--embed_model_path', type=str, default="/root/autodl-tmp/BAAI/bge-base-en-v1___5")
    args.add_argument('--bert_model_path', type=str, default="/root/autodl-tmp/BAAI/bge-base-en-v1___5")

    # training hyper-params
    args.add_argument('--learning_rate', type=float, default=1e-4)
    args.add_argument('--gradient_accumulation_steps', type=int, default=16)
    args.add_argument('--warmup_steps', type=int, default=100)
    args.add_argument('--num_epochs', type=int, default=5)
    
    # debug mode:
    args.add_argument('--debug', action='store_true', help='Use only 300 samples for quick testing')

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
