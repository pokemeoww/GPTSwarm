import copy
import json
import os

import random

import torch
from sentence_transformers import SentenceTransformer

from icl_retrieval.utils.existing_demos import list_propose_demos, list_if_correct_demos, list_suggest_demos, \
    list_value_demos
from icl_retrieval.utils.io_utils import load_json
from icl_retrieval.utils.logger_utils import LOG
from icl_retrieval.utils.device_utils import get_device
from icl_retrieval.reranker import RerankerModel
from swarm.graph import Node
from swarm.llm.format import Message

embedding_model_path = os.environ["EMBEDDING_MODEL_PATH"]

from icl_retrieval.utils.retrievers import FaissRetriever

# Get device (MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback)
device = get_device()
print(f"[ICL CrosswordsOperation] Using device: {device}")

embed_model = SentenceTransformer(
            embedding_model_path
        ).to(device)
embed_model.eval()

# Base path for demo files
demo_base_path = os.environ.get("DEMO_BASE_PATH", "data/final_demo_dec_13")

demo_retriever = FaissRetriever(
    os.path.join(demo_base_path, "direct_demos.json"),
    os.path.join(demo_base_path, "direct_demos.npy"),
    embed_model=embed_model
) if os.path.exists(os.path.join(demo_base_path, "direct_demos.json")) else None

# retriever for demos
propose_demo_retriever = FaissRetriever(
    os.path.join(demo_base_path, "list_demos_propose.json"),
    os.path.join(demo_base_path, "list_demos_propose.npy"),
    embed_model=embed_model
)
if_correct_demo_retriever = FaissRetriever(
    os.path.join(demo_base_path, "list_demos_if_correct.json"),
    os.path.join(demo_base_path, "list_demos_if_correct.npy"),
    embed_model=embed_model
)
suggest_demo_retriever = FaissRetriever(
    os.path.join(demo_base_path, "list_demos_suggest.json"),
    os.path.join(demo_base_path, "list_demos_suggest.npy"),
    embed_model=embed_model
)
value_demo_retriever = FaissRetriever(
    os.path.join(demo_base_path, "list_demos_value.json"),
    os.path.join(demo_base_path, "list_demos_value.npy"),
    embed_model=embed_model
)

# reranker
demo_reranker = None
reranker_ckpt_path = os.environ.get("reranker_ckpt_path")
if reranker_ckpt_path:
    demo_reranker = RerankerModel(
        bert_model_path=reranker_ckpt_path
    )


def prompt_type_cls(prompt):

    if "list all possible answers for unfilled or changed words" in prompt:
        prompt_type = "propose"
    elif "Respond only Yes or No" in prompt:
        prompt_type = "if_correct"
    elif "Write a plan for the next time." in prompt:
        prompt_type = "suggest"
    elif "Evaluate if there exists a five letter word of some meaning that fit some letter constraints" in prompt:
        prompt_type = "value"
    else:
        raise ValueError("Invalid prompt")

    return prompt_type


class CrosswordsOperation(Node):
    async def llm_query_with_cache(self, prompt):
        cache = self.memory.query_by_id("cache")
        if len(cache) == 0:
            cache = {}
            self.memory.add("cache", cache)
        else:
            cache = cache[0]

        # print("cache: ", len(cache))
        # print("cache: ", cache[0])
        # print("cache: ", cache.keys())
        # cache_tmp = {}
        # if os.path.exists("llm_query_caches.json"):
        #     cache_tmp = load_json(
        #         "llm_query_caches.json"
        #     )
        # cache_tmp.update(cache)
        # with open("llm_query_caches.json", "w", encoding="utf-8") as f:
        #     json.dump(
        #         cache_tmp,
        #         f,
        #         ensure_ascii=True,
        #         indent=2
        #     )

        prompt_0 = copy.copy(prompt)
        response_ = None
        if not prompt in cache:

            prompt_1 = copy.copy(prompt)
            if os.environ["add_direct_cues"] == "true":
                # 在prompt中添加 direct cues, 帮助提升效果
                if "<current query>" in prompt:
                    search_input = prompt.split("<current query>")[1].strip()
                else:
                    search_input = prompt
                retrieved_demos = demo_retriever.search_once(
                    search_input, top_k=18
                )["query2query"]
                random.shuffle(retrieved_demos)
                retrieved_demos = retrieved_demos[:10]

                cue_str = "<cue>\n"
                for r_d in retrieved_demos:
                    # print("r_d: ", r_d)
                    input_ = r_d["sample"]["input"]
                    cue_str += f"{input_}\n\n"
                cue_str += "</cue>\n\n"
                prompt_1 = cue_str + prompt

            prompt_type = prompt_type_cls(prompt_1)
            top_k = int(os.environ["TOP_K"])
            if os.environ["demo_method"] == "fixed":
                print("DEBUG: USING FIXed demos")

                if prompt_type == "propose":
                    demos_tmp = copy.deepcopy(list_propose_demos[ :top_k])
                elif prompt_type == "if_correct":
                    demos_tmp = copy.deepcopy(list_if_correct_demos[ :top_k])
                elif prompt_type == "suggest":
                    demos_tmp = copy.deepcopy(list_suggest_demos[ :top_k])
                elif prompt_type == "value":
                    demos_tmp = copy.deepcopy(list_value_demos[ :top_k])
                else:
                    raise ValueError("Invalid prompt_type")

                demo_str = ""
                for r_d in demos_tmp:
                    # print("r_d: ", r_d)
                    input_ = r_d["input"]
                    output_ = r_d["output"]
                    demo_str += f"{input_}\n{output_}\n\n"
                
                print("DEBUG: FIXED demo string is: ", demo_str)

            elif os.environ["demo_method"] in ["retrieved", "reranked"]:
                if "<current query>" in prompt:
                    search_input = prompt.split("<current query>")[1].strip()
                else:
                    search_input = prompt

                if prompt_type == "propose":
                    demos_tmp = propose_demo_retriever.search_once(
                        search_input, top_k=top_k
                    )["query2query"]
                elif prompt_type == "if_correct":
                    demos_tmp = if_correct_demo_retriever.search_once(
                        search_input, top_k=top_k
                    )["query2query"]
                elif prompt_type == "suggest":
                    demos_tmp = suggest_demo_retriever.search_once(
                        search_input, top_k=top_k
                    )["query2query"]
                elif prompt_type == "value":
                    demos_tmp = value_demo_retriever.search_once(
                        search_input, top_k=top_k
                    )["query2query"]
                else:
                    raise ValueError("Invalid prompt_type")

                if os.environ["demo_method"] == "reranked":
                    # rerank demo to top_k_2
                    top_k_2 = int(os.environ["TOP_K_2"])
                    demonstrations, reranked_scores = demo_reranker.predict(search_input, demos_tmp)
                    print("reranked_scores: ", reranked_scores)
                    demos_tmp = demonstrations[: top_k_2]

                demo_str = ""
                for r_d in demos_tmp:
                    # print("r_d: ", r_d)
                    input_ = r_d["sample"]["input"]
                    output_ = r_d["sample"]["output"]
                    demo_str += f"{input_}\n{output_}\n\n"

            else:
                raise ValueError("Invalid demo_method")

            prompt_2 = prompt_1.replace(
                "<placeholder>", demo_str
            )
            # print("prompt: ", prompt)

            response_ = await self.llm.agen([Message(role="user", content=prompt_2)], temperature=0.0)
            # prompt_ = prompt.split("</cue>")[1].strip()

            cache[prompt_0] = response_

            # if prompt_type in ["if_correct", "value"]:
            #     cache[prompt_0] = response_
            # else:
            #     cache[prompt_2.split("</cue>")[1].strip()] = response_

            LOG.info(f'prompt_message: {json.dumps({"input": prompt_2, "output": response_}, ensure_ascii=False)}')

        else:
            response_ = cache[prompt_0]
        return response_
