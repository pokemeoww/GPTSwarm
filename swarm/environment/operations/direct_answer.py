#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from collections import defaultdict
from datetime import datetime

import os
import re
from swarm.environment.operations.experience_store_utils import store_experience
from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from typing import List, Any, Optional
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer


from icl_retrieval.utils.retrievers import FaissRetriever
from sentence_transformers import SentenceTransformer
from icl_retrieval.reranker import RerankerModel
from dotenv import load_dotenv

load_dotenv()

# Check if we need to load demo-related models
use_demo = os.environ.get("MMLU_PRO_USE_DEMO", "false")

print("[Debug]use_demo is ", use_demo)

# Only load embedding model and retrievers if using ICL
if use_demo.lower() == "true":
    print(f"[MMLU_PRO] Loading demo models for method: {use_demo}")
    
    embedding_model_path = os.environ["EMBEDDING_MODEL_PATH"]
    model_dir = embedding_model_path
    
    from icl_retrieval.utils.retrievers import FaissRetriever
    
    # Get device
    device = 'cuda'
    print(f"[MMLU_PRO] Using device for embedding model: {device}")
    
    embed_model = SentenceTransformer(
                model_dir if model_dir else embedding_model_path
            ).to(device)
    embed_model.eval()

    demo_base_path = os.environ.get("MMLU_PRO_DEMO_BASE_PATH", "data/final_demo_dec_13")

    demo_retriever = FaissRetriever(
        os.path.join(demo_base_path, "list_demos_mmlu_pro.json"),
        os.path.join(demo_base_path, "list_demos_mmlu_pro.npy"),
        embed_model=embed_model
    ) if os.path.exists(os.path.join(demo_base_path, "list_demos_mmlu_pro.json")) else None

    demo_reranker = RerankerModel(bert_model_path=os.environ.get("mmlu_pro_reranker_ckpt_path"))


class DirectAnswer(Node): 
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str],
                 operation_description: str = "Directly output an answer.",
                 max_token: int = 400, 
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.max_token = max_token
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

        self.experience_file = f"data/experiences/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/direct_answer_demos.jsonl"

        self.use_demo = os.environ.get("MMLU_PRO_USE_DEMO", "false").lower() == "true"

    @property
    def node_name(self):
        return self.__class__.__name__
    
    async def node_optimize(self, input, meta_optmize=False):
        task = input["task"]
        self.prompt_set = PromptSetRegistry.get(self.domain)
        role = self.prompt_set.get_role()
        constraint = self.prompt_set.get_constraint()

        if meta_optmize:
            update_role = role 
            node_optmizer = MetaPromptOptimizer(self.model_name, self.node_name)
            update_constraint = await node_optmizer.generate(constraint, task)
            return update_role, update_constraint

        return role, constraint


    async def _execute(self, inputs: List[Any] = [], **kwargs):
        
        node_inputs = self.process_input(inputs)
        outputs = []

        for input in node_inputs:
            task = input["task"]
            role, constraint = await self.node_optimize(input, meta_optmize=False)
            prompt = self.prompt_set.get_answer_prompt(question=task)   

            if self.use_demo:
                reranker_demos = self.get_demos(task)
                print("Reranker demos: ", reranker_demos)
                print("format demos: ", self.format_demos(reranker_demos))
                demo_instruct_prompt = "I will provide you with some examples of how to answer similar questions. Please carefully study the pattern and reasoning.\n\n"
                prompt = prompt + demo_instruct_prompt + self.format_demos(reranker_demos)

            message = [Message(role="system", content=f"You are a {role}. {constraint}"),
                       Message(role="user", content=prompt)]
            
            answer, reasoning = await self.llm.agen(message, max_tokens=self.max_token)
            #answer, reasoning, valid = self.parse_model_response_with_reasoning(response_raw)

            execution = {
                "operation": self.node_name,
                "task": task,
                "files": input.get("files", []),
                "input": task,
                "role": role,
                "constraint": constraint,
                "prompt": prompt,
                "output": answer,
                "ground_truth": input.get("GT", []),
                "format": "natural language"
            }
            outputs.append(execution)
            self.memory.add(self.id, execution)

            # Save experience
            print("DEBUG: answer is: ", answer)
            print("DEBUG: reasoning is: ", reasoning)
            print("DEBUG: ground truth is: ", input.get("GT", []))

            store_experience(input['task'], answer, answer==input.get("GT", []), reasoning, self.experience_file) 

        # self.log()
        return outputs 
    
    def get_demos(self, task: str) -> List[dict]:
        """Get demonstrations based on the demo method."""
        if not self.use_demo:
            return []
        
        if self.use_demo:
            demos_tmp = demo_retriever.search_once(task, top_k=15)["query2query"]
            demonstrations, reranked_scores = demo_reranker.predict(task, demos_tmp)
            print("[Debug]Reranked_scores: ", reranked_scores)

            human_eval_top_k = int(os.environ.get("MMLU_PRO_TOP_K", "3"))
            demos_tmp = demonstrations[:human_eval_top_k]

            # Extract the actual demo content
            return demos_tmp


    # def parse_model_response_with_reasoning(self, text):
    #     text = text.strip()
        
    #     # 提取ANSWER部分
    #     answer = ""
        
    #     # 方法1: 查找 ANSWER: X 格式
    #     answer_patterns = [
    #         r'ANSWER:\s*([A-Z])',  # ANSWER: A
    #         r'Answer:\s*([A-Z])',  # Answer: A
    #     ]
        
    #     for pattern in answer_patterns:
    #         match = re.search(pattern, text, re.IGNORECASE)
    #         if match:
    #             answer = match.group(1).upper()
    #             print("DEBUG: found answer using pattern:", pattern, " answer:", answer)
    #             break
        
    #     # 提取REASONING部分
    #     reasoning = ""
        
    #     # 查找 REASONING: 部分
    #     reasoning_patterns = [
    #         r'REASONING:\s*(.+?)(?=ANSWER:|Answer:|$)',  # REASONING: ... 直到 ANSWER:
    #     ]
        
    #     for pattern in reasoning_patterns:
    #         match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    #         if match:
    #             reasoning = match.group(1).strip()
    #             break
        
        
    #     # 验证答案是否有效
    #     is_valid = False
    #     if answer is not None:
    #         is_valid = answer.isalpha() and answer.isupper()
        
    #     return answer, reasoning, is_valid

    def parse_model_response_with_reasoning(self, text):
        if not text or not isinstance(text, str):
            return "", "", False
        
        text = text.strip()
        answer = ""
        reasoning = ""
        
        # 1. 尝试匹配新格式: ANSWER: X REASONING: Y
        pattern = r'(?i)ANSWER:\s*([A-Z])(?:\s|$)(?:REASONING:\s*(.+))?'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            answer = match.group(1).upper()
            if match.group(2):
                reasoning = match.group(2).strip()
        else:
            # 2. 如果没有匹配到，尝试找单个大写字母
            letters = re.findall(r'\b([A-Z])\b', text)
            if letters:
                answer = letters[-1]  # 取最后一个大写字母
                # 尝试提取其他文本作为推理
                reasoning = re.sub(rf'\b{answer}\b', '', text).strip()
        
        # 3. 验证答案
        is_valid = answer.isalpha() and answer.isupper() if answer else False
        
        return answer, reasoning, is_valid

    # def format_demos(self, demos: List[dict]) -> str:
    #     # format using "=== Example ===\n" input output
    #     formatted_demos = ""
    #     for demo in demos:
    #         text = demo['sample']
    #         formatted_demos += (
    #                     "=== Example ===\n"
    #                     "Input:\n"
    #                     f"{text['input']}\n\n"
    #                     "Output:\n"
    #                     "```txt\n"
    #                     f"{text['output']}\n"
    #                     "```\n"
    #                     "=== End example ===\n\n"
    #                 )
    #     return formatted_demos
    
    def format_demos(self, demos: List[dict]) -> str:
        # format using "## EXAMPLE ##\n" input output
        formatted_demos = ""
        for demo in demos:
            text = demo['sample']
            formatted_demos += (
                f"## EXAMPLE ##\n"
                f"**Question:**\n"
                f"{text['input']}\n\n"
                f"**Correct Answer:**\n"
                f"```\n"
                f"{text['output']}\n"
                f"```\n"
                f"## END EXAMPLE ##\n\n"
            )
        return formatted_demos
        