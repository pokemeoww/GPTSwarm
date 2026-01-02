#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import asyncio
from collections import defaultdict
import random
import os
import json
from typing import List, Any, Optional

from swarm.llm.format import Message
from swarm.graph import Node
from swarm.memory.memory import GlobalMemory
from swarm.utils.log import logger, swarmlog
from swarm.utils.globals import Cost
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm import LLMRegistry
from swarm.optimizer.node_optimizer import MetaPromptOptimizer
from swarm.environment.tools.coding.python_executor import PyExecutor
from swarm.environment.operations.optimizable_operation import OptimizableOperation

# Check if we need to load demo-related models
use_demo = os.environ.get("use_demo", "false")

# Only load embedding model and retrievers if using ICL
if use_demo.lower() == "true":
    print(f"[CodeWriting] Loading demo models for method: {use_demo}")
    
    demo_method = os.environ.get("demo_method", "fixed")
    
    if demo_method in ["retrieved", "reranked"]:
        from sentence_transformers import SentenceTransformer
        from icl_retrieval.utils.retrievers import FaissRetriever
        from icl_retrieval.utils.device_utils import get_device
        
        # Get device - use CPU for embedding model to save MPS memory
        device = 'cpu'
        print(f"[CodeWriting] Using device for embedding model: {device}")
        
        embedding_model_path = os.environ.get("EMBEDDING_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2")
        embed_model = SentenceTransformer(embedding_model_path).to(device)
        embed_model.eval()
        
        # Base path for demo files
        demo_base_path = os.environ.get("DEMO_BASE_PATH", "data/humaneval_demos")
        
        # retriever for demos
        code_demo_retriever = FaissRetriever(
            os.path.join(demo_base_path, "code_demos.json"),
            os.path.join(demo_base_path, "code_demos.npy"),
            embed_model=embed_model
        )
        
        # reranker
        if demo_method == "reranked":
            from icl_retrieval.reranker import RerankerModel
            reranker_ckpt_path = os.environ.get("reranker_ckpt_path")
            if reranker_ckpt_path:
                demo_reranker = RerankerModel(bert_model_path=reranker_ckpt_path)
            else:
                demo_reranker = None
                print("[CodeWriting] Warning: reranked method selected but no reranker_ckpt_path provided")
        else:
            demo_reranker = None
        
        print(f"[CodeWriting] Demo models loaded successfully for method: {demo_method}")
    else:
        embed_model = None
        code_demo_retriever = None
        demo_reranker = None
else:
    # Baseline mode: don't load any demo-related models
    print(f"[CodeWriting] Baseline mode (use_demo={use_demo}), skipping demo model loading")
    embed_model = None
    code_demo_retriever = None
    demo_reranker = None


class CodeWriting(OptimizableOperation):
    def __init__(self, 
                 domain: str,
                 model_name: Optional[str] = None,
                 operation_description: str = "a Python code generator",
                 id=None):
        prompt = "You are an AI that only responds with only Python code. "
        prompt += "You will be given a function signature and its docstring by the user. "
        prompt += "Write your full implementation (restate the function signature). "
        prompt += "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        super().__init__(domain, False, prompt, model_name, operation_description, id)
        self.domain = domain
        self.model_name = model_name
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()
        
        # Load demos if use_demo is enabled
        self.use_demo = os.environ.get("use_demo", "false").lower() == "true"
        self.demo_method = os.environ.get("demo_method", "fixed")
        
        if self.use_demo and self.demo_method == "fixed":
            # Load fixed demos
            demo_base_path = os.environ.get("DEMO_BASE_PATH", "data/humaneval_demos")
            demo_path = os.path.join(demo_base_path, "code_demos.json")
            if os.path.exists(demo_path):
                with open(demo_path, 'r') as f:
                    demos = json.load(f)
                # Use all available demos for fixed mode
                self.domenstrations = demos
                print(f"[CodeWriting] Loaded {len(self.domenstrations)} fixed demos")
            else:
                print(f"[CodeWriting] Warning: Demo file not found at {demo_path}")
                self.domenstrations = []

    @property
    def node_name(self):
        """Return the class name."""
        return self.__class__.__name__


    def extract_example(self, prompt: str) -> list:
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results

    def get_demos(self, task: str) -> List[dict]:
        """Get demonstrations based on the demo method."""
        if not self.use_demo:
            return []
        
        if self.demo_method == "fixed":
            # Return all fixed demos
            return self.domenstrations
        
        elif self.demo_method in ["retrieved", "reranked"]:
            top_k = int(os.environ.get("TOP_K", "5"))
            
            # Retrieve demos using FaissRetriever
            demos_tmp = code_demo_retriever.search_once(task, top_k=top_k)["query2query"]
            
            if self.demo_method == "reranked" and demo_reranker is not None:
                # Rerank demos
                top_k_2 = int(os.environ.get("TOP_K_2", "3"))
                demonstrations, reranked_scores = demo_reranker.predict(task, demos_tmp)
                print(f"[CodeWriting] Reranked scores: {reranked_scores}")
                demos_tmp = demonstrations[:top_k_2]
            
            # Extract the actual demo content
            return [d["sample"] for d in demos_tmp]
        
        return []

    async def _execute(self, inputs: List[Any] = [], max_tries: int = 1, **kwargs):
        """
        Execute the node with the given inputs.
        """

        node_inputs = self.process_input(inputs)
        node_outputs = []

        for input in node_inputs:
            if input.get('is_solved', False):
                execution = deepcopy(input)
            else:
                task = input["task"]
                if 'feedback' in input.keys():
                    input = self.prompt_set.get_react_prompt(task, input["output"], input["feedback"])
                else:
                    input = input["task"]
                self.internal_tests = self.extract_example(task)
                
                # Get demos dynamically based on the task
                if self.use_demo and self.demo_method in ["retrieved", "reranked"]:
                    current_demos = self.get_demos(task)
                else:
                    current_demos = self.domenstrations
                
                message = self.get_messages(input, self.prompt, current_demos)
                
                response = await self.llm.agen(message)
                response = response.strip("```python\n").strip("```")
                is_solved, feedback, _ = PyExecutor().execute(response, self.internal_tests, timeout=10)
                execution = {
                    "operation": self.node_name,
                    "task": task, 
                    "input": input,
                    "feedback": feedback,
                    "output": response,
                    "format": "python code",
                    "is_solved": is_solved,
                }

            self.memory.add(self.id, execution)
            node_outputs.append(execution)

        return node_outputs

    def get_messages(self, task, prompt, domenstrations):
        messages = []
        messages.append(Message(role="system", content=prompt))
        for domenstration in domenstrations:
            messages.append(Message(role="user", content=domenstration['input']))
            messages.append(Message(role="assistant", content=domenstration['output']))
        messages.append(Message(role="user", content=task))
        return messages

    async def evaluate(self, candidate):
        prompt, domenstrations = candidate
        inputs = self.memory.query_by_id(self.id)
        inputs = [record for record in self.memory.query_by_id(self.id)[-10:]]#random.sample(inputs, min(10, len(inputs)))#
        score = 0
        tasks = []
        for input in inputs:
            message = self.get_messages(input['input'], prompt, domenstrations)
            response = await self.llm.agen(message)
            response = response.strip("```python\n").strip("```")
            tests = self.extract_example(input['task'])
            is_solved, _, _ = PyExecutor().execute(response, tests, timeout=10)
            score += is_solved

        return score / len(inputs)
