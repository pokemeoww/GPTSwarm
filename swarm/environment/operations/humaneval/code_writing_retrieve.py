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
        device = 'cuda'
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