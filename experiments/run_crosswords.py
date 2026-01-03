import json
from tqdm import tqdm
import asyncio
import numpy as np
from copy import deepcopy
import pickle
import torch
import sys
import random
import argparse
import os
from datetime import datetime

from swarm.environment.domain.crosswords.env import MiniCrosswordsEnv
from swarm.environment.agents.agent_registry import AgentRegistry
from swarm.graph.swarm import Swarm
from swarm.optimizer.edge_optimizer.optimization import optimize
from swarm.environment.domain.crosswords.evaluator import CrosswordsEvaluator


def parse_args():
    """Parse command line arguments for crosswords experiments."""
    parser = argparse.ArgumentParser(
        description='Run crosswords experiments with configurable model and ICL settings'
    )
    
    # Model selection
    parser.add_argument(
        '--model', 
        type=str, 
        #default='gpt-3.5-turbo-1106',
        # TODO: this is for vllm hosting
        #default='/root/autodl-tmp/qwen_4b_model/qwen/Qwen3-4B-Instruct-2507',
        default='/root/autodl-tmp/Qwen/Qwen2.5-7B-Instruct',
        help='Model name. Examples: gpt-4, gpt-3.5-turbo-1106, hf:gpt2, hf:meta-llama/Llama-2-7b-chat-hf'
    )
    
    # ICL control
    parser.add_argument(
        '--use_icl', 
        action='store_true',
        help='Enable ICL (In-Context Learning) retrieval. If not set, uses baseline without ICL.'
    )
    
    parser.add_argument(
        '--icl_method',
        type=str,
        choices=['retrieved', 'reranked'],
        default='retrieved',
        help='ICL method to use (only applies if --use_icl is set). retrieved: FAISS only, reranked: FAISS + reranker'
    )
    
    # Dataset selection
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Debug mode: use single_example.json for quick testing'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['debug', 'dev', 'test', 'train', 'mini'],
        default='dev',
        help='Dataset to use. debug: single example, dev: development set, test: test set, train: training set, mini: mini0505_0_100_5'
    )
    
    # Experiment settings
    parser.add_argument(
        '--exp_id', 
        type=int, 
        default=0,
        help='Experiment ID for random seed and result directory naming'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for evaluation. If not set, uses dataset-specific default.'
    )
    
    parser.add_argument(
        '--num_iter',
        type=int,
        default=2,
        help='Number of optimization iterations'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output showing detailed puzzle information and answers'
    )
    
    return parser.parse_args()


def get_dataset_path(dataset_choice, debug_mode):
    """Get the appropriate dataset path based on user choice."""
    if debug_mode or dataset_choice == 'debug':
        # return "datasets/crosswords/single_example.json"
        return "datasets/crosswords/mini_0505_dev.json"
    elif dataset_choice == 'dev':
        return "datasets/crosswords/mini_0505_dev.json"
    elif dataset_choice == 'test':
        return "datasets/crosswords/mini0505_test.json"
    elif dataset_choice == 'train':
        return "datasets/crosswords/mini0505_train.json"
    elif dataset_choice == 'mini':
        return "datasets/crosswords/mini0505_0_100_5.json"
    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice}")


def setup_icl_environment(use_icl, icl_method):
    """Configure environment variables for ICL."""
    if use_icl:
        os.environ['demo_method'] = icl_method
        print(f"✓ ICL enabled with method: {icl_method}")
    else:
        os.environ['demo_method'] = 'fixed'
        print("✓ ICL disabled (baseline mode)")


def save_experiment_config(args, file_path, experiment_id):
    """Save experiment configuration to a JSON file."""
    config = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'model': args.model,
        'use_icl': args.use_icl,
        'icl_method': args.icl_method if args.use_icl else None,
        'dataset': file_path,
        'debug_mode': args.debug,
        'batch_size': args.batch_size,
        'num_iter': args.num_iter,
        'exp_id': args.exp_id,
    }
    
    # Create results directory
    result_dir = f"result/{experiment_id}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Save config
    config_path = f"{result_dir}/config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Experiment config saved to: {config_path}")
    return result_dir


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.exp_id)
    np.random.seed(args.exp_id)
    random.seed(args.exp_id)
    
    # Setup ICL environment
    setup_icl_environment(args.use_icl, args.icl_method)
    
    # Get dataset path
    file_path = get_dataset_path(args.dataset, args.debug)
    
    # Determine batch size
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.debug or args.dataset == 'debug':
        # TODO: batch size decides how many puzzles to evaluate in parallel
        batch_size = 2
    else:
        batch_size = 20
    
    # Generate experiment ID
    model_name = args.model.replace('/', '_').replace(':', '_')
    icl_suffix = f"_icl_{args.icl_method}" if args.use_icl else "_baseline"
    dataset_suffix = f"_{args.dataset}" if not args.debug else "_debug"
    experiment_id = f"{model_name}{icl_suffix}{dataset_suffix}_exp{args.exp_id}"
    
    print("=" * 80)
    print(f"Experiment ID: {experiment_id}")
    print(f"Model: {args.model}")
    print(f"Dataset: {file_path}")
    print(f"Batch size: {batch_size}")
    print(f"ICL: {'Enabled (' + args.icl_method + ')' if args.use_icl else 'Disabled'}")
    print("=" * 80)
    
    # Load data
    with open(file_path, "r") as file:
        test_data = json.load(file)
    
    print(f"✓ Loaded {len(test_data)} examples from {file_path}")
    
    # Save experiment configuration
    result_dir = save_experiment_config(args, file_path, experiment_id)
    
    # Setup optimization parameters
    init_connection_probability = .1
    use_learned_order = False
    include_inner_agent_connections = True
    connect_output_nodes_to_final_node = True
    window_size = 10
    
    # Create evaluator
    evaluator = CrosswordsEvaluator(
        test_data, 
        batch_size=batch_size, 
        metric="words", 
        window_size=window_size, 
        init_socre=0.4, 
        use_init_score=True,
        verbose=args.verbose
    )
    
    # Create swarm with specified model
    print(f"✓ Initializing swarm with model: {args.model}")
    swarm = Swarm(
        ["CrosswordsReflection", "CrosswordsToT", "CrosswordsBruteForceOpt"], 
        "crosswords", 
        args.model,  # Use command-line specified model
        final_node_class="ReturnAll", 
        final_node_kwargs={},
        edge_optimize=True,
        init_connection_probability=init_connection_probability, 
        connect_output_nodes_to_final_node=connect_output_nodes_to_final_node, 
        include_inner_agent_connections=include_inner_agent_connections
    )
    
    # Run optimization
    if args.debug:
        num_iter = 3
    print(f"✓ Starting optimization for {num_iter} iterations...")
    optimize(
        swarm, 
        evaluator, 
        batch_size=batch_size, 
        num_iter=num_iter, 
        display_freq=1, 
        record=True,
        experiment_id=experiment_id, 
        lr=.4, 
        use_learned_order=use_learned_order
    )
    
    print("=" * 80)
    print(f"✓ Experiment completed: {experiment_id}")
    print(f"✓ Results saved to: {result_dir}")
    print("=" * 80)
