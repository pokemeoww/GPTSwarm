# Crosswords Experiment Guide

This guide explains how to run crosswords experiments with different configurations.

## Quick Start

### Basic Usage

```bash
# Show help
python3 experiments/run_crosswords.py --help

# Run with default settings (GPT-3.5, dev dataset, no ICL)
python3 experiments/run_crosswords.py
```

## Command Line Arguments

### 1. Model Selection (`--model`)

Choose which LLM to use:

```bash
# OpenAI models
python3 experiments/run_crosswords.py --model "gpt-4"
python3 experiments/run_crosswords.py --model "gpt-3.5-turbo-1106"

# HuggingFace models (local)
python3 experiments/run_crosswords.py --model "hf:gpt2"
python3 experiments/run_crosswords.py --model "hf:meta-llama/Llama-2-7b-chat-hf"
python3 experiments/run_crosswords.py --model "hf:Qwen/Qwen2.5-7B-Instruct"
```

**Default**: `gpt-3.5-turbo-1106`

### 2. ICL Control (`--use_icl`, `--icl_method`)

Enable or disable In-Context Learning:

```bash
# Disable ICL (baseline)
python3 experiments/run_crosswords.py

# Enable ICL with FAISS retrieval
python3 experiments/run_crosswords.py --use_icl

# Enable ICL with reranker
python3 experiments/run_crosswords.py --use_icl --icl_method reranked
```

**Options for `--icl_method`**:
- `retrieved` (default): Use FAISS similarity search only
- `reranked`: Use FAISS + trained reranker

### 3. Dataset Selection (`--dataset`, `--debug`)

Choose which dataset to use:

```bash
# Debug mode (single example)
python3 experiments/run_crosswords.py --debug

# Development set
python3 experiments/run_crosswords.py --dataset dev

# Test set
python3 experiments/run_crosswords.py --dataset test

# Training set
python3 experiments/run_crosswords.py --dataset train

# Mini dataset (100 examples)
python3 experiments/run_crosswords.py --dataset mini
```

**Dataset paths**:
- `debug`: `datasets/crosswords/single_example.json`
- `dev`: `datasets/crosswords/mini_0505_dev.json`
- `test`: `datasets/crosswords/mini0505_test.json`
- `train`: `datasets/crosswords/mini0505_train.json`
- `mini`: `datasets/crosswords/mini0505_0_100_5.json`

### 4. Other Options

```bash
# Set experiment ID (for random seed)
python3 experiments/run_crosswords.py --exp_id 42

# Set batch size
python3 experiments/run_crosswords.py --batch_size 10

# Set number of optimization iterations
python3 experiments/run_crosswords.py --num_iter 20
```

## Your 4 Planned Experiments

### Experiment 1: Baseline (Debug)

```bash
python3 experiments/run_crosswords.py \
  --model "hf:gpt2" \
  --debug
```

**What this does**:
- Uses GPT-2 model (local)
- Single example from `single_example.json`
- No ICL (baseline)
- Batch size: 1

**Expected output**:
```
Experiment ID: hf_gpt2_baseline_debug_exp0
Model: hf:gpt2
Dataset: datasets/crosswords/single_example.json
Batch size: 1
ICL: Disabled
```

### Experiment 2: Baseline + Retriever (Debug)

```bash
python3 experiments/run_crosswords.py \
  --model "hf:gpt2" \
  --use_icl \
  --debug
```

**What this does**:
- Uses GPT-2 model (local)
- Single example from `single_example.json`
- ICL enabled with FAISS retrieval
- Batch size: 1

**Expected output**:
```
Experiment ID: hf_gpt2_icl_retrieved_debug_exp0
Model: hf:gpt2
Dataset: datasets/crosswords/single_example.json
Batch size: 1
ICL: Enabled (retrieved)
```

### Experiment 3: Baseline (Dev)

```bash
python3 experiments/run_crosswords.py \
  --model "hf:gpt2" \
  --dataset dev
```

**What this does**:
- Uses GPT-2 model (local)
- Development set from `mini_0505_dev.json`
- No ICL (baseline)
- Batch size: 20

**Expected output**:
```
Experiment ID: hf_gpt2_baseline_dev_exp0
Model: hf:gpt2
Dataset: datasets/crosswords/mini_0505_dev.json
Batch size: 20
ICL: Disabled
```

### Experiment 4: Baseline + Retriever (Dev)

```bash
python3 experiments/run_crosswords.py \
  --model "hf:gpt2" \
  --use_icl \
  --dataset dev
```

**What this does**:
- Uses GPT-2 model (local)
- Development set from `mini_0505_dev.json`
- ICL enabled with FAISS retrieval
- Batch size: 20

**Expected output**:
```
Experiment ID: hf_gpt2_icl_retrieved_dev_exp0
Model: hf:gpt2
Dataset: datasets/crosswords/mini_0505_dev.json
Batch size: 20
ICL: Enabled (retrieved)
```

## Advanced Examples

### Using Reranker

```bash
# Make sure you have trained the reranker first
python3 experiments/run_crosswords.py \
  --model "hf:gpt2" \
  --use_icl \
  --icl_method reranked \
  --dataset dev
```

### Multiple Runs with Different Seeds

```bash
# Run 3 experiments with different random seeds
for i in 0 1 2; do
  python3 experiments/run_crosswords.py \
    --model "hf:gpt2" \
    --use_icl \
    --dataset dev \
    --exp_id $i
done
```

### Comparing Models

```bash
# GPT-2
python3 experiments/run_crosswords.py --model "hf:gpt2" --debug

# Llama-2
python3 experiments/run_crosswords.py --model "hf:meta-llama/Llama-2-7b-chat-hf" --debug

# GPT-3.5
python3 experiments/run_crosswords.py --model "gpt-3.5-turbo-1106" --debug
```

## Results

### Output Location

Results are saved to:
```
result/{experiment_id}/
├── config.json          # Experiment configuration
├── optimization_log.json # Optimization progress
└── final_results.json   # Final evaluation results
```

### Experiment ID Format

```
{model}_{icl_status}_{dataset}_exp{id}
```

Examples:
- `hf_gpt2_baseline_debug_exp0`
- `hf_gpt2_icl_retrieved_dev_exp0`
- `gpt-3.5-turbo-1106_icl_reranked_test_exp1`

### Config File Example

```json
{
  "experiment_id": "hf_gpt2_icl_retrieved_dev_exp0",
  "timestamp": "2025-12-14T10:55:00",
  "model": "hf:gpt2",
  "use_icl": true,
  "icl_method": "retrieved",
  "dataset": "datasets/crosswords/mini_0505_dev.json",
  "debug_mode": false,
  "batch_size": 20,
  "num_iter": 11,
  "exp_id": 0
}
```

## Troubleshooting

### ICL Not Working

**Problem**: ICL is enabled but demos are not being retrieved.

**Solution**:
1. Make sure you have generated embeddings:
   ```bash
   python3 scripts/generate_embeddings.py --demo_dir data/final_demo_dec_13
   ```

2. Check your `.env` file has the correct paths:
   ```bash
   EMBEDDING_MODEL_PATH="BAAI/bge-base-en-v1.5"
   DEMO_BASE_PATH="data/final_demo_dec_13"
   ```

### HuggingFace Model Not Found

**Problem**: `Model 'hf:xxx' not found`

**Solution**:
1. Make sure the model name is correct
2. Check you have internet connection (for first download)
3. Try a smaller model first (e.g., `hf:gpt2`)

### Out of Memory

**Problem**: `CUDA out of memory` or similar

**Solution**:
1. Use a smaller model
2. Reduce batch size: `--batch_size 5`
3. Use CPU: Set device in code or use smaller model

## Tips

### Quick Testing

Always test with `--debug` first:
```bash
python3 experiments/run_crosswords.py --model "hf:gpt2" --debug
```

### Comparing ICL vs Baseline

Run both and compare results:
```bash
# Baseline
python3 experiments/run_crosswords.py --model "hf:gpt2" --dataset dev --exp_id 0

# With ICL
python3 experiments/run_crosswords.py --model "hf:gpt2" --dataset dev --use_icl --exp_id 1
```

### Batch Processing

Create a script to run multiple experiments:
```bash
#!/bin/bash
# run_all_experiments.sh

models=("hf:gpt2" "gpt-3.5-turbo-1106")
datasets=("debug" "dev")
icl_flags=("" "--use_icl")

for model in "${models[@]}"; do
  for dataset in "${datasets[@]}"; do
    for icl in "${icl_flags[@]}"; do
      python3 experiments/run_crosswords.py \
        --model "$model" \
        --dataset "$dataset" \
        $icl
    done
  done
done
```

## Summary

The updated `run_crosswords.py` provides:
- ✅ Flexible model selection (OpenAI or HuggingFace)
- ✅ Easy ICL on/off switching
- ✅ Multiple dataset options
- ✅ Debug mode for quick testing
- ✅ Automatic experiment tracking
- ✅ Configuration saving

All controlled through simple command-line arguments!
