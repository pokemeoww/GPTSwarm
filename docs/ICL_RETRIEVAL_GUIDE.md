# ICL Retrieval System - Complete Usage Guide

## Overview

This guide explains how to use the In-Context Learning (ICL) retrieval system for the crosswords task in GPTSwarm. The system dynamically retrieves and ranks demonstration examples to improve LLM performance.

## System Architecture

The ICL retrieval system consists of:

1. **Demo Storage**: JSONL files containing input-output pairs
2. **Embeddings**: Vector representations of demos (.npy files)
3. **FAISS Retriever**: Fast similarity search
4. **Reranker** (optional): BERT-based model to refine demo selection
5. **Integration**: Automatic demo insertion into prompts

## Quick Start

### Step 1: Convert Your Demo Files

Your demo files have been converted from the cleaned format to the required format:

```bash
python3 scripts/convert_demos.py
```

**Output**: `/Users/sihua/personal/GPTSwarm/data/final_demo_dec_13/`
- `list_demos_if_correct.json` (72 samples)
- `list_demos_suggest.json` (276 samples)
- `list_demos_value.json` (873 samples)
- `list_demos_propose.json` (680 samples)

### Step 2: Generate Embeddings

Create embeddings for each demo file:

```bash
python3 -m poetry run python3 scripts/generate_embeddings.py \
  --demo_dir /Users/sihua/personal/GPTSwarm/data/final_demo_dec_13 \
  --embed_model BAAI/bge-base-en-v1.5
```

This will create `.npy` files alongside each `.json` file.

### Step 3: Set Environment Variables

Create a `.env` file or export these variables:

```bash
# Required for retrieval
export EMBEDDING_MODEL_PATH="BAAI/bge-base-en-v1.5"
export TOP_K=10
export demo_method="retrieved"  # Options: "fixed", "retrieved", "reranked"

# Optional: for reranked mode
export TOP_K_2=3
export reranker_ckpt_path="/path/to/trained/reranker"

# Optional: additional cues
export add_direct_cues="false"
```

### Step 4: Run Crosswords Task

The ICL retrieval will automatically activate when you run crosswords tasks:

```python
from swarm.graph.swarm import Swarm

swarm = Swarm(["TOT"], "crosswords")
# The system will automatically use ICL retrieval based on your env vars
```

## Environment Variables Explained

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `EMBEDDING_MODEL_PATH` | Path or name of sentence transformer model | `BAAI/bge-base-en-v1.5` |
| `TOP_K` | Number of demos to retrieve | `10` |
| `demo_method` | Demo selection method | `retrieved` or `reranked` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOP_K_2` | Number of demos after reranking (if using reranked mode) | `3` |
| `reranker_ckpt_path` | Path to trained reranker checkpoint | None |
| `add_direct_cues` | Add extra hint cues to prompts | `false` |

### Demo Methods

1. **`fixed`**: Use pre-defined demos from `existing_demos.py` (currently empty)
2. **`retrieved`**: Use FAISS to retrieve similar demos
3. **`reranked`**: Retrieve with FAISS, then rerank with trained BERT model

## Training the Reranker (Optional)

The reranker improves demo selection by learning which demos are most helpful.

### Prerequisites

1. Converted demo files (already done ✓)
2. LLM access for scoring demos
3. GPU with CUDA

### Training Command

```bash
export model_name="gpt-4"  # or your LLM model

python3 -m poetry run python3 icl_retrieval/run_reranker_ft.py \
  --demo_data_path /Users/sihua/personal/GPTSwarm/data/final_demo_dec_13 \
  --output_dir ./reranker_output \
  --embed_model_path BAAI/bge-base-en-v1.5 \
  --bert_model_path BAAI/bge-base-en-v1.5 \
  --learning_rate 1e-4 \
  --num_epochs 2
```

### What Happens During Training

1. **Data Split**: Demos split 90/10 into train/test
2. **Embedding Generation**: Creates `.npy` files automatically
3. **LLM Scoring**: Uses LLM to score demo quality (slow, cached)
4. **Training**: Trains reranker to predict LLM scores
5. **Checkpointing**: Saves best model to `output_dir/demo_reranker_*`

### Using the Trained Reranker

```bash
export demo_method="reranked"
export reranker_ckpt_path="./reranker_output/demo_reranker_100"
export TOP_K=10      # Retrieve 10 candidates
export TOP_K_2=3     # Rerank to top 3
```

## File Structure

```
GPTSwarm/
├── data/
│   └── final_demo_dec_13/
│       ├── list_demos_propose.json
│       ├── list_demos_propose.npy
│       ├── list_demos_if_correct.json
│       ├── list_demos_if_correct.npy
│       ├── list_demos_suggest.json
│       ├── list_demos_suggest.npy
│       ├── list_demos_value.json
│       └── list_demos_value.npy
├── icl_retrieval/
│   ├── reranker.py
│   ├── run_reranker_ft.py
│   ├── crosswords_operation.py
│   └── utils/
│       ├── io_utils.py
│       ├── encode_data.py
│       ├── retrievers.py
│       ├── logger_utils.py
│       └── existing_demos.py
└── scripts/
    ├── convert_demos.py
    └── generate_embeddings.py
```

## How It Works

### 1. Prompt Types

The system recognizes 4 prompt types:
- **propose**: Suggesting word candidates
- **if_correct**: Validating word-meaning pairs
- **suggest**: Generating improvement plans
- **value**: Evaluating word feasibility

### 2. Demo Retrieval Flow

```
User Query → Prompt Type Detection → FAISS Retrieval → [Optional: Reranking] → Demo Insertion → LLM
```

### 3. Special Markers

The system uses special markers in prompts:
- `<placeholder>`: Where demos are inserted
- `<current query>`: Marks the actual query for extraction
- `<cue>`: Optional hints (if `add_direct_cues=true`)

**Note**: These markers are handled by `crosswords_operation.py`, not in the prompt templates.

## Troubleshooting

### Issue: "EMBEDDING_MODEL_PATH not found"

**Solution**: Set the environment variable:
```bash
export EMBEDDING_MODEL_PATH="BAAI/bge-base-en-v1.5"
```

### Issue: "No module named 'icl_retrieval'"

**Solution**: Make sure you're running from the project root:
```bash
cd /Users/sihua/personal/GPTSwarm
python3 -m icl_retrieval.run_reranker_ft ...
```

### Issue: Embeddings not found

**Solution**: Generate embeddings first:
```bash
python3 scripts/generate_embeddings.py --demo_dir /path/to/demos
```

### Issue: CUDA out of memory during reranker training

**Solution**: Reduce batch size or use smaller model:
```bash
python3 icl_retrieval/run_reranker_ft.py \
  --gradient_accumulation_steps 32  # Increase from 16
```

## Performance Tips

1. **Start with `retrieved` mode**: Simpler and faster than reranking
2. **Tune TOP_K**: Try 5, 10, 15 to find optimal number
3. **Use reranking for best results**: But requires training time
4. **Cache is your friend**: Prompts are cached automatically

## Next Steps

1. ✅ Demo files converted
2. ⏳ Generate embeddings (run `generate_embeddings.py`)
3. ⏳ Set environment variables
4. ⏳ Test with `demo_method="retrieved"`
5. ⏳ (Optional) Train reranker
6. ⏳ (Optional) Switch to `demo_method="reranked"`

## Questions?

- Check the code in `icl_retrieval/` for implementation details
- Review `crosswords_operation.py` for integration logic
- See `run_reranker_ft.py` for training details
