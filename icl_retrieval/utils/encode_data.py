"""Utilities for encoding text data into embeddings."""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List
from icl_retrieval.utils.io_utils import load_jsonl
from icl_retrieval.utils.device_utils import get_device


def encode_corpus(
    corpus_path: str,
    model_name_or_path: str,
    to_path: str,
    batch_size: int = 64
) -> None:
    """Encode a corpus of text samples into embeddings and save as .npy file.
    
    Args:
        corpus_path: Path to JSONL file containing samples with 'input' field
        model_name_or_path: Path or name of sentence transformer model
        to_path: Path to save the embeddings (.npy file)
        batch_size: Batch size for encoding
    """
    # Load corpus
    corpus = load_jsonl(corpus_path)
    
    # Extract input texts
    texts = [sample.get("input", "") for sample in corpus]
    
    # Load embedding model and move to device
    device = get_device()
    print(f"Loading embedding model: {model_name_or_path}")
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name_or_path)
    model = model.to(device)
    
    # Encode in batches
    print(f"Encoding {len(texts)} samples...")
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    
    # Save to file
    print(f"Saving embeddings to {to_path}")
    np.save(to_path, all_embeddings)
    print(f"Saved {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")
