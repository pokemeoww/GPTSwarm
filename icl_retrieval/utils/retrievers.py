"""FAISS-based retriever for similarity search."""

import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from icl_retrieval.utils.io_utils import load_jsonl


class FaissRetriever:
    """FAISS-based retriever for finding similar examples."""
    
    def __init__(
        self,
        corpus_path: str,
        embeddings_path: str,
        embed_model: Optional[SentenceTransformer] = None
    ):
        """Initialize the FAISS retriever.
        
        Args:
            corpus_path: Path to JSONL file containing the corpus
            embeddings_path: Path to .npy file containing embeddings
            embed_model: SentenceTransformer model for encoding queries
        """
        # Load corpus
        self.corpus = load_jsonl(corpus_path)
        
        # Load embeddings
        self.embeddings = np.load(embeddings_path)
        
        # Store embedding model
        self.embed_model = embed_model
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Loaded {len(self.corpus)} samples with {dimension}-dim embeddings")
    
    def search_once(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search for similar examples given a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            Dictionary with 'query2query' key containing list of retrieved samples
        """
        if self.embed_model is None:
            raise ValueError("embed_model is required for encoding queries")
        
        # Encode query
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.corpus):
                result = {
                    "sample": self.corpus[idx],
                    "score": float(score)
                }
                results.append(result)
        
        # print(f"Retrieved {len(results)} results for query: {query[:50]}...")
        return {"query2query": results}
