"""
Embedding Engine for TriTopic
==============================

Handles document embedding with support for multiple models:
- Sentence-BERT models (default)
- Instructor models (task-specific)
- BGE models (multilingual)
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from tqdm import tqdm


class EmbeddingEngine:
    """
    Generate document embeddings using transformer models.
    
    Supports various embedding models optimized for different use cases.
    
    Parameters
    ----------
    model_name : str
        Name of the sentence-transformers model. Popular choices:
        - "all-MiniLM-L6-v2": Fast, good quality (default)
        - "all-mpnet-base-v2": Higher quality, slower
        - "BAAI/bge-base-en-v1.5": State-of-the-art
        - "BAAI/bge-m3": Multilingual
        - "hkunlp/instructor-large": Task-specific (use with instruction)
    batch_size : int
        Batch size for encoding. Default: 32
    device : str or None
        Device to use ("cuda", "cpu", or None for auto).
    show_progress : bool
        Show progress bar. Default: True
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.show_progress = show_progress
        
        self._model = None
        self._is_instructor = "instructor" in model_name.lower()
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return
        
        from sentence_transformers import SentenceTransformer
        
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
    
    def encode(
        self,
        documents: list[str],
        instruction: str | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode documents to embeddings.
        
        Parameters
        ----------
        documents : list[str]
            List of document texts.
        instruction : str, optional
            Instruction for Instructor models (e.g., "Represent the topic of this document:").
        normalize : bool
            Whether to L2-normalize embeddings. Default: True
            
        Returns
        -------
        embeddings : np.ndarray
            Document embeddings of shape (n_docs, embedding_dim).
        """
        self._load_model()
        
        # Handle instructor models
        if self._is_instructor and instruction:
            documents = [[instruction, doc] for doc in documents]
        
        # Encode in batches
        embeddings = self._model.encode(
            documents,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        
        return embeddings
    
    def encode_with_pooling(
        self,
        documents: list[str],
        pooling: Literal["mean", "max", "cls"] = "mean",
    ) -> np.ndarray:
        """
        Encode with custom pooling strategy.
        
        Parameters
        ----------
        documents : list[str]
            Document texts.
        pooling : str
            Pooling strategy: "mean", "max", or "cls".
            
        Returns
        -------
        embeddings : np.ndarray
            Pooled embeddings.
        """
        # For now, use default pooling from model
        # Custom pooling would require access to token-level embeddings
        return self.encode(documents)
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()
    
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.
        
        Parameters
        ----------
        embeddings1 : np.ndarray
            First set of embeddings.
        embeddings2 : np.ndarray, optional
            Second set. If None, compute pairwise similarity of embeddings1.
            
        Returns
        -------
        similarity : np.ndarray
            Similarity matrix.
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        if embeddings2 is None:
            return cosine_similarity(embeddings1)
        return cosine_similarity(embeddings1, embeddings2)


class MultiModelEmbedding:
    """
    Combine embeddings from multiple models.
    
    Useful for ensemble approaches where different models capture
    different aspects of document semantics.
    """
    
    def __init__(
        self,
        model_names: list[str],
        weights: list[float] | None = None,
        batch_size: int = 32,
    ):
        self.model_names = model_names
        self.weights = weights or [1.0 / len(model_names)] * len(model_names)
        self.batch_size = batch_size
        
        self._engines = [
            EmbeddingEngine(name, batch_size=batch_size)
            for name in model_names
        ]
    
    def encode(
        self,
        documents: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode using all models and combine.
        
        Parameters
        ----------
        documents : list[str]
            Document texts.
        normalize : bool
            Normalize final embeddings.
            
        Returns
        -------
        embeddings : np.ndarray
            Combined embeddings (concatenated).
        """
        all_embeddings = []
        
        for engine, weight in zip(self._engines, self.weights):
            emb = engine.encode(documents, normalize=True)
            all_embeddings.append(emb * weight)
        
        # Concatenate
        combined = np.hstack(all_embeddings)
        
        if normalize:
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            combined = combined / (norms + 1e-10)
        
        return combined
