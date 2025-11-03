"""
Base Retriever Interface and Implementations

This module provides a common interface for different retrieval methods,
making it easy to switch between BM25, FAISS, and other retrievers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import os

class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""

    def __init__(self, dataset_dir: str = "dataset", cache_dir: str = "cache", **kwargs):
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    @abstractmethod
    def build_index(self) -> None:
        """Build the retrieval index
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        pass

from faiss_retriever import FAISSRetriever
from bm25_retriever import BM25Retriever

# Factory function for easy retriever creation
def create_retriever(
    retriever_type: str,
    dataset_dir: str = "dataset",
    cache_dir: str = "cache",
    model_name: Optional[str] = None,
) -> BaseRetriever:
    """
    Factory function to create retrievers

    Args:
        retriever_type: "bm25", "faiss", or "hybrid"
        **kwargs: Additional arguments for the retriever (cache_dir, batch_size, etc.)
                 Note: dataset_dir should be provided separately to build_index()

    Returns:
        Configured retriever instance
    """
    retriever_type = retriever_type.lower()

    if retriever_type == "bm25":
        return BM25Retriever(dataset_dir=dataset_dir, cache_dir=cache_dir)
    elif retriever_type == "faiss":
        return FAISSRetriever(dataset_dir=dataset_dir, cache_dir=cache_dir, model_name=model_name)
    else:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. Choose from 'bm25', 'faiss'")

