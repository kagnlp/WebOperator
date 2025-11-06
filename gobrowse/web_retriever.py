#!/usr/bin/env python3
"""
Simple Smart Query Router for Website-Specific Retrieval

Routes queries to appropriate website-specific indices based on URL patterns.
"""

import os
import re
from pathlib import Path
from beartype.typing import Dict, List, Optional, Tuple, Any
from base_retriever import create_retriever
from website_configs import WEBSITE_CONFIGS
from base_retriever import BaseRetriever


class WebRetriever():
    """Simple router for website-specific retrieval"""

    # automatically set relative to current file
    current_dir: Path = Path(__file__).resolve().parent
    base_dataset_dir: Path = current_dir / "websites"
    cache_base_dir: Path = current_dir / "website_indices"
    cache_base_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    retrievers_cache: Dict[str, BaseRetriever] = {}
    
    # @classmethod
    # def configure(cls,
    #              retriever_type: str = "bm25",
    #              model_name: Optional[str] = None,
    #              top_k: int = 5):
    #     """Initialize the router"""
    #     current_dir = Path(__file__).resolve().parent  # directory of current file
    #     cls.base_dataset_dir = current_dir / "websites"
    #     cls.cache_base_dir = current_dir / "website_indices"
    #     cls.cache_base_dir.mkdir(parents=True, exist_ok=True)

    # @staticmethod
    # def detect_website_from_query(query: str) -> Optional[str]:
    #     """Detect website from query based on simple patterns"""
    #     query_lower = query.lower()

    #     for website, url in {
    #         "gitlab": os.environ["WA_GITLAB"],
    #         "reddit": os.environ["WA_REDDIT"],
    #         "shopping": os.environ["WA_SHOPPING"],
    #         "shopping_admin": os.environ["WA_SHOPPING_ADMIN"],
    #         "wikipedia": os.environ["WA_WIKIPEDIA"],
    #         "map": os.environ["WA_MAP"],
    #     }:
    #         if url in query_lower:
    #             return website
    #     return None

    @classmethod
    def get_or_create_retriever(cls, website: str, model_name: str, retriever_type: str):
        """Get or create retriever for website"""
        cache_key = f"{retriever_type}-{model_name}-{website}"
        try:
            if cache_key in cls.retrievers_cache:
                return cls.retrievers_cache[cache_key]

            # Check if dataset exists
            dataset_dir = cls.base_dataset_dir / website
            if not dataset_dir.exists():
                print(f"Dataset directory {dataset_dir} does not exist")
                return None

            # Create cache directory
            cache_dir = cls.cache_base_dir / retriever_type / model_name / website
            
            # Create retriever
            retriever = create_retriever(retriever_type, dataset_dir, cache_dir, model_name=model_name)

            try:
                retriever.build_index()
                cls.retrievers_cache[cache_key] = retriever
                return retriever
            except ZeroDivisionError as e:
                print(f"Division by zero while building index for {website}: {e}")
                return None
            except Exception as e:
                print(f"Error building index for {website}: {e}")
                return None
                
        except Exception as e:
            print(f"Error creating retriever for {website}: {e}")
            return None

    @classmethod
    def get_fallback_retriever(cls, model_name: str, retriever_type: str):
        """Get fallback retriever for all websites"""
        cache_key = f"{retriever_type}-{model_name}-fallback"
        try:
            if cache_key in cls.retrievers_cache:
                return cls.retrievers_cache[cache_key]

            cache_dir = cls.cache_base_dir / retriever_type / model_name / "fallback"

            retriever = create_retriever(retriever_type, dataset_dir=str(cls.base_dataset_dir), cache_dir=str(cache_dir), model_name=model_name)

            try:
                retriever.build_index()
                cls.retrievers_cache[cache_key] = retriever
                return retriever
            except ZeroDivisionError as e:
                print(f"Division by zero while building fallback index: {e}")
                return None
            except Exception as e:
                print(f"Error building fallback index: {e}")
                return None
                
        except Exception as e:
            print(f"Error creating fallback retriever: {e}")
            return None

    @classmethod
    def search(cls, query: Dict[str, Any], model_name: str, top_k: int, retriever_type: str, website: str) -> List[Tuple[Dict[str, Any], float, str]]:
        """Search with automatic website routing"""
        try:
            # Try website-specific search
            # website = cls.detect_website_from_query(query["axtree_txt"])

            if website:
                try:
                    retriever = cls.get_or_create_retriever(website, model_name, retriever_type)
                    if retriever is None:
                        print(f"Website-specific retriever for {website} is None, falling back to general search")
                    else:
                        results = retriever.search(query, top_k=top_k)
                        # Convert to tuple format
                        return [(result.get('metadata', result),
                                result.get('score', 0.0),
                                f"{result.get('filename', 'unknown')}@{website}")
                                for result in results]
                except ZeroDivisionError as e:
                    print(f"Division by zero in website-specific search for {website}: {e}")
                    # Fall through to fallback search
                except Exception as e:
                    print(f"Error in website-specific search for {website}: {e}")
                    # Fall through to fallback search

            # Fallback to search all
            try:
                retriever = cls.get_fallback_retriever(model_name, retriever_type)
                if retriever is None:
                    print("Fallback retriever is None, returning empty results")
                    return []
                    
                results = retriever.search(query, top_k=top_k)
                return [(result.get('metadata', result),
                        result.get('score', 0.0),
                        f"{result.get('filename', 'unknown')}@fallback")
                        for result in results]
            except ZeroDivisionError as e:
                print(f"Division by zero in fallback search: {e}")
                return []  # Return empty results
            except Exception as e:
                print(f"Error in fallback search: {e}")
                return []  # Return empty results
                
        except Exception as e:
            print(f"Unexpected error in smart_search: {e}")
            return []  # Return empty results
