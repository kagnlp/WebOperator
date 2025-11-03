import re
import pickle
import json
import os
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from base_retriever import BaseRetriever

class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever implementation"""

    def __init__(
        self,
        dataset_dir: str = "dataset",
        cache_dir: str = "bm25_cache",
        batch_size: int = 800,
        max_memory_mb: int = 4000,
    ):
        super().__init__(dataset_dir, cache_dir)
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb

        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.metadata = []
        self.bm25 = None
        self.index_file = os.path.join(cache_dir, "bm25_index.pkl")
        self.metadata_file = os.path.join(cache_dir, "metadata.pkl")

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Import and initialize the memory-efficient BM25 retriever
        # self._retriever = BM25RetrieverImpl(
        #     dataset_dir=dataset_dir,
        #     batch_size=batch_size,
        #     cache_dir=cache_dir,
        #     max_memory_mb=max_memory_mb
        # )
    
    # def extract_text_from_json(self, json_data: Dict) -> str:
    #     """Extract searchable text from JSON data"""
    #     text_parts = []

    #     def extract_recursive(obj):
    #         if isinstance(obj, dict):
    #             for key, value in obj.items():
    #                 if key in ['prompt', 'completion', 'content', 'observation', 'action', 'goal', 'instruction', 'text']:
    #                     if isinstance(value, str):
    #                         text_parts.append(value)
    #                     elif isinstance(value, list):
    #                         for item in value:
    #                             if isinstance(item, dict) and 'content' in item:
    #                                 text_parts.append(item['content'])
    #                 extract_recursive(value)
    #         elif isinstance(obj, list):
    #             for item in obj:
    #                 extract_recursive(item)

    #     extract_recursive(json_data)
    #     return " ".join(text_parts)

    def extract_text_from_json(self, json_data: dict) -> str:
        """Extract labeled text from WebArena JSON data (flat structure)"""
        keys = ['goal', 'axtree_txt']
        text_parts = [
            f"{k}: {json_data[k].strip()}" 
            for k in keys 
            if k in json_data and json_data[k].strip()
        ]
        return ", ".join(text_parts)

    def preprocess_text(self, text: str) -> List[str]:
        """Simple text preprocessing"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

    def build_index(self) -> None:
        """Build BM25 index"""

        # Try to load cached index first
        if self.load_cached_index():
            return

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}")

        # Collect JSON files from all subdirectories (for fallback) or current directory
        all_json_files = []
        
        # Check if this is a fallback retriever (dataset_dir contains multiple website folders)
        subdirs = [d for d in os.listdir(self.dataset_dir) 
                  if os.path.isdir(os.path.join(self.dataset_dir, d))]
        
        if subdirs:
            # This is the fallback case - collect from all website subdirectories
            print(f"Fallback retriever: collecting from subdirectories: {subdirs}")
            for subdir in tqdm(subdirs, desc="Scanning directories", unit="dirs"):
                subdir_path = os.path.join(self.dataset_dir, subdir)
                try:
                    json_files_in_subdir = [
                        (os.path.join(subdir_path, f), f, subdir) 
                        for f in os.listdir(subdir_path) 
                        if f.endswith('.json')
                    ]
                    all_json_files.extend(json_files_in_subdir)
                    print(f"Found {len(json_files_in_subdir)} JSON files in {subdir}")
                except Exception as e:
                    print(f"Error scanning directory {subdir}: {e}")
                    continue
        else:
            # This is a website-specific case - collect from current directory
            print(f"Website-specific retriever: collecting from {self.dataset_dir}")
            json_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.json')]
            all_json_files = [(os.path.join(self.dataset_dir, f), f, os.path.basename(self.dataset_dir)) 
                             for f in json_files]

        if not all_json_files:
            print(f"No JSON files found in {self.dataset_dir}")
            # Create a dummy document to avoid division by zero
            self.metadata = []
            self.bm25 = BM25Okapi([["dummy"]])
            return

        print(f"Processing {len(all_json_files)} JSON files total")

        all_docs = []
        all_metadata = []

        # Use tqdm to show progress while processing files
        with tqdm(all_json_files, desc="Building BM25 index", unit="files") as pbar:
            for filepath, filename, source_dir in pbar:
                pbar.set_postfix({"Current": filename[:30] + "..." if len(filename) > 30 else filename})
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                    text_content = self.extract_text_from_json(json_data["step_data"])
        
                    if text_content.strip():
                        tokens = self.preprocess_text(text_content)
                        if tokens:  # Only add if tokens are not empty
                            all_docs.append(tokens)
                            all_metadata.append({
                                'filename': filename,
                                'filepath': filepath,
                                'source_dir': source_dir,  # Track which website this came from
                                'metadata': json_data,
                                'preview': text_content[:200]
                            })
                except Exception as e:
                    print(f"\nError processing {filepath}: {e}")
                    continue

        # Check if we have any valid documents
        if not all_docs:
            print(f"No valid documents found in {self.dataset_dir}")
            # Create a dummy document to avoid division by zero
            self.metadata = []
            self.bm25 = BM25Okapi([["dummy"]])
            return

        # Check if all documents are empty
        non_empty_docs = [doc for doc in all_docs if doc]
        if not non_empty_docs:
            print(f"All documents are empty in {self.dataset_dir}")
            # Create a dummy document to avoid division by zero
            self.metadata = []
            self.bm25 = BM25Okapi([["dummy"]])
            return

        try:
            self.metadata = all_metadata
            self.bm25 = BM25Okapi(all_docs)
            self.save_index_to_cache()
            print(f"Successfully built BM25 index with {len(all_docs)} documents from {self.dataset_dir}")
            if subdirs:
                print(f"Documents by source: {dict((src, len([m for m in all_metadata if m['source_dir'] == src])) for src in set(m['source_dir'] for m in all_metadata))}")
        except ZeroDivisionError as e:
            print(f"Division by zero error in BM25Okapi: {e}")
            print(f"Number of documents: {len(all_docs)}")
            print(f"Sample document lengths: {[len(doc) for doc in all_docs[:5]]}")
            # Create a dummy document to avoid complete failure
            self.metadata = []
            self.bm25 = BM25Okapi([["dummy"]])
        except Exception as e:
            print(f"Error creating BM25 index: {e}")
            # Create a dummy document to avoid complete failure
            self.metadata = []
            self.bm25 = BM25Okapi([["dummy"]])

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")

        # If we only have dummy documents, return empty results

        results = []
        if not self.metadata:
            return []

        text_content = self.extract_text_from_json(query)
        query_tokens = self.preprocess_text(text_content)
        if not query_tokens:
            return []

        try:
            scores = self.bm25.get_scores(query_tokens)

            # Convert to numpy array if needed
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)

            # Check for valid scores
            if len(scores) == 0 or np.all(scores == 0):
                return []

            # Get top-k results
            top_indices = np.argsort(scores)[-top_k:][::-1]

            for idx in top_indices:
                if idx < len(self.metadata):  # Safety check
                    results.append({
                        'metadata': self.metadata[int(idx)]['metadata'],
                        'score': float(scores[idx]),
                        'filename': self.metadata[int(idx)]['filename']
                    })

        except Exception as e:
            print(f"Error during BM25 search: {e}")
            return []

        # Standardize result format
        standardized_results = []
        for result in results:
            standardized_result = {
                'filename': result.get('filename', ''),
                'filepath': result.get('filepath', ''),
                'score': result.get('score', 0.0),
                'retriever_type': 'BM25',
                'metadata': result
            }
            standardized_results.append(standardized_result)

        return standardized_results
    
    def load_cached_index(self) -> bool:
        """Try to load cached index"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                with open(self.index_file, 'rb') as f:
                    self.bm25 = pickle.load(f)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                return True
        except Exception:
            pass
        return False

    def save_index_to_cache(self):
        """Save index to cache"""
        try:
            with open(self.index_file, 'wb') as f:
                pickle.dump(self.bm25, f)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")
