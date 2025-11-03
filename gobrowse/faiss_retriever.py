import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from tqdm import tqdm
from base_retriever import BaseRetriever


class FAISSRetriever(BaseRetriever):
    """FAISS dense retriever implementation"""

    def __init__(
        self,
        dataset_dir: str = "dataset",
        cache_dir: str = "faiss_cache",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        max_memory_mb: int = 4000,
        use_gpu: bool = None,
        **kwargs
    ):
        super().__init__(dataset_dir, cache_dir)
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.use_gpu = use_gpu
        
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.index = None
        self.metadata = []

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        self.index_file = os.path.join(cache_dir, "faiss_index.bin")
        self.metadata_file = os.path.join(cache_dir, "metadata.pkl")

        # Import and initialize the FAISS retriever
        # self._retriever = FAISSRetrieverImpl(
        #     dataset_dir=dataset_dir,
        #     cache_dir=cache_dir,
        #     model_name=model_name,
        #     batch_size=batch_size,
        #     max_memory_mb=max_memory_mb,
        #     use_gpu=use_gpu if use_gpu is not None else True
        # )

    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def extract_text_from_json(self, json_data: dict) -> str:
        """Extract labeled text from WebArena JSON data (flat structure)"""
        keys = ['goal', 'axtree_txt']
        text_parts = [
            f"{k}: {json_data[k].strip()}" 
            for k in keys 
            if k in json_data and json_data[k].strip()
        ]
        return ", ".join(text_parts)
    
    def load_cached_index(self) -> bool:
        """Try to load cached index"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                return True
        except Exception:
            pass
        return False

    def save_index_to_cache(self):
        """Save index to cache"""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def build_index(self) -> None:
        """Build FAISS index"""
        # Try to load cached index first
        if self.load_cached_index():
            return

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}")

        # Load model
        self.load_model()

        # Collect JSON files from all subdirectories (for fallback) or current directory
        all_json_files = []
        
        # Check if this is a fallback retriever (dataset_dir contains multiple website folders)
        subdirs = [d for d in os.listdir(self.dataset_dir) 
                  if os.path.isdir(os.path.join(self.dataset_dir, d))]
        
        if subdirs:
            # This is the fallback case - collect from all website subdirectories
            print(f"Fallback FAISS retriever: collecting from subdirectories: {subdirs}")
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
            print(f"Website-specific FAISS retriever: collecting from {self.dataset_dir}")
            json_files = [f for f in os.listdir(self.dataset_dir) if f.endswith('.json')]
            all_json_files = [(os.path.join(self.dataset_dir, f), f, os.path.basename(self.dataset_dir)) 
                             for f in json_files]

        if not all_json_files:
            print(f"No JSON files found in {self.dataset_dir}")
            # Create dummy data for empty directories
            self.metadata = []
            # Create a minimal FAISS index with dummy embedding
            dummy_embedding = np.array([[0.0] * 384], dtype=np.float32)  # MiniLM dimension
            self.index = faiss.IndexFlatL2(384)
            self.index.add(dummy_embedding)
            return

        print(f"Processing {len(all_json_files)} JSON files total")

        documents = []
        metadata = []

        # Use tqdm to show progress while processing files
        with tqdm(all_json_files, desc="Building FAISS index", unit="files") as pbar:
            for filepath, filename, source_dir in pbar:
                pbar.set_postfix({"Current": filename[:30] + "..." if len(filename) > 30 else filename})
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)

                    text_content = self.extract_text_from_json(json_data["step_data"])
                    if text_content.strip():
                        documents.append(text_content)
                        metadata.append({
                            'filename': filename,
                            'filepath': filepath,
                            'source_dir': source_dir,  # Track which website this came from
                            'metadata': json_data,
                            'preview': text_content[:200]
                        })
                except Exception as e:
                    print(f"\nError processing {filepath}: {e}")
                    continue

        if not documents:
            print(f"No valid documents found in {self.dataset_dir}")
            # Create dummy data for empty documents
            self.metadata = []
            # Create a minimal FAISS index with dummy embedding
            dummy_embedding = np.array([[0.0] * 384], dtype=np.float32)  # MiniLM dimension
            self.index = faiss.IndexFlatL2(384)
            self.index.add(dummy_embedding)
            return

        try:
            self.metadata = metadata

            # Generate embeddings
            embeddings = self.model.encode(documents, convert_to_numpy=True)
            # response = requests.post(
            #     os.environ["SENTENCE_TRANSFORMER_API_SERVER"] + "/api/v1/encode",
            #     json={"documents": documents, "model": self.model_name}
            # )
            # embeddings = np.array(response.json()["embeddings"], dtype=np.float32)

            # Build FAISS index
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings.astype(np.float32))

            # Save to cache
            self.save_index_to_cache()
            print(f"Successfully built FAISS index with {len(documents)} documents from {self.dataset_dir}")
            if subdirs:
                print(f"Documents by source: {dict((src, len([m for m in metadata if m['source_dir'] == src])) for src in set(m['source_dir'] for m in metadata))}")
        except Exception as e:
            print(f"Error creating FAISS index: {e}")
            # Create dummy data to avoid complete failure
            self.metadata = []
            # Create a minimal FAISS index with dummy embedding  
            dummy_embedding = np.array([[0.0] * 384], dtype=np.float32)  # MiniLM dimension
            self.index = faiss.IndexFlatL2(384)
            self.index.add(dummy_embedding)

    def search(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """Search using FAISS"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        if self.model is None:
            self.load_model()

        results = []
        # If we only have dummy data, return empty results
        if not self.metadata:
            return []

        try:
            # Encode query
            text_content = self.extract_text_from_json(query)
            # response = requests.post(
            #     os.environ["SENTENCE_TRANSFORMER_API_SERVER"] + "/api/v1/encode",
            #     json={"documents": [text_content], "model": self.model_name}
            # )
            # query_embedding = np.array(response.json()["embeddings"], dtype=np.float32)
            query_embedding = self.model.encode([text_content], convert_to_numpy=True)

            # Search
            scores, indices = self.index.search(
                query_embedding.astype(np.float32), top_k)

            # Format results
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx < len(self.metadata):  # Safety check
                    results.append({
                        'metadata': self.metadata[int(idx)]['metadata'],
                        'score': float(score),
                        'filename': self.metadata[int(idx)]['filename']
                    })
            
        except Exception as e:
            print(f"Error during FAISS search: {e}")
            return []

        # Standardize result format
        standardized_results = []
        for result in results:
            # Handle different result formats from FAISS retriever
            if isinstance(result, tuple) and len(result) >= 2:
                metadata, score = result[:2]
                standardized_result = {
                    'filename': metadata.get('filename', ''),
                    'filepath': metadata.get('filepath', ''),
                    'score': float(score),
                    'retriever_type': 'FAISS',
                    'metadata': metadata
                }
            else:
                # Handle dict format
                standardized_result = {
                    'filename': result.get('filename', ''),
                    'filepath': result.get('filepath', ''),
                    'score': result.get('score', 0.0),
                    'retriever_type': 'FAISS',
                    'metadata': result
                }
            standardized_results.append(standardized_result)

        return standardized_results