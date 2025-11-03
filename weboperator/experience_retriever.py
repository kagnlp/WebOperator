from beartype.typing import Dict, List, Optional, Tuple, Any
import os
class ExperienceRetriever():
    """Simple router for website-specific retrieval"""
    retriever_type: str = "bm25"
    model_name: Optional[str] = None
    top_k: int = 5

    @classmethod
    def configure(cls,
                 retriever_type: str = "bm25",
                 model_name: Optional[str] = None,
                 top_k: int = 5):
        """Initialize the router"""
        cls.retriever_type = retriever_type
        cls.model_name = model_name
        cls.top_k = top_k
    
    @classmethod
    def get_examples(cls, goal: str, obs: str):
        """Get examples based on query"""
        import requests
        response = requests.post(
            os.environ["RETRIEVER_API_SERVER"] + "/api/v1/search",
            json={"goal": goal, "obs": obs, "model": cls.model_name, "top_k": cls.top_k, "r_type": cls.retriever_type},
        )
        if response.status_code == 200:
            return response.json()["examples"]
        return []