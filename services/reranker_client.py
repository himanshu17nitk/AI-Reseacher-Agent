import requests
from typing import List
from pydantic import BaseModel
from config import RERANKER_API_KEY, BASE_RERANKER_URL

class RerankedChunk(BaseModel):
    index: int
    text: str
    score: float

class RerankerClient:
    def __init__(self, model: str = "rerank-english-v2.0"):
        self.api_key = RERANKER_API_KEY
        self.base_url = BASE_RERANKER_URL 
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def rerank(self, query: str, texts: List[str], top_k: int = 5) -> List[str]:
        """Returns top_k most relevant chunks based on reranking."""
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,
            "top_n": top_k
        }

        response = requests.post(
            url=self.base_url,
            headers=self.headers,
            json=payload
        )

        if response.status_code != 200:
            raise RuntimeError(f"Rerank API failed: {response.text}")

        results = response.json().get("results", [])

        # Return top_k text entries sorted by score
        return [texts[item["index"]] for item in sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_k]]
        