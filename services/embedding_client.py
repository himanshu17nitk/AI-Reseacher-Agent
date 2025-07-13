import requests
import time
from typing import List
from pydantic import BaseModel
from config import API_KEY, BASE_EMBEDDING_URL
from utils.logger import api_logger

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]  # Each embedding is a list of floats

class EmbeddingClient:  
    def __init__(self, model: str = "text-embedding-3-small"):
        self.api_key = API_KEY
        self.base_url = BASE_EMBEDDING_URL
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        api_logger.debug(f"EMBEDDING: Initialized EmbeddingClient | Model: {self.model} | Base URL: {self.base_url}")

    def embed_text(self, text: str) -> List[float]:
        start_time = time.time()

        try:
            api_logger.info(f"EMBEDDING: Starting single text embedding | Text length: {len(text)} | Model: {self.model}")
            api_logger.debug(f"EMBEDDING: Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")

            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": text
                }
            )
            api_time = time.time() - start_time

            if response.status_code != 200:
                api_logger.error(f"EMBEDDING: Single text embedding failed in {api_time:.3f}s | Status: {response.status_code} | Response: {response.text}")
                raise RuntimeError(f"Embedding failed: {response.text}")

            embedding = response.json()["data"][0]["embedding"]

            api_logger.info(f"EMBEDDING: Single text embedding completed in {api_time:.3f}s")
            return embedding

        except Exception as e:
            api_time = time.time() - start_time
            api_logger.error(f"EMBEDDING: Single text embedding failed in {api_time:.3f}s | Text length: {len(text)} | Model: {self.model}", exc_info=True)
            raise

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        start_time = time.time()

        try:
            if not texts:
                api_logger.warning("EMBEDDING: Empty input list received.")
                return []

            api_logger.info(f"EMBEDDING: Starting batch text embedding | Texts count: {len(texts)} | Model: {self.model}")
            api_logger.debug(f"EMBEDDING: First text preview: {texts[0][:100]}{'...' if len(texts[0]) > 100 else ''}")
            total_text_length = sum(len(text) for text in texts)
            api_logger.debug(f"EMBEDDING: Total text length: {total_text_length} characters")

            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json={
                    "model": self.model,
                    "input": texts
                }
            )
            api_time = time.time() - start_time

            if response.status_code != 200:
                api_logger.error(f"EMBEDDING: Batch embedding failed in {api_time:.3f}s | Status: {response.status_code} | Response: {response.text}")
                raise RuntimeError(f"Embedding failed: {response.text}")

            embedding_list = [item["embedding"] for item in response.json()["data"]]

            api_logger.info(f"EMBEDDING: Document embedding completed in {api_time:.3f}s")
            return embedding_list

        except Exception as e:
            api_time = time.time() - start_time
            api_logger.error(f"EMBEDDING: Batch embedding failed in {api_time:.3f}s | Texts count: {len(texts)} | Model: {self.model}", exc_info=True)
            raise
        