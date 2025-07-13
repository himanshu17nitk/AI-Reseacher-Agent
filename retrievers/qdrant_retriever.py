from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from services.embedding_client import EmbeddingClient
from utils.logger import api_logger


class RetrieverService:
    def __init__(self, collection_name: str = "rag_chunks"):
        self.embedding_client = EmbeddingClient()
        self.collection_name = collection_name
        self.qdrant = QdrantClient(path="./qdrant_data")

        if self.collection_name not in {c.name for c in self.qdrant.get_collections().collections}:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

    def store_chunks(self, texts: List[str], metadatas: List[dict] | None = None):
        """Store text chunks with their metadata in Qdrant."""
        try:
            if not texts:
                api_logger.warning("No texts provided to store_chunks")
                return
            
            # Validate and prepare metadata
            if metadatas is None:
                metadatas = [{} for _ in texts]
            elif len(metadatas) != len(texts):
                api_logger.warning(f"Metadata length ({len(metadatas)}) doesn't match texts length ({len(texts)}). Padding metadata.")
                # Pad metadata if it's shorter than texts
                while len(metadatas) < len(texts):
                    metadatas.append({})
                # Truncate if it's longer
                metadatas = metadatas[:len(texts)]
            
            api_logger.debug(f"Storing {len(texts)} chunks with {len(metadatas)} metadata entries")
            
            # Get embeddings
            embeddings = self.embedding_client.embed_texts(texts)
            
            if len(embeddings) != len(texts):
                api_logger.error(f"Embedding count ({len(embeddings)}) doesn't match text count ({len(texts)})")
                return
            
            # Create points
            points = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                try:
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={**{"text": text}, **metadatas[i]},
                    )
                    points.append(point)
                except Exception as e:
                    api_logger.error(f"Error creating point {i}: {e}")
                    continue
            
            if points:
                self.qdrant.upsert(collection_name=self.collection_name, points=points)
                api_logger.info(f"Successfully stored {len(points)} chunks in Qdrant")
            else:
                api_logger.warning("No valid points to store")
                
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                "Error storing chunks in Qdrant",
                "Check Qdrant connection and embedding service"
            )

    def retrieve_similar_chunks(self, query: str, k: int = 10) -> List[str]:
        """Retrieve similar chunks based on query."""
        try:
            api_logger.debug(f"Retrieving {k} similar chunks for query: {query[:100]}...")
            
            query_vector = self.embedding_client.embed_text(query)
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
            )
            
            chunks = [r.payload.get("text", "") for r in results]
            api_logger.debug(f"Retrieved {len(chunks)} chunks from Qdrant")
            return chunks
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                f"Error retrieving chunks for query: {query[:50]}...",
                "Check Qdrant connection and embedding service"
            )
            return []
