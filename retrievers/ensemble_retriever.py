from typing import List, Dict
from collections import defaultdict
from retrievers.bm25_retriever import BM25Retriever
from retrievers.qdrant_retriever import RetrieverService
from utils.logger import api_logger


class EnsembleRetriever:
    def __init__(self, bm25_weight=0.4, qdrant_weight=0.6, top_k=10):
        self.bm25 = BM25Retriever()
        self.qdrant = RetrieverService()
        self.bm25_weight = bm25_weight
        self.qdrant_weight = qdrant_weight
        self.top_k = top_k
        api_logger.info(f"EnsembleRetriever initialized with weights: BM25={bm25_weight}, Qdrant={qdrant_weight}")

    def store_chunks(self, chunks: List[str], metadata: List[Dict] | None = None):
        """Store chunks in both BM25 and Qdrant retrievers."""
        try:
            if not chunks:
                api_logger.warning("No chunks provided to store_chunks")
                return
            
            api_logger.debug(f"Storing {len(chunks)} chunks in ensemble retriever")
            
            # Validate metadata
            if metadata is None:
                metadata = [{} for _ in chunks]
            elif len(metadata) != len(chunks):
                api_logger.warning(f"Metadata length ({len(metadata)}) doesn't match chunks length ({len(chunks)}). Padding metadata.")
                # Pad metadata if it's shorter than chunks
                while len(metadata) < len(chunks):
                    metadata.append({})
                # Truncate if it's longer
                metadata = metadata[:len(chunks)]
            
            # Store in BM25 (no metadata needed)
            try:
                self.bm25.store_chunks(chunks)
                api_logger.debug(f"Stored {len(chunks)} chunks in BM25")
            except Exception as e:
                api_logger.log_error_with_recovery(
                    e,
                    "Error storing chunks in BM25",
                    "BM25 storage failed, continuing with Qdrant"
                )
            
            # Store in Qdrant (with metadata)
            try:
                self.qdrant.store_chunks(chunks, metadata)
                api_logger.debug(f"Stored {len(chunks)} chunks in Qdrant")
            except Exception as e:
                api_logger.log_error_with_recovery(
                    e,
                    "Error storing chunks in Qdrant",
                    "Qdrant storage failed"
                )
            
            api_logger.info(f"Successfully stored {len(chunks)} chunks in ensemble retriever")
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                "Error in ensemble store_chunks",
                "Check both BM25 and Qdrant services"
            )

    def retrieve_chunks(self, query: str, k: int = 10) -> List[str]:
        """Retrieve chunks using ensemble of BM25 and Qdrant."""
        try:
            api_logger.debug(f"Retrieving {k} chunks for query: {query[:100]}...")
            
            # Get results from both retrievers
            bm25_chunks = []
            qdrant_chunks = []
            
            try:
                bm25_chunks = self.bm25.retrieve_chunks(query, k=2 * k)
                api_logger.debug(f"BM25 retrieved {len(bm25_chunks)} chunks")
            except Exception as e:
                api_logger.log_error_with_recovery(
                    e,
                    "Error retrieving from BM25",
                    "BM25 retrieval failed, using only Qdrant"
                )
            
            try:
                qdrant_chunks = self.qdrant.retrieve_similar_chunks(query, k=2 * k)
                api_logger.debug(f"Qdrant retrieved {len(qdrant_chunks)} chunks")
            except Exception as e:
                api_logger.log_error_with_recovery(
                    e,
                    "Error retrieving from Qdrant",
                    "Qdrant retrieval failed, using only BM25"
                )
            
            # If both failed, return empty list
            if not bm25_chunks and not qdrant_chunks:
                api_logger.warning("Both retrievers failed to return results")
                return []
            
            # Score and combine results
            score_map = defaultdict(float)

            for i, chunk in enumerate(bm25_chunks):
                score_map[chunk] += self.bm25_weight * (1 - i / len(bm25_chunks))

            for i, chunk in enumerate(qdrant_chunks):
                score_map[chunk] += self.qdrant_weight * (1 - i / len(qdrant_chunks))

            sorted_chunks = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
            result = [chunk for chunk, _ in sorted_chunks[:k]]
            
            api_logger.info(f"Ensemble retrieval completed. Returned {len(result)} chunks")
            return result
            
        except Exception as e:
            api_logger.log_error_with_recovery(
                e,
                f"Error in ensemble retrieval for query: {query[:50]}...",
                "Ensemble retrieval failed"
            )
            return []
