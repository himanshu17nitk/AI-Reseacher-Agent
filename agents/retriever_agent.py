from typing import List, Dict
from retrievers.ensemble_retriever import EnsembleRetriever


class RetrieverAgent:
    def __init__(self, top_k: int = 5):
        """
        Args:
            retriever: The ensemble retriever instance.
            top_k (int): Number of top chunks to retrieve per question.
        """
        self.retriever = EnsembleRetriever()
        self.top_k = top_k

    def retrieve(self, question: str) -> List[str]:
        """
        Retrieves top relevant chunks for each question.

        Args:
            questions (List[str]): List of sub-questions to retrieve info for.

        Returns:
            Dict[str, List[str]]: Mapping of question to retrieved chunks.
        """
        chunks = self.retriever.retrieve_chunks(query=question, k=self.top_k)
        return chunks
