from rank_bm25 import BM25Okapi
from typing import List
import re


class BM25Retriever:
    def __init__(self):
        self.tokenized_corpus = []
        self.corpus_chunks = []
        self.bm25 = None

    def preprocess(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def store_chunks(self, chunks: List[str]):
        self.corpus_chunks = chunks
        self.tokenized_corpus = [self.preprocess(doc) for doc in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve_chunks(self, query: str, k: int = 10) -> List[str]:
        if not self.bm25:
            return []

        tokenized_query = self.preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.corpus_chunks[i] for i in top_indices]
