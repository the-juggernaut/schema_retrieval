"""
Retriever Module
This module provides functionality to retrieve relevant document segments based on user queries.
"""

from typing import List, Optional
from schema_processor import FieldGroup
from document_segmenter import Segment

import numpy as np

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss


class FaissIndex:
    """
    Dense vector index using FAISS for fast similarity search. Accepts real embeddings.
    """

    def __init__(self, embedding_dim: Optional[int] = None):
        self.segments = []
        self.embeddings = None
        self.index = None
        self.embedding_dim = embedding_dim

    def build(self, segments: List[Segment], embeddings: np.ndarray):
        """
        Build FAISS index from provided segments and their dense embeddings.
        embeddings: np.ndarray of shape (num_segments, embedding_dim)

        """
        assert embeddings.shape[0] == len(
            segments
        ), "Embeddings and segments must match."
        self.segments = segments
        self.embeddings = embeddings.astype("float32")
        self.embedding_dim = embeddings.shape[1]
        # faiss-cpu is used by default unless faiss-gpu is installed
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Segment]:
        """
        query_embedding: np.ndarray of shape (embedding_dim,) or (1, embedding_dim)
        Returns top-k segments by similarity.
        """
        if self.index is None or self.embeddings is None or not self.segments:
            return []
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype("float32")
        scores, indices = self.index.search(query_embedding, k)
        top_indices = indices[0]
        return [self.segments[i] for i in top_indices if i < len(self.segments)]


class BM25Index:
    """
    Sparse BM25 index for text retrieval using TF-IDF scoring.
    """

    def __init__(self):
        self.segments = []
        self.texts = []
        self.vectorizer = None
        self.tfidf_matrix = None

    def build(self, segments: List[Segment]):
        self.segments = segments
        self.texts = [s.text for s in segments]
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=512)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, k: int = 10) -> List[Segment]:
        if self.tfidf_matrix is None or not self.segments or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = self.tfidf_matrix.dot(query_vec.T).toarray().flatten()
        top_indices = scores.argsort()[::-1][:k]
        return [self.segments[i] for i in top_indices]


class Retriever:
    """
    Hybrid retrieval engine supporting sparse (BM25), dense (FAISS), and hybrid strategies.
    Implements score normalization and merging with token-level statistics.
    """

    def __init__(self, embedding_dim: Optional[int] = None):
        self.bm25 = BM25Index()
        self.faiss = FaissIndex(embedding_dim=embedding_dim)
        self.segments: List[Segment] = []
        self.embeddings: Optional[np.ndarray] = None
        self._num_retrievals = 0
        self._total_retrieved_tokens = 0

        import tiktoken

        self._encoding = tiktoken.get_encoding("cl100k_base")

    @property
    def compute_stats(self):
        return {
            "num_retrievals": self._num_retrievals,
            "total_retrieved_tokens": self._total_retrieved_tokens,
        }

    def build_indexes(self, segments: List[Segment], embeddings: np.ndarray):
        """
        Initializes both BM25 and FAISS indexes from shared segments and dense embeddings.
        """
        self.segments = segments
        self.embeddings = embeddings
        self.bm25.build(segments)
        self.faiss.build(segments, embeddings)

    def retrieve(
        self,
        group: FieldGroup,
        k: int = 10,
        mode: str = "hybrid",
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Segment]:
        """
        Retrieve top-k segments relevant to a schema field group query using the selected mode.

        Args:
            group: FieldGroup containing shared context or query terms.
            k: Number of segments to return.
            mode: 'bm25', 'vector', or 'hybrid'.
            query_embedding: Required for 'vector' and 'hybrid' modes.

        Returns:
            A ranked list of Segment objects.
        """
        query = group.shared_context or " ".join(group.query_terms)

        if mode == "bm25":
            return self._finalize(self.bm25.search(query, k))

        if mode == "vector":
            if query_embedding is None:
                raise ValueError("query_embedding required for vector search")
            return self._finalize(self.faiss.search(query_embedding, k))

        if mode == "hybrid":
            bm25_scores = self._get_scores_bm25(query)
            faiss_scores = (
                self._get_scores_faiss(query_embedding)
                if query_embedding is not None
                else {}
            )
            merged = self._merge_scores(bm25_scores, faiss_scores)
            top_ids = sorted(merged, key=merged.get, reverse=True)[:k]
            id_to_seg = {s.id: s for s in self.segments}
            segments = [id_to_seg[i] for i in top_ids if i in id_to_seg]
            return self._finalize(segments)

        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def _get_scores_bm25(self, query: str) -> dict:
        if self.bm25.tfidf_matrix is None or self.bm25.vectorizer is None:
            return {}
        query_vec = self.bm25.vectorizer.transform([query])
        scores = self.bm25.tfidf_matrix.dot(query_vec.T).toarray().flatten()
        return {self.bm25.segments[i].id: float(scores[i]) for i in range(len(scores))}

    def _get_scores_faiss(self, query_embedding: np.ndarray) -> dict:
        if self.faiss.index is None or self.embeddings is None:
            return {}
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype("float32")
        scores, indices = self.faiss.index.search(query_embedding, len(self.segments))
        return {
            self.segments[int(indices[0][i])].id: float(scores[0][i])
            for i in range(len(self.segments))
        }

    def _merge_scores(
        self, sparse_scores: dict, dense_scores: dict, alpha: float = 0.5
    ) -> dict:
        """
        Merges and normalizes scores from both retrieval strategies.
        Avoids divide-by-zero even if all scores are zero.
        """
        merged = defaultdict(float)

        sparse_max = max(sparse_scores.values(), default=0.0)
        dense_max = max(dense_scores.values(), default=0.0)

        # Error handling for empty scores
        # Avoid zero division
        sparse_max = sparse_max if sparse_max > 0 else 1e-6
        dense_max = dense_max if dense_max > 0 else 1e-6

        for seg_id, score in sparse_scores.items():
            merged[seg_id] += (score / sparse_max) * alpha

        for seg_id, score in dense_scores.items():
            merged[seg_id] += (score / dense_max) * (1 - alpha)

        return dict(merged)

    def _finalize(self, segments: List[Segment]) -> List[Segment]:
        """
        Tracks token usage and retrieval count for performance metrics.
        """
        self._num_retrievals += 1
        self._total_retrieved_tokens += sum(
            len(self._encoding.encode(seg.text)) for seg in segments
        )
        return segments
