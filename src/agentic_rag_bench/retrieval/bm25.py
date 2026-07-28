"""Okapi BM25 implemented in-repo: deterministic, dependency-free, tested."""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by BM25 and the reranker."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 (k1/b defaults per Robertson & Zaragoza 2009)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._doc_freqs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0

    def __len__(self) -> int:
        return len(self._doc_ids)

    def fit(self, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError(f"{len(doc_ids)} ids but {len(texts)} texts")
        self._doc_ids = list(doc_ids)
        self._doc_freqs = []
        self._doc_lens = []
        self._df = Counter()
        for text in texts:
            tokens = tokenize(text)
            freqs = Counter(tokens)
            self._doc_freqs.append(freqs)
            self._doc_lens.append(len(tokens))
            self._df.update(freqs.keys())
        total = sum(self._doc_lens)
        self._avgdl = total / len(self._doc_lens) if self._doc_lens else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._doc_ids)
        df = self._df.get(term, 0)
        # BM25+-style floor at 0 avoids negative IDF for very common terms.
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return up to ``top_k`` (doc_id, score) pairs with score > 0, best first."""
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not self._doc_ids or self._avgdl == 0.0:
            return []
        query_terms = tokenize(query)
        scores = [0.0] * len(self._doc_ids)
        for term in query_terms:
            idf = self._idf(term)
            if idf == 0.0:
                continue
            for i, freqs in enumerate(self._doc_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * self._doc_lens[i] / self._avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(
            ((self._doc_ids[i], s) for i, s in enumerate(scores) if s > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:top_k]
