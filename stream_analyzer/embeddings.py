"""Sentence embeddings (MiniLM / BGE) with an offline TF-IDF + LSA fallback."""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .text import STOP, clean

class Embedder:
    def __init__(self, name: str, use_transformer: bool = True):
        self.model, self.name, self.error = None, "TF-IDF + LSA (offline fallback)", ""
        if not use_transformer:
            return
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(name, device=dev)
            self.model.max_seq_length = 128
            if dev == "cuda":
                self.model.half()
            self.name = name.split("/")[-1]
        except Exception as e:
            self.error = str(e)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        tf = TfidfVectorizer(stop_words=list(STOP), sublinear_tf=True, ngram_range=(1, 2),
                             token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
        try:
            X = tf.fit_transform([clean(t) for t in texts])
        except ValueError:
            return np.ones((len(texts), 1), dtype=np.float32)
        k = min(100, X.shape[1] - 1, X.shape[0] - 1)
        return normalize(X.toarray() if k < 2 else TruncatedSVD(k, random_state=0).fit_transform(X)).astype(np.float32)
