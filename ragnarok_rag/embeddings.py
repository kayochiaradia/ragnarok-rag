"""Backends de embedding.

Dois backends, ambos sem chave de API e sem chamada a modelo de linguagem:

- ``local``: sentence-transformers, multilingual MiniLM (384 dims). Baixa o
  modelo uma vez e depois roda offline. Melhor qualidade semântica.
- ``hash``:  embedding determinístico por hashing de n-gramas de caractere e
  de palavras, projetado em ``hash_dims`` dimensões e normalizado em L2.
  Zero download, zero dependência além de numpy. Serve de rede de segurança
  para o POC rodar em qualquer máquina.

``auto`` tenta o local e cai para o hash.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from .text import char_ngrams, tokenize


class Embedder(ABC):
    name: str
    dims: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Retorna matriz (n, dims) já normalizada em L2."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder(Embedder):
    """Embedding determinístico, offline, sem modelo treinado.

    Combina dois sinais no mesmo vetor:
      * n-gramas de caractere (robusto a erro de digitação e a flexão)
      * palavras inteiras com peso maior (sinal léxico forte)
    """

    name = "hash"

    def __init__(self, dims: int = 768, ngram: int = 4) -> None:
        self.dims = dims
        self.ngram = ngram

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.dims
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def _encode_single(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dims, dtype=np.float32)

        for gram in char_ngrams(text, self.ngram):
            index, sign = self._bucket(f"c:{gram}")
            vector[index] += sign

        for token in tokenize(text):
            index, sign = self._bucket(f"w:{token}")
            vector[index] += sign * 3.0

        return vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.vstack([self._encode_single(t) for t in texts]) if texts else np.zeros((0, self.dims), np.float32)
        return _l2_normalize(matrix.astype(np.float32))


class SentenceTransformerEmbedder(Embedder):
    name = "local"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # import tardio

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dims = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def build_embedder(backend: str, model_name: str, hash_dims: int) -> Embedder:
    backend = (backend or "auto").lower()

    if backend == "hash":
        return HashingEmbedder(dims=hash_dims)

    if backend in {"local", "auto"}:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # noqa: BLE001 - queremos o fallback sempre
            if backend == "local":
                raise RuntimeError(
                    f"Backend 'local' indisponível ({exc}). "
                    "Instale com: pip install sentence-transformers"
                ) from exc
            print(
                f"[embeddings] sentence-transformers indisponível ({type(exc).__name__}); "
                "usando o backend 'hash' offline."
            )
            return HashingEmbedder(dims=hash_dims)

    raise ValueError(f"Backend de embedding desconhecido: {backend}")
