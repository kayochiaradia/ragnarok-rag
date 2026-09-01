"""Recuperação híbrida: vetorial + lexical BM25, fundidas por RRF.

Busca puramente vetorial erra em nome próprio e sigla ("Ghostring", "GTB").
Busca puramente lexical erra em paráfrase ("como faço dinheiro" vs "zeny").
A fusão das duas por Reciprocal Rank Fusion resolve os dois casos sem
precisar de nenhum modelo de linguagem.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from .embeddings import Embedder
from .store import Hit, VectorStore
from .text import expand_query, tokenize

RRF_K = 60  # constante padrão do Reciprocal Rank Fusion
HEADING_BOOST = 0.3  # desempate moderado; uma palavra genérica não domina o corpo
TAG_BOOST = 0.15  # classificação do documento, limitada a três casamentos
LEXICAL_STRENGTH = 0.5  # preserva a vantagem de um casamento BM25 muito forte
HEADING_MIN_IDF = 2.0  # termos comuns como "dano" não decidem sozinhos


@dataclass
class RetrievedChunk:
    chunk_id: str
    score: float
    vector_rank: int | None
    lexical_rank: int | None
    vector_score: float
    lexical_score: float
    text: str
    metadata: dict[str, Any]

    @property
    def body(self) -> str:
        return self.metadata.get("body") or self.text

    @property
    def breadcrumb(self) -> str:
        return self.metadata.get("breadcrumb", "")

    @property
    def doc_titulo(self) -> str:
        return self.metadata.get("doc_titulo", "")

    @property
    def heading(self) -> str:
        return self.metadata.get("heading", "")

    @property
    def source(self) -> str:
        return self.metadata.get("source", "")


class BM25Index:
    """BM25 Okapi mínimo, construído em memória a partir do vector store."""

    def __init__(self, records: list[Hit], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.ids = [r.chunk_id for r in records]
        self.records = records
        # ``record.text`` contém breadcrumb, tags e corpo para o embedding. No
        # ramo lexical, porém, repetir as tags do documento em cada seção faz
        # todas elas parecerem igualmente relevantes. O BM25 deve medir o
        # conteúdo específico do chunk; cabeçalho e tags entram depois como
        # campos separados e limitados.
        self.docs: list[list[str]] = [
            tokenize(r.metadata.get("body") or r.text) for r in records
        ]
        self.doc_len = np.array([len(d) or 1 for d in self.docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 1.0

        self.term_freq: list[Counter[str]] = [Counter(d) for d in self.docs]
        doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc))

        total = max(len(self.docs), 1)
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def search(self, tokens: list[str], k: int) -> list[tuple[int, float]]:
        if not self.docs or not tokens:
            return []
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for term in tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.term_freq):
                freq = freqs.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        vector_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self._bm25: BM25Index | None = None

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.store.all_records())
        return self._bm25

    def search(self, query: str, *, top_k: int = 5, candidate_k: int = 20) -> list[RetrievedChunk]:
        if not query.strip():
            return []

        # O embedding hash é lexical: remover stopwords e expandir termos o
        # torna estável entre "qual o limite" e "limite". Embeddings neurais
        # recebem a pergunta natural para preservar o sinal semântico.
        tokens = expand_query(query)
        vector_query = " ".join(tokens) if self.embedder.name == "hash" else query

        # --- ramo vetorial ---
        query_vector = self.embedder.encode_one(vector_query)
        vector_hits = self.store.query(query_vector, candidate_k)

        # --- ramo lexical (com expansão de sinônimos do domínio) ---
        lexical_hits = self.bm25.search(tokens, candidate_k)

        fused: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(vector_hits):
            fused[hit.chunk_id] = {
                "hit": hit,
                "vector_rank": rank,
                "vector_score": hit.score,
                "lexical_rank": None,
                "lexical_score": 0.0,
                "score": self.vector_weight / (RRF_K + rank + 1),
            }

        max_lexical = max((s for _, s in lexical_hits), default=0.0) or 1.0
        for rank, (index, raw) in enumerate(lexical_hits):
            record = self.bm25.records[index]
            entry = fused.get(record.chunk_id)
            normalized_lexical = raw / max_lexical
            # RRF puro considera apenas posição: o primeiro e o quarto lugar
            # ficam quase empatados mesmo quando o BM25 mostra uma diferença
            # grande. Esta parcela limitada preserva essa evidência sem tornar
            # scores brutos (dependentes da consulta) comparáveis entre si.
            contribution = self.lexical_weight / (RRF_K + rank + 1)
            contribution += (
                self.lexical_weight
                * LEXICAL_STRENGTH
                * normalized_lexical
                / (RRF_K + 1)
            )
            if entry is None:
                fused[record.chunk_id] = {
                    "hit": record,
                    "vector_rank": None,
                    "vector_score": 0.0,
                    "lexical_rank": rank,
                    "lexical_score": normalized_lexical,
                    "score": contribution,
                }
            else:
                entry["lexical_rank"] = rank
                entry["lexical_score"] = normalized_lexical
                entry["score"] += contribution

        # Casamento de cabeçalho é um sinal forte de intenção: quem pergunta
        # "o que é WoE" quer a seção chamada "O que é a Guerra do Emperium",
        # não uma frase solta em outra seção do mesmo documento.
        query_set = set(tokens)
        heading_query_set = {
            token
            for token in query_set
            if self.bm25.idf.get(token, 0.0) >= HEADING_MIN_IDF
        }
        for entry in fused.values():
            heading_tokens = set(tokenize(entry["hit"].metadata.get("heading_path", "")))
            matches = len(heading_query_set & heading_tokens)
            if matches:
                entry["heading_matches"] = matches
                entry["score"] *= 1 + HEADING_BOOST * (1 - 1 / (1 + matches))

            # Tags classificam o documento inteiro. O bônus pequeno e
            # limitado ajuda a escolher o documento certo sem permitir que a
            # tag global vença uma seção cujo corpo realmente responde.
            tag_tokens = set(tokenize(entry["hit"].metadata.get("tags", "")))
            tag_matches = len(query_set & tag_tokens)
            if tag_matches:
                entry["tag_matches"] = tag_matches
                entry["score"] *= 1 + TAG_BOOST * min(tag_matches, 3)

        ranked = sorted(fused.values(), key=lambda e: e["score"], reverse=True)

        results: list[RetrievedChunk] = []
        seen_headings: set[str] = set()
        for entry in ranked:
            hit: Hit = entry["hit"]
            # Evita devolver duas fatias da mesma seção; diversidade importa
            # mais que redundância numa resposta curta de WhatsApp.
            key = hit.metadata.get("breadcrumb", hit.chunk_id)
            if key in seen_headings:
                continue
            seen_headings.add(key)
            results.append(
                RetrievedChunk(
                    chunk_id=hit.chunk_id,
                    score=float(entry["score"]),
                    vector_rank=entry["vector_rank"],
                    lexical_rank=entry["lexical_rank"],
                    vector_score=float(entry["vector_score"]),
                    lexical_score=float(entry["lexical_score"]),
                    text=hit.text,
                    metadata=hit.metadata,
                )
            )
            if len(results) >= top_k:
                break

        return results
