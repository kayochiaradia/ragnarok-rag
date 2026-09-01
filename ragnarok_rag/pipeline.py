"""Pipeline de ponta a ponta: ingestão e consulta."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .answer import Answer, compose
from .chunker import chunk_documents
from .config import CONFIG, Config
from .embeddings import build_embedder
from .loader import load_documents
from .retriever import HybridRetriever, RetrievedChunk
from .store import build_store

MANIFEST_NAME = "manifest.json"


class RagnarokRAG:
    """Fachada única do sistema. Sem LLM em nenhum ponto do caminho."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or CONFIG
        self.config.ensure_dirs()
        self.embedder = build_embedder(
            self.config.embedding_backend,
            self.config.embedding_model,
            self.config.hash_dims,
        )
        self.store = build_store(
            self.config.store_backend,
            self.config.index_dir,
            self.config.collection,
        )
        self.retriever = HybridRetriever(
            self.store,
            self.embedder,
            vector_weight=self.config.vector_weight,
            lexical_weight=self.config.lexical_weight,
        )

    # ------------------------------------------------------------------ #
    # Ingestão
    # ------------------------------------------------------------------ #
    def ingest(self, *, verbose: bool = True) -> dict[str, Any]:
        started = time.perf_counter()

        docs = load_documents(self.config.corpus_dir)
        chunks = chunk_documents(
            docs,
            self.config.chunk_target_chars,
            self.config.chunk_overlap_chars,
        )
        if verbose:
            print(f"[ingest] {len(docs)} documentos -> {len(chunks)} chunks")
            print(f"[ingest] embedding backend: {self.embedder.name} ({self.embedder.dims} dims)")
            print(f"[ingest] vector store: {self.store.name}")

        texts = [c.embed_text() for c in chunks]
        vectors = self.embedder.encode(texts)

        self.store.reset(self.embedder.dims)
        self.store.add(
            ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            texts=texts,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "doc_titulo": c.doc_titulo,
                    "categoria": c.categoria,
                    "heading": c.heading,
                    "heading_path": " > ".join(c.heading_path),
                    "breadcrumb": c.breadcrumb,
                    "tags": ", ".join(c.tags),
                    "source": c.meta.get("source", ""),
                    "ordinal": c.ordinal,
                    "body": c.text,
                }
                for c in chunks
            ],
        )
        self.store.persist()
        self.retriever._bm25 = None  # força reconstrução do índice lexical

        manifest = {
            "documentos": len(docs),
            "chunks": len(chunks),
            "embedding_backend": self.embedder.name,
            "embedding_dims": self.embedder.dims,
            "store_backend": self.store.name,
            "chunk_target_chars": self.config.chunk_target_chars,
            "corpus_dir": str(self.config.corpus_dir),
            "gerado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duracao_s": round(time.perf_counter() - started, 2),
            "por_documento": {d.doc_id: d.titulo for d in docs},
        }
        (Path(self.config.index_dir) / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if verbose:
            print(f"[ingest] concluído em {manifest['duracao_s']}s -> {self.config.index_dir}")
        return manifest

    # ------------------------------------------------------------------ #
    # Consulta
    # ------------------------------------------------------------------ #
    def search(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.search(
            query,
            top_k=top_k or self.config.top_k,
            candidate_k=self.config.candidate_k,
        )

    def ask(self, query: str, *, top_k: int | None = None) -> Answer:
        chunks = self.search(query, top_k=top_k)
        return compose(
            query,
            chunks,
            max_chars=self.config.answer_max_chars,
            related_count=self.config.related_count,
            min_score=self.config.min_score,
        )

    # ------------------------------------------------------------------ #
    # Estado
    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, Any]:
        manifest_path = Path(self.config.index_dir) / MANIFEST_NAME
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        return {
            "chunks_indexados": self.store.count(),
            "embedding_backend": self.embedder.name,
            "embedding_dims": self.embedder.dims,
            "store_backend": self.store.name,
            "index_dir": str(self.config.index_dir),
            "manifest": manifest,
        }

    def is_ready(self) -> bool:
        return self.store.count() > 0


_SINGLETON: RagnarokRAG | None = None


def get_rag() -> RagnarokRAG:
    """Instância compartilhada — evita recarregar o modelo a cada requisição."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = RagnarokRAG()
    return _SINGLETON


def answer_to_dict(answer: Answer) -> dict[str, Any]:
    return asdict(answer)
