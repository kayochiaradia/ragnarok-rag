"""Vector store persistente.

Dois backends com a mesma interface:

- ``chroma``: ChromaDB PersistentClient, espaço cosseno. É o backend "de
  vitrine" do POC, porque é o vector database que a maioria dos times já
  reconhece.
- ``numpy``: matriz .npy + metadados .json. Sem dependência externa, carrega
  em milissegundos e faz busca exata por produto interno (os vetores já vêm
  normalizados em L2, então produto interno == cosseno).

``auto`` usa chroma se estiver instalado.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorStore(ABC):
    name: str

    @abstractmethod
    def reset(self, dims: int) -> None: ...

    @abstractmethod
    def add(self, ids: Sequence[str], vectors: np.ndarray, texts: Sequence[str],
            metadatas: Sequence[dict[str, Any]]) -> None: ...

    @abstractmethod
    def query(self, vector: np.ndarray, k: int) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def all_records(self) -> list[Hit]: ...

    def persist(self) -> None:  # opcional
        return None


class NumpyStore(VectorStore):
    name = "numpy"

    def __init__(self, index_dir: Path, collection: str) -> None:
        self.dir = Path(index_dir)
        self.collection = collection
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._records: list[dict[str, Any]] = []
        self._load()

    @property
    def _vec_path(self) -> Path:
        return self.dir / f"{self.collection}.vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.dir / f"{self.collection}.records.json"

    def _load(self) -> None:
        if self._vec_path.exists() and self._meta_path.exists():
            self._vectors = np.load(self._vec_path)
            self._records = json.loads(self._meta_path.read_text(encoding="utf-8"))

    def reset(self, dims: int) -> None:
        self._vectors = np.zeros((0, dims), dtype=np.float32)
        self._records = []
        for path in (self._vec_path, self._meta_path):
            if path.exists():
                path.unlink()

    def add(self, ids, vectors, texts, metadatas) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        self._vectors = vectors if self._vectors is None or self._vectors.size == 0 else np.vstack([self._vectors, vectors])
        for chunk_id, text, meta in zip(ids, texts, metadatas):
            self._records.append({"id": chunk_id, "text": text, "metadata": meta})

    def persist(self) -> None:
        if self._vectors is None:
            return
        np.save(self._vec_path, self._vectors)
        self._meta_path.write_text(
            json.dumps(self._records, ensure_ascii=False), encoding="utf-8"
        )

    def query(self, vector: np.ndarray, k: int) -> list[Hit]:
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []
        scores = self._vectors @ np.asarray(vector, dtype=np.float32)
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            Hit(
                chunk_id=self._records[i]["id"],
                score=float(scores[i]),
                text=self._records[i]["text"],
                metadata=self._records[i]["metadata"],
            )
            for i in top
        ]

    def count(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def all_records(self) -> list[Hit]:
        return [
            Hit(chunk_id=r["id"], score=0.0, text=r["text"], metadata=r["metadata"])
            for r in self._records
        ]


class ChromaStore(VectorStore):
    name = "chroma"

    def __init__(self, index_dir: Path, collection: str) -> None:
        import chromadb  # import tardio
        from chromadb.config import Settings

        self.dir = Path(index_dir) / "chroma"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self._client = chromadb.PersistentClient(
            path=str(self.dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self, dims: int) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - coleção pode não existir ainda
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, vectors, texts, metadatas) -> None:
        # Chroma exige metadados escalares.
        flat = [{k: (v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False))
                 for k, v in meta.items()} for meta in metadatas]
        batch = 256
        vectors = np.asarray(vectors, dtype=np.float32)
        for start in range(0, len(ids), batch):
            end = start + batch
            self._collection.add(
                ids=list(ids[start:end]),
                embeddings=vectors[start:end].tolist(),
                documents=list(texts[start:end]),
                metadatas=flat[start:end],
            )

    def query(self, vector: np.ndarray, k: int) -> list[Hit]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[Hit] = []
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(Hit(chunk_id=chunk_id, score=1.0 - float(dist), text=doc, metadata=dict(meta)))
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def all_records(self) -> list[Hit]:
        if self.count() == 0:
            return []
        data = self._collection.get(include=["documents", "metadatas"])
        return [
            Hit(chunk_id=i, score=0.0, text=d, metadata=dict(m))
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])
        ]


def build_store(backend: str, index_dir: Path, collection: str) -> VectorStore:
    backend = (backend or "auto").lower()

    if backend == "numpy":
        return NumpyStore(index_dir, collection)

    if backend in {"chroma", "auto"}:
        try:
            return ChromaStore(index_dir, collection)
        except Exception as exc:  # noqa: BLE001
            if backend == "chroma":
                raise RuntimeError(
                    f"Backend 'chroma' indisponível ({exc}). Instale com: pip install chromadb"
                ) from exc
            print(
                f"[store] chromadb indisponível ({type(exc).__name__}); usando o store numpy local."
            )
            return NumpyStore(index_dir, collection)

    raise ValueError(f"Backend de vector store desconhecido: {backend}")
