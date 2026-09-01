"""Configuração central do RAG de Ragnarok Online.

Tudo é resolvido por variável de ambiente com um default sensato, para que o
projeto rode sem nenhum arquivo .env e sem nenhuma chave de API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- caminhos ---
    corpus_dir: Path = field(default_factory=lambda: Path(_env("RAG_CORPUS_DIR", str(ROOT / "corpus"))))
    index_dir: Path = field(default_factory=lambda: Path(_env("RAG_INDEX_DIR", str(ROOT / ".index"))))

    # --- chunking ---
    # Alvo em caracteres. Chunks pequenos demais perdem contexto; grandes demais
    # diluem o vetor e pioram a precisão da busca.
    chunk_target_chars: int = field(default_factory=lambda: _env_int("RAG_CHUNK_CHARS", 1100))
    chunk_overlap_chars: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 150))

    # --- embeddings ---
    # "auto"  -> tenta sentence-transformers, cai para "hash" se não houver
    # "local" -> sentence-transformers (offline após o primeiro download)
    # "hash"  -> embedding determinístico por n-grama, 100% offline, zero download
    embedding_backend: str = field(default_factory=lambda: _env("RAG_EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(
        default_factory=lambda: _env("RAG_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    )
    hash_dims: int = field(default_factory=lambda: _env_int("RAG_HASH_DIMS", 768))

    # --- vector store ---
    # "auto" -> chromadb se instalado, senão o store numpy persistido em .npz
    store_backend: str = field(default_factory=lambda: _env("RAG_STORE_BACKEND", "auto"))
    collection: str = field(default_factory=lambda: _env("RAG_COLLECTION", "ragnarok"))

    # --- busca ---
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 20))
    # Peso da busca vetorial contra a busca lexical na fusão RRF.
    vector_weight: float = field(default_factory=lambda: _env_float("RAG_VECTOR_WEIGHT", 1.0))
    lexical_weight: float = field(default_factory=lambda: _env_float("RAG_LEXICAL_WEIGHT", 1.0))
    # Abaixo disso a resposta é tratada como "não sei".
    min_score: float = field(default_factory=lambda: _env_float("RAG_MIN_SCORE", 0.16))

    # --- resposta (extrativa, sem LLM) ---
    answer_max_chars: int = field(default_factory=lambda: _env_int("RAG_ANSWER_CHARS", 1200))
    related_count: int = field(default_factory=lambda: _env_int("RAG_RELATED", 3))

    # --- whatsapp ---
    wa_verify_token: str = field(default_factory=lambda: _env("WHATSAPP_VERIFY_TOKEN", "ragnarok-poc"))
    wa_access_token: str = field(default_factory=lambda: _env("WHATSAPP_ACCESS_TOKEN", ""))
    wa_phone_number_id: str = field(default_factory=lambda: _env("WHATSAPP_PHONE_NUMBER_ID", ""))
    # A Meta aposenta versões periodicamente; live mode exige uma versão
    # escolhida no painel/documentação atual em vez de congelar um default.
    wa_api_version: str = field(default_factory=lambda: _env("WHATSAPP_API_VERSION", ""))
    wa_max_query_chars: int = field(default_factory=lambda: _env_int("WHATSAPP_MAX_QUERY_CHARS", 500))
    wa_max_chars: int = field(default_factory=lambda: _env_int("WHATSAPP_MAX_CHARS", 3500))
    wa_rate_limit_per_min: int = field(default_factory=lambda: _env_int("WHATSAPP_RATE_LIMIT", 20))
    web_host: str = field(default_factory=lambda: _env("RAG_WEB_HOST", "127.0.0.1"))
    web_port: int = field(default_factory=lambda: _env_int("RAG_WEB_PORT", 8000))

    def ensure_dirs(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
