from __future__ import annotations

from pathlib import Path

import pytest

from ragnarok_rag.config import Config
from ragnarok_rag.pipeline import RagnarokRAG


ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def deterministic_rag(tmp_path_factory: pytest.TempPathFactory) -> RagnarokRAG:
    config = Config(
        corpus_dir=ROOT / "corpus",
        index_dir=tmp_path_factory.mktemp("rag-index"),
        embedding_backend="hash",
        store_backend="numpy",
        candidate_k=30,
        min_score=0.16,
    )
    rag = RagnarokRAG(config)
    rag.ingest(verbose=False)
    return rag
