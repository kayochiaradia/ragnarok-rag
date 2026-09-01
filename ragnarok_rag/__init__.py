"""RAG vetorial de Ragnarok Online — sem LLM, apenas recuperação."""

from .answer import Answer, Source, compose
from .config import CONFIG, Config
from .pipeline import RagnarokRAG, get_rag
from .retriever import HybridRetriever, RetrievedChunk

__all__ = [
    "Answer",
    "Source",
    "compose",
    "CONFIG",
    "Config",
    "RagnarokRAG",
    "get_rag",
    "HybridRetriever",
    "RetrievedChunk",
]

__version__ = "1.0.0"
