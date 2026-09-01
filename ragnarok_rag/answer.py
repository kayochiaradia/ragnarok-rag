"""Composição da resposta a partir dos trechos recuperados.

Importante: aqui NÃO existe modelo de linguagem. A resposta é extrativa —
montada com o texto literal do corpus, recortado na janela de frases mais
relevante para a pergunta. O que o usuário lê veio, palavra por palavra, do
documento citado na fonte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .retriever import RetrievedChunk
from .text import expand_query, sentences, tokenize


@dataclass
class Source:
    doc_titulo: str
    heading: str
    source: str
    score: float


@dataclass
class Answer:
    query: str
    found: bool
    text: str
    confidence: float
    sources: list[Source] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


def _coverage(chunk: RetrievedChunk, query_tokens: list[str]) -> float:
    """Fração dos tokens da pergunta (expandidos) presentes no trecho."""
    if not query_tokens:
        return 0.0
    chunk_tokens = set(tokenize(f"{chunk.breadcrumb} {chunk.body}"))
    present = sum(1 for token in set(query_tokens) if token in chunk_tokens)
    return present / len(set(query_tokens))


def score_confidence(chunk: RetrievedChunk, query: str) -> float:
    """Confiança em 0..1, comparável entre backends de embedding.

    Combina o melhor sinal de ranking disponível com a cobertura literal dos
    termos da pergunta. A cobertura é o que impede o sistema de responder com
    firmeza a uma pergunta fora do domínio só porque algum vetor ficou perto.
    """
    tokens = expand_query(query)
    ranking = max(chunk.vector_score, chunk.lexical_score)
    ranking = max(0.0, min(1.0, ranking))
    return round(0.6 * ranking + 0.4 * _coverage(chunk, tokens), 4)


def _best_window(body: str, query_tokens: list[str], max_chars: int) -> str:
    """Janela contígua de frases com maior densidade de termos da pergunta."""
    if len(body) <= max_chars:
        return body

    parts = sentences(body)
    if not parts:
        return body[:max_chars].rstrip() + "…"

    wanted = set(query_tokens)
    weights = []
    for part in parts:
        part_tokens = set(tokenize(part))
        weights.append(len(wanted & part_tokens))

    best_start, best_end, best_score = 0, 1, -1.0
    for start in range(len(parts)):
        length = 0
        score = 0
        for end in range(start, len(parts)):
            length += len(parts[end]) + 1
            if length > max_chars:
                break
            score += weights[end]
            # Desempata por janela mais longa quando a pontuação empata.
            normalized = score + (length / max_chars) * 0.5
            if normalized > best_score:
                best_score, best_start, best_end = normalized, start, end + 1

    window = " ".join(parts[best_start:best_end]).strip()
    prefix = "…" if best_start > 0 else ""
    suffix = "…" if best_end < len(parts) else ""
    return f"{prefix}{window}{suffix}"


def compose(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    max_chars: int = 1200,
    related_count: int = 3,
    min_score: float = 0.22,
) -> Answer:
    if not chunks:
        return Answer(
            query=query,
            found=False,
            text="",
            confidence=0.0,
            debug={"reason": "sem_resultados"},
        )

    top = chunks[0]
    confidence = score_confidence(top, query)

    if confidence < min_score:
        return Answer(
            query=query,
            found=False,
            text="",
            confidence=confidence,
            related=[c.breadcrumb for c in chunks[:related_count] if c.breadcrumb],
            debug={"reason": "confianca_baixa", "min_score": min_score},
        )

    query_tokens = expand_query(query)
    body = _best_window(top.body, query_tokens, max_chars)

    related: list[str] = []
    for chunk in chunks[1: 1 + related_count]:
        crumb = chunk.breadcrumb
        if crumb and crumb != top.breadcrumb:
            related.append(crumb)

    sources = [
        Source(
            doc_titulo=chunk.doc_titulo,
            heading=chunk.heading,
            source=chunk.source,
            score=round(chunk.score, 5),
        )
        for chunk in chunks[:3]
    ]

    return Answer(
        query=query,
        found=True,
        text=body,
        confidence=confidence,
        sources=sources,
        related=related,
        debug={
            "top_chunk": top.chunk_id,
            "vector_rank": top.vector_rank,
            "lexical_rank": top.lexical_rank,
            "vector_score": round(top.vector_score, 4),
            "lexical_score": round(top.lexical_score, 4),
        },
    )
