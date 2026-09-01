"""Comportamento de chat compartilhado pelo simulador, Meta e Twilio."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

from ragnarok_rag.answer import Answer


NOT_FOUND_TEXT = (
    "Não encontrei isso na base de conhecimento. "
    "Tente usar o nome do item, monstro, mapa, classe ou habilidade."
)


class AnswersQuestions(Protocol):
    def ask(self, query: str) -> Answer: ...


@dataclass(frozen=True)
class ChatReply:
    text: str
    ok: bool
    found: bool
    confidence: float = 0.0
    error_code: str | None = None


class SlidingWindowRateLimiter:
    """Limite em memória, por chave, usando uma janela temporal deslizante."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("limit e window_seconds devem ser positivos")
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, *, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else now
        cutoff = timestamp - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(timestamp)
            return True


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 1:
        return "…"[:limit]
    return text[: limit - 1].rstrip() + "…"


def _source_block(answer: Answer) -> str:
    if not answer.sources:
        return ""
    citations = [
        f"{source.doc_titulo} > {source.heading} ({source.source})"
        for source in answer.sources[:3]
    ]
    label = "Fonte" if len(citations) == 1 else "Fontes"
    return f"📚 {label}: " + "\n• ".join(citations)


def format_answer(answer: Answer, *, max_chars: int = 3500) -> str:
    """Formata uma resposta extrativa e preserva a citação no limite do canal."""
    if not answer.found:
        return _truncate(NOT_FOUND_TEXT, max_chars)

    source_block = _source_block(answer)
    if not source_block:
        return _truncate(answer.text.strip(), max_chars)

    separator = "\n\n"
    body_budget = max_chars - len(separator) - len(source_block)
    if body_budget < 1:
        return _truncate(source_block, max_chars)
    body = _truncate(answer.text.strip(), body_budget)
    return f"{body}{separator}{source_block}"


class ChatService:
    def __init__(
        self,
        rag: AnswersQuestions,
        *,
        max_query_chars: int = 500,
        max_output_chars: int = 3500,
        rate_limit_per_min: int = 20,
        limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self.rag = rag
        self.max_query_chars = max_query_chars
        self.max_output_chars = max_output_chars
        self.limiter = limiter or SlidingWindowRateLimiter(rate_limit_per_min)

    def reply(self, sender_id: str, message: str) -> ChatReply:
        query = (message or "").strip()
        if not query:
            return ChatReply(
                text="Envie uma pergunta sobre Ragnarok Online.",
                ok=False,
                found=False,
                error_code="empty_message",
            )
        if len(query) > self.max_query_chars:
            return ChatReply(
                text=f"Sua pergunta é muito longa. Use até {self.max_query_chars} caracteres.",
                ok=False,
                found=False,
                error_code="message_too_long",
            )
        if not self.limiter.allow(sender_id or "anonymous"):
            return ChatReply(
                text="Você enviou muitas perguntas. Aguarde um minuto e tente novamente.",
                ok=False,
                found=False,
                error_code="rate_limited",
            )

        answer = self.rag.ask(query)
        return ChatReply(
            text=format_answer(answer, max_chars=self.max_output_chars),
            ok=True,
            found=answer.found,
            confidence=answer.confidence,
        )
