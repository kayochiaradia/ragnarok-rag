from __future__ import annotations

from ragnarok_rag.answer import Answer, Source
from whatsapp.service import ChatService, SlidingWindowRateLimiter, format_answer


class StubRAG:
    def __init__(self, answer: Answer) -> None:
        self.answer = answer

    def ask(self, query: str) -> Answer:
        return self.answer


def found_answer(text: str = "A resposta veio literalmente do corpus.") -> Answer:
    return Answer(
        query="pergunta",
        found=True,
        text=text,
        confidence=0.82,
        sources=[
            Source(
                doc_titulo="Cartas: slots, efeitos e as mais procuradas",
                heading="Cartas de escudo mais importantes",
                source="05-cartas.md",
                score=0.05,
            )
        ],
    )


def test_format_answer_includes_extractive_text_and_source() -> None:
    formatted = format_answer(found_answer(), max_chars=500)

    assert "A resposta veio literalmente do corpus." in formatted
    assert "Fonte:" in formatted
    assert "Cartas de escudo mais importantes" in formatted


def test_format_answer_respects_channel_limit_and_keeps_citation() -> None:
    formatted = format_answer(found_answer("Texto longo. " * 100), max_chars=180)

    assert len(formatted) <= 180
    assert formatted.endswith("(05-cartas.md)")
    assert "…" in formatted


def test_service_returns_explicit_not_found_message() -> None:
    rag = StubRAG(Answer(query="x", found=False, text="", confidence=0.04))

    reply = ChatService(rag).reply("user-1", "receita de lasanha")

    assert reply.ok is True
    assert reply.found is False
    assert "Não encontrei" in reply.text
    assert reply.error_code is None


def test_service_rejects_empty_and_oversized_questions() -> None:
    service = ChatService(StubRAG(found_answer()), max_query_chars=12)

    empty = service.reply("user-1", "   ")
    oversized = service.reply("user-1", "x" * 13)

    assert (empty.ok, empty.error_code) == (False, "empty_message")
    assert (oversized.ok, oversized.error_code) == (False, "message_too_long")


def test_sliding_window_rate_limiter_expires_old_entries() -> None:
    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)

    assert limiter.allow("user-1", now=100.0) is True
    assert limiter.allow("user-1", now=101.0) is True
    assert limiter.allow("user-1", now=102.0) is False
    assert limiter.allow("user-2", now=102.0) is True
    assert limiter.allow("user-1", now=161.0) is True


def test_service_reports_rate_limit_without_querying_the_rag() -> None:
    limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
    service = ChatService(StubRAG(found_answer()), limiter=limiter)

    assert service.reply("user-1", "primeira").ok is True
    blocked = service.reply("user-1", "segunda")

    assert blocked.ok is False
    assert blocked.error_code == "rate_limited"
    assert "muitas perguntas" in blocked.text
