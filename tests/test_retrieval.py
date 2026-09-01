from __future__ import annotations

import pytest

from ragnarok_rag.pipeline import RagnarokRAG


@pytest.mark.parametrize(
    ("query", "expected_doc", "expected_heading"),
    [
        ("o que e woe", "woe-pvp", "O que é a Guerra do Emperium"),
        ("como pegar homunculo", "companheiros", "Homunculus"),
        (
            "fogo faz quanto de dano em terra",
            "elementos",
            "Regras principais da tabela elemental",
        ),
        ("qual o limite seguro de refino", "refino", "Limites seguros"),
        ("limite seguro de refino", "refino", "Limites seguros"),
        ("o que a carta gtb faz", "cartas", "Cartas de escudo mais importantes"),
    ],
)
def test_player_question_returns_intended_section_first(
    deterministic_rag: RagnarokRAG,
    query: str,
    expected_doc: str,
    expected_heading: str,
) -> None:
    top = deterministic_rag.search(query, top_k=1)[0]

    assert top.metadata["doc_id"] == expected_doc
    assert top.heading == expected_heading


def test_empty_question_returns_no_chunks(deterministic_rag: RagnarokRAG) -> None:
    assert deterministic_rag.search("   ") == []


def test_beginner_class_question_answers_about_the_requested_class(
    deterministic_rag: RagnarokRAG,
) -> None:
    answer = deterministic_rag.ask("gatuno é uma boa classe para começar?")

    assert answer.found is True
    assert answer.sources[0].heading == "Qual classe eu devo escolher para começar?"
    assert "Gatuno também é uma boa classe para começar" in answer.text
    assert "Job 50" not in answer.text
