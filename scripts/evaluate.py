"""Avaliação determinística do ranking com perguntas reais de jogadores."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Iterable

from ragnarok_rag.config import CONFIG
from ragnarok_rag.console import configure_utf8_output
from ragnarok_rag.pipeline import RagnarokRAG


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_doc: str
    expected_heading: str


@dataclass(frozen=True)
class EvaluationResult:
    case: EvaluationCase
    actual_doc: str
    actual_heading: str
    passed: bool


CASES = (
    EvaluationCase("o que e woe", "woe-pvp", "O que é a Guerra do Emperium"),
    EvaluationCase("como pegar homunculo", "companheiros", "Homunculus"),
    EvaluationCase(
        "fogo faz quanto de dano em terra",
        "elementos",
        "Regras principais da tabela elemental",
    ),
    EvaluationCase("qual o limite seguro de refino", "refino", "Limites seguros"),
    EvaluationCase("limite seguro de refino", "refino", "Limites seguros"),
    EvaluationCase(
        "o que a carta gtb faz",
        "cartas",
        "Cartas de escudo mais importantes",
    ),
    EvaluationCase("onde acho o baphomet", "mvps", "Baphomet"),
    EvaluationCase("como ganhar zeny rapido", "economia", "Zeny"),
    EvaluationCase("qual build de mestre pra asura", "builds", "Asura Strike"),
)


def evaluate(
    rag: RagnarokRAG,
    cases: Iterable[EvaluationCase] = CASES,
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for case in cases:
        hits = rag.search(case.query, top_k=1)
        top = hits[0] if hits else None
        actual_doc = top.metadata.get("doc_id", "") if top else ""
        actual_heading = top.heading if top else ""
        results.append(
            EvaluationResult(
                case=case,
                actual_doc=actual_doc,
                actual_heading=actual_heading,
                passed=(
                    actual_doc == case.expected_doc
                    and actual_heading == case.expected_heading
                ),
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia o ranking do Ragnarok RAG")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="usa o índice de avaliação existente",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    config = replace(
        CONFIG,
        embedding_backend="hash",
        store_backend="numpy",
        collection="ragnarok-evaluation",
    )
    rag = RagnarokRAG(config)
    if not args.skip_ingest or not rag.is_ready():
        rag.ingest(verbose=True)

    results = evaluate(rag)
    for result in results:
        marker = "PASS" if result.passed else "FAIL"
        print(f"[{marker}] {result.case.query}")
        print(f"       esperado: {result.case.expected_doc} > {result.case.expected_heading}")
        print(f"       obtido:   {result.actual_doc} > {result.actual_heading}")

    passed = sum(result.passed for result in results)
    print(f"\n{passed}/{len(results)} consultas corretas")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
