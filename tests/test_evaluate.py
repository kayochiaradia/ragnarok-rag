import subprocess
import sys

from scripts.evaluate import CASES, evaluate


def test_default_evaluation_cases_all_match(deterministic_rag) -> None:
    results = evaluate(deterministic_rag, CASES)

    assert results
    assert all(result.passed for result in results), [
        (result.case.query, result.actual_doc, result.actual_heading)
        for result in results
        if not result.passed
    ]


def test_evaluation_cli_help_emits_utf8_when_captured() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.evaluate", "--help"],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0
    assert "índice de avaliação" in result.stdout
