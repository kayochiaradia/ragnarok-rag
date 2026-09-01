from __future__ import annotations

import sys
import subprocess
from argparse import Namespace
from types import SimpleNamespace

from ragnarok_rag.cli import build_parser, cmd_serve


def test_parser_exposes_core_and_web_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["ingest"]).func.__name__ == "cmd_ingest"
    assert parser.parse_args(["ask", "o que e woe"]).func.__name__ == "cmd_ask"
    assert parser.parse_args(["stats"]).func.__name__ == "cmd_stats"
    serve = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
    assert serve.func.__name__ == "cmd_serve"
    assert (serve.host, serve.port, serve.reload) == ("0.0.0.0", 9000, False)


def test_serve_runs_uvicorn_with_requested_network_options(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    fake_uvicorn = SimpleNamespace(
        run=lambda target, **kwargs: calls.append((target, kwargs))
    )
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    result = cmd_serve(Namespace(host="127.0.0.1", port=8765, reload=True))

    assert result == 0
    assert calls == [
        (
            "whatsapp.app:app",
            {"host": "127.0.0.1", "port": 8765, "reload": True},
        )
    ]


def test_module_cli_emits_utf8_when_output_is_captured() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ragnarok_rag.cli", "--help"],
        capture_output=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0
    assert "índice" in result.stdout
    assert "Modo interativo" in result.stdout
