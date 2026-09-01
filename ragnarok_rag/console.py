"""Compatibilidade de saída UTF-8 para comandos em diferentes consoles."""

from __future__ import annotations

import sys


def configure_utf8_output() -> None:
    """Mantém acentos legíveis em pipes e consoles Windows legados."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
