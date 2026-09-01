"""Leitura do corpus em markdown com front-matter YAML simplificado."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Document:
    doc_id: str
    titulo: str
    categoria: str
    tags: list[str]
    body: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    return raw


def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parser mínimo de front-matter: chave: valor, com listas em linha.

    Deliberadamente não usamos PyYAML para manter o projeto sem dependência
    extra — o formato do corpus é controlado por nós.
    """
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        line = line.rstrip()
        if not line or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _parse_scalar(value)
    return meta, text[match.end():]


def load_documents(corpus_dir: Path) -> list[Document]:
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Diretório de corpus não encontrado: {corpus_dir}")

    docs: list[Document] = []
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name.startswith("_"):
            continue
        text = path.read_text(encoding="utf-8")
        meta, body = _parse_front_matter(text)
        tags = meta.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        docs.append(
            Document(
                doc_id=str(meta.get("id") or path.stem),
                titulo=str(meta.get("titulo") or path.stem),
                categoria=str(meta.get("categoria") or "geral"),
                tags=list(tags),
                body=body.strip(),
                path=path,
                meta=meta,
            )
        )
    if not docs:
        raise FileNotFoundError(f"Nenhum documento .md encontrado em {corpus_dir}")
    return docs


def iter_documents(corpus_dir: Path) -> Iterator[Document]:
    yield from load_documents(corpus_dir)
