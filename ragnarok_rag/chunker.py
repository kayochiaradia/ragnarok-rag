"""Chunking consciente de cabeçalhos markdown.

A qualidade da recuperação depende muito mais do chunking do que do modelo de
embedding. A estratégia aqui:

1. Quebrar o documento pelos cabeçalhos ## e ###, preservando a hierarquia.
2. Seções maiores que o alvo são divididas por parágrafo, com sobreposição.
3. Cada chunk carrega o caminho de cabeçalhos no próprio texto indexado
   ("Documento > Seção > Subseção"), o que dá contexto ao vetor e melhora
   muito a busca por termos que só aparecem no título da seção.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .loader import Document

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_titulo: str
    categoria: str
    tags: list[str]
    heading_path: list[str]
    text: str
    ordinal: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def heading(self) -> str:
        return self.heading_path[-1] if self.heading_path else self.doc_titulo

    @property
    def breadcrumb(self) -> str:
        return " > ".join([self.doc_titulo, *self.heading_path])

    def embed_text(self) -> str:
        """Texto efetivamente vetorizado: breadcrumb + tags + corpo."""
        tag_line = " ".join(self.tags)
        return f"{self.breadcrumb}\n{tag_line}\n{self.text}"


def _split_sections(body: str) -> list[tuple[list[str], str]]:
    """Divide o markdown em (caminho_de_cabecalhos, texto)."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            # O H1 é o título do documento, que já viaja separado no breadcrumb;
            # incluí-lo aqui duplicaria o título em toda migalha de pão.
            sections.append(([title for level, title in stack if level > 1], text))
        buffer.clear()

    for line in body.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buffer.append(line)
    flush()
    return sections


def _split_long_text(text: str, target: int, overlap: int) -> list[str]:
    """Quebra por parágrafo, agrupando até o alvo, com sobreposição de cauda."""
    if len(text) <= target:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    size = 0

    for para in paragraphs:
        # Parágrafo isolado maior que o alvo: quebra por frase.
        if len(para) > target:
            if current:
                pieces.append("\n\n".join(current))
                current, size = [], 0
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf: list[str] = []
            buf_len = 0
            for sentence in sentences:
                if buf and buf_len + len(sentence) > target:
                    pieces.append(" ".join(buf))
                    buf, buf_len = [], 0
                buf.append(sentence)
                buf_len += len(sentence) + 1
            if buf:
                pieces.append(" ".join(buf))
            continue

        if current and size + len(para) > target:
            pieces.append("\n\n".join(current))
            tail = current[-1]
            current = [tail[-overlap:]] if overlap and len(tail) > overlap else []
            size = sum(len(p) for p in current)

        current.append(para)
        size += len(para) + 2

    if current:
        pieces.append("\n\n".join(current))

    return [p.strip() for p in pieces if p.strip()]


def chunk_document(doc: Document, target: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for heading_path, text in _split_sections(doc.body):
        for piece in _split_long_text(text, target, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{ordinal:03d}",
                    doc_id=doc.doc_id,
                    doc_titulo=doc.titulo,
                    categoria=doc.categoria,
                    tags=doc.tags,
                    heading_path=heading_path,
                    text=piece,
                    ordinal=ordinal,
                    meta={"source": doc.path.name},
                )
            )
            ordinal += 1
    return chunks


def chunk_documents(docs: list[Document], target: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, target, overlap))
    return out
