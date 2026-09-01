"""CLI: python -m ragnarok_rag.cli <comando>"""

from __future__ import annotations

import argparse
import json
import sys

from .config import CONFIG
from .console import configure_utf8_output
from .pipeline import RagnarokRAG, answer_to_dict


def cmd_ingest(args: argparse.Namespace) -> int:
    rag = RagnarokRAG()
    manifest = rag.ingest(verbose=not args.quiet)
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    rag = RagnarokRAG()
    if not rag.is_ready():
        print("Índice vazio. Rode primeiro: python -m ragnarok_rag.cli ingest", file=sys.stderr)
        return 1

    answer = rag.ask(" ".join(args.pergunta), top_k=args.top_k)

    if args.json:
        print(json.dumps(answer_to_dict(answer), ensure_ascii=False, indent=2))
        return 0

    if not answer.found:
        print(f"Não encontrei isso na base. (confiança {answer.confidence:.2f})")
        if answer.related:
            print("\nTalvez você queira perguntar sobre:")
            for item in answer.related:
                print(f"  - {item}")
        return 0

    print(answer.text)
    print()
    print(f"[confiança {answer.confidence:.2f}]")
    for source in answer.sources:
        print(f"  fonte: {source.doc_titulo} > {source.heading}  ({source.source})")
    if answer.related:
        print("  relacionado: " + "; ".join(answer.related))
    if args.debug:
        print("  debug: " + json.dumps(answer.debug, ensure_ascii=False))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    rag = RagnarokRAG()
    if not rag.is_ready():
        print("Índice vazio. Rode primeiro: python -m ragnarok_rag.cli ingest", file=sys.stderr)
        return 1
    for i, chunk in enumerate(rag.search(" ".join(args.pergunta), top_k=args.top_k), 1):
        print(f"{i:>2}. [{chunk.score:.5f}] {chunk.breadcrumb}")
        print(f"    vetor={chunk.vector_score:.3f} (rank {chunk.vector_rank})  "
              f"lexical={chunk.lexical_score:.3f} (rank {chunk.lexical_rank})")
        preview = chunk.body.replace("\n", " ")[:160]
        print(f"    {preview}…\n")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    print(json.dumps(RagnarokRAG().stats(), ensure_ascii=False, indent=2))
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    rag = RagnarokRAG()
    if not rag.is_ready():
        print("Índice vazio. Rode primeiro: python -m ragnarok_rag.cli ingest", file=sys.stderr)
        return 1
    print("RAG de Ragnarok Online. Digite a pergunta (ou 'sair').\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"sair", "exit", "quit", ""}:
            return 0
        answer = rag.ask(question)
        if answer.found:
            print(f"\n{answer.text}\n")
            src = answer.sources[0]
            print(f"[{src.doc_titulo} > {src.heading} | confiança {answer.confidence:.2f}]\n")
        else:
            print("\nNão encontrei isso na base.\n")


def cmd_serve(args: argparse.Namespace) -> int:
    """Inicia a API e o simulador; import tardio mantém o core leve."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            'Dependências web ausentes. Instale com: pip install -e ".[web]"'
        ) from exc

    uvicorn.run(
        "whatsapp.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragnarok_rag", description="RAG vetorial de Ragnarok Online")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_ingest = sub.add_parser("ingest", help="Indexa o corpus no vector store")
    p_ingest.add_argument("--json", action="store_true")
    p_ingest.add_argument("--quiet", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Faz uma pergunta e devolve a resposta extrativa")
    p_ask.add_argument("pergunta", nargs="+")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.add_argument("--json", action="store_true")
    p_ask.add_argument("--debug", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_search = sub.add_parser("search", help="Mostra os chunks recuperados e as pontuações")
    p_search.add_argument("pergunta", nargs="+")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_stats = sub.add_parser("stats", help="Estado do índice")
    p_stats.set_defaults(func=cmd_stats)

    p_repl = sub.add_parser("repl", help="Modo interativo no terminal")
    p_repl.set_defaults(func=cmd_repl)

    p_serve = sub.add_parser("serve", help="Inicia API, webhooks e simulador local")
    p_serve.add_argument("--host", default=CONFIG.web_host)
    p_serve.add_argument("--port", type=int, default=CONFIG.web_port)
    p_serve.add_argument("--reload", action="store_true", help="reinicia ao editar arquivos")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_output()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
