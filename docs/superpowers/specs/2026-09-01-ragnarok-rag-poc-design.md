# Ragnarok RAG + WhatsApp POC Design

## Objective

Deliver a runnable proof of concept that answers Ragnarok Online questions from a local, curated Markdown corpus and exposes the same answer flow through a browser simulator, Meta WhatsApp Cloud API, and Twilio WhatsApp webhook.

## Approved constraints

- No LLM or generative API anywhere in the answer path.
- The local demo must run without API keys and without downloading a model.
- Retrieval is hybrid: deterministic local embeddings plus lexical search.
- Answers are extractive, preserve corpus wording, and cite their source.
- ChromaDB and sentence-transformers remain optional quality upgrades; NumPy and hashing remain the zero-download fallback.
- Meta and Twilio credentials are only required for their live delivery paths.

## Architecture

The `ragnarok_rag` package owns corpus loading, Markdown-aware chunking, indexing, retrieval, confidence scoring, and extractive answer composition. Ranking uses body text as the primary lexical signal and treats headings and document tags as separate, bounded field signals so document-wide tags cannot make every section look equally relevant.

The `whatsapp` package owns channel-neutral formatting and request handling. A FastAPI application exposes health, local chat, Meta verification/inbound webhooks, and a Twilio-compatible TwiML webhook. The browser simulator calls the same local chat service used by the channel adapters. Live Meta delivery is an outbound HTTP call; Twilio replies inline with TwiML and therefore needs no Twilio SDK.

## Error handling and safety

- Empty or oversized questions receive a stable user-facing validation response.
- Low-confidence retrieval returns an explicit “not found in the base” response rather than guessing.
- Duplicate Meta message IDs are ignored in memory.
- A per-sender sliding-window limit protects the POC from accidental webhook loops.
- Missing live-channel credentials produce a clear configuration error while local chat and health remain available.
- Meta payloads without a supported text message are acknowledged and ignored.

## Verification

Automated tests cover the known ranking regressions, extractive formatting, validation, rate limiting, Meta payload parsing and verification, Twilio XML, local API behavior, and live-channel configuration failures. A fixed evaluation script reports retrieval hits for representative Portuguese player questions. The final verification runs the full test suite, compiles the package, rebuilds the index, runs the evaluation set, and smoke-tests the HTTP application.
