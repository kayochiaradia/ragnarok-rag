# Ragnarok RAG + WhatsApp POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the offline Ragnarok Online RAG and expose it as a tested local, Meta WhatsApp, and Twilio WhatsApp proof of concept.

**Architecture:** Keep retrieval and answer composition in `ragnarok_rag`; add a channel-neutral service and thin FastAPI/Twilio/Meta adapters in `whatsapp`. Preserve the dependency-free hash/NumPy path and make web and higher-quality vector backends optional.

**Tech Stack:** Python 3.10+, NumPy, FastAPI, Uvicorn, HTTPX, pytest; optional ChromaDB and sentence-transformers.

**Spec:** `docs/superpowers/specs/2026-09-01-ragnarok-rag-poc-design.md`

## Global Constraints

- No LLM or generative API anywhere in the answer path.
- Local ingest, query, evaluation, and simulator operation require no API key.
- Tests force the deterministic hash embedder and NumPy store.
- Production files are changed only after a focused failing test demonstrates the missing behavior.
- The directory has no Git metadata, so commit steps are omitted until the user initializes a repository.

---

### Task 1: Ranking regressions and deterministic evaluation

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_text.py`
- Create: `tests/test_retrieval.py`
- Create: `scripts/evaluate.py`
- Modify: `ragnarok_rag/text.py`
- Modify: `ragnarok_rag/retriever.py`
- Modify: `corpus/16-companheiros.md`

**Interfaces:**
- Consumes: `Config`, `RagnarokRAG.search(query, top_k)` and persisted Markdown corpus.
- Produces: stable normalization and ranking where definition, acquisition, elemental, refino, card, economy, and MVP questions return the intended source section first.

- [ ] **Step 1: Write failing normalization and retrieval tests**

  Add literal expectations for plural normalization and top breadcrumbs for `o que e woe`, `como pegar homunculo`, `fogo faz quanto de dano em terra`, and the already-correct baseline questions. Use a temporary index configured with `embedding_backend="hash"` and `store_backend="numpy"`.

- [ ] **Step 2: Run the focused tests and verify the three known queries fail for the expected top breadcrumb**

  Run: `python -m pytest tests/test_text.py tests/test_retrieval.py -v`

- [ ] **Step 3: Separate lexical body, heading, and tag signals**

  Build BM25 from `metadata["body"]`, calculate bounded heading and tag overlap independently, and retain RRF for vector/body fusion. Expand acquisition vocabulary and make the Homunculus section heading explicitly describe obtaining and invoking it.

- [ ] **Step 4: Run the focused tests and verify they pass**

  Run: `python -m pytest tests/test_text.py tests/test_retrieval.py -v`

- [ ] **Step 5: Add the evaluation script and run its fixed query set**

  `scripts/evaluate.py` ingests with configured backends, prints expected versus actual breadcrumbs, and exits nonzero on a mismatch.

### Task 2: Channel-neutral chat behavior

**Files:**
- Create: `whatsapp/__init__.py`
- Create: `whatsapp/service.py`
- Create: `tests/test_whatsapp_service.py`

**Interfaces:**
- Consumes: any object implementing `ask(query: str) -> Answer`.
- Produces: `ChatService.reply(sender_id: str, message: str) -> ChatReply`, `format_answer(answer, max_chars) -> str`, and `SlidingWindowRateLimiter.allow(key) -> bool`.

- [ ] **Step 1: Write failing tests for successful formatting, low confidence, empty/oversized input, source citation, and rate limiting**

  Tests construct real `Answer` values and a small fake RAG boundary; they assert final user-visible strings, not mock calls.

- [ ] **Step 2: Run tests and verify imports/behaviors are missing**

  Run: `python -m pytest tests/test_whatsapp_service.py -v`

- [ ] **Step 3: Implement the minimal channel-neutral service**

  Add immutable `ChatReply`, bounded WhatsApp-safe formatting, validation, and a monotonic-time sliding window limiter.

- [ ] **Step 4: Run the service tests and verify they pass**

  Run: `python -m pytest tests/test_whatsapp_service.py -v`

### Task 3: Meta, Twilio, local API, and browser simulator

**Files:**
- Create: `whatsapp/meta.py`
- Create: `whatsapp/app.py`
- Create: `whatsapp/static/index.html`
- Create: `tests/test_meta.py`
- Create: `tests/test_web_app.py`
- Modify: `ragnarok_rag/config.py`
- Modify: `ragnarok_rag/cli.py`

**Interfaces:**
- Consumes: `ChatService`; Meta JSON payloads; Twilio form fields `From` and `Body`.
- Produces: `parse_meta_messages(payload) -> list[InboundMessage]`, `send_meta_text(...)`, FastAPI `app`, `/health`, `/api/chat`, `/webhook/meta`, `/webhook/twilio`, `/`, and CLI command `serve`.

- [ ] **Step 1: Write failing adapter and route tests**

  Cover Meta handshake success/failure, nested message extraction, status-only payloads, missing Meta credentials, local JSON chat, Twilio escaped XML, and health readiness.

- [ ] **Step 2: Run route tests and verify the missing adapters/routes fail**

  Run: `python -m pytest tests/test_meta.py tests/test_web_app.py -v`

- [ ] **Step 3: Implement Meta parsing and outbound delivery**

  Parse only inbound text messages, preserve message IDs for deduplication, and send the official Graph API text payload using `httpx.AsyncClient`. Reject live sends without phone ID or access token.

- [ ] **Step 4: Implement the FastAPI routes and static simulator**

  Use dependency injection for tests, return TwiML inline, acknowledge Meta payloads promptly, and have the simulator call `/api/chat`.

- [ ] **Step 5: Add the lazy `serve` CLI command and configuration fields**

  Import Uvicorn only when serving so the core CLI remains usable with only NumPy installed.

- [ ] **Step 6: Run adapter and route tests and verify they pass**

  Run: `python -m pytest tests/test_meta.py tests/test_web_app.py -v`

### Task 4: Reproducible setup and operator documentation

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `README.md`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: the completed CLI and web app.
- Produces: editable installation with `web`, `quality`, and `dev` extras; documented zero-key quick start; Meta/Twilio/ngrok setup; test and evaluation commands.

- [ ] **Step 1: Write failing CLI parser tests for `ingest`, `ask`, `stats`, and `serve`**

- [ ] **Step 2: Run the CLI tests and confirm `serve` is the only missing contract**

  Run: `python -m pytest tests/test_cli.py -v`

- [ ] **Step 3: Add packaging metadata and install the project in a local virtual environment**

  Run: `python -m venv .venv`, then `.venv\\Scripts\\python -m pip install -e ".[web,dev]"`.

- [ ] **Step 4: Write concise setup, demo, webhook, configuration, and architecture documentation**

  Document that corpus facts can differ by server/episode and that the POC cites its local source rather than claiming universal game truth.

- [ ] **Step 5: Run CLI tests and all documented quick-start commands**

### Task 5: Final verification

**Files:**
- Modify only files proven faulty by the checks below.

**Interfaces:**
- Consumes: all earlier deliverables.
- Produces: fresh evidence that the POC is complete.

- [ ] **Step 1: Run the full automated suite**

  Run: `.venv\\Scripts\\python -m pytest -q`

- [ ] **Step 2: Compile all Python modules**

  Run: `.venv\\Scripts\\python -m compileall -q ragnarok_rag whatsapp scripts tests`

- [ ] **Step 3: Rebuild the deterministic local index and run evaluation**

  Run: set `RAG_EMBEDDING_BACKEND=hash` and `RAG_STORE_BACKEND=numpy`, then execute the ingest and evaluation commands.

- [ ] **Step 4: Smoke-test the application in process**

  Use FastAPI TestClient to confirm `/`, `/health`, `/api/chat`, Meta verification, and Twilio responses on the final installed environment.

- [ ] **Step 5: Review the final tree for secrets, placeholders, generated caches, and accidental LLM dependencies**

  Search for API-key values, `TODO`/`TBD`, Anthropic/OpenAI imports, and ensure generated files are ignored.
