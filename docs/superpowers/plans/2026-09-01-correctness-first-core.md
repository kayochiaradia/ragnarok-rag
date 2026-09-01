# Correctness-First RAG Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace closest-passage answers with a source-backed, deterministic record engine that answers the Gatuno question correctly and abstains from unsupported questions.

**Architecture:** Load validated JSON knowledge records and source metadata, analyze Portuguese queries into intents and entities, retrieve broad candidates, and authorize an answer only through an independent evidence gate. Keep the current embedding/vector-store abstractions for candidate recall, but never treat retrieval scores as answer confidence.

**Tech Stack:** Python 3.10+, standard-library dataclasses/JSON/hashlib/datetime, NumPy, existing hash embeddings and vector-store interfaces, pytest, FastAPI channel adapters.

**Spec:** `docs/superpowers/specs/2026-09-01-correctness-first-rag-design.md`

## Global Constraints

- Runtime answers use no LLM, generative API, network request, or automatic web scraping.
- The active ruleset identifier is exactly `ro-latam-renewal`.
- Official GNJOY sources take precedence; current bROWiki LATAM records use `community-reviewed` with lower reliability.
- The archived bRO corpus cannot authorize a current RO LATAM answer.
- Unsupported, ambiguous, stale, conflicting, or volatile questions abstain.
- Every displayed citation must be attached to evidence used in displayed answer text.
- Tests use hash embeddings and the NumPy store; optional backends cannot be required.
- This plan implements the correctness core. Full subject-area migration and channel security are separate subprojects because each has an independent acceptance boundary.

## File map

- `ragnarok_rag/knowledge.py`: typed source/record models and corpus validation.
- `ragnarok_rag/query_analysis.py`: deterministic Portuguese intent/entity analysis.
- `ragnarok_rag/record_retriever.py`: fielded BM25 plus vector candidate generation.
- `ragnarok_rag/evidence.py`: absolute support scoring, ambiguity, and abstention.
- `ragnarok_rag/answer.py`: public answer/citation contract and record composition.
- `ragnarok_rag/index_manifest.py`: corpus/config fingerprint and readiness result.
- `ragnarok_rag/pipeline.py`: ingestion and end-to-end orchestration.
- `corpus/sources.json`: authoritative source catalog.
- `corpus/records/*.json`: independently answerable, source-backed records.
- `tests/fixtures/knowledge/`: controlled valid and invalid corpora.
- `tests/test_knowledge.py`, `test_query_analysis.py`, `test_record_retriever.py`, `test_evidence.py`, `test_index_manifest.py`: focused unit tests.
- `tests/test_retrieval.py`, `test_cli.py`, `test_web_app.py`, `test_whatsapp_service.py`: public-contract regressions.
- `evaluation/core_cases.json`, `scripts/evaluate.py`, `tests/test_evaluate.py`: held-out core metrics.

---

### Task 1: Validated knowledge records and sources

**Files:**
- Create: `ragnarok_rag/knowledge.py`
- Create: `tests/test_knowledge.py`
- Create: `tests/fixtures/knowledge/sources.json`
- Create: `tests/fixtures/knowledge/records/classes.gatuno.beginner.json`

**Interfaces:**
- Consumes: a directory containing `sources.json` and `records/*.json`.
- Produces: `load_knowledge_base(root: Path, *, today: date | None = None) -> KnowledgeBase` and `CorpusValidationError`.

- [ ] **Step 1: Write validation tests that name the breaks**

```python
def test_load_knowledge_base_resolves_evidence_source(knowledge_fixture: Path) -> None:
    base = load_knowledge_base(knowledge_fixture, today=date(2026, 9, 1))
    record = base.records_by_id["classes.gatuno.beginner"]
    assert record.ruleset == "ro-latam-renewal"
    assert base.sources_by_id[record.evidence[0].source_id].url.scheme == "https"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["evidence"][0].update(source_id="missing"), "fonte inexistente"),
        (lambda data: data.update(ruleset="bro-renewal"), "ruleset incompatível"),
        (lambda data: data.update(review_status="unknown"), "review_status inválido"),
    ],
)
def test_invalid_record_stops_ingestion(tmp_path: Path, mutation, message: str) -> None:
    root = copy_valid_fixture(tmp_path)
    mutate_record(root, "classes.gatuno.beginner", mutation)
    with pytest.raises(CorpusValidationError, match=message):
        load_knowledge_base(root, today=date(2026, 9, 1))
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_knowledge.py -v`

Expected: collection fails because `ragnarok_rag.knowledge` does not exist.

- [ ] **Step 3: Implement the typed schema and aggregate validation**

```python
@dataclass(frozen=True)
class SourceRecord:
    id: str
    title: str
    url: ParseResult
    publisher: str
    source_type: Literal["official", "community-reviewed"]
    accessed_at: date
    ruleset: str
    reliability: float


@dataclass(frozen=True)
class Evidence:
    source_id: str
    locator: str
    excerpt: str


@dataclass(frozen=True)
class Validity:
    episode: str | None
    valid_from: date | None
    valid_until: date | None
    volatile: bool


@dataclass(frozen=True)
class KnowledgeRecord:
    schema_version: int
    id: str
    title: str
    intent: str
    entities: tuple[str, ...]
    aliases: tuple[str, ...]
    ruleset: str
    review_status: Literal["active", "conflicting"]
    validity: Validity
    answer: str
    evidence: tuple[Evidence, ...]
    tags: tuple[str, ...]
    related_ids: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeBase:
    sources_by_id: dict[str, SourceRecord]
    records_by_id: dict[str, KnowledgeRecord]

    @property
    def records(self) -> tuple[KnowledgeRecord, ...]:
        return tuple(self.records_by_id.values())
```

Reject duplicate IDs, non-HTTP(S) URLs, reliability outside `0..1`, future access dates, empty answers/evidence, unknown related IDs, source/ruleset mismatch, invalid review status, and schema versions other than `1`. Expired, volatile, and conflicting records remain loadable so the evidence gate can return the correct typed abstention. Raise one `CorpusValidationError` containing every issue with its file path.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_knowledge.py -v`

Expected: PASS.

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: the pre-existing Gatuno regression is the only failure.

- [ ] **Step 5: Commit**

```powershell
git add ragnarok_rag/knowledge.py tests/test_knowledge.py tests/fixtures/knowledge
git commit -m "feat: validate source-backed knowledge records"
```

### Task 2: Deterministic intent and entity analysis

**Files:**
- Create: `ragnarok_rag/query_analysis.py`
- Create: `tests/test_query_analysis.py`

**Interfaces:**
- Consumes: `KnowledgeBase` from Task 1 and raw query text.
- Produces: `QueryAnalyzer(base: KnowledgeBase).analyze(query: str) -> QueryAnalysis`.

- [ ] **Step 1: Write query-analysis regressions**

```python
@pytest.mark.parametrize("query", [
    "gatuno é uma boa classe para começar?",
    "vale a pena iniciar de gatuno?",
    "gattuno serve para primeiro personagem?",
])
def test_beginner_gatuno_resolves_entity_and_intent(analyzer, query: str) -> None:
    analysis = analyzer.analyze(query)
    assert analysis.intent == "class.beginner_suitability"
    assert analysis.entities == ("class:gatuno",)
    assert analysis.ambiguities == ()


def test_out_of_domain_question_has_no_supported_intent(analyzer) -> None:
    analysis = analyzer.analyze("qual é a capital do Brasil?")
    assert analysis.intent is None
    assert analysis.entities == ()


def test_short_ambiguous_alias_is_not_guessed(analyzer_with_bb_aliases) -> None:
    analysis = analyzer_with_bb_aliases.analyze("como usar bb?")
    assert analysis.entities == ()
    assert analysis.ambiguities == ("bb",)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_query_analysis.py -v`

Expected: collection fails because `QueryAnalyzer` does not exist.

- [ ] **Step 3: Implement explicit intent rules and conservative typo resolution**

```python
@dataclass(frozen=True)
class QueryAnalysis:
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    requested_ruleset: str
    intent: str | None
    entities: tuple[str, ...]
    ambiguities: tuple[str, ...]


INTENT_PATTERNS = {
    "class.beginner_suitability": (
        frozenset({"boa", "classe", "comecar"}),
        frozenset({"iniciar", "classe"}),
        frozenset({"primeiro", "personagem"}),
        frozenset({"iniciante", "classe"}),
    ),
    "class.overview": (
        frozenset({"como", "gatuno"}),
        frozenset({"sobre", "classe"}),
    ),
}
```

Build the alias map from records. Exact aliases win. For tokens with at least five characters, accept one canonical alias only when `SequenceMatcher(None, token, alias).ratio() >= 0.84`; never fuzzy-match aliases shorter than five characters. When one alias maps to multiple entity IDs, return it in `ambiguities` unless other exact context terms leave one record family.

Default `requested_ruleset` to `ro-latam-renewal`. Set it to `pre-renewal` when the normalized query contains `pre renewal`, `pre-renewal`, `pré-renovação`, or `pre renovacao`; the evidence gate will return `version_mismatch` instead of silently applying current rules.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_query_analysis.py tests/test_text.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add ragnarok_rag/query_analysis.py tests/test_query_analysis.py
git commit -m "feat: analyze Ragnarok intents and entities"
```

### Task 3: Record candidate retrieval without confidence leakage

**Files:**
- Create: `ragnarok_rag/record_retriever.py`
- Create: `tests/test_record_retriever.py`
- Modify: `ragnarok_rag/retriever.py`

**Interfaces:**
- Consumes: `KnowledgeBase`, `VectorStore`, `Embedder`, and `QueryAnalysis`.
- Produces: `RecordRetriever.search(analysis: QueryAnalysis, *, top_k: int, candidate_k: int) -> list[RecordCandidate]`.

- [ ] **Step 1: Write ranking-signal tests**

```python
def test_gatuno_beginner_record_ranks_above_generic_job_change(record_retriever, analyzer) -> None:
    hits = record_retriever.search(
        analyzer.analyze("gatuno é uma boa classe para começar?"),
        top_k=3,
        candidate_k=10,
    )
    assert hits[0].record.id == "classes.gatuno.beginner"


def test_retrieval_scores_are_raw_diagnostics_not_confidence(record_retriever, analyzer) -> None:
    hits = record_retriever.search(analyzer.analyze("capital do Brasil"), top_k=3, candidate_k=10)
    assert all(hit.confidence is None for hit in hits)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_record_retriever.py -v`

Expected: collection fails because `RecordRetriever` does not exist.

- [ ] **Step 3: Implement fielded BM25 and vector RRF**

```python
@dataclass(frozen=True)
class RecordCandidate:
    record: KnowledgeRecord
    rank_score: float
    lexical_score: float
    vector_score: float
    lexical_rank: int | None
    vector_rank: int | None
    confidence: None = None


FIELD_WEIGHTS = {
    "title": 2.5,
    "aliases": 3.0,
    "entities": 3.0,
    "answer": 1.0,
    "tags": 0.5,
}
```

Use raw BM25 scores per field and merge lexical/vector rank positions with RRF `1 / (60 + rank)`. Do not normalize the best lexical hit to `1.0`. Store one vector per record using `title + aliases + tags + answer`; keep `record_id` as the store ID and resolve authoritative content from `KnowledgeBase`, not vector metadata.

- [ ] **Step 4: Run focused and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_record_retriever.py tests/test_retrieval.py -v`

Expected: the new retriever tests pass; the old pipeline Gatuno assertion remains RED until Task 5.

- [ ] **Step 5: Commit**

```powershell
git add ragnarok_rag/record_retriever.py ragnarok_rag/retriever.py tests/test_record_retriever.py
git commit -m "feat: retrieve structured knowledge candidates"
```

### Task 4: Independent evidence gate and typed abstention

**Files:**
- Create: `ragnarok_rag/evidence.py`
- Create: `tests/test_evidence.py`

**Interfaces:**
- Consumes: `QueryAnalysis`, ranked `RecordCandidate` values, source reliability, and date.
- Produces: `EvidenceGate.select(analysis: QueryAnalysis, candidates: Sequence[RecordCandidate], sources: Mapping[str, SourceRecord], *, today: date) -> EvidenceSelection`.

- [ ] **Step 1: Write support and abstention tests**

```python
def test_entity_and_intent_are_both_required(gate, gatuno_beginner_candidate, analysis_factory) -> None:
    wrong_intent = analysis_factory(intent="class.leveling", entities=("class:gatuno",))
    selection = gate.select(wrong_intent, [gatuno_beginner_candidate])
    assert selection.status == "not_found"


@pytest.mark.parametrize("query", [
    "qual é a capital do Brasil?",
    "onde upar gatuno nível 20?",
    "quanto HP eu preciso?",
])
def test_unsupported_intent_abstains(engine_fixture, query: str) -> None:
    selection = engine_fixture.select(query)
    assert selection.record is None
    assert selection.status == "not_found"


def test_two_supported_answers_inside_margin_are_ambiguous(gate, tied_candidates, analysis) -> None:
    selection = gate.select(analysis, tied_candidates)
    assert selection.status == "ambiguous"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evidence.py -v`

Expected: collection fails because `EvidenceGate` does not exist.

- [ ] **Step 3: Implement hard gates before absolute scoring**

```python
@dataclass(frozen=True)
class EvidenceSelection:
    status: Literal[
        "answered", "not_found", "ambiguous", "version_mismatch",
        "volatile", "conflicting_sources", "index_stale"
    ]
    record: KnowledgeRecord | None
    confidence: float
    reason: str


class EvidenceGate:
    MIN_CONFIDENCE = 0.80
    MIN_MARGIN = 0.08
```

Return `ambiguous` before candidate scoring when `analysis.ambiguities` is non-empty, and `version_mismatch` when `analysis.requested_ruleset != "ro-latam-renewal"`. Return `conflicting_sources` for a matching record with `review_status == "conflicting"`, and `volatile` for an expired or volatile matching record. Hard-reject unknown intent, incomplete entity coverage, ruleset mismatch, and missing evidence sources. For eligible records calculate `0.45` exact intent + `0.30` complete entity coverage + up to `0.15` literal non-stopword coverage + `0.10 * source.reliability`. Ranking scores never enter this formula. Require `0.80` and a `0.08` lead over a different eligible answer.

- [ ] **Step 4: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evidence.py -v`

Expected: PASS, including all three false-answer regressions.

- [ ] **Step 5: Commit**

```powershell
git add ragnarok_rag/evidence.py tests/test_evidence.py
git commit -m "feat: gate answers on explicit evidence"
```

### Task 5: Seed the authoritative RO LATAM corpus

**Files:**
- Create: `corpus/sources.json`
- Create: `corpus/records/classes.gatuno.beginner.json`
- Create: `corpus/records/classes.gatuno.overview.json`
- Create: `corpus/records/classes.progression.json`
- Create: `tests/test_corpus.py`

**Interfaces:**
- Consumes: schema from Task 1.
- Produces: the first production-valid `KnowledgeBase` for `ro-latam-renewal`.

- [ ] **Step 1: Write production-corpus validation tests**

```python
def test_production_corpus_is_valid_and_source_backed(project_root: Path) -> None:
    base = load_knowledge_base(project_root / "corpus", today=date(2026, 9, 1))
    assert "classes.gatuno.beginner" in base.records_by_id
    assert all(record.evidence for record in base.records)
    assert all(source.url.scheme == "https" for source in base.sources_by_id.values())


def test_gatuno_beginner_answer_is_cautious_and_does_not_claim_best(project_root: Path) -> None:
    record = load_knowledge_base(project_root / "corpus").records_by_id["classes.gatuno.beginner"]
    assert "pode ser uma opção" in record.answer.lower()
    assert "melhor classe" not in record.answer.lower()
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_corpus.py -v`

Expected: FAIL because `corpus/sources.json` and `corpus/records/` do not exist.

- [ ] **Step 3: Add the reviewed sources and records**

Use these source entries with `accessed_at: 2026-09-01`:

| ID | Type | Reliability | URL |
|---|---|---:|---|
| `gnjoy.portal` | `official` | `1.0` | `https://www.gnjoylatam.com/pt` |
| `gnjoy.classes.gatuno` | `official` | `1.0` | `https://ro.gnjoylatam.com/pt/intro/class/thief/guillotinecross?v=true` |
| `browiki.gatunos` | `community-reviewed` | `0.8` | `https://browiki.org/wiki/Gatunos` |

Use this answer for `classes.gatuno.beginner`:

> Gatuno pode ser uma opção para começar se você prefere combate ágil: a classe se destaca por esquiva e ataques rápidos e evolui para Mercenário ou Arruaceiro. As fontes consultadas não afirmam que seja a melhor classe para iniciantes, então a escolha depende do seu estilo.

Attach the intent `class.beginner_suitability`, entity `class:gatuno`, aliases `gatuno`, `gatunos`, `thief`, tags `classes`, `iniciante`, `agilidade`, and short evidence locators from the official class tree and the bROWiki introduction. Keep each copied excerpt below 25 words.

- [ ] **Step 4: Validate the production corpus**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_corpus.py tests/test_knowledge.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add corpus/sources.json corpus/records tests/test_corpus.py
git commit -m "data: add sourced RO LATAM class records"
```

### Task 6: Integrate the correctness engine and faithful citations

**Files:**
- Modify: `ragnarok_rag/answer.py`
- Modify: `ragnarok_rag/pipeline.py`
- Modify: `ragnarok_rag/config.py`
- Modify: `ragnarok_rag/cli.py`
- Modify: `whatsapp/service.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_retrieval.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_web_app.py`
- Modify: `tests/test_whatsapp_service.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: backward-compatible `Answer.found/text/confidence/sources` plus explicit `status/reason`; all channels use the same result.

- [ ] **Step 1: Replace the Gatuno regression with the approved public contract**

```python
def test_beginner_class_question_answers_about_the_requested_class(deterministic_rag) -> None:
    answer = deterministic_rag.ask("gatuno é uma boa classe para começar?")
    assert answer.status == "answered"
    assert answer.found is True
    assert "Gatuno pode ser uma opção para começar" in answer.text
    assert "Job 50" not in answer.text
    assert {source.record_id for source in answer.sources} == {"classes.gatuno.beginner"}
    assert "https://browiki.org/wiki/Gatunos" in {source.url for source in answer.sources}


@pytest.mark.parametrize("query", [
    "qual é a capital do Brasil?",
    "onde upar gatuno nível 20?",
    "quanto custa a carta hidra hoje?",
])
def test_unsupported_questions_abstain(deterministic_rag, query: str) -> None:
    answer = deterministic_rag.ask(query)
    assert answer.found is False
    assert answer.status in {"not_found", "volatile"}
    assert answer.sources == []
```

- [ ] **Step 2: Run public-contract tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_whatsapp_service.py tests/test_web_app.py -v`

Expected: FAIL because the pipeline still composes a Markdown chunk and cites three retrieval hits.

- [ ] **Step 3: Implement typed answers and pipeline orchestration**

```python
@dataclass(frozen=True)
class Source:
    record_id: str
    title: str
    url: str
    locator: str
    accessed_at: str
    source_type: str


@dataclass
class Answer:
    query: str
    found: bool
    text: str
    confidence: float
    status: str = "not_found"
    reason: str = ""
    sources: list[Source] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
```

`RagnarokRAG.ingest()` loads the knowledge base, embeds one text per record, writes `record_id` store entries, and rebuilds `RecordRetriever`. `ask()` performs analyze → retrieve → evidence select → compose. The composer uses only `selection.record.answer`; construct citations only from that record's evidence. `format_answer()` prints source titles and public URLs. Do not emit candidate records as sources or related citations.

- [ ] **Step 4: Run all channel and core tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_retrieval.py tests/test_cli.py tests/test_web_app.py tests/test_whatsapp_service.py -v`

Expected: PASS, including the original Gatuno bug.

- [ ] **Step 5: Commit**

```powershell
git add ragnarok_rag/answer.py ragnarok_rag/pipeline.py ragnarok_rag/config.py ragnarok_rag/cli.py whatsapp/service.py tests
git commit -m "feat: answer only from supported records"
```

### Task 7: Fingerprinted index and readiness diagnostics

**Files:**
- Create: `ragnarok_rag/index_manifest.py`
- Create: `tests/test_index_manifest.py`
- Modify: `ragnarok_rag/store.py`
- Modify: `ragnarok_rag/pipeline.py`
- Modify: `whatsapp/app.py`
- Modify: `tests/test_web_app.py`

**Interfaces:**
- Consumes: sorted authoritative corpus bytes, schema/ruleset/retrieval config, store count/dimensions.
- Produces: `compute_index_fingerprint(corpus_root: Path, config: Config) -> str` and `RagnarokRAG.readiness() -> Readiness`.

- [ ] **Step 1: Write stale/corrupt index tests**

```python
def test_corpus_change_makes_existing_index_stale(rag, corpus_root: Path) -> None:
    rag.ingest(verbose=False)
    mutate_answer(corpus_root, "classes.gatuno.overview")
    assert rag.readiness() == Readiness(ready=False, reason="index_stale")


def test_record_vector_count_mismatch_is_not_ready(rag) -> None:
    rag.ingest(verbose=False)
    remove_one_metadata_record(rag.config.index_dir)
    fresh = RagnarokRAG(rag.config)
    assert fresh.readiness().reason == "index_invalid"
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_index_manifest.py -v`

Expected: FAIL because readiness only checks `store.count() > 0`.

- [ ] **Step 3: Implement fingerprint, validation, and atomic writes**

```python
@dataclass(frozen=True)
class Readiness:
    ready: bool
    reason: str


def compute_index_fingerprint(corpus_root: Path, config: Config) -> str:
    digest = hashlib.sha256()
    for path in sorted([corpus_root / "sources.json", *(corpus_root / "records").glob("*.json")]):
        digest.update(path.relative_to(corpus_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(json.dumps({
        "schema_version": 1,
        "ruleset": config.ruleset,
        "embedding_backend": config.embedding_backend,
        "embedding_model": config.embedding_model,
        "hash_dims": config.hash_dims,
    }, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
```

Write NumPy vectors and metadata to sibling `.tmp` files, validate dimensions/count/IDs, then `os.replace()` data files and write the manifest last. `readiness()` recomputes the fingerprint and validates manifest/store counts and dimensions. `/health` returns process status separately from `rag_ready` and `rag_reason`.

- [ ] **Step 4: Run readiness and full tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_index_manifest.py tests/test_web_app.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add ragnarok_rag/index_manifest.py ragnarok_rag/store.py ragnarok_rag/pipeline.py whatsapp/app.py tests/test_index_manifest.py tests/test_web_app.py
git commit -m "feat: reject stale or invalid indexes"
```

### Task 8: Core held-out evaluation and documentation

**Files:**
- Create: `evaluation/core_cases.json`
- Modify: `scripts/evaluate.py`
- Modify: `tests/test_evaluate.py`
- Modify: `README.md`
- Modify: `.env.example`

**Interfaces:**
- Consumes: public `RagnarokRAG.ask()` results.
- Produces: `EvaluationMetrics` with answer precision, answerable recall, citation faithfulness, abstention precision/recall, and OOD false-answer rate.

- [ ] **Step 1: Write metric tests with hand-derived expectations**

```python
def test_metrics_count_supported_answers_and_abstentions() -> None:
    cases = [
        EvaluationCase("q1", "answered", "r1"),
        EvaluationCase("q2", "not_found", None),
        EvaluationCase("q3", "answered", "r2"),
        EvaluationCase("q4", "not_found", None),
    ]
    predictions = [
        Prediction("answered", "r1", citation_faithful=True),
        Prediction("not_found", None, citation_faithful=True),
        Prediction("not_found", None, citation_faithful=True),
        Prediction("answered", "r1", citation_faithful=False),
    ]
    metrics = calculate_metrics(cases, predictions)
    assert metrics.supported_answer_precision == 0.5
    assert metrics.answerable_recall == 0.5
    assert metrics.citation_faithfulness == 0.5
    assert metrics.ood_false_answer_rate == 0.5
```

- [ ] **Step 2: Run the metric test and verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_evaluate.py -v`

Expected: FAIL because the old evaluator only compares Markdown headings.

- [ ] **Step 3: Implement metrics and a 30-case independent core set**

Create ten answerable Gatuno/class questions, ten paraphrase/typo variants, and ten mandatory abstentions. The abstention set must include the exact questions `qual é a capital do Brasil?`, `onde upar gatuno nível 20?`, `quanto HP eu preciso?`, `quanto custa a carta hidra hoje?`, `gatuno é melhor que arqueiro?`, `como chegar em Geffen?`, `qual servidor tem mais jogadores hoje?`, `receita de lasanha`, `como virar transclasse?`, and `qual carta resiste a fogo?`. Do not reuse these 30 strings in intent-rule unit tests.

The CLI exits nonzero unless citation faithfulness and supported-answer precision are `1.0`, OOD false-answer rate is `0.0`, and answerable recall is at least `0.85`. Label these as core metrics; the full corpus-migration subproject expands the held-out set beyond 100 reviewed questions before release.

- [ ] **Step 4: Update operator documentation and verify from a clean index**

Document the `ro-latam-renewal` scope, public-source JSON format, strict abstention behavior, legacy Markdown exclusion, index-stale recovery, and the fact that source URLs are shown to users.

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

Run: `.\.venv\Scripts\python.exe -m ragnarok_rag.cli ingest`

Expected: exit `0`, three authoritative records indexed, fingerprint printed.

Run: `.\.venv\Scripts\python.exe -m scripts.evaluate`

Expected: exit `0`, precision `1.0`, citation faithfulness `1.0`, OOD false-answer rate `0.0`, answerable recall at least `0.85`.

Run: `.\.venv\Scripts\python.exe -m ragnarok_rag.cli ask "gatuno é uma boa classe para começar?"`

Expected: cautious Gatuno answer, no `Job 50`, and only the evidence URLs attached to `classes.gatuno.beginner`.

- [ ] **Step 5: Commit and push the completed core**

```powershell
git add evaluation/core_cases.json scripts/evaluate.py tests/test_evaluate.py README.md .env.example
git commit -m "test: add correctness-first evaluation"
git push origin main
```

## Core completion boundary

This plan is complete only when the Gatuno regression is green, all unsupported adversarial questions abstain, citations are faithful, stale indexes are rejected, the full pytest suite passes, and the core evaluator meets its thresholds. The next independently reviewable subprojects are: source-reviewed migration of the remaining subject areas to reach the 100+ held-out set, then Meta/Twilio signature and delivery hardening.
