# Ragnarok RAG — Correctness-First Redesign

## Objective

Replace the proof-of-concept behavior that returns the closest passage with a deterministic, evidence-gated question-answering system for the current Brazilian Ragnarok Online Renewal ruleset. The runtime remains fully local and uses no LLM or generative API.

The system must prefer an explicit refusal over an unsupported answer. Every factual answer must be traceable to the exact curated record used to produce it and, through that record, to a public source.

## Scope

The primary scope is the current bRO Renewal ruleset. Facts that depend on an episode, event, server, date, or economic state must declare that limitation. Pre-Renewal and private-server information is excluded from the primary answer path unless a future caller explicitly selects another ruleset.

Official public sources have the highest priority. A recognized community source may fill a gap when official documentation does not cover the fact, provided the record identifies the source type, access date, applicability, and reduced reliability. Conflicting sources do not produce an automatic answer; the conflict is recorded for editorial review.

The redesign covers corpus provenance, ingestion, retrieval, evidence validation, answers, citations, evaluation, index lifecycle, and the existing browser/Meta/Twilio delivery paths. It does not introduce conversational generation, live market-price lookup, automated web scraping, a distributed queue, or a complete knowledge graph.

## Correctness contract

An answer is allowed only when all of the following are true:

1. The question has a supported intent or contains enough literal evidence for a safe document lookup.
2. Every required named entity is resolved unambiguously.
3. A candidate record matches the active ruleset and covers the required intent and entities.
4. The record contains an explicit answer and at least one public source supporting that answer.
5. The evidence score passes an absolute threshold and the leading candidate is not materially ambiguous with another candidate.
6. The information is not marked expired, conflicting, or volatile without a current source.

Otherwise, the result is an abstention with a stable reason: `not_found`, `ambiguous`, `version_mismatch`, `volatile`, `conflicting_sources`, or `index_stale`.

The answer text is never synthesized from unrelated passages. Multiple records may be combined only for a predefined multi-part intent, and every included record must contribute visible answer text.

## Knowledge-record format

The authoritative corpus becomes a set of reviewable JSON files under `corpus/records/`, one record per independently answerable fact. JSON keeps the schema explicit, works with the Python standard library, and avoids parsing loosely structured Markdown front matter.

Each record has the following shape. The values below illustrate the schema and are not authoritative corpus content:

```json
{
  "schema_version": 1,
  "id": "classes.gatuno.iniciante",
  "title": "Gatuno para jogadores iniciantes",
  "intent": "class.beginner_suitability",
  "entities": ["class:gatuno"],
  "aliases": ["gatuno", "thief"],
  "ruleset": "bro-renewal",
  "validity": {
    "episode": null,
    "valid_from": null,
    "valid_until": null,
    "volatile": false
  },
  "answer": "Texto editorial curto, estritamente sustentado pela evidência.",
  "evidence": [
    {
      "source_id": "source.example",
      "locator": "seção ou fragmento verificável",
      "excerpt": "Trecho curto necessário para auditoria."
    }
  ],
  "tags": ["classes", "iniciante"],
  "related_ids": []
}
```

Source metadata lives in `corpus/sources.json` and includes `id`, `title`, `url`, `publisher`, `source_type`, `accessed_at`, `ruleset`, and `reliability`. Accepted source types are `official` and `community-reviewed`. URLs must be HTTP(S), source identifiers must resolve, and access dates cannot be in the future.

The `answer` is human-curated text, not runtime generation. `evidence.excerpt` is kept short for audit and copyright safety. A record may cite multiple sources, but the response cites only the sources connected to the evidence actually used.

The existing Markdown corpus is moved to a legacy/reference location and excluded from authoritative answers until its facts are migrated into validated records. It must not silently coexist with the new records in the same answer index.

## Components and boundaries

### Schema and validation

A corpus module owns typed record/source models, JSON loading, cross-reference validation, ruleset validation, duplicate-ID detection, URL validation, and validity checks. Invalid corpus data stops ingestion with actionable diagnostics; it is never skipped silently.

### Query analysis

A deterministic analyzer normalizes Portuguese text, resolves entities through canonical names and aliases, tolerates conservative spelling errors through character similarity, and classifies known intents through explicit phrase/token rules. Ambiguous aliases such as `BB`, `HP`, or `ME` remain ambiguous unless surrounding words resolve them.

The analyzer returns a structured query containing normalized text, resolved entities, unresolved terms, candidate intents, ruleset, and ambiguity information. It does not infer gameplay facts.

### Candidate retrieval

Candidate generation remains hybrid. Fielded BM25 searches title, answer, aliases, and tags. The existing hash embeddings may provide a secondary recall signal; sentence-transformers remains an optional backend. Candidate generation is allowed to be broad because it cannot authorize an answer.

Per-query normalization is not used as confidence. Retrieval scores are diagnostic ranking signals only.

### Evidence gate

The evidence gate independently checks candidate support. Required features are intent compatibility, complete entity coverage, active ruleset, valid source references, and record validity. Optional bounded features include literal query-term coverage, phrase coverage, alias quality, and the margin from the next eligible candidate.

Each feature has an absolute value with a documented range. The gate produces a decision plus reasons; it does not reuse the top candidate's relative lexical score as proof. Unknown intent requires stricter literal coverage than a recognized intent. Ambiguous entities force abstention unless exactly one candidate resolves the ambiguity using context.

### Answer composition

The composer consumes only eligible records. Its result contains `status`, `text`, `confidence`, `reason`, `citations`, and optional related questions. A citation contains the record ID, source title, public URL, locator, access date, and source type.

Related questions are suggestions only and are never presented as supporting sources. The renderer may format the same result for CLI, browser, Meta, and Twilio, but channel adapters cannot alter factual content.

## Query flow

1. Validate length and normalize the incoming question.
2. Resolve ruleset, intent, entities, spelling variants, and ambiguity.
3. Retrieve a broad candidate set.
4. Remove candidates with incompatible ruleset, invalid dates, missing sources, or unsupported intent/entities.
5. Score the remaining evidence with absolute features.
6. Answer only if the evidence threshold and ambiguity margin pass.
7. Render answer text and citations from the eligible record set.
8. Return a typed abstention for every other case.

For “Gatuno é uma boa classe para começar?”, the analyzer must identify `class:gatuno` and `class.beginner_suitability`. A generic class description or a Job 50 FAQ cannot pass the evidence gate. If no sourced Gatuno suitability record exists, the correct result is `not_found`.

## Index lifecycle

The manifest includes a SHA-256 fingerprint derived from the sorted authoritative corpus bytes, source catalog, schema version, ruleset, retrieval configuration, embedding backend, and embedding model identifier. Readiness requires the current fingerprint to equal the manifest fingerprint.

NumPy index output is written to a temporary directory, validated, and atomically swapped into place. Loading validates vector dimensions, record/vector counts, unique IDs, and manifest compatibility. Backend fallback is permitted only for an explicitly unavailable optional dependency; corrupt data and invalid configuration are surfaced as errors.

The health endpoint distinguishes process health from answer readiness and reports a machine-readable non-secret reason when the index is stale or invalid.

## Delivery and operational safety

The browser, CLI, Meta, and Twilio paths call one channel-neutral answer service. Blocking local retrieval is moved off the asynchronous event loop.

Meta requests verify the request signature with an explicitly configured application secret. Twilio requests verify `X-Twilio-Signature` using the supported signing implementation. Production-channel startup fails when required secrets are missing; there are no usable default tokens.

Inbound message deduplication uses a bounded TTL store interface. The local implementation is in-memory; a shared implementation can be supplied for multi-process deployment. Meta processing may use the framework's post-response background mechanism for the POC, while the documentation states that durable production delivery requires an external queue. Rate limiting remains bounded and keyed by sender without exposing message contents.

Errors are separated into validation, not-ready, upstream-delivery, and internal failures. Users receive stable Portuguese messages without stack traces. Logs contain correlation IDs and decisions but redact tokens, signatures, phone numbers, and full message bodies.

## Testing and evaluation

Implementation follows test-driven development. Each behavior is introduced by a test that fails for the expected reason before production code changes.

Unit tests cover schema validation, source references, query intent, entity resolution, conservative typo handling, ambiguity, evidence features, absolute thresholds, citation construction, fingerprints, atomic index validation, signatures, and bounded TTL deduplication.

Integration tests build a real deterministic hash/NumPy index from controlled records and exercise the public answer service. They include exact questions, paraphrases, spelling mistakes, entity-correct/intent-wrong cases, version mismatches, volatility, source conflicts, and unrelated questions.

The evaluation set is separate from rule-tuning fixtures and contains at least 100 reviewed Portuguese questions across:

- directly answerable facts;
- paraphrases and colloquial forms;
- conservative spelling errors;
- ambiguous abbreviations;
- questions outside Ragnarok;
- unsupported current prices or events;
- correct entities paired with unsupported intents;
- incompatible rulesets.

Reported metrics are top-record accuracy, supported-answer precision, answerable recall, citation faithfulness, abstention precision/recall, and false-answer rate for out-of-domain questions. Release acceptance requires 100% citation faithfulness and supported-answer precision on the reviewed set, zero out-of-domain factual answers, at least 90% top-record accuracy, and at least 85% answerable recall. Metric computation itself is tested against hand-derived fixtures.

The existing Gatuno regression remains mandatory. The old nine-question evaluation is retained only as a historical regression subset and is not reported as overall quality.

## Migration and rollout

Implementation proceeds in dependency order:

1. Introduce record/source schemas and validation alongside the legacy loader.
2. Add representative sourced records and build the deterministic analyzer/evidence gate through TDD.
3. Replace answer composition and citation behavior while keeping channel-facing result compatibility where practical.
4. Add corpus fingerprints, atomic persistence, and readiness diagnostics.
5. Build the reviewed evaluation set and calibrate only against separate development fixtures.
6. Migrate the remaining subject areas from Markdown after source review; unmigrated topics abstain.
7. Harden Meta/Twilio boundaries and update operational documentation.
8. Remove the legacy authoritative path after all acceptance checks pass.

During migration, a feature flag may select the legacy engine for comparison, but public-facing defaults use the correctness-first engine only after it satisfies the acceptance metrics. The flag cannot merge legacy passages into validated answers.

## Success criteria

- Runtime answers use no LLM or generative API.
- The active corpus is scoped to bRO Renewal and every authoritative record resolves to public source metadata.
- Unsupported questions abstain instead of returning merely similar passages.
- “Gatuno é uma boa classe para começar?” never returns the Job 50 answer.
- Every displayed citation contributed evidence to the displayed answer.
- Corpus or retrieval changes make an old index unready.
- The complete automated suite, corpus validator, clean index build, evaluation thresholds, and HTTP smoke tests pass.
- Documentation clearly distinguishes the local POC boundary from production requirements.
