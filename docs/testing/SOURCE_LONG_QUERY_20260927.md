# r149 complete long source queries

Root cause: chat admits 8,000 characters but source sparse search rejected
queries over 2,000 characters or 128 unique terms. SourceMemoryService caught
the exception as retrieval_error. Increasing model context cannot repair this.

Search now accepts the full 8,000-character admitted query. Terms are passed
as a single bound JSON array: SQLite json_each / PostgreSQL
jsonb_array_elements_text. Ranking remains one scope-filtered SQL snapshot,
with complete terms and existing erasure fences; no prefix selection, extra
model call, per-chunk top-k approximation or source-body scan was introduced.

Four new tests initially failed, then passed. The prior 2,001-character
rejection test was updated to the 8,001-character admission boundary.
45 focused tests passed, followed by 235 combined source/context/reviewer
tests (16.30 seconds, .tmp/r149-source-suite.xml). SQLite query-plan assertion
still confirms the scope+term index. Changed standalone modules pass Ruff;
pg_database.py has existing whole-file lint debt, not cleaned in this patch.

Real remote SQLite/PostgreSQL parity ran on a disposable cluster under
/home/boot/lhm/multipersonal-runtime/evaluations/r110pg.eT4h9f, then stopped
successfully. Both engines passed long text and 600-term tail-target search,
scope exclusion, old-source recall beyond 260 rows, source windows, erasure
fences and 13 lifecycle contracts including concurrent capture/erase.
The broad erasure fence contract is preserved, not claimed as ideal UX.
Artifact: .tmp/r149-pg-parity.json. No production database was used.

Real 64K DeepSeek replay: three HTTP 200 turns, ten calls (3 policy, 3 answer,
2 writer, 2 selector), zero local calls. Long first turn now produces source
no_match rather than retrieval_error; the complete long source is available
on following turns. Old workplace becomes superseded; replacement active;
cold-question history is empty and the answer correctly gives the new place.
Selector was exercised; semantic reviewer was not triggered.
Artifacts: .tmp/r149-manifest.json, .tmp/r149-traces.jsonl.
Remote snapshot mechanism-r149-source; output r147-long-source-query uses
the replay's existing allowed prefix.

Not complete: source-only targeted erasure and overly broad owner fences,
warm-history erasure behavior, acknowledgment before writer commit, RAG
sufficiency and remaining dynamic-context real-call/ablation coverage.
The sparse query is still lexical, not semantic proof of source relevance.
