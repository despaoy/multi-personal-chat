# Project simplification audit

Baseline: `8511e95d1a27bfc726a4b82ebeb49ce3fd30cdf9`. Date: 2026-10-25.

This is a bounded, contract-preserving inventory, not a claim that every possible
refactor is complete. Failure must not become fabricated success. Optional modes,
domain abstention, authorization, provenance and cleanup are not redundant merely
because they have fallback-like names.

## Inventory and decisions

| Subsystem | Removed or consolidated | Deliberately retained |
| --- | --- | --- |
| API and generation | Repeated provider arguments; duplicated filtered admin queries; invented authenticated username | Provider/LoRA choice, scoped SQL parameters, branch exclusions, request limits, persistence |
| Inference and bot | Duplicated async-to-sync bridges; repeated vLLM retry accounting; configured tokenizer errors replaced by estimates | Bounded transient retry, actual exception and cancellation, explicit no-tokenizer mode, provider protocols and usage |
| App, middleware, infra, cache | Duplicate JWT decoding, invalid cache-value clamping, unusable HTTP pool fallback | Valid API-key path, nonempty JWT principal, startup ownership, resource limits, all-resource cleanup with propagated errors |
| Database and repositories | Duplicate task decoding and LoRA metadata readers; exception-driven duplicate migration; corrupt JSON replaced by defaults | Savepoints/rollback, legacy schemas, erasure/receipt integrity, strict metadata, unknown rather than fabricated progress |
| Knowledge and character | Broad-reader compatibility branch; duplicate runtime selection; incomplete evidence defaults; eager model loading | Source ownership, conflict retries, required model/index failure, approved evidence, genuine insufficient-evidence abstention |
| Training and evaluation | Repeated cancellation; database-to-memory fallback; stale/failed/blank replies scored as success; invalid Gold/cleaner inputs accepted | Explicit in-memory/mock/ablation modes, task lifecycle, partial-turn receipts, valid refusals, hardware protection |
| Services | Mandatory generation evidence read directly | Completion admission, delivery leases, model-use accounting, subprocess limits, search errors, independent source authority |
| Frontend and gateway | Duplicate authentication HTTP handling and write-only user cache; feedback errors replaced by empty data; unreachable hash dedup | Shared API errors, explicit event identity, delivery acknowledgements and retry receipts |
| Scripts and deployment | Obsolete container-name polling; automatic model-source switching; duplicate launcher settings | Explicit source selection, health-checked Compose startup, bare deployment, canonical launcher |
| Dependencies and files | Orphan UI files/dependencies, unused development CLIs, unreferenced historical micro-reports | Component source, on-demand CLI workflow, licenses, provenance, operational evidence and regression coverage |

Historical Alembic migrations remain deployment contracts. `interfaces` remains
a package boundary. Root scripts, CI, public assets, documentation and archives
were checked for references before deletion. Runtime models, indexes, logs and
user databases are not source-cleanup targets. Game text and training data retain
attribution; absence of application imports is not evidence that a reproducibility
CLI, dataset or offline experiment is disposable.

## Important behavior changes

- vLLM releases active-request accounting before backoff. Exhaustion preserves the
  original error; permanent HTTP/protocol failures do not switch models or invent
  replies. Streaming and cancellation are not replayed.
- Configured training-store read failures propagate. Invalid task configuration
  fails before other fields are written. Explicit `db=None` remains supported.
- Required SQLite migrations roll back and fail startup rather than leaving a
  partially initialized database. Pool shutdown attempts every close and then
  propagates errors. Unexpected authentication faults cannot authorize a request.
- Missing LoRA trainer metadata leaves unknown progress at zero. Malformed metadata
  fails; weights alone do not imply completed training. Final/latest numeric
  checkpoint discovery remains supported.
- Character RAG construction captures configuration without loading a provider.
  The required load boundary initializes provider and service under the existing
  lock and publishes them atomically. Out-of-scope routing and warm/cold timeout
  checks require no model I/O; actual retrieval still requires valid artifacts.
- Rule-memory writes use the repository's existing scoped reader. Conflict retries
  reread authoritative state; they never fall back to scanning all memory.
- Indexed evidence uses its declared schema rather than missing-attribute defaults.
  Explicitly empty valid content is distinct from malformed evidence.
- Evaluation counts only successful, nonblank responses and exposes
  `scored_responses` and `generation_errors`. Failed turns stop the case and retain
  only earlier completed turns. Nonempty refusals/abstentions remain valid outputs.
  Missing/corrupt Gold data and failed safety/persona checks are not empty success.
- Download CLI sources are explicit; removed `--source auto` was not used by callers.
  Compose startup fails when health readiness fails instead of reporting success.

## Why other branches remain

- Retrieval snapshots, generation-bound cache keys, model/vector validation,
  source revisions and complete-query obligations prevent stale or unauthorized
  evidence. Mixed domains cannot borrow authority from one another.
- Approved-evidence fallback preserves already approved evidence when narrowing
  cannot match; it neither invents a quote nor widens approved spans. Missing raw
  source files and invalid ranges still fail.
- Character strategy selection, unresolved-risk handling and evidence abstention
  are policy outcomes, not hidden provider substitution.
- Memory observations, superseded records, pending conflicts, temporal projections
  and protected claims represent distinct history. Confirmed writes remain
  reportable when a later operation fails or is cancelled.
- Background delivery failures keep explicit retry receipts. Optional diagnostic
  logging does not invalidate an otherwise completed review. Dashboard snapshots
  copy cached state and reject invalid configuration.
- Offline probes, replay tools, experiment fixtures and benchmark compatibility
  entrypoints retain distinct schemas. Explicit mock/ablation choices remain;
  they do not activate after real generation fails.
- TRL contract checks, GPTQ's third-party renamed-symbol alias and optional GPU
  telemetry/thermal protection remain. Removing them would require separate
  package or hardware validation. Already-running training logs secondary storage
  failures without hiding the original cancellation/failure.

## Deletion audit

Removed 90 unreferenced historical micro-change reports, verified byte-identical
to the baseline before deletion. Exact paths, rationale and recovery commit are
in [the deletion manifest](../reports/cleanup-20261025.json). Preserved linked
reports, unique limitations, benchmark/model-run evidence, licenses, provenance
and both archives with their checksum manifests.

Also removed these unreferenced UI files and their exclusive dependencies:

- `src/hooks/use-mobile.ts`
- `src/components/ui/popover.tsx`
- `src/components/ui/dropdown-menu.tsx`

Unused `shadcn` and `only-allow` development dependencies were removed, preserving
component source and documented on-demand generation. Compatible Next.js, sharp
and js-yaml updates address the checked dependency advisories.

The initial pass retained all test files. A subsequent input-aware review found
limited exact duplication and fragmented guard contracts; the changes below
supersede that initial assessment. Formatting in the 34-file CI retrieval scope
was verified AST-identical. Separate subsystem audit notes were consolidated here.

## Test integrity and reproducibility

- Routing fixtures explicitly test real local and cloud budget contracts, including
  local `review=None`, rather than ambient machine configuration. Existing source,
  privacy, failure and answer-completeness assertions remain.
- Evidence fixtures use canonical document/candidate schemas. Generic vector tests
  write explicitly synthetic temporary bundles through the real artifact writer
  and loader; all existing vector-error/no-fallback assertions remain unchanged.
  The actual 1,098-document corpus integration assertions were not altered.
- Tests default `HF_HOME` to isolated temporary storage while honoring caller
  overrides. The ASGI export fixture sends a completed request once, then waits,
  rather than starving Starlette with endless immediate request events. Cleanup,
  cancellation, send-failure and admission assertions remain.
- The optional preference stack is already pinned in
  `backend/requirements-preference.txt`. Its datasets 4.1.1 pin conflicts with the
  production datasets 5 range. CI explicitly replaces the exact training-package
  declarations in its existing generated CPU manifest and resolves that manifest
  together with the preference pins. Production requirements remain unchanged.
- PostgreSQL integration uses isolated databases owned by a non-superuser fixture
  role. CI waits for the official PostgreSQL 17 service before provisioning them.

The tracked approved corpus reproduces 1,098 documents and 409 exact original
spans. Generated vectors are intentionally absent from source. Validation uses
only the official public, ungated Apache-2.0-declared encoder
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, pinned at revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Its five required files total
479,724,528 bytes; snapshot metadata and content hashes were verified. No remote
custom code or credentials are used. The existing offline CPU builder generated
all nine genuine partition files in 90 seconds at approximately 1.15 GiB peak RSS.
CI caches the pinned encoder, verifies it and builds the real index before tests.
Weights and indexes remain ignored and are not distributed in this patch.

Final aggregate counts belong to the validation report, not overlapping focused
run totals. Local tiny-model CPU integration is not user-model training; genuine
index rebuilding is not a production promotion or proof of chat-answer quality.
No paid provider validation or production deployment was performed. Browser smoke
still requires a functioning browser sandbox; successful HTTP smoke is not a
browser test. GitHub-hosted workflow execution must be verified after publication.


## Follow-up: test value and shared evaluation probes

This pass starts from PR commit `2e0a212` rather than assuming that earlier
completion notes prove every part of the repository is already minimal.

- Removed four redundant executed cases: one identical task-dependency input,
  one enum-membership assertion already implied by the exact enum list, and two
  erasure-selection input/assertion pairs repeated in the same module. Retained
  the invalid-type, foreign-source, and unauthorized-source security boundaries.
- Consolidated `test_nominal_preference_guard.py`,
  `test_preference_assertion_boundary.py`, `test_polar_question_guard.py`, and
  `test_advisory_preference_guard.py` into `test_user_fact_guard.py` plus
  `test_reply_guard_replay.py`. All 90 executed cases and distinct linguistic
  boundaries remain. The shared generation harness retains all five original
  message/reply pairs and their assertions. Old files are recoverable at the
  starting commit; their deletion is relocation, not loss of coverage.
- The two 455-line document-import evaluation probes differed only in seed-call
  expectation and session identity. They now use `document_import_probe.py`,
  retaining both documented module entrypoints and their distinct defaults.
  Normalized AST comparison confirms that the shared runner preserves each
  original body. Four new contract cases protect stage binding, CLI behavior,
  and invalid-phase rejection before creating output.

A broad static scan did not justify mass deletion: identical assertion bodies
often cover different semantic inputs. Test count is not a quality target.
Baseline collection contained 10,901 executed cases; four duplicate removals
and four new probe contracts leave that total unchanged. Collection equivalence
is not a claim of runtime success; aggregate test results are recorded separately.
The live provider/PostgreSQL evaluator scenarios themselves remain unexecuted.


## Integration review and observation evidence ownership (2026-10-25)

PR #3 was reviewed and merged at `edec840`. Its exact head `34daed4` has verified
CI results: 10,873 backend passes, one applicability skip, eight warnings, and
43 frontend passes. A separate server snapshot passed 145 targeted checks;
17 affected frontend checks passed locally. An extra Windows backend attempt
stalled during collection and was terminated without completed results.

The subsequent memory change keeps full observation sources solely in evidence,
with observation time and uncertainty in display content. Retrieval text and
persisted/admin records are unchanged. Revocation cleanup uses original evidence
rather than the common display label, preserving unrelated history. Production
code is two lines shorter, with no new semantic rule or abstraction.

After integration, 235 related checks and six-file syntax/Ruff checks passed.
Two new DeepSeek Flash requests verified complete synthetic identity, alias,
negative-alias and location evidence through stored recall, actual selection,
reference compilation and readback (3,755 tokens). The fixed sample's reference
shrank from 3,911 to 3,581 characters and selector input from 5,993 to 5,663;
other packet fields and stored memory remained unchanged. An earlier preflight
copied SQLite over an already opened destination and made zero provider calls;
the corrected test copies before importing the database singleton.

The server frontend was rebuilt with locked Next.js 16.4.1 and both services
restarted. A cross-filesystem rename interrupted the initial artifact switch;
explicit cross-filesystem copying completed it. Readiness/login return 200 and
unauthenticated protected access returns 401 with nosniff. Configuration and
other services are unchanged. These checks do not prove browser rendering,
external-platform delivery, sustained load, large-index performance or Windows
process cleanup. Broad model quality and remaining architectural work stay open.


## Validate proposal evidence before admission (2026-10-25)

A synthetic SQLite reproduction showed a valid candidate plus an invented quote
returning `saved` with one persisted fact, silently omitting the invalid candidate.
The existing normalized-source check now precedes business admission and raises
the existing `MemoryResponseError` for `evidence`. The batch fails before fact
writes while retaining the captured original source. Valid low-confidence and
empty outputs remain business outcomes. Production code is two lines shorter;
no semantic keyword rule, additional reviewer, wrapper or repair path was added.

The latest full-file results cover 82 related parser/scheduler caller files:
1,489 passes, plus syntax/Ruff for 18 modified Python files and repository
integrity. Initial test-collection errors were editing mistakes, corrected before
results were counted. Only corrected files were rerun after the first directed
checks; no manual whole-backend repeat was performed.

The prior commit's CI run 40186588600 actually finished with 10,841 passes,
33 failures, one skip and eight warnings; frontend passed and Docker was skipped.
All 33 failures were in seven tests still expecting raw observation sources in
display content. Updated assertions retain full source evidence, temporal and
subject constraints, version chains and actual compiled references. The original
failed run remains evidence, not a claimed green baseline. Tests for authorization
now supply the actual current quotation and still require denial; invented or
copied historical quotes explicitly fail the model protocol instead of looking
like no new information.

One new DeepSeek Flash call (2,720 tokens) saved and recalled two complete
synthetic identity/study facts through actual SQLite and reference compilation.
This checks writer success, not final-answer accuracy. Backend restart readiness
and login are 200; unauthenticated protected access is 401 with nosniff. Frontend
process and configuration are unchanged. Current-source whole-CI results, browser,
external-platform, sustained-load and remaining architecture work stay separate.
