# API context working budget

DeepSeek official Models & Pricing page was fetched on 2026-09-27:
https://api-docs.deepseek.com/quick_start/pricing
It lists deepseek-v4-pro (DeepSeek-V4-Pro-0813) with 1M context.
This is provider capacity, not a reason to fill every request with 1M tokens.

The isolated web replay now defaults to a 65,536-token working window for both
answer and memory writer. `--cloud-context-tokens` permits 8,192–1,000,000.
Local vLLM configuration is unchanged. Provider selection alone previously
replaced the completion function but left the local 8K request budget in place.

At the default cloud budget, the context service reads up to 128 messages /
65,536 characters and packs complete source records within 16,384 characters.
The final answer evidence cap is also 16,384 characters. These are loading
budgets, not exact token measurements; full request token estimation remains.
All overrides are instance/request-local, not process-global monkeypatches.

Validation: 57 focused tests passed, covering cloud adapter, writer budget,
context service and generation packet budget. A >12K Chinese-character input
is preserved at the answer/writer builders with 64K and rejected by the 8K
answer budget. A >9K-character source is preserved with the enlarged budget.
These are local mechanism tests, NOT a fresh real API or HTTP acceptance run.

Still open: upstream API admission limits, internal selector/policy/semantic
history budgets, upstream RAG packing and sparse-query limits. Enlarging the
answer cap cannot restore evidence those stages have already dropped.
No production deployment or server configuration change was made.

## r155: database history loading no longer silently clamps to 50 turns

The context service's cloud request for 128 stored turns was silently reduced
to 50 in both database implementations. Both now use a shared bounded loader
limit (1–500 stored question/answer turns). Default local loading remains 8;
cloud loading remains 128. Character budgets, whole-turn assembly, source
revocation filtering and final model-token budgets remain in force.

Validation: 24 history tests passed, 14 PostgreSQL tests skipped because this
local interpreter lacks asyncpg; 60 additional context, generation, Web history
and writer tests passed. Ruff on the shared helper/test and git diff --check
passed. New SQLite cases cover private/group scopes and requested limits of
8/128/1000. PostgreSQL code is updated but not newly runtime-verified this round.
This is a local mechanism regression, not a fresh live API run or deployment.
The previously verified 64K cloud working window is unchanged. Remaining
independent writer/RAG/admission caps still require purpose-specific review.

## r156: upstream RAG packing uses an instance-local evidence budget

`RoutedMultiScaleService` and its runtime now accept a positive integer
`context_max_chars` (default 6000, preserving local serving). Cloud Web replay
injects a separate runtime with `cloud_context_tokens // 4`, matching final
answer evidence allocation: 16,384 characters for the 65,536-token window.
The manifest records this budget. No global runtime configuration is mutated;
the evaluation getter override is scoped to its existing isolated replay.
Complete evidence packets, attached citations, background dependencies and
knowledge visibility gates are preserved. No extra LLM call or prompt edit.

Local focused regression: 132 passed, 3 skipped. New tests exercise >6000-char
complete evidence, retained late qualifications, independent local/cloud
instances, runtime forwarding and invalid configuration. Ruff passed.
First remote run: 168 passed, 5 failed because embedding model location was
not configured for the isolated snapshot. This is not a retrieval quality
result. Rerun uses the existing model directory explicitly, without downloading
or changing production configuration. Configured remote rerun: **173 passed**,
4 dependency warnings, 37.60 seconds; report
`evaluations/r156-focused-configured.xml`, snapshot `mechanism-r156-source`.
This includes the previously skipped PostgreSQL adapter tests, whose database
session is mocked: it is not a live PostgreSQL parity check. SQLite storage and
index loading run for real, while retrieval tests use deterministic embedding
doubles. No fresh real API acceptance yet. Production remains unchanged.

## r148: internal budgets and real capacity replay

Selector, contextual policy and semantic estimator now accept an immutable,
instance-local ReviewContextBudget. Cloud injection passes the 64K working
window to all three. Their serialized system/payload input, output allowance
and safety margin are counted together. Legacy local defaults remain unchanged.
Semantic history now uses complete user-led turns rather than a raw six-message
tail slice. No prompt changes or additional review calls were introduced.

189 focused tests passed (7.71 seconds); Ruff passed. This is not a fresh full
project regression. Tests include >12K-character query preservation, >10K
history preservation, 8K overflow rejection, complete-turn boundaries, local
configuration isolation and a single-call policy request.

Remote snapshot `mechanism-r148-source`, output directory `r147-cloud64k`
(the older prefix is required by the replay allowlist). Four ASGI HTTP turns
all returned 200. Twelve real deepseek-v4-pro calls: 4 answer, 4 contextual
policy, 4 writer; zero unexpected local calls. All calls completed.

The final question consumed 12,110 actual prompt tokens for answer, 11,597 for
policy, and 8,315 for writer. All three source messages (6,118 / 5,893 / 5,668
characters) were verified verbatim in both final answer and policy inputs.
The comparison correctly distinguished conflict preservation, revision history,
and missing-source annotation. No personal claims were stored in any turn.

This repeated-paragraph fixture is a capacity probe, not a natural dialogue
quality benchmark. Semantic review was not needed and selector had no candidates;
neither has new real-call evidence in this run. Writer retains a four-message
recent-turn view, so its input does not contain all three archive messages.
RAG was enabled but found no documents; this does not verify long RAG evidence.
Upstream admission, source sparse-query limits, source-only erasure and broader
module acceptance remain open. Artifacts: `.tmp/r148-manifest.json` and
`.tmp/r148-traces.jsonl`. Production remains unchanged; goal active.
