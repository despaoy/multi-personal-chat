# r158: real retrieval-count comparison and output-budget failure

## Changes and test conditions

Added `RecordedRetrieval` for isolated replay: original query, requested and
effective top_k, complete bundle, elapsed time and failure/cancellation state.
`--rag-top-k` permits an explicit 1–32 experiment; omission preserves the
production count. No production ranking or retrieval default was changed.

Added configurable `--answer-max-tokens`: cloud default 1024, local default
256; explicit 32–8192 accepted. This is output allowance, independent of the
64K input/output working window. Final generation budgeting already accounts
for this output allowance. Manifest records both settings. No prompt changes,
extra reviewers, model switching or answer-specific runtime exceptions.

34 focused tests passed; Ruff and diff checks passed. Fixture validation was
also added after the initial empty-turn fixture was rejected before model use.

## Actual failure that required the output change

The first valid k=3 replay failed on a multi-stage identity question:
two real answer calls both requested max_tokens=256, returned 256 tokens with
finish_reason=length, and led to HTTP 500. The alternate generation path retried
the same cloud model with the same insufficient budget. Input was only 5,421
provider tokens, far below 64K. This was not an input-window overflow.
Artifact: `.tmp/r158-k3-failed-traces.jsonl`.

The adapter correctly did not present a truncated response as complete, but
unchanged-budget retry is wasteful; that retry policy remains unresolved.

## Real comparison after output allowance repair

Snapshot: `/home/boot/lhm/multipersonal-runtime/evaluations/mechanism-r158-source`.
Output directories `r147-r158-output1024-k3` and `r147-r158-output1024-k8`.
Fixtures are identical, each with 2 cases / 4 ASGI HTTP turns and isolated
SQLite state; fixed evaluation identity and inline queue adapter, no LoRA.
Real configured index and embedding model, real deepseek-v4-pro for all
invoked LLM roles. Each run: 4 answers, 4 policies, 2 writers, 2 semantic reviews.
All 24 calls completed, all 8 HTTP turns returned 200, zero local fallbacks,
no structured personal claims. Reviewer/writer non-invocation on other turns
is native gating, not a substitute model. One sample per arm is exploratory,
not statistically controlled proof; preparatory assistant wording varied.

| Question | k=3 evidence chars / prompt tokens | k=8 evidence chars / prompt tokens |
| --- | --- | --- |
| Returned character identity and relationship stages | 6997 / 5430 | 16098 / 10432 |
| Ruby deception, evidence and consequences | 7427 / 5780 | 16246 / 10730 |

Every admitted evidence packet was present verbatim in actual answer input.
k=8 skipped one background block in each question under the 16,384-character
budget; all eight selected document IDs remained admitted. This verifies a
real >6000-character RAG path without synthetic padding, but not 64K saturation.
Identity answer completions used 436/538 tokens; Ruby answers used 221/253.
No length termination after the 1024-token repair.

## Content review and decision

Ruby answers correctly described the fabricated book/story, absence of forced
mutual romantic feelings, and enjoyment of school life as her own experience.
Checked against `gametext/纸上魔法使/2红宝石的天作之合.txt`, around lines 2500–2685.
Increasing k did not add the later forgiveness/return-home consequences.
The k=3 third candidate already came from White Pearl, not Ruby. k=8 added more
White Pearl plus Black Onyx, Rose Quartz and Lapis Lazuli candidates. The model
mostly ignored them in this sample, but this is real retrieval noise, not
evidence that adding documents inherently improves coverage.

Both identity answers distinguished original character from reconstructed
existence, checked against `8萤石的时空残影.txt`, around lines 615–807.
k=8 added a later confession, but labelled its section “妃死前对琉璃的告白”
while the section itself correctly described the post-death constructed
character. That heading/body contradiction fails temporal consistency.
Do not claim k=8 is better. Retain experimental switch only, production k=3
unchanged. No sample-specific guard or prompt repair was introduced.

Next root-cause work: explicit story scope currently gives ranking preference
rather than reliable coverage boundaries; near-duplicate passages consume
space; chapter-level temporal/viewpoint metadata can be too coarse. Scoping
must preserve legitimate cross-story comparison and retrospective evidence,
not blindly filter by filename. Larger models still make synthesis errors.

Artifacts: `.tmp/r158-k3-manifest.json`, `.tmp/r158-k3-traces.jsonl`,
`.tmp/r158-k8-manifest.json`, `.tmp/r158-k8-traces.jsonl`.
Production untouched. Full objective remains active and unachieved.
