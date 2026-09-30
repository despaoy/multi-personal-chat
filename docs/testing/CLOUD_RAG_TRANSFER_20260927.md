# r157: actual cloud dialogue transfer after budget plumbing

Remote snapshot: `evaluations/mechanism-r156-source`. Output:
`evaluations/r147-cloud-rag-transfer-r157` (legacy replay directory allowlist).
Fixture: `backend/evaluation/fixtures/cloud_rag_transfer_20260927.json`.
Local artifacts: `.tmp/r157-manifest.json`, `.tmp/r157-traces.jsonl`.
Production unchanged; goal remains active.

## What actually ran

10 ASGI HTTP turns, all 200, with real knowledge index/embedding retrieval,
isolated SQLite persistence and real deepseek-v4-pro calls. No LoRA. Fixed
evaluation login and inline admission are test adapters, not production auth
or queue acceptance. Browser history was echoed; server history remained
authoritative. All 33 provider calls completed: 10 answers, 10 writers,
10 contextual policies, 2 memory selectors, 1 semantic review. Zero unexpected
local calls; no guard retries. Working window 65,536 tokens, RAG evidence budget
16,384 characters. Answer input actually ranged 1,269–4,377 provider tokens.

## Reviewed outcomes and attribution

- User's sister 若杉 stored as one user fact and recalled separately from the
  character's family; follow-up kept ownership separate. Writer made no
  additional claims from character lore or assistant answers.
- Hypothetical watch-shop scenario stayed hypothetical in both later answers;
  no user occupation or canonical shop claim persisted. Original utterances
  remained in source storage; absence of claims is not absence of storage.
- Relationship explanation quoted supplied evidence. Broad character overview
  also used stable profile relationships (彼方/理央/夜子), not only retrieved
  passages. Thus this is not evidence-only factual QA or independent validation
  of every statement in the profile.
- After lore discussion, explicit listening-only request led to stay_present,
  subsequent frustration to acknowledge_emotion, goodbye to graceful_close.
  Replies did not add advice, questions or unrelated lore.
- Every admitted evidence packet was present verbatim in the actual answer
  provider messages (0 missing across all turns). Citations/packet structure
  were inspected alongside replies, not scored solely via answer keywords.

## What this does not establish / next work

Maximum actual RAG evidence was only 4,234 characters. This run does NOT validate
the newly available >6,000-character path or prove quality gains from 64K.
HTTP currently requests a fixed top_k=3, a separate candidate coverage limit.
Long-evidence capacity and natural broad-question coverage need independent
tests, not padded fixtures presented as real conversational quality.

The listening request containing a negated request for methods still triggered
generic RAG and got no documents. Existing conservative local-task grammar plus
keyword routing can waste retrieval work despite correct final behavior. Do
not fix this by exempting any turn containing one negative instruction: mixed
external questions must continue to retrieve.

One retrieved friendship packet inherits chapter-level `viewpoint` metadata
labelled 琉璃第一人称, although the local excerpt reads as 妃's narration. This
requires source/metadata-granularity audit; it is not proof of a new model error.
No prompt changes or per-fixture runtime rules were made in this round.

Provider latency in this small run: answer median (lower order statistic) 1.50s,
maximum 6.11s; policy median 1.49s/max 3.01s; writer median 0.91s/max 2.18s.
These are per-call latency observations, not end-to-end p95 or comparative
performance claims.

## Full backend regression and failure triage

`pytest backend/tests -q`: **4,312 passed, 8 failed, 29 skipped**, 308.33s.
Authoritative result `.tmp/r157-full.xml`. This is not an all-green regression.

One failure was an inconsistent evidence contract in
`test_mechanism_integrity`: the old test expected an omitted-subject workplace
claim to discard its owner-bearing preceding clause. Current extractor and
writer lifecycle tests intentionally retain the original complete sentence
for inherited ownership so the evidence can be independently reparsed. Changed
the narrow-evidence case to explicit self ownership and added three separate
inherited-owner cases. No production parser behavior was weakened or changed.
Focused rerun with lifecycle, possessive coordination and rule correctness:
**78 passed**; Ruff passed. Full suite has not been rerun after this test change.

Seven other failures remain in training modules outside this no-LoRA chain:
four PPO tests fail with policy/reference initial-weight inequality; two DPO
tests cannot import FSDPModule from the installed torch; one ORPO test reports
that installed TRL does not expose ORPO. These are unresolved, not silently
skipped or classified as passing. No training implementation or dependencies
were changed. Passing inference tests does not certify training functionality.
