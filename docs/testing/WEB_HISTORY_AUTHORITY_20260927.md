# r152 server-owned stateful web history

Root: request.history bypassed the database history projection at context
preparation and answer generation. A stale browser transcript could restore
source text removed by r151, including in internal reviewers and writer input.

Authenticated web character requests now clear client history at the same
boundary that assigns authenticated user identity. Completed web turns already
persist server-side; all downstream stages therefore use the scoped server
projection. No text matching, extra model calls or additional validation stage.
Stateless management and external bot history contracts are not changed.
This means arbitrary imported transcripts are not authoritative history for
stateful web character chat; such import needs a separate explicit workflow.

Two regression cases first failed (explicit character and web-character
adapter); a stateless compatibility case passed. Final combined regression:
130 passed, 11 skipped, 3 warnings, 5.95 seconds (.tmp/r152-suite.xml).
Ruff and diff checks passed. No production change.

The isolated replay supports --echo-client-history to resend the whole browser
transcript, including erased turns. It records the submitted history count;
the existing prepared/model traces show what the backend actually consumed.
This option is opt-in and does not alter the default comparison baseline.

Real server replay completed six HTTP 200 turns and 20 DeepSeek calls:
6 policy, 5 answer, 6 writer, 1 semantic review, 2 memory selection; zero local
generation calls. For deletion, submitted history counts 0/2/4 became prepared
counts 0/0/2. Deleted city was absent from post-operation history and final
policy/writer inputs; deleted claim stayed absent. The no-delete control kept
history 0/2/4 and its active major; final answer correctly recalled geology.
Artifacts .tmp/r152-manifest.json and .tmp/r152-traces.jsonl.
Snapshot mechanism-r152-source, output r147-client-history, production unchanged.

Still open: bot paths without authoritative source IDs, independent later
paraphrases, source-only targeted deletion, excessive owner fences, pending
writer acknowledgments, and outstanding RAG/dynamic-context acceptance.
