# r150 revoked source identity cannot recreate claims

Audit initially targeted broad erasure fences and warm history. It found an
additional concrete lifecycle defect: append checked receipt age but did not
check a source identity's revoked state when linking a newly inserted claim.
A missing or fresh timestamp could therefore recreate a normalized claim from
an erased source. Legacy UPSERT did not participate in source linking/locking.

Repair: shared link_plan checks source state while the caller holds the owner
lock, raising ClaimSourceRevokedError on revoked identities. SQLite rolls back
the entire insert/update transaction; PostgreSQL session rollback does the same.
Legacy UPSERT now acquires that lock and links the source before commit. Exact
identity is checked; no content fingerprints, model review or prompt changes.

Initial matrix failed in all four executions (one legacy timestamp combination
was redundant and removed). Final targeted tests cover missing/new receipts,
legacy writes, clean rollback, subsequent new speech, and preserving the old
valid value when a legacy overwrite is rejected.

115 combined lifecycle/source/legacy/conflict tests passed (9.01 seconds,
.tmp/r150-suite.xml). A subsequently added overwrite-rollback test passed in a
four-test focused rerun (0.36 seconds); counts overlap and must not be summed.
Changed standalone modules passed Ruff; database modules retain prior lint debt.
git diff --check passed.

Remote disposable SQLite/PostgreSQL parity passed all 14 lifecycle contracts,
including revoked identity retries and concurrent capture/erase. Long query,
scope and source-window checks also passed. PostgreSQL cluster
r110pg.oWfOmK was stopped successfully. Artifact .tmp/r150-pg-parity.json.

Real 64K DeepSeek regression: six HTTP 200 turns, 19 provider calls (6 policy,
5 answer, 6 writer, 2 selection); no local calls. Residence deletion returned
an actual erased receipt, removed both claim and original source, and the cold
question did not recover the location. A negative instruction to NOT delete a
major preserved the active claim and subsequent cold question answered it.
The deletion acknowledgment itself did not invoke the answer model. This replay
checks ordinary workflow regression; the retry defect is proven by database
tests, not by asking a model to manufacture a retry.
Artifacts .tmp/r150-manifest.json, .tmp/r150-traces.jsonl. Remote snapshot
mechanism-r150-source; isolated output r147-revoked-retry.

Still open: excessive owner-wide fencing on targeted deletion, source-only
target selection, warm-history reintroduction, acknowledgment before writer
commit, and outstanding RAG/dynamic-context acceptance. This change does not
resolve those broader problems. Production unchanged; goal remains active.
