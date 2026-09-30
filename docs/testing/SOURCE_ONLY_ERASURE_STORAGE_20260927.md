# r153 unlinked source erasure storage contract

An original utterance can be stored without any normalized claim. The existing
claim-target eraser cannot address that source directly. Added a shared atomic
erase_unlinked_plan with SQLite, PostgreSQL, sync facade and repository APIs.
Natural-language operation selection is NOT wired yet; the user-visible
source-only deletion gap is not claimed fixed by this storage work.

The caller supplies 1..200 explicit source IDs in an exact authenticated scope.
Source keys are derived, not accepted from the model. The transaction takes
the same owner lock as capture and claim linking, rejects an entire batch if
any source acquired a claim, removes source text/postings, and retains identity
revocation anchors to block retries. Unknown/already-revoked IDs count zero.
It does not advance the owner-wide fence or suppress unrelated historical
speech. Linked sources require the existing claim lifecycle eraser, not a
partial erase that leaves normalized claims alive.

Added source-to-claim reverse index in SQLite schema, ORM and migration 014.
Index plan checked locally. No production migration/deployment performed.

Seven initial tests failed due to missing capability. Final 81 related tests
passed (5.78 seconds, .tmp/r153-final.xml), including full-batch rollback,
isolation, idempotency, delayed capture/claim refusal, repository routing and
12 concurrent write/delete races. No extra model calls or prompt changes.

Remote SQLite/PostgreSQL parity passed the new source erasure suite plus
existing lifecycle/search/window contracts. For every concurrent race the
only allowed outcomes were (saved, linked-conflict) or (revoked, erased), with
matching final source/claim state. Artifact .tmp/r153-pg-parity.json. Disposable
cluster r110pg.PltzGa stopped successfully; snapshot mechanism-r153-source.
This round made no real model calls because no new dialogue path is wired.
Standalone changed modules pass Ruff; repository/database files retain existing
whole-file lint debt. git diff --check passed.

Next required integration: bounded complete source candidates alongside claim
targets, explicit source-target proposals, current-user deletion authorization,
commit-conflict handling, and receipts distinguishing source records from claim
operations. Then repeat pet-name source-only deletion and independent negative,
scope and mixed-task cases through the actual stronger-model HTTP chain.
Do not use lexical similarity alone to authorize deletion, or silently claim a
partial candidate search erased all occurrences. Global source-only selection,
linked/unlinked repetitions, owner-fence granularity and wider module acceptance
remain open. Goal active; production unchanged.
