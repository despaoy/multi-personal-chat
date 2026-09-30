# r151 model history honors explicit source revocation

Final real replay after canonical-scope repair: three HTTP 200 turns, eight
DeepSeek calls (3 policy, 2 answer, 3 writer), no local generation calls.
Artifacts .tmp/r151-final-manifest.json and .tmp/r151-final-traces.jsonl.
Deleted claim remains absent and the subsequent storage-status answer is correct.
Prepared history counts are 0 / 0 / 2: the deleted turn is removed while the
deletion-and-seasons turn remains. The deleted city is absent from both final
policy and writer inputs. Deletion itself necessarily still uses pre-operation
evidence to identify the target; this is distinct from post-operation reuse.

Warm-history baseline returned correct saved-memory status, but traces still
contained the deleted source and its assistant reply. Correct output did not
prove correct input isolation.

Database model-history readers now batch-check exact source keys and exclude
both sides of explicitly revoked stored turns before whole-turn budgeting.
Chat storage is not deleted; unrelated turns remain. This is an indexed identity
lookup, not a text match, fingerprint, model reviewer or owner-wide history wipe.

The first local fix passed tests yet FAILED its real warm-history replay:
messages retain the front-end session ID, while private memory scope uses sender
ID. The fix now reuses build_user_scope and checks canonical plus legacy exact
identities. A separate source-scope case was added, rather than changing the
test fixture to match the faulty implementation.

Final local result: 152 passed, 11 skipped (7.46 seconds), .tmp/r151-scope.xml.
Skipped cases include locally unavailable PostgreSQL dependencies. Initial
remote SQLite/PostgreSQL parity passed 15 lifecycle contracts including history
projection; cluster r110pg.0MQPBX stopped successfully. This parity preceded the
canonical private-scope adjustment. Artifact .tmp/r151-pg-parity.json.
Ruff for changed standalone modules and git diff --check passed.

Baseline and failed first real replay are preserved in
.tmp/r151-baseline-traces.jsonl and .tmp/r151-traces.jsonl. The latter contains
six HTTP turns and 20 DeepSeek calls (6 policy, 5 answer, 6 writer, 1 semantic,
2 selection). Its no-delete case retained the correct major, but its deletion
input isolation must not be reported as passing.

Remaining scope: client-supplied history without source identities, independent
later paraphrases/repetitions, source-only deletion targets, excessive owner
fences, concurrent already-prepared generation, and complete RAG/dynamic-context
acceptance. Legacy history with no source identity cannot be retrospectively
authorized by guessing from its text. Production unchanged; goal active.

An independent model-quality issue was observed in the seasons answer: it
described 23.5 degrees relative to the orbital plane rather than its normal.
Do not equate successful task splitting with fully accurate scientific content,
or assume a stronger model removes all model-side errors.
