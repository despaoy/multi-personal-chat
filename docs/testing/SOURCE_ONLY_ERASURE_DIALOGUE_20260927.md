# r154 source-only erasure reaches real dialogue execution

The existing writer call now receives complete source candidates for explicit
deletion requests. Sources come from scope-local sparse search plus recent
records; known active-claim primary sources use the normal claim eraser instead.
The bounded set is never described as exhaustive. Complete source packets fit
within half the configured writer budget; omitted packets are counted, not
prefix-truncated. Final full-request budgeting still applies.

The writer output contract is extended with optional erase_source_ids. This is
an executable target schema extension, not a stylistic prompt workaround.
Only current-user authorized deletion and exact supplied IDs can reach the
transactional unlinked-source eraser. Search similarity does not authorize a
mutation. A concurrent claim yields conflict; original source retrieval failure
does not disable an otherwise valid claim erasure. No extra reviewer/model call.
Confirmed source counts propagate to the operation receipt and rendered result.

151 related tests passed (3 warnings, 5.25 seconds, .tmp/r154-final.xml).
Tests cover candidate completeness/budget, unknown targets, no authorization,
one-call scheduling, actual source deletion preserving unrelated speech,
receipt accuracy and source-read failure with a valid claim operation.
Standalone changed modules passed Ruff and diff checks.

First real replay (warm pet + negated major, six turns) did NOT exercise the
new source-only branch: this model run saved a normalized pet claim. Preserve
that result as normal deletion/negative regression, not source-only evidence.
Artifacts .tmp/r154-claims-manifest.json and .tmp/r154-claims-traces.jsonl.

Independent transfer replay used a hypothetical future bookstore name followed
by a separate museum preference and deletion of the bookstore idea. Five HTTP
200 turns, 16 DeepSeek calls (5 policy, 4 answer, 4 writer, 3 selector), zero
local generation calls. The hypothetical source had no claim. The actual writer
returned memories=[] and exactly one erase_source_id. Database receipt recorded
source_erased=1, persisted=1, status=erased. The original bookstore source was
removed, the museum source and claim survived, and neither subsequent history
nor provider inputs contained the deleted name. Museum recall answered correctly.
Client stale history was echoed throughout. No whole-owner source fence advanced.
Artifacts .tmp/r154-manifest.json and .tmp/r154-traces.jsonl; snapshot
mechanism-r154-source, output r147-source-only-idea. Production untouched.

The deterministic receipt was subsequently shortened locally to avoid implying
that source-only execution necessarily deleted a normalized claim. Its message
still explicitly states the bounded match and retained chat archive; this wording
change did not require a fresh provider call and is covered by focused tests.

Remaining: exhaustive historical repeats, semantically distant old sources
outside candidate recall, sources mixing requested-erasure and retained facts,
inactive linked sources/conflict resolution, and original claim eraser's broad
owner fence. This successful transfer is not proof of full deletion semantics,
nor of global memory/RAG/dynamic-context acceptance. Goal remains active.
