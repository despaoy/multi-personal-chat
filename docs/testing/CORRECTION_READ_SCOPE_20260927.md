# r147 correction source scope

Shared memory statement parsing now recognizes one leading correction marker
and fully matched same-slot withdrawal of the prior workplace/residence.
The original complete evidence remains intact; arbitrary trailing negations,
other subjects, hypothetical/future/conditional statements are not discarded.

13 targeted cases: five positives initially failed, eight negatives passed;
all passed after repair. Broad regression: 2502 passed, 1 skipped, 1784
deselected, 4 warnings (141.83 seconds). Ruff and diff checks passed.

Isolated remote replay: three HTTP 200 turns, nine DeepSeek calls (three
policy, two answer, two writer, two selector), no local generation calls.
The old workplace was superseded by the full correction. The cold-history
question read asserted_state and returned the current workplace directly,
without an unnecessary external-verification disclaimer. Its answer was a
deterministic memory response, not an additional model call.

Artifacts: `.tmp/r147-wide.xml`, `.tmp/r147-manifest.json`,
`.tmp/r147-traces.jsonl`. Remote snapshot: mechanism-r147-source.
This run still used the explicit 8192-token comparison baseline. Production
was not modified; complete project acceptance remains unfinished.
