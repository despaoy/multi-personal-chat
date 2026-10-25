# Backend test contracts

A passing case count is not a quality score. Keep a test when it protects a
specific public behavior, regression, data invariant, or security boundary.

## Choosing the right level

- Test a shared parser's input classes at its boundary. At callers, test field
  forwarding, aliases, modes, and error propagation rather than repeating the
  entire parser matrix through every caller.
- Test a list endpoint's filtered total using the real endpoint and storage.
  Running the same SQL in the test only proves the database works.
- Execute adapters and inspect forwarded arguments/results. Method names,
  signatures, and source substrings alone do not prove correct behavior.
- Keep source checks for genuine structural constraints (for example forbidden
  SQL placeholders), but do not present them as runtime correctness evidence.
- Retain distinct language boundaries, authorization, erasure, transactions,
  cancellation, and resource cleanup. Similar assertions are not proof that the
  inputs are interchangeable.
- A deliberate broken implementation should fail the relevant test. Small
  mutation experiments can expose weak assertions; a few hand-picked mutants
  are not a suite-wide mutation score.

## Running the suite

Run from `backend` with the supported dependency stack installed:

```sh
python -m pytest tests --collect-only -q
python -m pytest tests -q --tb=short --basetemp=/tmp/multipersonal-pytest
```

The full suite requires the real pinned encoder/index and isolated PostgreSQL
fixture databases prepared by `.github/workflows/ci.yml`. Without them, report
missing prerequisites rather than calling an incomplete run a full pass.
`fault_injection_test.py` is a separate executable against a running server,
excluded by `conftest.py`; it is not silently skipped unit-test coverage.

Frontend contracts run from the repository root:

```sh
node --test scripts/tests/*.test.mjs
```

Those tests execute real client/callback bodies with controlled dependencies;
they do not replace React/browser lifecycle testing. Recording PostgreSQL
sessions prove operation order and decisions, not concurrent database behavior.

## Review checklist

1. Name the failure the test would detect and which production entrypoint runs.
2. Check that assertions cover returned data and side effects, not only mocks.
3. For removed cases, identify the remaining contract and the equivalence
   assumption. Do not drop a failure merely to make CI green.
4. Verify both representative positive and negative paths. Distinct input
   values alone do not justify the full Cartesian product.
5. Record current commands, collected/executed counts, skips, and limitations.
   Recheck current code; historical reports are leads, not verification.
