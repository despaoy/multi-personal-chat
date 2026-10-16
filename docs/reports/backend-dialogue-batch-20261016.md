# Exact dialogue batches without output repair

## Problem and resulting behavior

The training dialogue generator repaired JSON wrappers, filtered messages, searched for new user/assistant pairs, truncated extra turns and dialogues, and retried low-quality batches. After a later batch failed repeatedly, any previously accepted dialogue was enough to return success and 100% progress. A complete isolated two-dialogue baseline produced one valid dialogue, made three additional synthetic model calls for malformed output, and returned success with total one and progress 100.

The route now follows its existing turn distribution using a fixed batch plan. A single batch parser requires a complete JSON text response with exactly the requested dialogues, each containing exactly the requested alternating user/assistant messages and nonblank string content. It rejects duplicate or unexpected fields rather than choosing an interpretation. The entire batch must pass before it is mapped to the persisted dialogue shape. Accepted message content is preserved verbatim. JSON repair, message filtering/re-pairing, truncation and quality retries were removed.

Invalid output immediately returns HTTP 502 with a safe reason, the failing batch and expected count/turns, and the already-completed/requested counts. Earlier fully validated batches remain available in generation progress, which is updated only from accepted batches. Invalid model text is not logged. Non-cancelled generation cannot return success with an incomplete count.

The prompt's format example is now valid complete JSON with the actual batch size and turn count. Previous examples contradicted one-dialogue requests and included ellipses for longer conversations. Dialogue counts must be positive integers and character descriptions must contain non-whitespace text, validated before research or generation admission. Existing turn weights, scenes, style and custom prompt functions remain.

## Focused validation

- Isolated baseline reproduced requested two / completed one / success true / progress 100 / four synthetic model calls. Real model and search calls: zero.
- Eleven new focused endpoint tests passed in 1.84 seconds with the actual route and batch parser.
- The successful 80-dialogue case independently checks the existing 16-batch plan, all one-to-eight-turn groups, complete JSON prompt examples, exact output count, preserved content including leading/trailing whitespace, and final state/result consistency.
- A bad second batch now fails after exactly two calls, retaining one complete dialogue and 50% progress. Seven malformed/extra/wrong-role/type cases fail the whole batch without repair, truncation or retry. Invalid request cases return 422 before research, model calls or state admission.
- Error responses and captured logs exclude the synthetic output marker. Syntax, incremental lint and whitespace checks passed. No unrelated regression suite was rerun.

## Limits and next phase

These are deterministic chain tests, not new live model-quality measurements. DeepSeek credentials were not retried. Correct structure does not guarantee factual or stylistic quality. Explicit user cancellation and force-reset ownership/frontend handling retain their previous behavior in this phase and are the next separate lifecycle change. Earlier complete batches are retained in the existing admin progress state; this phase does not introduce a new job store or partial-result UI.
