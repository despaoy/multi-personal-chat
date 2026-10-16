# Owned dialogue generation tasks and confirmed cancellation

## Problem and behavior

Force reset previously returned success and set generation idle while the original model call was still running. An isolated live-coroutine probe reproduced that state directly. The old task could then continue writing shared progress after a new request was admitted. User cancellation only set a flag, so it could not interrupt research or a pending model call. The frontend aborted the main request immediately and treated that as confirmed cancellation, discarding its partial response.

Generation admission now owns one actual asynchronous child task. Cancellation interrupts that task. Its started runner returns only complete, previously validated batches with success false and cancelled true for explicit user cancellation; unrelated external cancellation still propagates. A child cancelled before business starts returns a known zero-result cancellation, distinguished using its explicit cancellation message. The owner has one cleanup block, including this pre-start case. Task identity is checked only at ownership cleanup and reset, so old completion cannot clear a successor's state.

Force reset cancels and waits for its captured task to terminate outside the state lock. It resets shared state only if that task still owns it, and confirms that the original task terminated rather than claiming any newer task is idle. No job framework, per-update ownership checks or new persistence was added.

The training page waits for the original POST's confirmed cancellation result and displays its complete partial batch list. Polling and reconnection use the complete result snapshot instead of appending consumable new-batch lists, including terminal snapshots. Small shared callbacks centralize timer cleanup and progress projection. Controller identity checks at request/poll/reset boundaries reject stale updates. Force reset abandons the old UI request only after server confirmation. Progress-query failures stop polling and explicitly report that backend state is unconfirmed instead of silently retrying indefinitely.

## Focused validation

- Baseline: reset reported success and idle while the synthetic model coroutine remained running. The probe then cancelled and joined it. Real model calls: zero.
- Six focused actual-route tests passed in 1.82 seconds. Event gates cover cancellation after one complete batch, reset waiting for a slow coroutine cleanup, refusal of concurrent admission, successor isolation from old request/reset cleanup, external cancellation both alone and overlapping user cancellation, real asynchronous search transport/client closure, and cancellation before business starts.
- Cancellation after the first of two batches returns one complete dialogue, success false, cancelled true and 50% progress, with no quality retry. Reset waits for the child to terminate before allowing replacement, and old cleanup cannot alter the replacement's state.
- Changed-page ESLint passed. The production Next.js build, including its TypeScript check, passed. The previous build was retained outside the repository before rebuilding. No unrelated backend regression suite was rerun.
- Syntax, incremental Python lint and whitespace checks passed. Test transport and model boundaries were explicitly synthetic; no real search or model calls were made.

## Limits

No authenticated browser interaction test was performed; frontend validation is static checking and a production build. Live DeepSeek validation remains pending valid credentials. These cancellation guarantees cover asynchronous operations such as the configured cloud provider; cancelling a coroutine is not claimed to kill an already-running synchronous GPU thread in a different provider. Structured presentation of partial results from HTTP failure responses remains separate work; this phase retains earlier accepted batches in the existing admin progress state and handles explicit cancellation results.
