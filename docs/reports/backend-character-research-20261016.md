# Character research failure and async lifecycle

## Problem and behavior

The training dialogue endpoint treated failed character searches as an empty result, then invoked the model. An isolated baseline with both search sources returning HTTP 503 reached the model boundary once. The old configuration module owned an unclosed synchronous HTTP client and a thread executor whose shutdown could outlast its stated deadline. It selected the longer of two sources, truncated snippets, and described unverified search summaries as authoritative character facts. The parser dependency was also absent from the server environment and requirements.

Search now belongs to one asynchronous service, with a scoped HTTP client, concurrent independent sources, request timeouts and an overall six-second deadline. Both sources must return valid results; a valid empty response remains a normal business result. HTTP failures, unknown/challenge HTML, invalid JSON and invalid response structures raise a safe typed error. The route reports the source and reason as 502 or 504 before calling the model, without disclosing query text or upstream response bodies. Both sources' complete supplied snippets remain in deterministic order and are explicitly unverified reference material rather than instructions or official facts.

The endpoint awaits the model manager's existing async API. A single finally block releases generation admission after success, errors or actual coroutine cancellation. The import-time synchronous client and search executor were removed. BeautifulSoup is declared in requirements; the server installed beautifulsoup4 4.15.0 and soupsieve 3.0.3. No model, credential, deployment configuration or persistent user data changed.

## Focused validation

- Server baseline: both sources failed with HTTP 503, but the previous route still reached its synthetic model boundary once. Real model calls: zero.
- Eight new endpoint tests passed in 1.80 seconds, using the real FastAPI router, parsers, HTTP client lifecycle and output mapping with explicit synthetic transport/model boundaries.
- Covered complete long excerpts from both sources, late negation/conditions, untrusted-reference prompt framing, native async generation, valid empty search results, HTTP failure, unrecognized HTML, malformed JSON, timeout, cancellation during both source requests, cancellation during the model await and admission after cancellation.
- Failure tests assert that model calls are prevented, clients close and raw upstream diagnostics are absent from responses and captured logs. Cancellation tests assert that pending source/model coroutines terminate and the busy state is released.
- Syntax, incremental lint and whitespace checks passed. No unrelated regression suite was rerun.

## Limits and next work

This phase used isolated HTTP fixtures rather than live search services, and made no real DeepSeek calls. The previously observed DeepSeek credential failure remains unresolved. Prompt framing preserves provenance but cannot guarantee model behavior. Browser disconnection alone is not claimed to cancel a server coroutine. Legacy force-reset task ownership and dialogue batch validation remain separate work: malformed/short model output, repair/retry behavior and partial success must still be simplified and made explicit.
