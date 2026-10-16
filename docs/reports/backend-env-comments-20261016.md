# Align configuration guidance with current failure behavior

Updated only `.env.example` comments. The sample no longer promises rule fallback after semantic-review failures, silent memory omission on review failures, or automatic bigram degradation when required semantic embeddings fail. It describes the current full-turn budgeting, candidate/selection failure behavior, independent memory-writer provider configuration and explicit deletion timing. The OpenAI-compatible provider name is corrected to `openai_compat`. The preceding all-environment explicit encryption-key guidance is preserved.

The corresponding implementation was checked in the semantic-state refinement, contextual memory selection, decision policy, MemoryLlmConfig and hierarchical-memory semantic retrieval paths. No classifier, trigger rule, model choice or feature switch was added or changed by this phase.

Server verification compared all 181 active or commented example assignment lines byte-for-byte, all executable non-comment lines, and dotenv parsing with interpolation disabled. All three comparisons were identical. Whitespace checking passed, and the real backend configuration hash was unchanged. No functional test suite or model call was run for this comment-only change. No service was restarted; backend readiness and the frontend login page remained HTTP 200. Historical stage reports remain unchanged.
