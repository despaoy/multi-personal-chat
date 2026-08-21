# Scripts index

All active scripts live in this directory. Historical tools are under
`archive/` (indexed in `archive/INDEX.json`) and must not be used for formal
V4 experiments.

## Current V4 data workflow

| Script | Purpose |
|---|---|
| `extract_character_dialogues.py` | Extract attributable Kisaki lines from `gametext/` |
| `audit_kisaki_source_alignment.py` | Audit source-line alignment and RAG lineage |
| `build_kisaki_v4_canonical_draft.py` | Rebuild V4 canonical train/validation draft |
| `freeze_kisaki_v4_dataset.py` | Freeze reviewed train/validation |
| `finalize_kisaki_v4_dataset.py` | Advance canonical state after Gold approval |
| `reaudit_kisaki_gold_v3.py` | Re-audit approved Gold v3 against the current frozen train/validation splits |
| `build_kisaki_gold_v21.py` / `build_kisaki_gold_v3.py` | Build Gold sets and contamination audits |
| `build_kisaki_r1v4_configs.py` | Generate E1-E5 single-variable configs |
| `validate_kisaki_v4_training_gate.py` | Formal training gate |
| `run_kisaki_experiment.py` | Run one canonical experiment |

## V4.1 augmentation and review

| Script | Purpose |
|---|---|
| `generate_kisaki_v41_augmentation.py` | Generate candidate sessions with LLM |
| `run_kisaki_v41_user_simulation.py` | Run user-simulation rounds |
| `promote_kisaki_v41_round06.py` | Promote approved sessions into canonical train |
| `apply_kisaki_llm_persona_review.py` / `apply_kisaki_llm_full_dialogue_review.py` | Apply LLM review revisions |
| `build_kisaki_v4_chat_smoke.py` / `run_kisaki_v4_overfit_test.py` | Chat smoke and overfit smoke |

## RAG / routing / operations

`build_character_rag_eval.py`, `build_kisaki_rag_v2.py`, `freeze_kisaki_rag_v2.py`,
`import_kisaki_rag_evidence.py`, `enrich_kisaki_rag_evidence_lineage.py`,
`build_system_routing_eval.py`, `evaluate_system_routing.py`,
`restore_sqlite_backup.py`, `download_model.py`,
`local-verify.ps1`, `start-local-backend.ps1`.

## Lab / remote scripts

Lab scripts require environment variables rather than hard-coded personal paths.
Legacy `QQCHAT_*` variable names remain supported as fallback aliases during migration:

- `MULTIPERSONAL_LAB_ROOT`: lab root, e.g. `/lab` or `/data`
- `MULTIPERSONAL_REMOTE_ROOT`: remote repository root for SSH scripts
- `MULTIPERSONAL_REMOTE_PYTHON`: remote Python interpreter
- `MULTIPERSONAL_REMOTE_MODEL`: remote base model path
- `LAB_HOST` / `LAB_USER` / `LAB_SSH_KEY` or `LAB_PASS`: SSH credentials

Examples: `lab-run-kisaki-r2.sh`, `remote_kisaki_r1v4.py`,
`remote_kisaki_v4_overfit.py`, `merge_kisaki_adapter_for_eval.py`.

## Historical generation pipeline (kept for tests/provenance)

`build_few_shot_pool.py`, `build_v3_negative_pool.py`,
`build_kisaki_v4_quota_plan.py`, `generate_kisaki_llm_dialogues_v3.py`,
`generate_kisaki_llm_v4.py`, `judge_kisaki_llm_v4.py`,
`hard_gate_kisaki_v4.py`, `regen_kisaki_llm_pipeline.py`,
`kisaki_v4_llm_client.py`. These read archived V3 pipeline inputs and are kept
only for reproducing the historical generation phase or running its contract
tests.
