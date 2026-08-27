# Scripts index

本目录只保存当前可执行的研究、训练、评测、远程运行与仓库验证入口。FastAPI 服务维护工具见 [`../backend/scripts/README.md`](../backend/scripts/README.md)，历史工具见 [`archive/README.md`](archive/README.md)。正式实验不得调用归档脚本。

## 仓库与本地验证

- `check_repository_integrity.py`：核对冻结数据、API 挂载、前端导航、脚本索引、README 链接和归档完整性。
- `local-verify.ps1`：Windows 完整验证流水线。
- `start-local-backend.ps1`：以 mock 推理模式启动本地后端。
- `restore_sqlite_backup.py`：在路径和目标检查后恢复 SQLite 备份。
- `download_model.py`、`validate_lora_training.py`：模型下载和 LoRA 训练前只读检查。

## V4 canonical 数据治理

- 提取与来源：`extract_character_dialogues.py`、`audit_kisaki_source_alignment.py`。
- 构建与冻结：`build_kisaki_v4_canonical_draft.py`、`freeze_kisaki_v4_dataset.py`、`finalize_kisaki_v4_dataset.py`。
- 清理与契约：`build_kisaki_v4_cleanup_candidate.py`、`promote_kisaki_v4_cleanup_candidate.py`、`align_kisaki_v4_prompt_policy.py`、`apply_kisaki_v4_speaker_contract.py`、`apply_kisaki_v4_text_normalizations.py`。
- 审核材料：`build_kisaki_train_review.py`、`apply_kisaki_llm_persona_review.py`、`apply_kisaki_llm_full_dialogue_review.py`。
- Gold：`build_kisaki_gold_v21.py`、`build_kisaki_gold_v3.py`、`reaudit_kisaki_gold_v3.py`。
- 训练门禁与配置：`validate_kisaki_v4_training_gate.py`、`build_kisaki_r1v4_configs.py`。

## 训练、checkpoint 与生成审核

- 正式入口：`run_kisaki_experiment.py`、`aggregate_kisaki_repetitions.py`、`prepare_kisaki_dpo_v3.py`、`merge_kisaki_adapter_for_eval.py`。
- checkpoint：`build_kisaki_checkpoint_devset.py`、`build_kisaki_r1v4_blind_review.py`、`finalize_kisaki_r1v4_blind_review.py`。
- 过拟合链路：`build_kisaki_v4_overfit_test.py`、`run_kisaki_v4_overfit_test.py`、`generate_kisaki_v4_overfit_results.py`、`render_kisaki_v4_overfit_review.py`。
- chat smoke：`build_kisaki_v4_chat_smoke.py`、`generate_kisaki_v4_chat_smoke.py`、`render_kisaki_v4_chat_smoke_review.py`。

## RAG 与系统路由

- `build_character_rag_eval.py`、`build_kisaki_rag_v2.py`、`freeze_kisaki_rag_v2.py`。
- `import_kisaki_rag_evidence.py`、`enrich_kisaki_rag_evidence_lineage.py`。
- `build_system_routing_eval.py`、`evaluate_system_routing.py`。

## V4.1 与 V5 候选工作区

- V4.1：`generate_kisaki_v41_augmentation.py`、`run_kisaki_v41_user_simulation.py`、`promote_kisaki_v41_round06.py`。
- V5 盘点与模拟审核：`build_kisaki_v5_asset_inventory.py`、`build_kisaki_v5_candidate.py`、`collect_kisaki_v5_simulation_decisions.py`。
- V5 构造数据审核：`build_kisaki_v5_constructed_review.py`、`build_kisaki_v5_constructed_rewrites.py`。

V5 仍是候选工作区，不覆盖已冻结的 V4，也不是生产训练入口。

## 可复现但非当前主线的生成工具

以下脚本仍由契约测试覆盖，用于复现 V3/V4 生成阶段，不得据此覆盖 canonical 数据：

- `build_few_shot_pool.py`、`build_v3_negative_pool.py`、`build_kisaki_v4_quota_plan.py`。
- `generate_kisaki_llm_dialogues_v3.py`、`generate_kisaki_llm_v4.py`、`judge_kisaki_llm_v4.py`、`hard_gate_kisaki_v4.py`。
- `regen_kisaki_llm_pipeline.py`、`kisaki_v4_llm_client.py`、`review_preference_candidates.py`。

## 实验室与远程运行

- 环境：`setup_lab_env.sh`、`remote_config.py`。
- vLLM：`lab-start-vllm.sh`、`lab-start-vllm-daemon.sh`。
- 实验：`lab-run-kisaki-r2.sh`、`lab-run-kisaki-r3.sh`、`lab-run-kisaki-r4-dpo.sh`、`lab-select-kisaki-e1-checkpoint.sh`。
- 远程：`remote_kisaki_r1v4.py`、`remote_kisaki_v4_overfit.py`。

实验室脚本优先使用 `MULTIPERSONAL_LAB_ROOT`、`MULTIPERSONAL_LAB_PYTHON`、`MULTIPERSONAL_REMOTE_ROOT`、`MULTIPERSONAL_REMOTE_PYTHON` 和 `MULTIPERSONAL_REMOTE_MODEL`。旧 `QQCHAT_*` 名称仅作迁移兼容。

## 维护规则

1. 新活动脚本必须加入本索引并接入相应测试或验证链。
2. 一次性、旧数据或个人机器专用工具移入 `archive/`，同时登记来源和哈希。
3. 任何修改 canonical、Gold 或审核决定的脚本必须默认 fail closed，并保留 provenance。
4. mock、候选和历史输出不得表述为正式实验结论。
