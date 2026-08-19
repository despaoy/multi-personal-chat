# 月社妃 V4 人工审核指南

Game Train 上下文质量复审已经完成。正式训练仍须先通过当前环境训练门禁；实际状态以 `review_manifest.json` 与 V4 canonical manifest 为准。

## 当前审核顺序

1. `04_CONSTRUCTED_TRAIN`、`05_VALIDATION`、`06_GOLD_V21` 和 `07_GOLD_V3` 已完成审核。
2. Game Train 上下文质量复审已完成，最终保留 522 条。
3. 重新运行训练门禁，再使用已生成的 E1-E5 正式配置。

`01_PROFILE_PROMPT` 已由项目负责人确认。`02_SOURCE_COVERAGE` 和 `08_EXCLUSIONS` 的可复现结果见 `02_SOURCE_COVERAGE/SOURCE_ALIGNMENT_AUDIT.json`。RAG evidence 已从 train 与 validation 候选中隔离。

## 回复格式

- `全部通过`
- `样本 ID：修改建议`
- `样本 ID：排除，原因`
- `样本 ID：需要更多原作上下文`

Gold 修改不得回流训练集。原作 assistant 台词不可改写；无法可靠配对的样本应排除，不需要维持固定数量。

当前计数：`{"source_lines": 1598, "game_train": 522, "constructed_train": 150, "reviewed_multiturn_augmentation": 276, "frozen_train": 948, "frozen_validation": 70, "rag_withheld_sft_records": 34, "gold_v21": 150, "gold_v3": 150, "game_train_total_exclusions": 130, "constructed_exclusions": 9, "validation_exclusions": 7}`
