# 月社妃 V4 人工审核指南

正式训练保持关闭，直到 Game Train 上下文质量复审完成且训练门禁重新通过。实际状态以 `review_manifest.json` 与 V4 canonical manifest 为准。

## 当前审核顺序

1. `04_CONSTRUCTED_TRAIN`、`05_VALIDATION`、`06_GOLD_V21` 和 `07_GOLD_V3` 已完成审核。
2. 当前复审 `10_FINAL_REVIEW/02_GAME_CONTEXT` 的 Game Train 上下文质量。
3. 复审关闭后重新运行训练门禁，再使用已生成的 E1-E5 正式配置。

`01_PROFILE_PROMPT` 已由项目负责人确认。`02_SOURCE_COVERAGE` 和 `08_EXCLUSIONS` 的可复现结果见 `02_SOURCE_COVERAGE/SOURCE_ALIGNMENT_AUDIT.json`。RAG evidence 已从 train 与 validation 候选中隔离。

## 回复格式

- `全部通过`
- `样本 ID：修改建议`
- `样本 ID：排除，原因`
- `样本 ID：需要更多原作上下文`

Gold 修改不得回流训练集。原作 assistant 台词不可改写；无法可靠配对的样本应排除，不需要维持固定数量。

当前计数：`{"source_lines": 1598, "game_train": 576, "constructed_train": 150, "reviewed_multiturn_augmentation": 4, "frozen_train": 730, "frozen_validation": 70, "rag_withheld_sft_records": 34, "gold_v21": 150, "gold_v3": 150, "game_train_exclusions": 76, "constructed_exclusions": 9, "validation_exclusions": 7}`
