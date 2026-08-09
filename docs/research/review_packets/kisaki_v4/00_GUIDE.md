# 月社妃 V4 人工审核指南

正式训练保持关闭，直到所有必需分类得到项目负责人明确批准。

## 建议顺序

1. `01_PROFILE_PROMPT`
2. `02_SOURCE_COVERAGE`
3. `03_GAME_TRAIN`
4. `04_CONSTRUCTED_TRAIN`
5. `05_VALIDATION`
6. `06_GOLD_V2`
7. 数据冻结后再生成 `07_GOLD_V3`
8. `08_EXCLUSIONS`
9. `09_EXPERIMENT_CONFIGS`

## 回复格式

- `全部通过`
- `样本 ID：修改建议`
- `样本 ID：排除，原因`
- `样本 ID：需要更多原作上下文`

Gold 修改不得回流训练集。原作 assistant 台词不可改写，只能修正构造的问题或排除错误提取。

当前计数：`{"source_lines": 1598, "game_train_candidates": 801, "constructed_train_candidates": 159, "legacy_validation": 92, "v5_draft_validation": 27, "gold_v2": 150, "gold_v3": 0, "exclusions": 117}`
