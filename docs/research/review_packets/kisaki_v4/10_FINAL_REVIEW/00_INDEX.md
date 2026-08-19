# 月社妃 V4 已完成审核索引

本目录仅保留构造集与 Validation 的最终审核快照。Game Train 逐批上下文包已在批准晋升后删除，其最终决定见 `backend/data/character_dialogues/experiments/v4/game_train_context_review_approval.json`。

## 1. 人物与系统提示词

- [人物设定](../01_PROFILE_PROMPT/01_character_profile.md)
- [System Prompt v3](../01_PROFILE_PROMPT/02_system_prompt_v3.md)

## 2. 构造训练集：150 条

- [batch_01](03_CONSTRUCTED/batch_01.md)
- [batch_02](03_CONSTRUCTED/batch_02.md)
- [batch_03](03_CONSTRUCTED/batch_03.md)

## 3. Validation：70 条

- [batch_01](04_VALIDATION/batch_01.md)
- [batch_02](04_VALIDATION/batch_02.md)

## 4. 已批准多轮增补：4 条完整五轮会话

已完成逐轮修订、人工批准和污染复审，仅供追溯，不重复审核：

- [Round 06 批准稿](../../../../../backend/data/character_dialogues/experiments/v4/augmentation_candidates/deepseek_user_simulation_round06/approved_sessions.json)
- [Round 06 晋升结果](../../../../../backend/data/character_dialogues/experiments/v4/augmentation_candidates/deepseek_user_simulation_round06/promotion_result.json)

## 5. Gold

- [Gold v2.1 第 1 批](../06_GOLD_V21/batch_01.md)
- [Gold v2.1 第 2 批](../06_GOLD_V21/batch_02.md)
- [Gold v2.1 第 3 批](../06_GOLD_V21/batch_03.md)
- [Gold v3 第 1 批](../07_GOLD_V3/batch_01.md)
- [Gold v3 第 2 批](../07_GOLD_V3/batch_02.md)
- [Gold v3 第 3 批](../07_GOLD_V3/batch_03.md)

## 6. 技术记录（无需重新判断角色效果）

- [20 条过拟合技术结论](../09_OVERFIT_TEST/technical_review_decision.json)

全部审核已经结束；当前计数与冻结状态以 canonical manifest 为准。
