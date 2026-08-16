# 月社妃 V4 最终总审核入口

只审核本目录列出的当前对象；不要再使用旧 `03_GAME_TRAIN` 或 `legacy_v3` 批次。

## 1. 人物与系统提示词

- [人物设定](../01_PROFILE_PROMPT/01_character_profile.md)
- [System Prompt v3](../01_PROFILE_PROMPT/02_system_prompt_v3.md)

## 2. Game Train：576 条

重点判断脱离原作现场后是否仍能构成可靠问答。

- [batch_01](02_GAME_CONTEXT/batch_01.md)
- [batch_02](02_GAME_CONTEXT/batch_02.md)
- [batch_03](02_GAME_CONTEXT/batch_03.md)
- [batch_04](02_GAME_CONTEXT/batch_04.md)
- [batch_05](02_GAME_CONTEXT/batch_05.md)
- [batch_06](02_GAME_CONTEXT/batch_06.md)
- [batch_07](02_GAME_CONTEXT/batch_07.md)
- [batch_08](02_GAME_CONTEXT/batch_08.md)
- [batch_09](02_GAME_CONTEXT/batch_09.md)
- [batch_10](02_GAME_CONTEXT/batch_10.md)
- [batch_11](02_GAME_CONTEXT/batch_11.md)
- [batch_12](02_GAME_CONTEXT/batch_12.md)

## 3. 构造训练集：150 条

- [batch_01](03_CONSTRUCTED/batch_01.md)
- [batch_02](03_CONSTRUCTED/batch_02.md)
- [batch_03](03_CONSTRUCTED/batch_03.md)

## 4. Validation：70 条

- [batch_01](04_VALIDATION/batch_01.md)
- [batch_02](04_VALIDATION/batch_02.md)

## 5. 已批准多轮增补：4 条完整五轮会话

已完成逐轮修订、人工批准和污染复审，仅供追溯，不重复审核：

- [Round 06 批准稿](../../../../../backend/data/character_dialogues/experiments/v4/augmentation_candidates/deepseek_user_simulation_round06/approved_sessions.json)
- [Round 06 晋升结果](../../../../../backend/data/character_dialogues/experiments/v4/augmentation_candidates/deepseek_user_simulation_round06/promotion_result.json)

## 6. Gold

- [Gold v2.1 第 1 批](../06_GOLD_V21/batch_01.md)
- [Gold v2.1 第 2 批](../06_GOLD_V21/batch_02.md)
- [Gold v2.1 第 3 批](../06_GOLD_V21/batch_03.md)
- [Gold v3 第 1 批](../07_GOLD_V3/batch_01.md)
- [Gold v3 第 2 批](../07_GOLD_V3/batch_02.md)
- [Gold v3 第 3 批](../07_GOLD_V3/batch_03.md)

## 7. 技术记录（无需重新判断角色效果）

- [20 条过拟合技术结论](../09_OVERFIT_TEST/technical_review_decision.json)

完成后请分别给出 Game Train、构造集、Validation、Gold v2.1、Gold v3 的通过、修改、排除数量；已批准多轮增补不重复计入待审核项。
