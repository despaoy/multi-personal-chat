# 历史训练数据归档

> 本目录只保存已退出活动工作面的历史/旧版本训练数据资产，用于审计追溯。
> 当前唯一活动主线是 `experiments/v4/` 下的 KISAKI-CANONICAL-V4；迁移明细和哈希见同目录 `INDEX.json`。

## 目录说明

| 目录 | 内容 | 历史定位 |
|---|---|---|
| `v2_canonical/` | V2 canonical train/eval（826 train / 92 eval，含 111 条已废弃 llm_v3_deepseek 样本） | V2 正式数据集，已被 V3/V4 取代 |
| `v2_quality_review/` | V2 排除样本、统计和人工评分记录 | V2 数据治理证据 |
| `v3_pipeline/` | V3 canonical manifest、715 原作 train + 84 eval，以及 `llm_v4_judged` 生成流水线输入 | V3 合成数据生成阶段输入；目录名仅表示流水线来源版本 |
| `v3_llm_generation/` | DeepSeek 生成的 119 条 llm_v3 样本（系统风格偏差后归档） | 已因 meta-narrative/“正因如此”/笑声单调等问题停止使用 |
| `v4_draft/` | V4 早期 blindfix train/eval 草案、整体检查清单和 token 统计 | 已被 `train_v5_clean.jsonl` 与 `v4/train.jsonl` 取代 |
| `game_extract_pre_v4/` | 月社妃、理央、夜子等游戏原文候选提取、合并与 manifest | V4 提取前的候选生成中间产物 |
| `generation_tools/` | V1 每日生成提示词池 | 历史生成工具输入 |
| `e2_supplement/` | E2 元叙事等手工补充 25 条 | 旧 E2 实验资产 |
| `legacy_rag/` | 旧版角色知识库 JSON | 已被 `experiments/research/character_rag_seed_documents.json` 取代 |

## 规则

1. 归档文件不再被正式训练、评测或 API 读取；需要历史结果时优先查 Git 历史。
2. 不得把归档数据重新混入 V4 train/validation 或 Gold 集。
3. 归档移动必须同步更新 `INDEX.json` 的 `original_path`、`archived_path` 和 SHA-256。
4. 如果旧工具脚本仍需要读取归档数据，必须使用 `archive/` 下的显式路径，并标注 `legacy_only`。
