# 月社妃角色数据

当前唯一活动研究体系为 KISAKI-LLM-RESEARCH-V4。人物画像、system prompt v3、训练集、验证集、Gold v2.1 和 Gold v3 均已审核；Game Train 上下文质量复审已经完成，canonical 数据集已重新冻结。

## 活动资产

| 内容 | 路径 | 用途 |
|---|---|---|
| 原始可追溯台词 | `tsukiyashiro_kisaki_raw.jsonl` | 来源覆盖和证据定位 |
| 原作训练候选 | `tsukiyashiro_kisaki_sft.json` | 自动来源审计后的训练候选 |
| 完整提取候选与排除记录 | `tsukiyashiro_kisaki_sft_full.json` / `tsukiyashiro_kisaki_excluded.jsonl` | 提取与排除 provenance |
| 构造训练候选 | `experiments/train_v5_clean.jsonl` | 150 条已批准构造数据的可追溯来源 |
| V4 审核包 | `../../../docs/research/review_packets/kisaki_v4/` | 用户逐批审核 |
| 人物提示词 | `kisaki_system_prompt_v3.txt` | 已确认的角色身份、关系、性格和表达 |
| V4 canonical 数据 | `experiments/v4/` | **948 条 train、70 条 validation**；状态 `frozen` |
| V4.1 增补证据 | `experiments/v4/augmentation_candidates/` | 276 个新生成五轮会话的生成、审核与晋升记录 |
| Gold v2.1 | `../../evaluation/kisaki_gold_set_v21_candidates.json` | 已批准的 development-only 评测集，不回流训练 |
| Gold v3 | `../../evaluation/kisaki_gold_set_v3.json` | 150 条最终盲测，已审核并冻结 |
| 历史数据归档 | `experiments/archive/` | 已退出活动工作面的 V2/V3/V4 草稿与旧实验资产 |

V4 train 的 canonical 来源分布：522 条原作提取 + 150 条既有已审核构造 + 4 条 DeepSeek round06 五轮会话 + 272 条 Codex 自动化批次五轮会话 = **948 条**。最终 Game Train 审核记录见 `experiments/v4/game_train_context_review_approval.json`。

## 三层提示词

1. 人物层：`kisaki_system_prompt_v3.txt`。
2. 全局事实与安全层：`backend/inference/prompt_policy.py`。
3. RAG 证据层：仅检索命中时由同一策略模块条件注入。

人物提示词不承载密钥保护、管理权限或引用格式；RAG 内容被标记为不可信证据，不能覆盖系统规则。

训练数据记录不内置人物 system prompt。训练器使用 `system_prompt_policy=replace`，统一注入已审核的 `kisaki_system_prompt_v3.txt`；推理阶段仍由同一人物提示词入口加载。

## 审核与训练门禁

```bash
python scripts/validate_kisaki_v4_training_gate.py
python scripts/validate_kisaki_v4_training_gate.py
```

原作台词归属、定位和覆盖由可复现审计负责。Game Train 上下文质量复审完成、canonical 状态恢复为最终 `frozen` 且训练门禁通过前，不得启动正式 R1V4 训练。旧实验资产统一在 `experiments/archive/` 归档，不再散落在活动目录；需要更早的历史版本时从 Git 历史追溯。
