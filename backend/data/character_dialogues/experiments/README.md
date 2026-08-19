# 月社妃实验资产

> 唯一活动主线是 KISAKI-LLM-RESEARCH-V4。历史数据已集中归档到 `archive/`；活动数据集中在 `v4/` 与 `research/`。

## 当前权威入口

| 作用 | 路径 | 状态 |
|---|---|---|
| 研究注册表 | `research/research_program_registry_v4.json` | authoritative |
| 人工审核清单 | `../../../../docs/research/review_packets/kisaki_v4/review_manifest.json` | Game Train 上下文质量复审中 |
| 人物提示词 | `../kisaki_system_prompt_v3.txt` | approved |
| V4 数据清单 | `v4/canonical_dataset_manifest.json` | **948 train / 70 validation**；当前 `frozen` |
| V4 canonical 训练/验证 | `v4/train.jsonl` / `v4/validation.jsonl` | train 含 276 条新晋升 V4.1 五轮会话 |
| V4 实验配置 | `v4/configs/kisaki_r1v4_e1.json` 至 `e5.json` | 数据冻结后生成 |
| V4.1 增补证据链 | `v4/augmentation_candidates/INDEX.json` | 68 个 automation 批次 + DeepSeek rounds 汇总 |
| Gold v2.1 | `../../../evaluation/kisaki_gold_set_v21_candidates.json` | 已批准的 development-only 集合 |
| Gold v3 | `../../../evaluation/kisaki_gold_set_v3.json` | 150 条最终 held-out，已冻结 |

正式训练只能通过：

```bash
python scripts/validate_kisaki_v4_training_gate.py
python scripts/run_kisaki_experiment.py --experiment e1 --seed 42
```

门禁未通过时，训练器必须拒绝启动。当前阶段只允许完成 Game Train 上下文质量复审、整理经批准的数据增补，以及运行不消耗 GPU 的契约测试。

## 目录布局

| 目录 | 内容 | 状态 |
|---|---|---|
| `v4/` | V4 canonical 数据、配置、overfit 冒烟与 V4.1 增补证据 | active |
| `research/` | RAG 证据、检索评测、系统路由评测、偏好配置与研究注册表 | active |
| `train_v5_clean.jsonl` | 150 条已批准构造数据的可追溯来源 | active provenance |
| `clean_v5_report.json` | train_v5_clean 清洗报告 | active provenance |
| `archive/v2_canonical/` | V2 train/eval | archived |
| `archive/v2_quality_review/` | V2 排除、统计与评分 | archived |
| `archive/v3_pipeline/` | V3 canonical 与 `llm_v4_judged` 流水线输入 | archived |
| `archive/v3_llm_generation/` | 已废弃的 llm_v3 DeepSeek 样本 | archived |
| `archive/v4_draft/` | V4 blindfix 草案与整体检查记录 | archived |
| `archive/game_extract_pre_v4/` | 游戏原文候选提取中间产物 | archived |
| `archive/e2_supplement/`、`archive/legacy_rag/`、`archive/generation_tools/` | 旧 E2、旧 RAG KB、旧提示词池 | archived |

归档迁移明细、原始路径与 SHA-256 见 `archive/INDEX.json`。

## 活动数据边界

- `train_v5_clean.jsonl` 是 150 条已批准构造数据的可追溯来源，正式训练读取 `v4/train.jsonl`。
- V4 冻结训练与独立验证分别为 `v4/train.jsonl` 和 `v4/validation.jsonl`。
- `archive/v3_pipeline/llm_v4_judged/` 是旧合成数据生成阶段的输入；已归档，不再作为 V4 活动输入。
- 只有 Gold v3 批准、正式配置生成且训练门禁通过后，才能启动 R1V4。
- Gold v2.1 用于开发比较但不得用于正式结论；Gold v3 保持最终 held-out 角色。
- 归档数据不得重新混入 V4 train/validation 或 Gold 集。

## 状态语义

- `human_review`：审核仍在进行。
- `frozen`：内容与哈希均已冻结。
- `training_complete`：仅表示 adapter 生成成功。
- `automatic_evaluation_passed`：自动指标通过。
- `blind_review_complete`：匿名人工 A/B 完成。
- `conclusion_ready`：受控实验、盲评和必要复现实验均完整。
