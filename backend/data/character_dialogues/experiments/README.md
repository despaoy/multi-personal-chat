# 月社妃实验资产

> 唯一活动主线是 KISAKI-LLM-RESEARCH-V4。历史契约、脚本、数据和结果仅用于追溯，活动代码不得读取归档目录。

## 当前权威入口

| 作用 | 路径 | 状态 |
|---|---|---|
| 研究注册表 | `research/research_program_registry_v4.json` | authoritative |
| 人工审核清单 | `../../../../docs/research/review_packets/kisaki_v4/review_manifest.json` | pending |
| 人物提示词 | `../kisaki_system_prompt_v3.txt` | pending human approval |
| V4 数据清单 | `v4/canonical_dataset_manifest.json` | 审核完成后生成并冻结 |
| V4 实验配置 | `v4/configs/kisaki_r1v4_e1.json` 至 `e5.json` | 数据冻结后生成 |
| Gold v2 | `../../../evaluation/kisaki_gold_set_v2.json` | development only |
| Gold v3 | `../../../evaluation/kisaki_gold_set_v3.json` | 训练数据冻结后建立 |

正式训练只能通过：

```bash
python scripts/validate_kisaki_v4_training_gate.py
python scripts/run_kisaki_experiment.py --experiment e1 --seed 42
```

门禁未通过时，训练器必须拒绝启动。当前阶段只允许审核、修订候选数据和运行不消耗 GPU 的契约测试。

## 活动数据边界

- `train_v5_clean.jsonl`、`combined_*.jsonl` 和原作提取文件是 V4 审核候选，不是正式训练集。
- `v3/llm_v4_judged/` 是合成数据生成阶段的可追溯输入；其目录名表示流水线来源版本，不代表正式 R1 版本。
- 只有 `v4/canonical_dataset_manifest.json` 状态为 `frozen` 且哈希匹配时，V4 数据才能用于正式训练。
- Gold v2 仅用于开发；Gold v3 只能在训练数据冻结后建立，避免反向泄漏。

## 归档边界

- `archive/contracts/`：旧 canonical manifest、registry 和 v2/v3 配置。
- `archive/prompts/`：prompt v1/v2。
- `archive/registries/`：研究注册表 v3。
- `results/archive/legacy_exploratory/`：旧 E1/E2/E2'/E2'' 结果。
- `../../archive/legacy_e2/`：旧 E2 数据增强训练集与补充数据。
- `scripts/archive/kisaki_legacy/`：旧构建、启动、评测和部署脚本。

归档内容统一为 `legacy_exploratory_non_comparable`。不得被运行时导入、被活动脚本调用，或进入 V4 正式对比表。

## 状态语义

- `human_review`：审核仍在进行。
- `frozen`：内容与哈希均已冻结。
- `training_complete`：仅表示 adapter 生成成功。
- `automatic_evaluation_passed`：自动指标通过。
- `blind_review_complete`：匿名人工 A/B 完成。
- `conclusion_ready`：受控实验、盲评和必要复现实验均完整。
