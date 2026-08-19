# KISAKI-CANONICAL-V4 数据目录

> 这是月社妃角色数据当前唯一的正式活动数据版本。权威计数、哈希和冻结状态以 `canonical_dataset_manifest.json` 为准；本 README 只做导航和速览。

## 当前快照

| 项目 | 值 |
|---|---|
| 训练记录 | **948 条**（2071 个有效 assistant 监督目标） |
| 验证记录 | **70 条**，内容与哈希冻结 |
| train SHA-256 | `903c07934f29ea764ea14b28208e2a78468e33fe6f7de0d53d85bd642358cfb8` |
| validation SHA-256 | `fb5cd5d93027b37be53327cda4e7c6137a7bae97201830fc8aa726727b9777b8` |
| 数据集状态 | `frozen`（Game Train 上下文复审已完成） |

训练来源分布（canonical 口径）：

| 来源 | 记录数 |
|---|---|
| 原作提取 `game_extraction_current_sft` | 522 |
| 既有已审核构造数据 `llm_v4_reviewed_constructed` | 150 |
| DeepSeek 用户模拟 round06 `deepseek_user_simulation_v41_reviewed` | 4 |
| Codex 自动化批次 001-068 `codex_user_simulation_v41_reviewed` | 272 |

## 目录布局

| 路径 | 用途 |
|---|---|
| `train.jsonl` | 正式训练读取的唯一 canonical 训练集 |
| `validation.jsonl` | 独立验证集，禁止进入训练 |
| `canonical_dataset_manifest.json` | 权威清单：计数、哈希、来源分布、审核状态与晋升 provenance |
| `game_train_context_review_approval.json` | Game Train 最终审核计数、决定 ID 与晋升哈希 |
| `split_seed.json` | 固定 seed 42 split 依据 |
| `configs/` | R1V4 E1-E5 正式实验配置（数据冻结后生成） |
| `overfit_20/` | 20 条 overfit 冒烟实验资产与结果 |
| `augmentation_candidates/` | V4.1 增补数据生成、审核与晋升的完整证据链 |

## 新增 V4.1 数据的整合规则

- 晋升单位：**完整五轮会话**，不生成前缀切片（`cumulative_prefix_records_allowed=false`）。
- DeepSeek round06：4 个会话 / 20 轮，经项目负责人批准后晋升。
- Codex automation_batch_001-068：每批 4 个会话 / 20 轮，全部审核批准并晋升；0 拒绝。
- 增补完成时 train 曾达到 1002 条；最终 Game Train 上下文复审排除 54 条并为 107 条补充历史，canonical 收敛为 948 条。validation、Gold v2.1 与 Gold v3 内容均未修改，Gold v3 污染复审为 `clean`。
- 批次级明细见 `augmentation_candidates/INDEX.json` 与 `augmentation_candidates/README.md`。

## 重要边界

- 正式训练前仍须运行 `scripts/validate_kisaki_v4_training_gate.py`，不得绕过门禁。
- Game Train 补上下文记录使用 `assistant_supervision=last`，只监督最终妃回复，不重复监督历史 assistant 回合。
- 历史版本数据已归档到 `../archive/`，不得重新混入本目录。
