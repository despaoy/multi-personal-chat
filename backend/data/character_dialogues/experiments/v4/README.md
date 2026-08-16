# KISAKI-CANONICAL-V4 数据目录

> 这是月社妃角色数据当前唯一的正式活动数据版本。权威计数、哈希和冻结状态以 `canonical_dataset_manifest.json` 为准；本 README 只做导航和速览。

## 当前快照

| 项目 | 值 |
|---|---|
| 训练记录 | **1002 条**（2125 个 assistant 回合） |
| 验证记录 | **70 条**，内容与哈希冻结 |
| train SHA-256 | `864b20e50d1d4dc4917e5d93be7c29da05e567edce07faf42526481b075b04ed` |
| validation SHA-256 | `fb5cd5d93027b37be53327cda4e7c6137a7bae97201830fc8aa726727b9777b8` |
| 数据集状态 | `frozen_under_reassessment`（正式训练门禁仍阻塞） |

训练来源分布（canonical 口径）：

| 来源 | 记录数 |
|---|---|
| 原作提取 `game_extraction_current_sft` | 576 |
| 既有已审核构造数据 `llm_v4_reviewed_constructed` | 150 |
| DeepSeek 用户模拟 round06 `deepseek_user_simulation_v41_reviewed` | 4 |
| Codex 自动化批次 001-068 `codex_user_simulation_v41_reviewed` | 272 |

## 目录布局

| 路径 | 用途 |
|---|---|
| `train.jsonl` | 正式训练读取的唯一 canonical 训练集 |
| `validation.jsonl` | 独立验证集，禁止进入训练 |
| `canonical_dataset_manifest.json` | 权威清单：计数、哈希、来源分布、审核状态与晋升 provenance |
| `train_candidate.jsonl` / `validation_candidate.jsonl` | 冻结前的候选与审核输入，保留可追溯性 |
| `split_seed.json` | 固定 seed 42 split 依据 |
| `configs/` | R1V4 E1-E5 正式实验配置（数据冻结后生成） |
| `overfit_20/` | 20 条 overfit 冒烟实验资产与结果 |
| `augmentation_candidates/` | V4.1 增补数据生成、审核与晋升的完整证据链 |

## 新增 V4.1 数据的整合规则

- 晋升单位：**完整五轮会话**，不生成前缀切片（`cumulative_prefix_records_allowed=false`）。
- DeepSeek round06：4 个会话 / 20 轮，经项目负责人批准后晋升。
- Codex automation_batch_001-068：每批 4 个会话 / 20 轮，全部审核批准并晋升；0 拒绝。
- 晋升后 train 从 730 逐批增长到 1002；validation、Gold v2.1 与 Gold v3 内容均未修改，Gold v3 污染复审为 `clean`。
- 批次级明细见 `augmentation_candidates/INDEX.json` 与 `augmentation_candidates/README.md`。

## 重要边界

- 训练门禁未通过前，不得启动 R1V4 seed 42 或 E2-E5。
- `frozen_under_reassessment` 是当前主动设置的保守状态；不要只根据 train 文件存在就推断门禁已通过。
- 历史版本数据已归档到 `../archive/`，不得重新混入本目录。
