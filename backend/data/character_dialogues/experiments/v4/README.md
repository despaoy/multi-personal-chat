# KISAKI-CANONICAL-V4 数据目录

> 这是月社妃角色数据当前唯一的正式活动数据版本。权威计数、哈希和冻结状态以 `canonical_dataset_manifest.json` 为准；本 README 只做导航和速览。

## 当前快照

| 项目 | 值 |
|---|---|
| 训练记录 | **926 条**（1961 个有效 assistant 监督目标） |
| 验证记录 | **70 条**，内容与哈希冻结 |
| train SHA-256 | `8791a57ad2b8bce7824f93185c99125b37b15e4a136bee28b2cfae1e609cd245` |
| validation SHA-256 | `fb5cd5d93027b37be53327cda4e7c6137a7bae97201830fc8aa726727b9777b8` |
| 数据集状态 | `frozen`（R0 数据、Prompt 与 Gold v3 已完成冻结） |

训练来源分布（canonical 口径）：

| 来源 | 记录数 |
|---|---|
| 原作提取 `game_extraction_current_sft` | 522 |
| 既有已审核构造数据 `llm_v4_reviewed_constructed` | 150 |
| DeepSeek 用户模拟 round06 `deepseek_user_simulation_v41_reviewed` | 3 |
| Codex 自动化批次 001-068 `codex_user_simulation_v41_reviewed` | 251 |

## 目录布局

| 路径 | 用途 |
|---|---|
| `train.jsonl` | 正式训练读取的唯一 canonical 训练集 |
| `validation.jsonl` | 独立验证集，禁止进入训练 |
| `canonical_dataset_manifest.json` | 权威清单：计数、哈希、来源分布、审核状态与晋升 provenance |
| `game_train_context_review_approval.json` | Game Train 最终审核计数、决定 ID 与晋升哈希 |
| `split_seed.json` | 固定 seed 42 split 依据 |
| `configs/` | 已绑定当前 frozen 数据与 Prompt 3.3.0 的 R1V4 E1-E5 正式配置 |
| `overfit_20/` | 20 条 overfit 冒烟实验资产与结果 |
| `augmentation_candidates/` | V4.1 增补数据生成、审核与晋升的完整证据链 |
| `cleanup/` | 长度清理策略、22 条技术辅助数据、候选与晋升授权 |
| `SPEAKER_CONTRACT.md` | `user/assistant` 协议角色与剧情人物身份的统一契约 |
| `text_normalizations.json` | 只修缺字或明显误字、不改变原作主题的 3 条表层规范化清单 |

## 新增 V4.1 数据的整合规则

- 晋升单位：**完整五轮会话**，不生成前缀切片（`cumulative_prefix_records_allowed=false`）。
- DeepSeek round06：4 个会话 / 20 轮，经项目负责人批准后晋升。
- Codex automation_batch_001-068：每批 4 个会话 / 20 轮，全部审核批准并晋升；0 拒绝。
- 增补完成时 train 曾达到 1002 条；最终 Game Train 上下文复审排除 54 条并为 107 条补充历史，canonical 一度收敛为 948 条。后续长度清理与说话者契约迁移使正式 train 收敛为 926 条；validation、Gold v2.1 与 Gold v3 内容均未修改。
- 批次级明细见 `augmentation_candidates/INDEX.json` 与 `augmentation_candidates/README.md`。
- 2026-08-21 长度复审将 4 条截断损伤样本和 18 条代码主导样本移至技术辅助集，并修复 8 条场景元数据乱码。原 948 条快照仍可由 Git 历史和 cleanup provenance 追溯。
- 训练前文字复核只规范化 3 处可由语法和上下文确认的缺字/误字；原始 gametext 与 raw 抽取保持不变，Gold 题目、人物选择和场景主题均未改写。

## 重要边界

- 正式训练前仍须运行 `scripts/validate_kisaki_v4_training_gate.py`，不得绕过门禁。
- 人物 Prompt 内容保持 v3 不变，已与 prompt policy `3.3.0`、动态上下文和说话者边界对齐；组合规则见 `../../PROMPT_COMPOSITION_CONTRACT.md`。
- Gold v3 已针对当前 926 条 train 与 70 条 validation 重新审计，996 条冻结参考中无精确/近似文本重叠、重复题目或 RAG 事件污染；审计见 `../../../../evaluation/kisaki_gold_set_v3_contamination_audit.json`。
- R0V4 已完成，R1V4 E1-E5 配置已重新生成并通过单变量检查：E2 仅启用 NEFTune，E3 仅启用 DoRA，E4 仅启用 RSLoRA，E5 仅启用 Packing；统一 `max_seq_length=1280`。
- 正式训练门禁当前通过。下一独立步骤是在服务器执行模型、依赖、磁盘与 GPU 预检，然后只启动 R1-E1 Seed 42 基线训练。
- Game Train 补上下文记录使用 `assistant_supervision=last`，只监督最终妃回复，不重复监督历史 assistant 回合。
- `assistant` 始终代表月社妃；`user` 只是对话协议角色。原作人物身份通过独立说话者元数据表达，构造数据不得依据文本中的人名猜测发言者。
- 说话者标签只进入 user 消息的不可信参考区，不得进入 system prompt。详见 `SPEAKER_CONTRACT.md`。
- 历史版本数据已归档到 `../archive/`，不得重新混入本目录。
