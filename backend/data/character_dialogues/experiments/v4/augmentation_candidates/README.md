# V4.1 增补候选数据：生成、审核与晋升证据

> 本目录保存 2026-08-15/16 由大模型生成的新训练数据的完整证据链。
> 机器可读汇总见 `INDEX.json`；晋升后的 canonical 训练集见 `../train.jsonl`。

## 总量

| 项目 | 数量 |
|---|---|
| 已晋升 automation 批次 | 68 批（batch_001-068） |
| 已晋升 automation 完整会话 | 272 个 / 1360 个 assistant 回合 |
| 已晋升 DeepSeek round06 会话 | 4 个 / 20 个 assistant 回合 |
| 新进入 V4.1 的会话合计 | **276 个 / 1380 个 assistant 回合** |
| 当前 canonical train | 948 条 / 2071 个有效 assistant 监督目标 |

V4.1 增补完成时 canonical 曾达到 1002 条；后续 Game Train 上下文复审排除 54 条原作记录，因此当前总数低于本目录各批晋升完成时的历史快照。

## 子目录约定

### `automation_batch_001/` … `automation_batch_068/`

每批标准文件：

| 文件 | 含义 |
|---|---|
| `candidates.json` | 原始生成候选（4 个完整五轮会话） |
| `review.json` | 逐轮审核结果（persona/对齐/事实/技术/多轮/安全等 rubric） |
| `approved_sessions.json` | 批准会话；`messages_mode=source_unchanged` 或含修订后 `messages` |
| `rejected_sessions.json` | 拒绝清单（当前全部为 `no_rejections` 或空） |
| `promotion_result.json` | 晋升事务：前序/结果 train 计数与 SHA-256 |
| `raw_candidates.json` / `revision_history.json` / `round_status.json` | 部分批次的原始草稿与修订痕迹 |

审核结论：272 个会话全部批准，0 拒绝；6 个 assistant 回合在 5 个批次中经修订后通过（batch_001/012/013/014/016）。

### `deepseek_round01/`

20 个候选，状态 `pending_human_review`（18 pending + 2 blocked），**未进入训练集**。

### `deepseek_user_simulation_round02/` 到 `round06/`

- round02：`rejected_quality_audit`。
- round03/04/05：分别被后续轮次取代（`superseded_by_round04/05/06`）。
- round06：4 个会话，`approved_after_revision` 并已晋升。其 `manifest.json` 保留的是生成阶段状态字段；晋升事实以 `approved_sessions.json` 和 `promotion_result.json` 为准。

### `llm_persona_review_20260816/` 与 `llm_full_dialogue_review_20260816/`

这是对 426 条 LLM 构造记录的两轮复核，不是新增会话：

| 复核 | 修订 |
|---|---|
| persona review | 39 条记录 / 39 个 assistant 回合 |
| full-dialogue review | 14 条记录 / 16 个 assistant 回合 |
| 拒绝 | 0 条 |
| validation / Gold v3 内容 | 均未修改，Gold v3 污染复审 `clean` |

## 状态判定优先级

1. `promotion_result.json`：是否进入 canonical train 的唯一事实来源。
2. `approved_sessions.json`：审核批准与逐轮修订 provenance。
3. `manifest.json`：生成时快照；可能与晋升后的最终状态不一致，只用于追溯生成条件。
