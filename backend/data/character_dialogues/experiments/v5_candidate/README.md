# KISAKI-V5 候选数据资产清单（阶段 1 产物）

本目录是 V5 重构的候选工作区，不覆盖 `experiments/v4/`（冻结，sha256 见下）。

## 生成方式

- 脚本：`scripts/build_kisaki_v5_asset_inventory.py`（只读分析，可重算）
- 数据源：`KISAKI-CANONICAL-V4` train.jsonl（926 条）/ validation.jsonl（70 条）
- 运行前后均校验 V4 冻结哈希，未修改任何 V4 文件
- token 估算：`chars * 0.65`（Qwen 系中文近似，沿用项目既有口径）；**占比一律以字符口径为准**

## 监督口径（与训练契约一致）

`backend/training/chat_dataset.py` 支持 `assistant_supervision = "all" | "last"`（缺省 `all`）：
107 条原作多轮记录设置为 `last`，**只监督每条记录的最后一条 assistant 消息**。
因此统计同时给出原始 assistant 消息数与实际监督目标数，只有被监督的回复
才计入监督字符。manifest 的 `assistant_supervision_targets: 1961` 即此口径，
与实测一致（2,068 原始消息 − 107 条 last 记录各去掉 1 条非末位消息）。

## 来源统计（监督口径）

| 来源 | 记录 | 原始 assistant 消息 | 实际监督目标 | 监督字符 | 监督字符占比 | 多轮记录 | 平均监督字符/目标 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原作提取 (game_extraction) | 522 | 629 | 522 | 16,433 | 14.00% | 107 | 31.5 |
| Codex 用户模拟 | 251 | 1,255 | 1,255 | 96,530 | **82.26%** | 251 | 76.9 |
| 短构造 (llm_v4_*) | 150 | 169 | 169 | 2,883 | 2.46% | 12 | 17.1 |
| DeepSeek 用户模拟 | 3 | 15 | 15 | 1,499 | 1.28% | 3 | 99.9 |
| **合计** | **926** | **2,068** | **1,961** | **117,345** | 100% | 373 | 59.9 |

按实际监督字符计算，Codex 模拟占 **82.26%**——长篇用户模拟的暴露问题
比原始口径（79.63%）更严重。字符占比不等于精确梯度权重，仅刻画数据组成。

## 逐条清单字段（asset_inventory.json）

`id` / `data_source`（原值）/ `source_group`（四类聚合）/ `status` / `scene` /
`interlocutor_kind` / `assistant_supervision` / `turns{user,assistant}` /
`multiturn` / `chars{raw_assistant,supervised_assistant,total}` /
`supervised_assistant_targets` / `est_tokens` / `avg_supervised_chars_per_target` /
`issues`

## 状态语义（本阶段只允许三种）

- `keep_core`：522 条原作提取（V4 manifest: approved_after_context_reaudit）
- `pending_review`：150 短构造（阶段 3 复审）+ 254 长模拟（阶段 2 筛选）
- `excluded_candidate`：本阶段未使用；排除决定由后续审核阶段写入

## 质量核验结果（inventory_issues.json）

926 条记录全部检查以下项，**0 条异常**：

- 重复 ID：无
- 空消息 / 未知 role：无
- system 消息不在首位 / 相邻同角色消息（user-user 与 assistant-assistant，
  与 `normalize_chat_record()` 规则一致）：无
- 末条非 assistant（无监督目标）：无
- 审核状态字段缺失（game 缺 final_review、构造/模拟缺 human_review）：无
- 与 validation ID 重叠：无（仅 ID 口径；文本级精确/近似重叠检查留待阶段 5
  与 validation 复查时进行）

## 代表样本（每来源按监督字符排序取最短/中位/最长）

- 原作：`tsukiyashiro_kisaki_sft_745b43ab63c84b25`（最短）、
  `tsukiyashiro_kisaki_sft_38a21f368c7aab0c`（中位）、
  `tsukiyashiro_kisaki_sft_e63133b385dce439`（最长）
- 短构造：`kisaki_llm_v4_life_0025` / `kisaki_llm_v4_0031` / `kisaki_llm_v4_0208`
- DeepSeek 模拟：`kisaki_v41_round06_daily_chat` /
  `kisaki_v41_round06_research_learning` / `kisaki_v41_round06_project_safety`
- Codex 模拟：`kisaki_v41_auto_b001_rainy_weekend`（雨天周末）/
  `kisaki_v41_auto_b033_deciding_whether_to_repot_an_overgrown_houseplant` /
  `kisaki_v41_auto_b023_webhook_retry_idempotency_and_signature_tests`
  （Webhook 排查，典型技术问答类，阶段 2 重点排查对象）

## 文件说明

| 文件 | 内容 |
|---|---|
| `asset_inventory.json` | 926 条逐条清单（含来源/场景/轮数/监督口径长度/状态） |
| `asset_stats.json` | 来源聚合统计（raw/supervised 双口径）+ 问题类型汇总 |
| `inventory_issues.json` | 异常记录明细（当前为空） |
| `README.md` | 本说明 |

单元测试：`backend/tests/test_kisaki_v5_asset_inventory.py`
（监督口径 all/last、未知来源、相邻同角色、ID 重叠、926 完整性、V4 只读）。

## 后续阶段（未开始）

阶段 2 筛选 254 条长模拟（目标保留 15–25 条）；阶段 3 复查 150 条短构造；
本阶段不删除数据、不训练。
