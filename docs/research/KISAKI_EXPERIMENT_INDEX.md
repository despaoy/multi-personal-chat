# 月社妃实验总览

> 当前执行入口：[月社妃 V4 人工审核与重训练](KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md)。旧 v3 主线仅保留历史设计与已完成结果。

## 当前状态

- 历史 `prompt-v1 pilot` 已归档：E1 胜 26、E2 胜 32、平局 62，双侧精确符号检验 `p=0.512`。该结果不可改写，也不进入 V4 正式结论。
- R0V4：正在人工审核 1,598 条原作覆盖、801 条原作训练候选、159 条构造候选、验证集与 Gold v2；正式训练被门禁阻塞。
- 旧 R1：E1-E5 Seed 42 均已训练和真实评测，E3 自动指标暂时最好；这些结果使用旧数据和旧标签逻辑，只作为历史对照。
- R1V4：等待审核、数据冻结、Qwen3 多轮监督 smoke 和 Gold v3 冻结，不得提前运行。
- R2：60 条 held-out 候选已生成，等待人工冻结。
- R3：真实 SSE TTFT 基准器已实现，等待隔离服务测试。
- R4：等待至少 100 条月社妃人工批准偏好对。
- S1：等待最终路由与 AstrBot 系统验收。

## 权威文件

| 作用 | 文件 |
|---|---|
| V4 人工审核入口 | [审核指南](review_packets/kisaki_v4/00_GUIDE.md) |
| 静态研究定义 | `backend/data/character_dialogues/experiments/research/research_program_registry_v4.json` |
| V4 正式训练门禁 | `scripts/validate_kisaki_v4_training_gate.py` |
| R1V4 E1-E5 配置 | 数据冻结后生成至 `backend/data/character_dialogues/experiments/v4/configs/` |
| 当前人物 prompt v3 | `backend/data/character_dialogues/kisaki_system_prompt_v3.txt`（待人工批准） |
| Gold v2 开发集 | `backend/evaluation/kisaki_gold_set_v2.json` |
| Seed 42 prompt-v1 pilot | [盲评结果](reviews/KISAKI_SEED42_BLIND_REVIEW_RESULT.md) |
| 历史资产索引 | `backend/data/character_dialogues/experiments/archive/legacy_result_index.json` |

## 规则

1. 设计注册表、服务器运行清单和实验结果分开保存。
2. mock 只能验证界面和流程，必须显示“演示数据”，不能进入报告。
3. 旧 E2/E2'/E2'' 是 `legacy_exploratory_non_comparable`。
4. RAG 引用属于结构化响应，不训练人物正文输出文档 ID。
5. 单种子只称 pilot；最终只为 E1 与最佳变体补 Seed 43/44。
6. V4 审核完成前禁止生成正式配置或启动 GPU 训练。
7. `experiments/archive`、`scripts/archive` 和 `docs/research/archive` 中的文件只能用于历史追溯，活动代码不得读取。
