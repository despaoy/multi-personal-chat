# 月社妃实验总览

> 当前执行入口：[月社妃 V4 人工审核与重训练](KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md)。工作树只保留 V4 主线。

## 当前状态

- R0V4：当前 948 条训练数据（522 原作 + 150 既有构造 + 276 条 V4.1 五轮会话）和 70 条 validation；Game Train 复审完成，Gold v2.1 已批准为开发集，Gold v3 已冻结。
- 数据复审：Game Train 上下文质量复审已关闭；V4.1 的 276 条五轮会话（DeepSeek round06 4 条 + Codex 自动化批次 272 条）均已修订/批准并完成 Gold 污染复审。
- R1V4：数据已冻结，重新通过当前环境训练门禁后可运行。
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
| V4 canonical 数据 | `backend/data/character_dialogues/experiments/v4/` |
| V4.1 增补证据链 | `backend/data/character_dialogues/experiments/v4/augmentation_candidates/INDEX.json` |
| 历史数据归档 | `backend/data/character_dialogues/experiments/archive/` |
| R1V4 E1-E5 配置 | 数据冻结后生成至 `backend/data/character_dialogues/experiments/v4/configs/` |
| 当前人物 prompt v3 | `backend/data/character_dialogues/kisaki_system_prompt_v3.txt`（已批准，训练策略为 `replace`） |
| Gold v2.1 开发集 | `backend/evaluation/kisaki_gold_set_v21_candidates.json`（已批准，禁止正式结论） |
| Gold v2.1 污染审计 | `backend/evaluation/kisaki_gold_set_v21_contamination_audit.json` |
| Gold v3 | `backend/evaluation/kisaki_gold_set_v3.json`（已冻结的最终 held-out） |

## 规则

1. 设计注册表、服务器运行清单和实验结果分开保存。
2. mock 只能验证界面和流程，必须显示“演示数据”，不能进入报告。
3. 旧实验只通过 Git 历史追溯，不作为工作树中的第二套入口。
4. RAG 引用属于结构化响应，不训练人物正文输出文档 ID。
5. 单种子只称 pilot；最终只为 E1 与最佳变体补 Seed 43/44。
6. V4 审核完成前禁止生成正式配置或启动 GPU 训练。
7. 正式训练仅使用 V4 manifest 指向的冻结资产。
