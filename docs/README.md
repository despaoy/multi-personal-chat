# QQChat Enhanced 文档中心

本文档目录只保存可维护的项目说明和可复现研究记录。模型权重、LoRA 产物、数据库、日志与向量索引不进入 Git。

## 阅读顺序

1. [项目 README](../README.md)：项目定位、架构和快速验证。
2. [保研项目答辩手册](research/POSTGRADUATE_RECOMMENDATION_DEFENSE_PLAYBOOK.md)：项目叙事、LLM 原理、追问与演示。
3. [部署指南](operations/DEPLOYMENT_GUIDE.md)：本地、实验室服务器和容器部署。
4. [服务器目录规范](operations/SERVER_LAYOUT.md)：`/home/szw/lhm2` 的唯一推荐布局。
5. [清理策略](operations/CLEANUP_POLICY.md)：哪些文件可以删除，哪些必须保留。
6. [代码知识库](architecture/CODE_WIKI.md)：模块职责、API 和调用链。
7. [可扩展性开发指南](architecture/EXTENSIBILITY_GUIDE.md)：新增功能时的分层边界与最短路径。
8. [生产准备审查](architecture/PRODUCTION_READINESS_REVIEW_2026-07-18.md)：当前风险、限制与后续工作。

## 文档分类

### `architecture/`

- `CODE_WIKI.md`：代码结构与关键调用链。
- `EXTENSIBILITY_GUIDE.md`：分层边界、新增后端/前端功能的最短路径与兼容性规则。
- `OPTIMIZATION_STRATEGY.md`：性能、可靠性、安全和部署优化基线。
- `PRODUCTION_READINESS_REVIEW_2026-07-18.md`：面向个人研究项目的部署前审查。

### `operations/`

- `DEPLOYMENT_GUIDE.md`：安装、配置、启动与验收。
- `SERVER_LAYOUT.md`：源码与运行时资产的目录边界。
- `CLEANUP_POLICY.md`：本地和服务器清理规则。

### `research/`

- `POSTGRADUATE_RECOMMENDATION_DEFENSE_PLAYBOOK.md`：保研答辩叙事、技术原理、高频追问和现场演示手册。
- `KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md`：当前月社妃数据审核、训练门禁和重训练入口。
- `review_packets/kisaki_v4/00_GUIDE.md`：按批次人工检查人物画像、原作、训练候选、验证集和 Gold v2。
- `KISAKI_EXPERIMENT_INDEX.md`：月社妃 R0V4-R4、S1、门禁和历史结果统一入口。
- `archive/KISAKI_E1_E2_CANONICAL_EXPERIMENT.md`：旧 R1 NEFTune 受控实验记录（只读归档）。
- `KISAKI_CHARACTER_PROFILE.md`：原作证据支持的人物画像。
- `archive/KISAKI_GOLD_V2_AI_PRESCREEN.md`：Gold v2 历史审核与冻结记录。
- `RESEARCH_AND_LEARNING_ROADMAP.md`：研究主线和学习路线。
- `BEGINNER_REAL_LLM_EXPERIMENT_GUIDE.md`：真实 LLM 实验操作指南。
- `archive/KISAKI_LORA_RETRAIN_PLAN.md`：历史 LoRA/DoRA/RSLoRA 消融计划（只读归档）。
- 旧 `KISAKI_E1_*`、`KISAKI_E2*` 报告统一位于 `research/archive/`，只作历史追溯。
- `archive/REAL_VLLM_BENCHMARK_REPORT.md`：Qwen2.5 历史基准（迁移对照，已归档）。

### `data/`

- `archive/dataset-card.md`：历史 v1 数据集卡。当前 V4 数据只有在审核完成后才会生成至 `backend/data/character_dialogues/experiments/v4/`。
- `human-scoring-rubric.md`：Gold Set 人工评分标准。

## 维护规则

- 所有 Markdown 使用 UTF-8（无 BOM）和 LF 换行。
- 当前部署事实只写在 README 与 `operations/`；历史报告不得改写真实实验条件。
- 报告必须区分 `planned`、`mock`、`measured`，不得把模拟值写成真实结果。
- 路径、端口和版本以环境变量及锁文件为准，文档中的值仅为推荐默认值。
- 架构、依赖、启动方式或数据目录发生变化时，同一提交必须更新相关文档。
