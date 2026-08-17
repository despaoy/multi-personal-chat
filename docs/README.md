# MultiPersonal Chat System 文档中心

本文档目录只保存可维护的项目说明和可复现研究记录。模型权重、LoRA 产物、数据库、日志与向量索引不进入 Git。

## 阅读顺序

1. [项目 README](../README.md)：项目定位、能力、架构和快速开始。
2. [部署指南](operations/DEPLOYMENT_GUIDE.md)：本地、服务器和容器部署。
3. [服务器目录规范](operations/SERVER_LAYOUT.md)：源码与运行时资产的目录边界。
4. [代码知识库](architecture/CODE_WIKI.md)：模块职责、API 和调用链。
5. [可扩展性开发指南](architecture/EXTENSIBILITY_GUIDE.md)：新增功能时的分层边界与最短路径。
6. [生产准备审查](architecture/PRODUCTION_READINESS_REVIEW_2026-07-18.md)：当前风险、限制与后续工作。
7. [清理策略](operations/CLEANUP_POLICY.md)：哪些文件可以删除，哪些必须保留。
8. [发布前检查清单](RELEASE_CHECKLIST.md)：可发布边界和验证命令。

## 文档分类

### `architecture/`

- `CODE_WIKI.md`：代码结构与关键调用链。
- `EXTENSIBILITY_GUIDE.md`：分层边界、新增后端/前端功能的最短路径与兼容性规则。
- `OPTIMIZATION_STRATEGY.md`：性能、可靠性、安全和部署优化基线。
- `PRODUCTION_READINESS_REVIEW_2026-07-18.md`：部署前风险与限制审查。

### `operations/`

- `DEPLOYMENT_GUIDE.md`：安装、配置、启动与验收。
- `SERVER_LAYOUT.md`：源码与运行时资产的目录边界。
- `CLEANUP_POLICY.md`：本地和服务器清理规则。

### `research/`

- `POSTGRADUATE_RECOMMENDATION_DEFENSE_PLAYBOOK.md`：研究叙事、技术原理和现场问答手册（作者个人答辩案例，可按需忽略）。
- `KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md`：当前月社妃数据审核、训练门禁和重训练入口。
- `review_packets/kisaki_v4/00_GUIDE.md`：V4 数据和 Gold 审核入口。
- `KISAKI_EXPERIMENT_INDEX.md`：月社妃 R0V4-R4、S1 和门禁统一入口。
- `KISAKI_CHARACTER_PROFILE.md`：原作证据支持的人物画像。
- `RESEARCH_AND_LEARNING_ROADMAP.md`：研究主线和学习路线。
- `BEGINNER_REAL_LLM_EXPERIMENT_GUIDE.md`：真实 LLM 实验操作指南。

### `data/`

- `human-scoring-rubric.md`：Gold Set 人工评分标准。

## 维护规则

- 所有 Markdown 使用 UTF-8（无 BOM）和 LF 换行。
- 当前部署事实只写在 README 与 `operations/`；旧版本通过 Git 历史追溯。
- 报告必须区分 `planned`、`mock`、`measured`，不得把模拟值写成真实结果。
- 路径、端口和版本以环境变量及锁文件为准，文档中的值仅为推荐默认值。
- 架构、依赖、启动方式或数据目录发生变化时，同一提交必须更新相关文档。

### `maintainer/`

- `PROJECT_CONTEXT.md`：维护者与 AI 助手使用的内部项目上下文，不属于最终用户阅读路径。
- [发布前检查清单](RELEASE_CHECKLIST.md)：仓库可发布边界、验证命令和当前 blocker。
