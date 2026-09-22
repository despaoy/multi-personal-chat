# 文档维护记录：2026-09-11

## 范围与方法

核对开始时docs共158个文件：129个Markdown、27个JSON、1个JSONL、1个HTML。对全部文档进行目录分类、文本读取及本地Markdown链接扫描，对JSON/JSONL检查解析，对HTML做只读解析。对活动说明按入口源码、配置和manifest复核；历史批次仅做导航与一致性风险扫描，不重新人工评分或批准。

本次未连接远程服务器，未运行真实vLLM、GPU、PostgreSQL、Redis或AstrBot验收。日期只表示文档核对，不替换历史测试日期。工作区已有6个记忆实现/测试文件修改以及一个未跟踪历史RAG报告，本次不改动它们。

## 活动文档逐项处理

| 文档 | 处理 |
| --- | --- |
| README.md | 增加当前事实和历史审核记录的阅读分流 |
| RELEASE_CHECKLIST.md | 区分历史完整验证与当前工作区；保留旧日期和旧head测量记录 |
| architecture/CODE_WIKI.md | 重写为可维护的目录和调用链说明，移除过时接口计数和冗长函数签名列表；更新008迁移、回执、LoRA选择、记忆与训练参数口径 |
| architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md | 更新聊天角色化弃答与独立证据问答的区别 |
| architecture/EXTENSIBILITY_GUIDE.md | 保留分层契约，统一前端pnpm命令 |
| architecture/OPTIMIZATION_STRATEGY.md | 修正路径映射、双知识库、发送重试与单worker约束；明确优化目标不等于已验收 |
| architecture/PRODUCTION_READINESS_REVIEW_2026-07-18.md | 标明历史审查，正文及当时结论保留 |
| data/human-scoring-rubric.md | 修正强制口癖/笑声禁用等人物评价偏差，区分通用105题与月社妃Gold v3，标记示例非实测 |
| maintainer/PROJECT_CONTEXT.md | 更新阅读入口、归档现状和服务器历史配置边界 |
| operations/DEPLOYMENT_GUIDE.md | 改用run.py，补迁移升级、单worker、发送回执和弃答验收 |
| operations/SERVER_LAYOUT.md | 将具体环境版本标为历史记录参考，目录规范保留 |
| operations/CLEANUP_POLICY.md | 保留；保护数据库、模型、审核与未归档数据的规则仍适用 |
| research/BEGINNER_REAL_LLM_EXPERIMENT_GUIDE.md | 删除重复命令和缺失历史报告路径；区分数据/模型门禁，修正共享GPU处置 |
| research/KISAKI_CHARACTER_PROFILE.md | 保留已批准人物画像；以其修正评分指南，不重新解释原作 |
| research/KISAKI_EXPERIMENT_INDEX.md | 修正“只保留V4”与归档说明，保留E1失败和E2-E5暂停状态 |
| research/KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md | 修正候选/审核包已删除的错误说明，明确数据冻结不等于adapter可发布 |
| research/RESEARCH_AND_LEARNING_ROADMAP.md | 更新为四条主线及当前起点，移除已过时的立即执行计划，区分SFT/LoRA/偏好训练和未来实验 |
| research/KISAKI_GAME_RAG_BASELINE_AUDIT.md | 标明历史阶段审计，数字和原结果保留 |
| research/KISAKI_GAME_RAG_FREEZE_READINESS.md | 标明P3.5/P3.6记录，批准和哈希保留 |
| research/KISAKI_GAME_RAG_PARSER.md | 标明阶段解析器记录，生产入口导向现行RAG文档 |
| research/KISAKI_GAME_RAG_SCENES.md | 标明阶段切分记录，避免旧“未冻结”描述成为当前状态 |
| research/KISAKI_GAME_RAG_SCENE_METADATA_CANDIDATE.md | 标明候选生成阶段记录，不把当时未调用LLM当作当前事实 |
| research/KISAKI_GAME_RAG_SCENE_METADATA_REVIEW.md | 标明审核阶段记录，批准契约与证据保留 |
| research/KISAKI_GAME_RAG_SCHEMA.md | 标明阶段schema记录，当前运行索引以架构文档和源码为准 |

新增CURRENT_STATE.md集中记录当前状态，architecture/CHARACTER_CONTEXT_AND_MEMORY.md补齐人物与两条记忆链路。新增review_packets/README.md并更新kisaki_v4/00_GUIDE.md的过时待办；其他逐条审核材料不修改。知识库与路线图的原全文仍可由Git历史追溯。

仓库完整性检查还发现backend/scripts/README.md遗漏现有build_legacy_chunk_baseline.py，已补充其独立历史对照用途。该修改仅为脚本索引补全。

## 验证与限制

- V4数据门禁：本次执行返回passed=true、blockers为空。模型质量门禁仍未通过，不能据此启动E2-E5或宣称adapter达标。
- 文档扫描：更新后共133个Markdown、27个JSON、1个JSONL、1个HTML；本地链接无失效项，JSON/JSONL解析通过。HTML只读解析，不重导出历史审核页面。
- 仓库完整性检查通过；git diff --check未发现空白格式错误（存在CRLF转LF提示，不影响检查结果）。
- 未重新运行完整后端/前端测试、模型训练、人物评分或生产验收。文档变化不产生新的模型效果数字。
- 记忆写入召回描述包含任务开始前已有的未提交实现。其全量扫描和embedding成本随数据规模增长，本次只说明实际行为，不证明规模性能。
