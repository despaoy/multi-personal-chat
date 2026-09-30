# r140：记忆管理包装与事实读视图

## 原因及修复

r139 数据库已保存 user_name，但证据“请记住，我叫夏澄。”未通过读侧完整自述识别，降为 observation，姓名存在性为 unknown。并非模型字段分类错误。

新增共享 memory_statement_body，仅移除一次开头的肯定“请记住/记下”包装。路由与 temporal_projection 共用；正文仍须通过完整主体、时态、条件及抽取匹配。不更改原始证据或数据库，不修改人物提示词，不增加模型调用。

## 验证

- 包装 × 姓名/住址/专业/工作地点，以及否定、假设、未来、引述、他人、后续限定负例；另覆盖召回→字段存在性→保存状态及姓名响应。
- 扩大本地回归 2338 passed、1 skipped、1902 deselected、4 依赖警告，147.38 秒；`.tmp/r140-wide.xml`。聚焦测试与其重叠，不累计。定向 Ruff 与 git diff --check 通过。
- 隔离服务器真实 API 6 轮均 HTTP 200，无 LoRA；19 次 DeepSeek 请求：6 policy、4 answer、5 writer、4 memory_selection。无本地生成，semantic_review 未触发，不能称其已通过实测。
- 姓名与工作地点读视图 asserted_state、存在性 true；保存状态直读成功，姓名混合人物问答同时回答保存姓名与兄妹关系，工作地点追问正确。个人任务不请求外部 RAG，混合人物任务保留 RAG。
- 证据：`.tmp/r140-manifest.json`、`.tmp/r140-traces.jsonl`。身份及准入采用测试适配器、数据库为隔离 SQLite，不代表生产鉴权/限流或 PostgreSQL 已验收。模型归因以 provider 记录为准，响应旧模型标签并非真实提供商。

## 边界

仅验证上述迁移场景，不代表三个模块完成。仍有 source-only 删除、自然概括准入、暖历史复述、动态上下文消融、RAG 证据充分性及生命周期问题。远端仅隔离快照，生产未修改，SSH 已关闭，持续目标保持 active。
