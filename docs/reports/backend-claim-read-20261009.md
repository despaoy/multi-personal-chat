# 记忆准确读取接口：第二十阶段（2026-10-09）

基线 b5240e16e59b09dacc8b03fefd3447fa7a23de21。服务器先修改验证，部署后同步本地及 GitHub。

## 问题和修改

`repositories/character_memory.py` 的列表接口会在缺少层级读取方法时退回旧列表方法，不能完整表达 scope_levels/include_inactive；按 ID 读取缺少方法时扫描前 500 条，未命中可能被误当作记录不存在。正式 SQLite/PostgreSQL 均已提供准确接口。

- 直接调用 list_character_memory_claims/get_character_memory_claim，缺少接口明确抛错，不扫描有限列表假装准确查询。
- 删除仓储第二遍 active 过滤。两种数据库都在实际查询处依据 include_inactive 过滤状态，仓储继续负责记录解码和输入范围校验。
- 来源窗口直接调用正式 memory_source_windows，删除多余接口探测。

业务代码净减少 24 行，无新增依赖、配置、数据迁移或 SQL 变更。正常无匹配仍返回空列表/None，权限拒绝的 ID 查询仍返回 None；基础设施故障与接口缺失不会被转换为无匹配。

## 验证

37 项相关测试一次通过，包括新增正式接口缺失不回退、实际 SQLite 的 active/pending/superseded 状态、历史 ID 查询和用户隔离，以及现有版本化记忆、来源窗口和结构化证据权限测试。

额外在既有隔离 PostgreSQL 测试库创建独立随机作用域的合成版本与待确认记录，验证 active 列表、全历史列表、历史 ID 查询、跨用户拒绝和真实无匹配，随后清除该作用域合成记录。未读写生产用户资料，不把这次接口验证称为全量 PostgreSQL 回归。

无真实模型调用，未跑全量回归。语法、同配置 Ruff 增量、Git 差异检查通过；部署核对后端就绪、其他服务、运行配置及文件哈希。整体优化目标继续；前阶段明确报告的旧键名语义绑定限制未在本阶段解决。暂不开放公网。
