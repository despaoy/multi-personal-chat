# 可扩展性开发指南

本文记录当前项目已经建立的扩展边界，以及新增功能时应遵循的最短路径。目标是在保持现有 API 兼容的同时，逐步减少路由层、数据库实现和页面组件之间的直接耦合。

## 1. 当前分层

后端新代码应优先遵循以下调用方向：

```text
FastAPI route
    ↓ 只处理 HTTP 参数、鉴权和错误映射
Application service
    ↓ 编排业务流程，不依赖 FastAPI
Repository Protocol
    ↓ 描述领域所需的持久化能力
Database adapter
    ↓ SQLite / PostgreSQL 兼容实现
```

关键入口：

- `backend/app/main.py:create_app`：应用组合根，负责装配 FastAPI、路由、中间件和生命周期。
- `backend/app/runtime.py:RuntimeContainer`：一个应用实例拥有的数据库、推理运行时和启动环境。
- `backend/app/providers.py`：FastAPI 依赖提供器集中入口，将运行时容器适配为 Service 或 Repository。
- `backend/services/`：与 HTTP 传输无关的业务用例。
- `backend/repositories/`：持久化 Protocol、查询对象以及数据库适配实现。

旧 API 模块仍有部分 `db` 全局导入。这是兼容层，不应作为新模块的实现模板；迁移时按领域逐个替换。

## 2. 新增后端功能

### 新增业务用例

1. 在 `backend/services/` 定义服务类，其构造参数只接收 Protocol、函数或明确的配置值。
2. 不要在服务中读取 `Request`、返回 `Response` 或抛出仅属于 HTTP 的异常。
3. 在 `backend/app/providers.py` 从 `RuntimeContainer` 构造服务。
4. 路由通过 `Depends(...)` 获取服务，只负责输入输出和 HTTP 错误映射。
5. 为服务使用 fake 依赖编写单元测试，再补一条路由注入测试。

聊天生成可参考 `ChatGenerationService`：安全策略、Trace ID、限流和排队均可替换，现有 `generate_reply_core` 只保留为兼容入口。

### 新增数据库操作

1. 先在对应 Repository `Protocol` 中声明领域操作，避免把通用 `execute_sql` 暴露给服务层。
2. 使用查询对象封装筛选条件，避免路由和数据库方法之间传递不断增长的参数列表。
3. 同步数据库调用必须在 Repository 内通过 `asyncio.to_thread` 执行，不能阻塞事件循环。
4. provider 必须从当前 `request.app.state.runtime_container` 取数据库，不要重新导入全局数据库单例。
5. SQLite 和 PostgreSQL 的 schema 以 `backend/db/models.py` 与 Alembic 迁移为演进基线；兼容旧 SQLite 数据时再保留显式迁移逻辑。

消息历史领域可参考 `MessageRepository`、`MessageQuery` 和 `DatabaseMessageRepository`。

## 3. 新增前端接口

前端接口变更应按以下顺序进行：

1. 在 `src/lib/api-contracts.ts` 定义共享请求、响应和领域记录类型。
2. 在 `src/lib/api.ts` 中仅实现 HTTP 调用，并复用共享契约；不要在方法签名中重复大型内联类型。
3. Hook 负责请求状态和缓存，页面组件只处理交互与展示。
4. 大型页面的新功能优先放入 `src/components/<domain>/`，通过明确 props 接入。
5. 后端字段变化时同时更新 Pydantic schema、共享前端契约和至少一条契约/静态检查。

`unknown` 应只出现在真正不可预知的 JSON 边界。读取前必须通过类型守卫收窄，不能使用 `any` 绕过检查。

## 4. 兼容性规则

- 对外 URL、字段名或模块级调用入口已有使用者时，先保留薄兼容函数，再迁移调用方。
- 不在一次重构中同时改变业务行为、数据格式和分层结构。
- 新依赖先通过构造参数或 provider 注入，不在模块导入阶段创建不可替换的网络客户端。
- 应用实例相关状态放入 `RuntimeContainer` 或 `app.state`，不要新增无界模块级可变字典。
- 新的路由策略统一登记在 `src/lib/proxy.ts`，避免白名单和超时规则分散。
- 新的页面持久化统一使用 `usePageData`，缓存键必须包含用户和页面维度。

### 扩展角色知识域

1. 在 `knowledge.retrieval_core.registry` 注册知识域、别名、来源根目录和叙事策略。
2. 将原始资料转换为经过审核的 story、scene、fact/relation/event 和 evidence 文档，不直接修改生成的向量文件。
3. 复用 `knowledge.multiscale_rag` 的结构化 bundle 契约；不要复制 runtime 创建新的版本号链路。
4. 为新知识域建立独立索引目录，并通过 manifest 记录索引格式、embedding 模型和文档数量。
5. 新增域门控、关系/事件路由、原文路径边界和无证据拒答测试。

通用用户文档仍进入 `rag_helper`/`vector_db` 路径。不要把用户上传资料写入角色作品索引，也不要让角色检索忽略 `knowledge_base_id`。

## 5. 验证清单

后端：

```powershell
py -3.12 -m pytest backend/tests -q --basetemp C:\tmp\qqchat-pytest
py -3.12 -m py_compile backend/app/main.py backend/app/runtime.py
```

前端：

```powershell
npm run ts-check
npm run lint
npm run build
```

提交前还应运行：

```powershell
git diff --check
```

涉及部署文件时，额外验证 Compose YAML、Nginx 配置和 `deploy/verify.sh` 语法；涉及数据库 schema 时，确认 Alembic 只有一个 head。

