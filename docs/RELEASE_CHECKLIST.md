# 发布前检查清单

> 本文定义仓库进入“可发布”状态时需要满足的最低事实和命令验证结果。
> 最新一次完整验证：见文末。

## 发布边界

**仓库内发布：** 源码、配置模板、可审计角色语料、canonical 数据与审核证据、文档、测试和部署编排。

**仓库外保留（Git 忽略）：** 模型权重、LoRA checkpoint、数据库、日志、向量索引、原始评测报告、个人笔记、`.env` 密钥、`runtime/`、`node_modules/`、个人 PPT 和编辑器缓存。

## 发布前必须为真

1. 不包含真实 `JWT_SECRET`、`ENCRYPTION_KEY`、API key 或 SSH 私钥。
2. `README.md` 中的技术基线、数据计数和门禁状态与当前事实一致。
3. 数据统计以 `backend/data/character_dialogues/experiments/v4/canonical_dataset_manifest.json` 为权威。
4. 活动脚本不包含个人机器路径；实验室路径一律使用 `MULTIPERSONAL_LAB_ROOT` / `MULTIPERSONAL_REMOTE_*` 环境变量。
5. 历史脚本和数据位于 `scripts/archive/` 与 `experiments/archive/`，并具有 README 说明。
6. 未把 mock 输出、旧版实验指标或未通过的训练结果表述为正式结论。
7. 生产文档只使用“多粒度角色知识检索”功能名；P6/P7 和实验 V3 不出现在活动 API、类名或部署入口中。
8. 生成的 `character_knowledge_index_v*/` 不进入 Git；发布部署能够按文档重建并只读挂载索引。
9. `python scripts/check_repository_integrity.py` 通过：冻结哈希/计数、API 挂载、前端导航、脚本索引、README 链接、归档和生成物边界一致。

## 验证命令

```bash
# 后端
python -m ruff check backend/api/ask.py backend/api/generate.py backend/api/knowledge.py backend/db/schemas.py backend/knowledge/multiscale_rag backend/knowledge/retrieval_core backend/knowledge/grounded_answer backend/scripts/build_character_rag_index.py
python -m ruff format --check backend/api/ask.py backend/api/generate.py backend/api/knowledge.py backend/db/schemas.py backend/knowledge/multiscale_rag backend/knowledge/retrieval_core backend/knowledge/grounded_answer backend/scripts/build_character_rag_index.py
python -m pytest backend/tests -q
python -m compileall -q backend scripts astrbot_plugins
python scripts/check_repository_integrity.py

# 前端
pnpm ts-check
pnpm lint
pnpm build

# 研究门禁（数据或审核状态变化后必须重新执行）
python scripts/validate_kisaki_v4_training_gate.py

# 单命令本地基线（Windows）
powershell -ExecutionPolicy Bypass -File scripts/local-verify.ps1 -Frontend
```

## 当前发布状态

- 源码、文档、冻结数据和归档索引已完成 2026-08-28 收尾审计，可作为源码封存候选。
- V4 canonical manifest 状态为 `frozen`：926 train / 70 validation；训练门禁 `passed=true`、无 blocker。
- R1V4 现有 E1 与 recovery checkpoint 均未通过生成/语义门禁，当前无正式 adapter，E2-E5 暂停。
- 本地环境未提供真实 Redis、vLLM、GPU、PostgreSQL 或 AstrBot 账号；这些属于部署环境验收，不得从本次源码验证推断为已通过。

## 最新验证记录（2026-08-28）

- 远端基线：收尾提交已合入 `main`；验证时本地工作区与 `origin/main` 一致。
- 仓库完整性：通过；全部 README 本地链接有效，20 个 FastAPI 路由模块均挂载，15 个管理页均进入导航，31 个归档源文件均有大小和 SHA-256（生成的 `pyc` 不归档）。
- 后端：`python -m pytest backend/tests -q` → **1457 passed, 12 skipped**；4 条警告均为第三方弃用提示。
- Python：`compileall`、Alembic head（`006_character_memory`）和 `pip check` 均通过。
- API 链路：本地 mock 冒烟通过；无 Redis/vLLM 时验证了受控降级路径。
- 训练门禁：`passed=true`，`blockers=[]`。
- 前端：TypeScript、ESLint、Next.js 16.2.11 生产构建均通过；构建生成 29 个页面/路由。
- 依赖安全：`pnpm audit --prod` → `No known vulnerabilities found`。
- 清洁度：验证生成的 `.next/`、`node_modules/`、TypeScript 缓存和 pytest 临时目录已删除；它们均可按锁文件重建。

## 角色知识检索迁移验证（2026-08-29）

- 生产入口统一为 `knowledge.multiscale_rag:MultiScaleRagRuntime`，共享基础组件统一为 `knowledge.retrieval_core`。
- 构建入口统一为 `backend/scripts/build_character_rag_index.py`，默认产物目录为 `character_knowledge_index_v3/`。
- 在线生成和 Grounded Answer 的相关回归测试为 **43 passed, 1 skipped**；真实关系查询返回结构化引用和上下文。
- P6 的 14 个归档代码文件均通过大小和 SHA-256 清单校验，活动代码不导入归档模块。
- 角色检索模块、公共知识 API 和索引构建入口已纳入 Ruff lint/format CI 门禁。
- Windows 与 Linux 的文本归档校验统一按仓库 LF 表示计算；仓库完整性检查通过。
- Python 3.10 完整后端测试为 **1407 passed, 16 skipped**；TypeScript、ESLint、Next.js 生产构建和生产依赖安全审计通过。

## 服务器上线核验（2026-08-28）

- 部署工作区：`/root/autodl-tmp/multi-personal-chat`，新鲜克隆并同步至 `main@e9ea119`；旧实验目录 `qqchat-enhanced` 的未提交代码、数据和评测报告保持原样，未被覆盖。
- 跨平台门禁：在 Linux 新鲜克隆中修正并验证归档文件 LF 哈希；`compileall` 后仓库完整性检查仍通过，生成的 `__pycache__` / `pyc` 不计入归档资产。
- 后端：Python 3.12.4，`1458 passed, 11 skipped`；`pip check`、Python 编译、训练门禁和 Alembic head `006_character_memory` 均通过。
- 前端：Node.js 22.23.1、pnpm 10.34.2；锁文件安装、TypeScript、ESLint、Next.js 生产构建及官方 npm 生产依赖审计通过，29 个页面/路由生成，无已知生产依赖漏洞。
- 真实推理：RTX 3090 上以 vLLM 0.10.2 成功加载 Qwen3-8B-Instruct BF16；模型列表、最小生成、FastAPI `/health` / `/ready` 均通过。
- 端到端：Next.js 入口与 API 代理、临时用户注册、管理员 Cookie 鉴权、真实生成、消息持久化查询和注销均通过；临时服务、数据库、Cookie 和响应文件已清理，GPU 与端口已释放。
- 当前生产阻断项：尚未配置 PostgreSQL/Redis、AstrBot 集成令牌与回调地址、正式域名 CORS、TLS/反向代理。旧裸机配置只能通过开发模式验证；在这些运营参数补齐前不得声明为公网生产就绪。
- 非阻断提示：裸机执行 `pnpm start` 可正常服务，但 Next.js 对 `output: standalone` 给出启动方式提示；Compose/Docker 镜像路径使用 standalone 产物，裸机长期运行应由正式进程守护配置明确启动命令。
