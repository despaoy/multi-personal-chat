# 发布前检查清单

> 本文定义仓库进入“可发布”状态时需要满足的最低事实和命令验证结果。
> 最新一次完整验证：见文末。

## 发布边界

**仓库内发布：** 源码、配置模板、可审计角色语料、canonical 数据与审核证据、文档、测试和部署编排。

**仓库外保留（Git 忽略）：** 模型权重、LoRA checkpoint、数据库、日志、向量索引、`.env` 密钥、`runtime/`、`node_modules/`、个人 PPT 和编辑器缓存。

## 发布前必须为真

1. 不包含真实 `JWT_SECRET`、`ENCRYPTION_KEY`、API key 或 SSH 私钥。
2. `README.md` 中的技术基线、数据计数和门禁状态与当前事实一致。
3. 数据统计以 `backend/data/character_dialogues/experiments/v4/canonical_dataset_manifest.json` 为权威。
4. 活动脚本不包含个人机器路径；实验室路径一律使用 `MULTIPERSONAL_LAB_ROOT` / `MULTIPERSONAL_REMOTE_*` 环境变量。
5. 历史脚本和数据位于 `scripts/archive/` 与 `experiments/archive/`，并具有 README 说明。
6. 未把 mock 输出、旧版实验指标或未通过的训练结果表述为正式结论。
7. `python scripts/check_repository_integrity.py` 通过：冻结哈希/计数、API 挂载、前端导航、脚本索引、README 链接、归档和生成物边界一致。

## 验证命令

```bash
# 后端
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

- 远端基线：本地 `206e523` 与远端 `codex/v4-mainline-repair` 一致；本节结果针对其上的收尾工作树。
- 仓库完整性：通过；全部 README 本地链接有效，20 个 FastAPI 路由模块均挂载，15 个管理页均进入导航，31 个归档源文件均有大小和 SHA-256（生成的 `pyc` 不归档）。
- 后端：`python -m pytest backend/tests -q` → **1457 passed, 12 skipped**；4 条警告均为第三方弃用提示。
- Python：`compileall`、Alembic head（`006_character_memory`）和 `pip check` 均通过。
- API 链路：本地 mock 冒烟通过；无 Redis/vLLM 时验证了受控降级路径。
- 训练门禁：`passed=true`，`blockers=[]`。
- 前端：TypeScript、ESLint、Next.js 16.2.11 生产构建均通过；构建生成 29 个页面/路由。
- 依赖安全：`pnpm audit --prod` → `No known vulnerabilities found`。
- 清洁度：验证生成的 `.next/`、`node_modules/`、TypeScript 缓存和 pytest 临时目录已删除；它们均可按锁文件重建。
