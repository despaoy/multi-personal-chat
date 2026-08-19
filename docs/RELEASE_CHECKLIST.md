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

## 验证命令

```bash
# 后端
python -m pytest backend/tests -q
python -m compileall -q backend scripts astrbot_plugins

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

- `game_train` 上下文质量复审已批准并晋升。
- V4 canonical manifest 状态为 `frozen`，无 freeze blocker。
- R1V4 仍须以本次工作树实际训练门禁和测试结果为准，未运行的实验不得写成正式结论。

## 最新验证记录

- 后端：`python -m pytest backend/tests -q` → 610 passed, 13 skipped（2026-08-19 工作树）
- 前端：`pnpm ts-check`、`pnpm lint`、`pnpm build` 均通过；`pnpm audit --prod` 无已知漏洞（2026-08-16 工作树）
- Python：`python -m compileall -q backend scripts astrbot_plugins` 通过
- 训练门禁：`passed=true`，无 blocker（2026-08-19 工作树）
