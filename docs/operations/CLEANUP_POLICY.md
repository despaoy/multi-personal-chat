# 文件清理策略

目标是保持源码、运行数据和研究证据相互独立，同时避免误删不可恢复的训练资产。

## 可直接清理

- Python：`__pycache__/`、`*.pyc`、`.pytest_cache/`。
- Next.js/TypeScript：停止前端后可清理 `.next/`、`*.tsbuildinfo`。
- 临时文件：`runtime/tmp/*`、`_output.txt`、`_temp_*.py`。
- 失效 PID：对应进程不存在时的 `*.pid`。
- 已完成模型下载留下的空 `._____temp/`。
- 本地 pnpm store；删除后会重新下载依赖。

## 仅在确认后清理

- `node_modules/`：可重建，但离线服务器应保留。
- `conda-pkgs/`：环境可继续运行，但离线重建会失去缓存。
- `runtime/logs/`：先归档与实验报告有关的日志。
- 训练 checkpoint：确认 `final/`、训练配置、指标和最佳 checkpoint 已保留后再删。
- 旧模型和旧 LoRA：迁移到 `runtime/archive/`，记录基座模型和兼容性后再决定删除。

## 永不自动清理

- `.env`、数据库、原始数据、Gold Set、人工标注、偏好数据。
- `runtime/models/` 中唯一的模型副本。
- `runtime/loras/*/final/`。
- 未提交或尚未归档的实验代码、评测结果和训练日志。

## 清理前检查

```bash
git status -sb
find "$MULTIPERSONAL_LAB_ROOT/multipersonal-chat-system" -type d \
  \( -name __pycache__ -o -name .pytest_cache \) -print
find "$MULTIPERSONAL_LAB_ROOT"/runtime -name '*.pid' -type f -print
du -sh "$MULTIPERSONAL_LAB_ROOT"/*
```

正在运行的前端需要 `.next` 和 `node_modules`；正在训练或推理时不能删除当前日志、PID、checkpoint、模型或 adapter。

## 目录迁移规则

- 移动而非直接删除旧资产。
- 在 `runtime/archive/<date>-<reason>/` 中保留说明或清单。
- 修改路径后立即检查 `.env`、启动脚本和软链接。
- 完成健康检查和一次真实推理后，才删除迁移前副本。