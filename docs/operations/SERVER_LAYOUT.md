# 服务器目录规范

适用范围：无 root 权限的实验室服务器，项目根目录固定为 `"$MULTIPERSONAL_LAB_ROOT"`。

## 规范布局

```text
"$MULTIPERSONAL_LAB_ROOT"/
├── activate_qqchat.sh          # 统一环境入口
├── multipersonal-chat-system/            # Git 源码及被忽略的分模式配置文件
├── envs/                       # 当前用户的 Conda/venv 环境
│   └── qqchat-py312/           # 推荐 Python 3.12 环境
├── tools/                      # Miniconda 等用户级工具
├── node/                       # Node.js 22 与 pnpm
├── conda-pkgs/                 # 离线可复用的 Conda 包缓存
└── runtime/                    # 所有可变数据
    ├── archive/                # 旧版本与迁移前快照
    ├── data/                   # 训练输入和导入数据
    ├── db/                     # SQLite 开发数据库
    ├── environment/            # 环境锁定和 pip freeze
    ├── hf-cache/               # Hugging Face/ModelScope 缓存
    ├── logs/                   # 服务与实验日志
    ├── loras/                  # LoRA 训练产物和可加载 adapter
    ├── models/                 # 基座、AWQ、Embedding、Reranker
    ├── rag/                    # 向量索引及知识库运行数据
    ├── results/                # 评测和基准结果
    └── tmp/                    # 可随时清理的临时文件
```

## 资产边界

| 资产 | 唯一位置 | 是否进 Git |
| --- | --- | --- |
| Python/TypeScript 源码 | `multipersonal-chat-system/` | 是 |
| 环境配置模板 | `.env.example` | 是 |
| 裸机后端实际配置 | `multipersonal-chat-system/backend/.env` | 否 |
| Compose 实际配置 | `multipersonal-chat-system/deploy/.env` | 否 |
| Next.js 本地配置 | `multipersonal-chat-system/.env.local` | 否 |
| 基座与量化模型 | `runtime/models/` | 否 |
| LoRA adapter | `runtime/loras/` | 否 |
| RAG 向量索引 | `runtime/rag/` | 否 |
| 数据库 | `runtime/db/` | 否 |
| 实验报告 JSON | `runtime/results/`；经筛选的报告可复制进仓库 | 视用途 |
| 日志/PID/临时文件 | `runtime/logs/`、`runtime/tmp/` | 否 |

## 当前推荐环境

```text
Python       3.12.13
PyTorch      2.8.0+cu128
vLLM         0.10.2
Transformers 4.57.6 (<5)
Node.js      22.14.0
pnpm         10.34.2
```

激活命令：

```bash
source "$MULTIPERSONAL_LAB_ROOT"/activate_qqchat.sh
```

激活脚本统一设置：`DATABASE_PATH`、`VECTOR_DB_PATH`、`LORA_PATH`、`VLLM_LORA_ROOT`、`AUDIT_LOG_DIR` 和 `HF_HOME`。

## 端口约定

| 服务 | 默认端口 | 说明 |
| --- | --- | --- |
| Next.js | 5000 | 管理台 |
| FastAPI | 8000 | 核心 API |
| vLLM | 8001 | 默认推理端口 |
| 实验 vLLM | 8002 | LoRA 对比实验使用，不写死到生产配置 |
| AstrBot | 6185 | 平台管理面板 |
| Redis | 6379 | 可选；不可用时退化到进程内缓存 |

## 禁止事项

- 不在 `"$MULTIPERSONAL_LAB_ROOT"` 之外创建、移动或删除项目文件。
- 不把模型、数据库或 LoRA 放入 Git 仓库。
- 不在服务运行时删除 `.next`、`node_modules`、当前日志或正在使用的 PID 文件。
- 不覆盖服务器未提交实验文件；同步前必须先提交、归档或逐文件比对。