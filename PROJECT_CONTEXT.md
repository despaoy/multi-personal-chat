# 项目上下文

> 本文件是 AI 助手和维护者进入项目时的当前事实入口。
>
> 更新时间：2026-07-31（HEAD `201dd2f`，工作区含分层架构与服务/仓储演进的未提交变更）。详细实验导航见 [月社妃实验总览](docs/research/KISAKI_EXPERIMENT_INDEX.md)。

## 1. 项目定位

QQChat Enhanced 是面向角色对话研究与保研展示的多平台 LLM 系统，覆盖数据治理、LoRA 微调、AWQ 推理、混合 RAG、评测、AstrBot 网关和 Web 管理台。

- 定位：单机可部署、证据驱动的研究原型。
- 主线：数据 -> 训练 -> 推理 -> 检索 -> 评测 -> 多平台交付。
- 目标：展示对 LLM 数据、训练、推理、评测和工程部署完整链路的理解。
- 边界：个人研究项目，不以堆叠云原生组件为目标。

## 2. 技术基线

| 模块 | 当前实现 |
|---|---|
| 基础模型 | Qwen3-8B-Instruct |
| 量化推理 | Qwen3-8B-Instruct-AWQ + vLLM 0.10.2 |
| 训练 | PyTorch 2.8、Transformers 4.57、PEFT、TRL |
| 后端 | FastAPI + Python 3.12 目标环境 |
| 前端 | Next.js 16 + React 19 + Node.js 22 |
| RAG | BGE-M3、FAISS/BM25 Hybrid、可选 Reranker |
| 平台 | AstrBot 网关，接入 QQ、微信系和 Telegram |
| 数据 | SQLite 本地开发；PostgreSQL 生产推荐 |
| 缓存 | Redis 可选；不可用时受限降级到进程内实现 |

## 3. 实验室服务器

所有操作限制在 `/home/szw/lhm2`。

| 用途 | 路径或状态 |
|---|---|
| 项目 | `/home/szw/lhm2/qqchat-enhanced` |
| 正式训练环境 | `/home/szw/lhm2/envs/qqchat-gpu-qwen3`，Python 3.11 |
| 后端测试/Embedding 环境 | `/home/szw/lhm2/envs/qqchat-gpu`，Python 3.10 |
| 基础模型 | `/home/szw/lhm2/runtime/models/Qwen3-8B-Instruct` |
| Embedding | `/home/szw/lhm2/runtime/models/bge-m3` |
| LoRA 输出 | `/home/szw/lhm2/runtime/loras/kisaki/canonical/` |
| 实验结果 | `/home/szw/lhm2/runtime/experiments/kisaki/` |
| 日志 | `/home/szw/lhm2/runtime/logs/` |
| GPU | 2 x RTX 3090，共享资源，只等待空闲，不抢占其他进程 |

运行时模型、checkpoint、日志、数据库和向量索引位于 `runtime/`，不得提交 Git。

## 4. 月社妃 Canonical E1/E2

### 4.1 当前状态

| 项目 | 当前事实 |
|---|---|
| 训练集 | 826 条，SHA256 记录于数据清单 |
| 固定验证集 | 92 条，E1/E2 共用 |
| Gold v2 | 150 条，五类各 30 条，已冻结 |
| 人工审核 | 150/150 approved |
| 文本泄漏审计 | passed |
| BGE-M3 语义泄漏审计 | passed，阈值 0.88，unresolved=0 |
| 实验预检 | ready_for_training |
| seed 42 | 等待源码提交同步后启动 |
| 正式结论 | 尚未形成 |

Gold v2 内容哈希：`1e3f6542bb46823c6074c6a64172a38b58942fd4f0dcdc491946bd4bfa4e4feb`。

### 4.2 严格对照

| 项目 | KISAKI-E1 | KISAKI-E2 |
|---|---|---|
| 方法 | 标准 LoRA | LoRA + NEFTune |
| `neftune_noise_alpha` | 0.0 | 5.0 |
| 数据、验证集、模型、种子 | 固定 | 与 E1 相同 |
| LoRA | r32、alpha64、7 modules | 与 E1 相同 |
| 训练轮数 | 3 | 3 |
| 生成参数 | temperature 0、thinking off | 与 E1 相同 |

配置差异检查必须证明唯一训练变量为 `neftune_noise_alpha`。

### 4.3 历史实验

旧 E1、E2、E2' Safety++ 和 E2'' RAG 保留为探索记录，状态统一为 `legacy_exploratory_non_comparable`。这些实验的数据量、提示词、评测脚本或生成参数存在变化，不进入新的 NEFTune 因果结论。

DoRA、RSLoRA、QLoRA、NEFTune 和 Sequence Packing 的后续研究应在当前 E1/E2 基线完成后另立实验 ID，每次只改变受控变量。

## 5. 人物与数据规则

### 5.1 人物画像

月社妃的稳定特征包括聪慧敏锐、自尊独立、克制深情、语言锋利、重视自主选择，以及在故事规则相关场景中的元叙事洞察。

- 日常回答通常简短，分析或情绪高潮可以更长。
- 反问、挖苦和反向表达是常见倾向，关键场景也可以坦率表达。
- “呼呼呼”“噗噗”“呵呵”“哈哈”“嘿嘿”均见于原作，使用频率和语境决定角色一致性。
- 琉璃是亲生哥哥和最重要的情感中心。
- 夜子是少数朋友和相似处境的共鸣者，同时存在冲突与竞争。
- 理央属于妃珍视的共同生活圈。
- 元叙事主要用于魔法之书、命运、角色和故事结构相关语境。

完整依据见 [人物画像](docs/research/KISAKI_CHARACTER_PROFILE.md)。

### 5.2 数据治理

- 原作直接台词必须保留来源定位。
- 训练/验证按完整对话切分，防止同一上下文跨集合泄漏。
- Gold v1 仅作开发集，Gold v2 不进入训练。
- 精确/文本相似泄漏直接阻断，语义相似度不低于 0.88 进入复核。
- RAG 角色正文保持自然语气，证据放入结构化 `citations`。
- 合成数据必须标记来源和审核状态，不能冒充原作台词。

## 6. 实验状态定义

- `ready_for_training`：数据、配置、模型与 Gold 契约通过。
- `training_complete`：训练运行成功并生成 adapter。
- `automatic_evaluation_passed`：自动指标与质量门通过。
- `blind_review_complete`：匿名 A/B 人工盲评完成。
- `conclusion_ready`：多随机种子、自动评测和人工盲评证据完整。

“训练完成”不等于“效果更好”。正式报告必须同时保留失败样本、均值、标准差、延迟和显存。

## 7. 常用命令

### 本地/服务器测试

```bash
python -m pytest backend/tests -q
python scripts/validate_kisaki_experiments.py
```

### 正式预检

```bash
python scripts/validate_kisaki_experiments.py --require-model --formal-eval
```

### seed 42 队列

```bash
bash scripts/lab-queue-kisaki-e1-e2.sh pilot
```

### seeds 43/44

seed 42 自动质量门和人工抽查通过后运行：

```bash
bash scripts/lab-queue-kisaki-e1-e2.sh replicate
```

队列使用原子锁和 GPU 连续空闲检查，只在 `/home/szw/lhm2` 写入运行资产。

## 8. 文档导航

| 入口 | 用途 |
|---|---|
| [项目 README](README.md) | 系统能力、结构与部署入口 |
| [文档中心](docs/README.md) | 全部文档分类 |
| [月社妃实验总览](docs/research/KISAKI_EXPERIMENT_INDEX.md) | E1/E2、Gold、脚本和历史结果统一索引 |
| [Canonical 实验设计](docs/research/KISAKI_E1_E2_CANONICAL_EXPERIMENT.md) | 正式研究问题与通过条件 |
| [实验资产 README](backend/data/character_dialogues/experiments/README.md) | 数据、配置和结果目录 |
| [服务器布局](docs/operations/SERVER_LAYOUT.md) | 源码与运行资产边界 |
| [研究路线图](docs/research/RESEARCH_AND_LEARNING_ROADMAP.md) | 后续 LLM 学习与实验方向 |

## 9. 工作规则

1. 先读取本文件和对应模块 README，再修改代码或实验资产。
2. 不覆盖用户未提交且与任务无关的改动。
3. 不将真实密钥、账号、日志、模型和 checkpoint 提交 Git。
4. 正式实验必须在干净、已同步的 commit 上运行。
5. 服务器共享 GPU 只等待空闲，不停止、不抢占现有进程。
6. 历史报告保留实际条件，不改写为当前实验结果。
7. 项目事实变化时，同一提交更新本文件和实验索引。

## 10. 最近验证

- 服务器后端测试：`121 passed, 3 warnings`。
- Gold v2 文本与 BGE-M3 语义泄漏审计：`passed`。
- 正式实验预检：`ready_for_training`，`errors=[]`。
- 当前源码提交基线：HEAD `201dd2f`（已推送至 origin/main）；工作区存在分层架构（services/repositories/app 运行时容器）演进的未提交变更，服务器运行前需先确认工作区状态再决定 `git pull` 或暂存本地变更。