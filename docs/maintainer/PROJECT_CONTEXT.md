# 维护者项目上下文

> 本文件是维护者和 AI 助手进入项目时的内部事实入口，不面向最终用户。
>
> 更新时间：2026-08-19。详细实验导航见 [月社妃实验总览](../research/KISAKI_EXPERIMENT_INDEX.md)。

## 1. 项目定位

MultiPersonal Chat System 是面向角色对话研究与保研展示的多平台 LLM 系统，覆盖数据治理、LoRA 微调、AWQ 推理、混合 RAG、评测、AstrBot 网关和 Web 管理台。

- 定位：单机可部署、证据驱动的研究原型。
- 主线：数据 -> 训练 -> 推理 -> 检索 -> 评测 -> 多平台交付。
- 目标：展示对 LLM 数据、训练、推理、评测和工程部署完整链路的理解。
- 边界：个人研究项目，不以堆叠云原生组件为目标。

## 2. 技术基线

| 模块 | 当前实现 |
|---|---|
| 基础模型 | 官方 `Qwen/Qwen3-8B`，本地别名 `Qwen3-8B-Instruct` |
| 量化推理 | 官方 `Qwen/Qwen3-8B-AWQ`，本地别名 `Qwen3-8B-Instruct-AWQ` |
| 训练 | PyTorch 2.8、Transformers 4.57、PEFT、TRL |
| 后端 | FastAPI + Python 3.12 目标环境 |
| 前端 | Next.js 16 + React 19 + Node.js 22 |
| RAG | BGE-M3、FAISS/BM25 Hybrid、可选 Reranker |
| 平台 | AstrBot 网关，接入 QQ、微信系和 Telegram |
| 数据 | SQLite 本地开发；PostgreSQL 生产推荐 |
| 缓存 | Redis 可选；不可用时受限降级到进程内实现 |

## 3. 实验室服务器

实验室根目录通过 `MULTIPERSONAL_LAB_ROOT` 注入；部署根不写入仓库。

| 用途 | 路径或状态 |
|---|---|
| 项目 | `$MULTIPERSONAL_LAB_ROOT/multi-personal-chat` |
| 正式训练环境 | `$MULTIPERSONAL_LAB_ROOT/envs/qqchat-gpu-qwen3`，Python 3.11 |
| 后端测试/Embedding 环境 | `$MULTIPERSONAL_LAB_ROOT/envs/qqchat-gpu`，Python 3.10 |
| 基础模型 | `$MULTIPERSONAL_LAB_ROOT/runtime/models/Qwen3-8B-Instruct` |
| Embedding | `$MULTIPERSONAL_LAB_ROOT/runtime/models/bge-m3` |
| LoRA 输出 | `$MULTIPERSONAL_LAB_ROOT/runtime/loras/kisaki/canonical/` |
| 实验结果 | `$MULTIPERSONAL_LAB_ROOT/runtime/experiments/kisaki/` |
| 日志 | `$MULTIPERSONAL_LAB_ROOT/runtime/logs/` |
| GPU | 2 x RTX 3090，共享资源，只等待空闲，不抢占其他进程 |

运行时模型、checkpoint、日志、数据库和向量索引位于 `runtime/`，不得提交 Git。

## 4. 月社妃 R0V4/R1V4

### 4.1 当前状态

| 项目 | 当前事实 |
|---|---|
| 人物画像 | 已确认 |
| system prompt v3 | 已批准，正式使用策略为 `replace` |
| V4 train | 当前 948 条；522 条原作 + 150 条既有构造 + 276 条经审核晋升的 V4.1 五轮会话（DeepSeek round06 4 条 + Codex 自动化批次 272 条） |
| V4 validation | 已冻结 70 条独立原作数据 |
| Gold v2.1 | 150 条，已批准为 development only |
| Gold v3 | 150 条最终盲测，已审核并冻结 |
| 实验预检 | Game Train 复审已完成；运行正式门禁确认当前环境 |
| R1V4 seed 42 | 门禁通过后可启动 |
| 正式结论 | 尚未形成 |

### 4.2 严格对照

R1V4 固定数据、验证集、基座模型、seed、LoRA r/alpha、target modules、训练预算和评测参数。E1 是标准 LoRA；E2 只启用 NEFTune；E3 只启用 DoRA；E4 只启用 RSLoRA；E5 只启用 Sequence Packing。V4 配置只能在数据冻结后生成至 `experiments/v4/configs/`。

### 4.3 历史实验

旧 E1、E2、E2' Safety++ 和 E2'' RAG 不进入 V4 结论。为避免仓库内形成第二套入口，旧数据、脚本和结果已从工作树移除，需要时从 Git 历史追溯。

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

完整依据见 [人物画像](../research/KISAKI_CHARACTER_PROFILE.md)。

### 5.2 数据治理

- 当前来源审计覆盖 1,598 条原作直接台词，并保留来源定位。
- 训练/验证按完整对话切分，防止同一上下文跨集合泄漏。
- Gold v2.1 是已批准的开发评测集，Gold v3 是已冻结的最终盲测集；二者均不进入训练。
- 精确/文本相似泄漏直接阻断，相似度不低于 0.90 进入复核。
- RAG 角色正文保持自然语气，证据放入结构化 `citations`。
- 合成数据必须标记来源和审核状态，不能冒充原作台词。

## 6. 实验状态定义

- `human_review`：审核尚未完成，训练门禁必须阻塞。
- `frozen`：内容和哈希均已冻结。
- `training_complete`：训练运行成功并生成 adapter。
- `automatic_evaluation_passed`：自动指标与质量门通过。
- `blind_review_complete`：匿名 A/B 人工盲评完成。
- `conclusion_ready`：多随机种子、自动评测和人工盲评证据完整。

“训练完成”不等于“效果更好”。正式报告必须同时保留失败样本、均值、标准差、延迟和显存。

## 7. 常用命令

### 本地/服务器测试

```bash
python -m pytest backend/tests -q
python scripts/validate_kisaki_v4_training_gate.py
```

### 正式预检

```bash
export MULTIPERSONAL_LAB_ROOT=/path/to/lab-root
python scripts/validate_kisaki_v4_training_gate.py --disk-path "$MULTIPERSONAL_LAB_ROOT"
```

### 门禁通过后的 seed 42

```bash
python scripts/run_kisaki_experiment.py --experiment e1 --seed 42
```

E2-E5 及补充随机种子只能在 E1 全链路验收后按实验注册表顺序运行。

## 8. 文档导航

| 入口 | 用途 |
|---|---|
| [项目 README](../../README.md) | 系统能力、结构与部署入口 |
| [文档中心](../README.md) | 全部文档分类 |
| [月社妃实验总览](../research/KISAKI_EXPERIMENT_INDEX.md) | E1/E2、Gold、脚本和历史结果统一索引 |
| [V4 审核与重训练](../research/KISAKI_V4_HUMAN_REVIEW_AND_RETRAINING.md) | 当前数据、门禁和训练入口 |
| [实验资产 README](../../backend/data/character_dialogues/experiments/README.md) | 数据、配置和结果目录 |
| [脚本索引](../../scripts/README.md) | 活动脚本、历史脚本归档与实验室环境变量 |
| [发布前检查清单](../RELEASE_CHECKLIST.md) | 可发布边界、验证命令与当前 blocker |
| [服务器布局](../operations/SERVER_LAYOUT.md) | 源码与运行资产边界 |
| [研究路线图](../research/RESEARCH_AND_LEARNING_ROADMAP.md) | 后续 LLM 学习与实验方向 |

## 9. 工作规则

1. 先读取本文件和对应模块 README，再修改代码或实验资产。
2. 不覆盖用户未提交且与任务无关的改动。
3. 不将真实密钥、账号、日志、模型和 checkpoint 提交 Git。
4. 正式实验必须在干净、已同步的 commit 上运行。
5. 服务器共享 GPU 只等待空闲，不停止、不抢占现有进程。
6. 历史报告保留实际条件，不改写为当前实验结果。
7. 项目事实变化时，同一提交更新本文件和实验索引。

## 10. 最近验证

当前重构后的完整测试结果以本次工作结束时的验证报告为准。任何历史测试计数、旧 commit 和旧 `ready_for_training` 状态都不再代表 V4 当前状态。
