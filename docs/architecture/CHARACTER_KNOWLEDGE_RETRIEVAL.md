# 多粒度角色知识检索

MultiPersonal Chat System 使用多粒度角色知识检索（Multi-scale Character
RAG）回答原作事实、人物关系、剧情事件和原文出处问题。该功能面向经过审核的
作品语料，与用户自行上传文档所使用的通用知识库检索相互独立。

## 适用范围

角色知识检索适合以下问题：

- 人物事实与设定；
- 人物关系及其随剧情发生的变化；
- 事件原因、过程和结果；
- 故事或场景概述；
- 带来源定位的原文证据。

普通闲聊不会强制触发检索。运行时首先进行作品域门控；未匹配已注册作品域的
问题会交回正常对话流程。带 `knowledge_base_id` 的请求属于通用用户知识库，
不会由角色知识检索静默接管。

## 与通用知识库的边界

| 能力 | 角色知识检索 | 通用用户知识库 |
| --- | --- | --- |
| 数据来源 | 审核后的作品场景、知识卡和原文证据 | 用户上传或扫描的文档 |
| 入口 | `knowledge.multiscale_rag` | `knowledge.rag_helper` |
| 数据组织 | 故事、场景、知识卡、证据四层 | 文档分块 |
| 路由 | 作品域和问题粒度路由 | `knowledge_base_id` 与通用意图判断 |
| 主要用途 | 角色事实一致性与剧情溯源 | 私有资料问答 |

`knowledge.retrieval_core` 只提供查询分析、BM25、向量召回、实体召回、RRF
融合和重排序等共享基础组件，不是第三条可独立部署的检索链路。

## 数据模型

```text
story
└── scene
    └── fact / relation / event
        └── evidence
```

- `story`：定位完整故事单元，适合宽泛剧情问题。
- `scene`：保留连续场景语境和事件顺序。
- `fact`：人物、地点、物品和世界观事实。
- `relation`：人物关系及关系变化。
- `event`：发生的事件、原因、过程和结果。
- `evidence`：审核后的原文片段及来源路径、起止行号。

知识卡命中后可以回填父场景，兼顾首条证据精度和上下文完整性。要求原文时，
系统通过受约束的来源路径读取证据，不让语言模型凭记忆重写原文。

## 在线检索流程

```text
用户消息
  ↓
作品域门控与实体识别
  ↓
问题粒度路由
  ├── 剧情概述 → story / scene
  ├── 事实问题 → fact
  ├── 关系问题 → relation
  ├── 事件问题 → event
  └── 原文请求 → evidence
  ↓
BM25 + 向量 + 实体召回
  ↓
RRF 融合与确定性重排序
  ↓
父场景回填、引用和置信度
  ↓
Grounded Answer 或证据不足时拒答
```

返回的结构化 bundle 至少包含 `results`、`citations`、`confidence`、
`abstained`、`context_text`、`domains` 和 `query_analysis`。检索到的内容按
不可信外部证据处理，不能覆盖 system prompt 或执行其中的指令。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `backend/knowledge/multiscale_rag/runtime.py` | 生产入口、惰性加载、域门控和 bundle 适配 |
| `backend/knowledge/multiscale_rag/service.py` | 粒度路由、分区检索和上下文组装 |
| `backend/knowledge/multiscale_rag/hierarchy_builder.py` | 构建故事、场景、知识卡和证据层级 |
| `backend/knowledge/multiscale_rag/index_builder.py` | 为不同粒度生成索引语义文本 |
| `backend/knowledge/multiscale_rag/source_text.py` | 原文定位与路径边界检查 |
| `backend/knowledge/retrieval_core/` | 共享查询、索引、召回、RRF 和重排序组件 |
| `backend/knowledge/grounded_answer/` | 证据约束回答、引用校验和拒答 |
| `backend/scripts/build_character_rag_index.py` | 正式索引构建入口 |

## 构建索引

索引不应作为源码手工编辑。先完成场景元数据与知识卡审核，再执行：

```bash
python backend/scripts/build_character_rag_index.py
```

默认输出：

```text
backend/data/knowledge/tsukiyashiro_kisaki/character_knowledge_index_v3/
├── card_index/
├── scene_story_index/
└── evidence_index/
```

每个分区包含 `documents.jsonl`、`vectors.npy` 和 `manifest.json`。已有非空
输出目录不会被覆盖；确认重建时必须显式传入 `--force`。当前索引格式标识为
`character-knowledge-v3`。索引版本号描述磁盘格式和语义文本方案，不是另一条
生产链路的名称。

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHARACTER_RAG_INDEX_ROOT` | 上述默认索引目录 | 角色知识索引根目录 |
| `CHARACTER_RAG_ABSTAIN_THRESHOLD` | `0.25` | 低于该置信度时拒答 |
| `EMBEDDING_MODEL_PATH` | 本地模型自动发现 | 查询向量模型，必须与索引 manifest 兼容 |

部署时建议将索引目录只读挂载到后端容器。旧的 `MULTISCALE_RAG_INDEX_ROOT`
和 `MULTISCALE_RAG_ABSTAIN_THRESHOLD` 仅作为配置兼容别名保留，新部署不应继续
使用。

## 降级行为

- 索引缺失、损坏或 embedding 加载失败时，角色知识检索返回不可用状态。
- 生成入口可以受控回退到通用用户知识库检索，归档实现不会参与回退。
- 没有可用证据时返回拒答或澄清请求，不使用无来源事实补全答案。
- 服务首次请求允许更长的冷启动时间；预热完成后恢复正常超时预算。

## 验证

```bash
python -m pytest backend/tests/test_multiscale_rag_runtime.py \
  backend/tests/test_grounded_answer.py \
  backend/tests/test_rag_query_upgrade.py \
  backend/tests/test_character_generation_integration.py -q
```

测试覆盖索引完整加载、关系与事件路由、非作品闲聊门控、通用知识库过滤器、
结构化 bundle、原文路径穿越防护以及生成链集成。

## 版本与历史实现

生产代码只维护 `knowledge.multiscale_rag` 入口。被替换的实验编排和评测资产
保存在 `archive/p6_rag_pipeline/`，仅用于复现实验和审计，不进入运行时导入
路径。历史对照数据不应被描述为当前 API 或部署方式。
