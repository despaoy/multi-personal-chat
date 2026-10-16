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

聊天入口根据完整问题与可用历史的语义来源规划决定是否公共检索，再用对象绑定选择资料来源。独立知识问答未选域时也执行这条语义准备；无历史且明确选域时直接使用该域。已确认角色对象不会再被问法词表否决，单字域问法词表已删除。未知指定域、审核故障和所需来源故障明确失败，正常无匹配或证据不足保持弃答语义。

带 `knowledge_base_id` 的请求属于通用用户知识库，不会由角色知识检索静默接管。聊天支持分别处理角色和业务来源；独立知识问答当前只支持角色来源，遇到混合来源任务会明确报错。

## 与通用知识库的边界

| 能力 | 角色知识检索 | 通用用户知识库 |
| --- | --- | --- |
| 数据来源 | 审核后的作品场景、知识卡和原文证据 | 用户上传或扫描的文档 |
| 入口 | `knowledge.multiscale_rag` | `knowledge.rag_helper` |
| 数据组织 | 故事、场景、知识卡、证据四层 | 文档分块 |
| 范围选择 | 显式作品域或已绑定角色／故事对象 | `knowledge_base_id` 与已绑定业务对象／任务 |
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
用户消息与可用历史
  ↓
完整来源规划与对象绑定（无历史且明确选域时直接使用指定域）
  ↓
作品域与登记名称对齐
  ↓
故事、场景、事实、关系和事件统一候选索引
  ↓
BM25 + 向量 + 实体召回 → RRF 融合与重排序
  ↓
当前原文读取、完整证据预算与引用绑定
  ↓
Grounded Answer 或证据不足时弃答
```

登记名称对应随查询分析进入独立回答提示，仅用于对齐用户称呼与资料登记名，不能补充原作事实或授予权限。模型、检索与证据来源保持各自记录，复用审核结果不计作新模型调用。

返回的结构化 bundle 至少包含 `results`、`citations`、`confidence`、
`abstained`、`context_text`、`domains` 和 `query_analysis`。检索到的内容按
不可信外部证据处理，不能覆盖 system prompt 或执行其中的指令。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `backend/knowledge/multiscale_rag/runtime.py` | 生产入口、惰性加载、域门控和 bundle 适配 |
| `backend/knowledge/multiscale_rag/service.py` | 统一候选检索、对象范围与上下文组装 |
| `backend/knowledge/multiscale_rag/hierarchy_builder.py` | 构建故事、场景、知识卡和证据层级 |
| `backend/knowledge/multiscale_rag/index_builder.py` | 为各层文档生成统一检索语义文本 |
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
- 聊天入口在证据不足时保留人物设定与所选 LoRA，让模型自然表达无法确定；低置信候选和引用不进入该生成请求。
- 角色化弃答仍返回 `abstained=true`、`answerMode=abstention` 和空引用。`modelInvoked=true` 仅表示尝试生成弃答表达，不表示事实回答成功；异常或空回复使用固定兜底。
- 独立 Grounded Answer 接口继续执行自己的澄清或拒答策略。提示词限制不能替代真实模型的防编造评测。
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


## 逐任务的语义检索范围

角色检索在现有问题对象审核中同时声明每项任务的 `retrieval_scope`，不再通过
身份或关系关键词判断整段问题。`identity`、`independent_identity` 和
`direct_relation` 只约束已登记人物的结构化归属；其他任务使用模型明确给出的
`open`。多项任务取各自范围的并集，避免一个双人关系任务删去另一项导师、
成因或经过任务需要的原文。未解决的来源分支保留未知，不虚构对象授权。

主聊天和独立知识问答都要求该语义契约，包括已经选择角色知识域的请求。
缺少范围、非法枚举和不符合人数约束的审核结果直接失败，不能自动改成开放
检索。通用业务来源的名称绑定仍使用明确声明的名称契约。范围判断不证明
事实或权限；结构化访问边界、原作快照核验和回答前证据审核继续独立执行。
历史关键词分类器只保留在 `experiments/query_scope.py`，用于离线对照。
