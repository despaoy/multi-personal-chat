# P6 统一知识 RAG 管线

P6 建设项目级、可扩展的 RAG 管线：统一知识文档契约、多知识域注册、
embedding Provider 抽象与缓存、持久化索引、稀疏+向量+实体混合召回、
RRF 融合、叙事层权重、重排、预算内上下文组装、引用溯源、业务接入与
离线评估。首个正式数据域为「纸上魔法使」（P5 approved 知识卡 409 张），
但架构不绑定任何具体作品——人物别名、卷名、叙事层策略全部收敛在
domain 配置中。

## 1. 分层架构

```
业务层    api/generate.py（_retrieve_rag_bundle）
              │ 域门控未命中 → 回退既有 RAGHelper 链路
门面层    knowledge/rag_pipeline/service.py（get_rag_pipeline_service）
编排层    knowledge/rag_pipeline/pipeline.py（RagPipeline）
              ├─ query.py      查询理解与别名归一（域选择/意图/叙事层）
              ├─ retrieval.py  混合检索（sparse+vector+entity 通道 → RRF+加权）
              ├─ rerank.py     重排（CrossEncoder 注入 / 确定性降级）
              ├─ context.py    预算内上下文组装 + Citation
              ├─ index.py      持久化索引（FAISS + BM25 + 实体索引）
              ├─ embedding.py  Provider 抽象 + npz 缓存
              ├─ loaders.py    approved 卡 loader（通用卡契约）
              ├─ documents.py  KnowledgeIndexDocument 契约
              ├─ registry.py   知识域注册表
              └─ domains/      各域专属配置（唯一允许出现作品词表的位置）
```

## 2. 关键设计

### 2.1 域门控（普通聊天不加载游戏库）
自动域选择需要以下信号之一：
1. 卷名/故事标题命中
2. ≥2 个不同实体命中
3. ≥2 字别名 token 命中（多字名/全名，如"月社妃"）
4. 单字实体命中 + 查询含类型意图词（"妃的哥哥是谁"）

未命中 → service 返回 None → 调用方回退既有链路。显式指定 domain_id
时始终执行检索。

### 2.2 三通道混合召回 + RRF
- sparse：BM25（归一查询 + 原始查询取最优名次；title/entities/keywords
  权重 ×3，防止长 evidence 稀释精确词）
- vector：FAISS IndexFlatIP（归一化内积 = 余弦）
- entity：实体精确索引（多实体查询 AND 语义优先，回退 ANY）
- 融合：RRF(k=60) + 实体命中加成（base 0.02/命中，cap 0.03）
  × 叙事层乘数（默认策略表；查询带偏好时 ±15%）× 类型意图乘数

### 2.3 叙事层（不写死作品规则）
`reality_status`（objective/fictional/character_claim/inferred）、
`temporal_scope`（current/flashback/reconstruction/hypothetical）、
`content_scope`（main_story/bonus_story）的取值与权重由 domain 配置
（`NarrativeLayerPolicy`）声明；查询分析识别意图词（"真相/梦里/小时候"）
后在 ±15% 区间调整，不做硬过滤——多层结果共存，metadata 区分。

### 2.4 重排
优先接入现有 `CrossEncoderReranker`（RERANKER_ENABLED 控制，与
RAGHelper 同一开关）；本地无 bge-reranker-base 时自动确定性降级：
实体重合率 0.30 + 类型匹配 0.15 + jieba 词覆盖 0.25 + 关系方向完整性
0.10 + 叙事层对齐 0.10 + 身份意图对齐 0.08 + 超长内容惩罚。全部为
通用规则，无作品特例。

### 2.5 增量构建与缓存
缓存键 = domain_id + 文档 ID + embedding_text 指纹 + 模型 ID + 模型
指纹（不同模型向量不混用）。未变化文档全量复用（409/409 实测），
消失文档的缓存项自动清理。索引写入顺序 documents → faiss → manifest
（最后替换，崩溃时 count 校验失败 → 索引判定不可用 → 服务降级）。

## 3. 入口

| 用途 | 命令 |
| --- | --- |
| 构建索引 | `python scripts/build_knowledge_index.py [--domain D] [--force/--full] [--dry-run] [--limit N]` |
| 离线评估 | `python scripts/evaluate_rag_retrieval.py --domain tsukiyashiro_kisaki [--report]` |
| 业务调用 | `knowledge.rag_pipeline.service.get_rag_pipeline_service().retrieve_with_citations(query, top_k)` |

索引产物：`backend/data/knowledge/<domain>/rag_index/`
（documents.jsonl / faiss.bin / index_manifest.json / embedding_cache.npz）

评估集：`backend/data/eval/rag_retrieval/<domain>.jsonl`（48 条 × 12
类别，期望通过结构化 criteria 解析为文档 ID，不用生成模型打分）

## 4. 新增知识域

1. 在 `domains/` 注册工厂函数：声明 domain_id、source_root、loader、
   别名表（alias→规范名）、卷名表、叙事层策略
2. 追加到 `_FACTORIES` 列表
3. 运行 `build_knowledge_index.py` 构建索引
4. （可选）在 `data/eval/rag_retrieval/` 添加该域评估集

其他来源（角色记忆、用户文档）实现同签名 loader（SourceLoader 协议：
`source_root → list[KnowledgeIndexDocument]`）即可接入。

## 5. 配置（环境变量）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| EMBEDDING_MODEL_PATH | 自动解析 | embedding 模型路径（HF 缓存/本地目录） |
| RAG_PIPELINE_EMBEDDING_MODEL_ID | paraphrase-multilingual-MiniLM-L12-v2 | 模型 ID |
| RAG_PIPELINE_EMBEDDING_DIM | 384 | 期望维度 |
| RAG_PIPELINE_EMBEDDING_BATCH | 32 | 批量大小 |
| RAG_PIPELINE_EMBEDDING_TIMEOUT | 300 | 编码超时（秒） |
| RAG_PIPELINE_WARMUP | true | 启动后台预热（索引 + 模型） |
| RERANKER_ENABLED | false | CrossEncoder 重排（与现有链路共用） |
| RAG_CITATIONS_ENABLED | true | 引用返回开关（generate.py 层） |

## 6. 运行环境注意

- 需要 faiss + sentence-transformers（本机 Python 3.12 环境齐备；
  Python 3.10 缺 faiss/sentence-transformers 时索引构建与向量检索
  不可用，服务自动降级为不可用状态，不影响聊天主流程）
- embedding 模型为本地缓存的 paraphrase-multilingual-MiniLM-L12-v2
  （384 维，与既有 VectorDatabase 同款，无下载、无外部 API）
- 单元测试使用 FakeEmbeddingProvider 注入，不依赖真实模型安装

## 7. 评估结果（2026-08-27，48 条 × 12 类别）

| 阶段 | Hit@1 | Hit@3 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| sparse_only | 0.521 | 0.688 | 0.729 | 0.606 |
| vector_only | 0.458 | 0.542 | 0.646 | 0.520 |
| hybrid | 0.667 | 0.854 | 0.896 | 0.750 |
| hybrid_rerank | 0.792 | 1.000 | 1.000 | 0.882 |

全类别 hit@5 ≥ 4/4，no_answer 类正确拒绝域路由。各层贡献单调递增。

## 8. 已知限制

- CrossEncoder（bge-reranker-base）本地不可用，生产重排走确定性
  降级路径；下载模型并置 RERANKER_ENABLED=true 可启用
- 单字人名（"妃"）的域门控依赖意图词，"王妃"类闲聊不会误路由，
  但纯名词型短查询（"妃"）不触发本域
- BM25 与实体索引在加载时确定性重建（409 文档 <1s）；十万级以上
  域需要改为持久化稀疏索引
- 角色长期记忆与作品知识索引完全隔离（character 包不走本服务），
  用户记忆与作品设定不会互相污染；后续接入记忆域时新增独立 domain
