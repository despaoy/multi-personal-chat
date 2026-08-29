# 月社妃全游戏文本 RAG 基线审计（P0）

- 阶段：P0 基线审计（只读）
- 日期：2026-08-24（P0.1、P0.2 修订：同日，按人工复核意见修正，见文末修订记录）
- 依据：《月社妃全游戏文本 RAG 实施规划 v1》
- 性质：本阶段未修改任何业务代码、数据文件和配置；仅新增本文档。
  统计通过临时只读脚本完成（`scripts/extract_character_dialogues.py` 的解析逻辑以 import 方式复用），临时脚本已删除。

> 本文是 P0 时点的审计记录，不描述当前生产命名或运行入口。当前架构见
> [多粒度角色知识检索](../architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md)。

---

## 1. 原始文本基线

### 1.1 文件清单与编码

`gametext/纸上魔法使/` 共 **17 个 `.txt` 文件**，全部：

- UTF-8 编码、无 BOM、解码无错误；
- CRLF 行尾（`\r\n`）；
- 无嵌套目录，扁平结构。

| 文件 | 字节 | 物理行 | 标签数 | 解析事件 | 非空叙述行* |
|---|---:|---:|---:|---:|---:|
| 1翡翠的排挤原理 | 245,557 | 4,521 | 1,991 | 1,990 | 2,528 |
| 2红宝石的天作之合 | 159,480 | 2,831 | 1,364 | 1,363 | 1,465 |
| 3蓝宝石的存在证明 | 233,545 | 4,321 | 1,779 | 1,779 | 2,540 |
| 4紫水晶的怪异传说 | 193,231 | 3,413 | 1,543 | 1,542 | 1,868 |
| 5磷灰石的怠惰现象 | 180,991 | 3,178 | 1,439 | 1,439 | 1,736 |
| 6芙蓉石的长年隔绝 | 204,348 | 3,735 | 1,612 | 1,612 | 2,121 |
| 6芙蓉石的终焉轮回 | 135,973 | 2,607 | 1,079 | 1,078 | 1,526 |
| 7黑珍珠的求爱信号 | 143,433 | 2,599 | 1,105 | 1,105 | 1,492 |
| 8萤石的怠惰现象 | 75,515 | 1,443 | 598 | 598 | 844 |
| 8萤石的时空残影 | 101,126 | 1,896 | 785 | 785 | 1,109 |
| 9白珍珠的泡沫爱慕 | 109,866 | 1,997 | 722 | 721 | 1,273 |
| 9绿幽灵水晶的命运连锁 | 130,075 | 2,503 | 1,099 | 1,099 | 1,402 |
| 10黑曜石的因果目录 | 50,030 | 926 | 165 | 165 | 761 |
| 11黑玛瑙的不在证明 | 92,806 | 1,669 | 614 | 614 | 1,053 |
| 12青金石的幻想图书馆 | 150,890 | 2,700 | 946 | 945 | 1,751 |
| 13璀璨的紫翠玉 | 64,731 | 1,146 | 284 | 284 | 861 |
| 日后谈 | 52,536 | 879 | 405 | 404 | 472 |
| **合计** | **2,234,107** | **42,364** | **17,530** | **17,523** | **24,802** |

\* "非空叙述行"= 不以 `[说话人]+引号` 开头的非空行；其中约 12 行实为跨行台词的续行（见 §2.2），纯叙述行约 24,790。

### 1.2 行分类总账

```text
42,364 物理行
 = 17,530 对话标签行（全部位于行首、全部后跟引号起始）
 + 24,802 非标签非空行（叙述 + 少量台词续行）
 +     32 空行
```

17,530 = 行首标签数 = 后跟引号的标签数（三种口径一致，无行内嵌套标签、无缺引号前缀的标签）。

### 1.3 对话与说话人分布

- 解析事件总数：**17,523**（`SCRIPT_RE` 匹配）
- 引号类型：直角引号 `「」` 17,515 条；弯引号 `“”` 8 条；顶层 `『』` 0 条
- 说话人共 **14 位**：

| 说话人 | 台词数 | 说话人 | 台词数 |
|---|---:|---|---:|
| 琉璃 | 6,019 | 汀 | 828 |
| 夜子 | 2,982 | 岬 | 352 |
| 彼方 | 2,395 | 奏 | 293 |
| 理央 | 1,934 | 暗子 | 189 |
| **妃** | **1,598** | 父亲 | 69 |
| 克丽索贝莉露 | 845 | 母亲 | 11 |
| | | 学生/医者 | 4/4 |

合计 17,523 ✓（与解析事件数一致）。

### 1.4 妃台词分布

- 妃（含 `月社妃` 别名）台词 **1,598 条**，说话人标签全部为 `妃`
- 与既有产物完全一致：
  - `backend/data/character_dialogues/tsukiyashiro_kisaki_raw.jsonl` = 1,598 条（全部 direct/canonical，无文本污染）
  - `manifest.json`：`canonical_direct_occurrences = 1598`；recommended SFT 768 条、覆盖 1,254 次出现
- **4 个卷妃零出场**（说话人均含克丽索贝莉露）：`11黑玛瑙的不在证明`、`13璀璨的紫翠玉`、`6芙蓉石的终焉轮回`、`9绿幽灵水晶的命运连锁`。与剧情结构一致（妃死亡后由"魔法重现"角色在场）。对 RAG 的含义：涉及这些卷的问题需要区分"妃不在场"与"检索失败"，克丽索贝莉露的台词不得归入妃。
- 妃的 8 条弯引号台词（现有解析已覆盖，P2 须保持）：`8萤石的怠惰现象:981,983`（告白场景）；`日后谈:657-660,663,816`（《萤色光景》猫语场景）。

---

## 2. 缺失 7 条对话的定位与根因

### 2.1 结论

**标签统计 17,530 与解析结果 17,523 之差 = 7，全部定位成功，根因单一：原文存在 7 处开引号 `「` 未闭合（原文瑕疵）。**

`SCRIPT_RE` 使用非贪婪 `.*?` + `re.DOTALL` 且不锚定行首。当某条台词开引号后本行未闭合时，匹配会一直延伸到**下一条完整台词的 `」`**，导致：

1. 下一条完整台词整行被并入前一条事件的文本（即"缺失"的 7 条）；
2. 前一条事件的文本被中间叙述行污染（吞没者文本内嵌 `[标签] 「`）。

### 2.2 七条明细

| # | 被吞台词位置 | 被吞说话人 | 吞没者（未闭合引号） | 说明 |
|---|---|---|---|---|
| 1 | 12青金石:2488 | 琉璃 | [夜子] 12青金石:2485 | 「就一点点…时间。 后未闭合 |
| 2 | 1翡翠:3894 | 岬 | [岬] 1翡翠:3892 | 「不是的…多爬了一次。 后未闭合 |
| 3 | 2红宝石:72 | 夜子 | [暗子] 2红宝石:70 | 「虽然琉璃是这么说…头绪吗？ 后未闭合 |
| 4 | 4紫水晶:328 | 彼方 | [彼方] 4紫水晶:326 | 「哎？你的朋友不是很多吗？ 后未闭合 |
| 5 | 6终焉轮回:230 | 琉璃 | [理央] 6终焉轮回:228 | 「呼呼呼…敬请期待！ 后未闭合 |
| 6 | 9白珍珠:1645 | 琉璃 | [琉璃] 9白珍珠:1641 | 「——无人使用的失律钢琴。 后未闭合 |
| 7 | 日后谈:186 | 彼方 | [彼方] 日后谈:183 | 「你之类的也是这样对不对？…风格呢。 后未闭合 |

影响评估：

- 被吞 7 条说话人分布：琉璃×3、彼方×2、夜子×1、岬×1；吞没者：夜子/岬/暗子/彼方/理央/琉璃/彼方。**妃的 1,598 条台词完全不受影响**（无妃事件被吞、无妃事件吞没他人），既有 raw/SFT 数据无污染。核验方式：`tsukiyashiro_kisaki_raw.jsonl` 全量检查 0 条文本含 `[`；另对 `backend/data/character_dialogues/tsukiyashiro_kisaki_sft.json` 与 `backend/data/character_dialogues/tsukiyashiro_kisaki_sft_full.json` 两个 SFT 文件做内嵌说话人标签模式扫描（检索 prompt/reply 文本中的 `[说话人]「台词」` 模式），均未发现污染。
- 16 条多行事件 = 7 条吞没污染 + **9 条合法跨行台词**（妃 3 条：1翡翠:46-47、59-60、3蓝宝石:2536-2537；其余为琉璃/暗子/汀/夜子/彼方）。P2 解析器必须保留合法跨行台词能力。
- P2 修复建议：采用**逐行状态机**解析（不能用"同一段落内找闭引号"——全库仅 32 个空行，段落边界基本不存在）：
  1. 行首遇到 `[说话人]` + 开引号即开启一条台词事件；
  2. 正常跨行台词持续逐行累计，直到读到闭引号为止（保留 9 条合法跨行台词）；
  3. 若在遇到**下一条说话人标签行**时仍未读到闭引号，判定为未闭合引号：输出显式告警清单（即上述 7 处），在标签行前截断该事件，**不允许静默吞并下一条台词**；
  4. 截断处可辅以人工补引号映射表校正，但映射表必须显式登记、可审计。

---

## 3. 文本结构特征（后续阶段输入）

1. **同编号多故事单元**：第 6、8、9 章各有两个同编号文件（6长年隔绝/6终焉轮回；8怠惰现象/8时空残影；9白珍珠/9绿幽灵水晶），开头视点、叙述起点均不同。**仅凭编号和开头内容不足以判定它们是平行路线、时间阶段差异还是视角差异——连续性关系（continuity）尚待剧情结构人工审核，不得按文件编号自动推断为平行/分支。** 建模时使用 `volume_number + story_unit_id + continuity_id + sequence_order` 中性结构；`route`（main/branch/parallel/unknown）保留为可空字段，仅在剧情结构审核完成后填写（回忆/重现等时间状态归 `temporal_scope`，不属于 route，见 §7.2）。P3 场景切分在 `continuity_id` 确定前不得跨同编号文件合并证据。
2. **日后谈双段结构**：第 1–37 行为**宣传元叙事**（体验版宣传、预约特典介绍、打破第四面墙的"导演"对话）；第 38 行空行 + 第 39 行 `※通关正编后的追加剧本。当心谜底揭开。` 起，为追加剧本**《萤色光景》正篇后续**（猫视角故事）。P3 必须按此边界切分 content_scope。
3. **场景切换少有空行**：全库仅 32 个空行，场景切换主要靠时间副词与叙述文本（`extract_character_dialogues.py` 的 `SCENE_RESET_RE` 词表即针对此设计）。P3 场景切分不能依赖空行。
4. **叙述占比高**：非标签行约占非空行的 58.6%（24,802/42,332），第一人称叙述（琉璃视点）承载大量剧情事实。P4 事实提取的主要来源是叙述而非台词，`speaker` 字段之外需要"叙述者"概念。

---

## 4. 现有 RAG 基础设施盘点

### 4.1 模块清单

| 模块 | 路径 | 现状 |
|---|---|---|
| RAG 主流程 | `backend/knowledge/rag_helper.py` | 查询扩展→多查询混合检索→区域加权→重排→归一化；top_k=5、recall×4、缓存 TTL 60s |
| 向量库 | `backend/knowledge/vector_db.py` | FAISS Flat/IVF/HNSW 自动迁移（阈值 1 万/10 万）；BM25(jieba)；混合检索 vector 0.7/BM25 0.3；元数据过滤 |
| 分块器 | `backend/knowledge/text_splitter.py` | `smart_text_split`：段落→句子→定长，默认 600 字/重叠 100 字符，非对话感知 |
| 导入器 | `backend/knowledge/importer.py` | `import_genshin_knowledge`：原神专用（characters/events/world 分类映射），600/150 分块 |
| 重排器 | `backend/knowledge/reranker.py` | `bge-reranker-base`（本地路径/`RERANKER_MODEL_PATH`），**默认关闭**（`RERANKER_ENABLED=false`），失败回退原序 |
| 纠正性 RAG | `backend/knowledge/corrective_rag.py` | 置信度<0.3 → 关键词重写 → 重试 1 次 → 弃答；`CORRECTIVE_RAG_ENABLED` 默认 false |
| 意图路由 | `backend/knowledge/intent_detector.py` | ML 多分类器（按 KB 名路由）+ 规则引擎兜底（含原神关键词） |
| KB 管理 API | `backend/api/knowledge.py` | KB/文件夹/文档/分块 CRUD，ZIP 上传与目录扫描入库，向量文档注入 `knowledge_base_id` 路由过滤 |
| 生成集成 | `backend/api/generate.py` L866-944 | `needs_rag`→KB 路由过滤→`_retrieve_rag_bundle`（top_k=3）→ abstained 时返回拒答模板且 `modelInvoked=false`；引用经 `citations` 结构化返回 |
| 评测 | `backend/evaluation/`、`backend/experiments/rag_ablation.py` | Recall@1/5、MRR、nDCG@5、P50/P95 延迟；4 变体消融（vector_only/bm25_only/hybrid/hybrid_reranker） |

### 4.2 模型与索引配置

- **运行时 embedding 无统一固定基线，实际模型必须由运行环境记录**。`vector_db.py` 的 `_find_local_embedding_model`（L176-190）按以下优先级动态解析：`EMBEDDING_MODEL_PATH` 环境变量 → 本地路径搜索（MiniLM 候选目录、HF 缓存，以及 **ModelScope 中文模型 `iic/nlp_corom_sentence-embedding_chinese-base`**）→ `ALLOW_REMOTE_EMBEDDING_MODEL=true` 时允许远程下载 MiniLM，否则抛 `FileNotFoundError`。即代码默认优先查找本地 MiniLM，但也可能回退到 ModelScope 中文模型，不存在"统一为 MiniLM"的事实。
- **P0 审计时的本地环境状态：未配置可直接加载的运行时 RAG embedding**（本地 MiniLM 候选路径不存在、`.env` 未设置 `EMBEDDING_MODEL_PATH`）。此为审计时点快照，后续本地配置变化不使本条失真，以 run manifest 实际记录为准。
- `KISAKI_EMBEDDING_MODEL_PATH=./models/bge-small-zh-v1.5` **只服务于 V4 数据审核流程**（`scripts/hard_gate_kisaki_v4.py`），不等于运行时 RAG 配置。
- **服务器侧 R2 脚本明确使用 BGE-M3**：`scripts/lab-run-kisaki-r2.sh:15` 默认 `EMBEDDING_MODEL_PATH=$ROOT/runtime/models/bge-m3`；部署文档（`docs/operations/DEPLOYMENT_GUIDE.md`）同样指向 `runtime/models/bge-m3`。
- 结论：embedding 模型随部署环境而变（本地未配置 / 服务器 BGE-M3），后续任何检索评测与对比实验都必须在 run manifest 中记录实际加载的模型路径与维度，不能统一称为 MiniLM。规划中的"BGE-M3/Qwen3-Embedding 对比"属于**后续检索模型与索引消融**（在 P10 正式评测前完成；P8 只负责查询扩展、Corrective RAG 和置信度相关修复），以此为对照基线。
- 重排器基线：`bge-reranker-base`；`bge-reranker-v2-m3` 未配置。

### 4.3 数据现状

- 本地 SQLite（`backend/qq_assistant.db`）：`knowledge_bases` / `knowledge_folders` / `knowledge_documents` / `knowledge_chunks` 四表**全部 0 行**。
- 向量库目录 `backend/knowledge/data/vector_db/` 存在但为空（无 faiss_index.bin / metadata.pkl / bm25_state.pkl）。
- 结论：**RAG 基础设施可用，但当前没有任何已入库的月社妃原作知识**。游戏文本 RAG 属于从零构建索引（模块复用、数据新建）。

### 4.4 既有月社妃 RAG 数据资产（静态文件）

| 文件 | 内容 | 性质 |
|---|---|---|
| `experiments/archive/legacy_rag/kisaki_knowledge_base.json` | 10 条人工撰写的妃档案卡（人物/关系/世界观的结论式摘要，无原文行号） | 旧 RAG 接地性评测的引用来源；非游戏文本知识 |
| `experiments/research/character_rag_seed_documents.json` | 30 条 held-out 证据文档，**带 `source_lineage`（游戏文本路径+行号+event id）** | 已冻结的 held-out 证据，禁止用于 SFT |
| `experiments/research/character_rag_retrieval_eval.json` | 30 题：question=对话上文，gold_answer=妃的下一句 | **续写式**检索评测（R2 v1） |
| `experiments/research/kisaki_rag_eval_v2_candidates.json` | 60 题：单证据 30 / 多证据 15 / 无答案 15，`status=pending_human_review`、`formal_use_allowed=false` | R2 v2 候选（仍是续写式），待人工冻结 |

评测框架已实现 Recall@1/5、MRR、nDCG@5、延迟分位与四组检索消融（`rag_ablation.py` 默认数据集即上述 v2 候选），formal 模式有 frozen 门禁。**但现有评测全部是续写式，没有事实型问答测试集**——P10 需按规划新建（单证据 25/多证据 15/时间线冲突 10/无答案 10），且不应把旧续写测试称为正式 RAG 评测。

---

## 5. 已发现问题（P8 输入）

1. **QueryExpander 原神词表无差别套用**（`rag_helper.py` L42-118）：`synonym_map`（胡桃/钟离/七七/魈/璃月…）、`region_keywords`（璃月/蒙德/稻妻/须弥）、`domain_keywords` 全部为原神领域。对任何查询（包括月社妃查询）启用：
   - 含"角色/剧情/故事/经历"等常用中文词的查询会被追加 `角色 xxx` / `剧情 xxx` 前缀变体；
   - 更严重的是 `extract_filters`（L120-158）把"剧情→事件、角色→角色、世界→世界"映射为**原神知识库分类过滤条件**——月社妃游戏 RAG 文档若不带这些 category，会被元数据过滤静默筛除；
   - `region_boost ±0.5` 加权只对原神区域生效（对妃查询无直接影响，但同一代码路径）。
   - P8 要求"分离原神与月社妃查询扩展"确认必要：需要按知识库域选择扩展器，而非全局单例。
2. **中文 Corrective RAG 查询重写失效**（`corrective_rag.py` L24-63）：`_tokenize` 把中文拆为单字，`reformulate_query` 中 `len(tok) > 1` 过滤后**纯中文 top 结果提取的关键词为空**，重写查询与原查询相同；第二次检索命中 RAGHelper 缓存（cache_key 含查询文本）返回同一结果，必然再次低于阈值而弃答。即"中文重写从未产生新查询"，P8 修复点明确：重写需产出实质性新查询（如 jieba 分词 + 去停用词 + 实体替换），且重试应绕过或失效缓存。
3. **intent_detector 规则引擎含原神关键词**（L57-66：蒙德/璃月/稻妻/须弥/枫丹…；L96：胡桃/往生堂）：通用知识词（"什么/为什么"等）仍工作，妃相关疑问句可触发 RAG，但规则原因字段会误导日志。ML 多分类器按 KB 路由是正确方向，需为月社妃游戏知识库补充训练样本或显式 KB 映射。
4. **分块器非对话感知**（`text_splitter.py`）：600 字固定分块 + 100 字符机械重叠，会把游戏台词拦腰截断、把不同说话人的台词混入同块。且参数不一致：`importer.py` 用 600/150，`api/knowledge.py` ZIP 上传用默认 600/100。当时规划的后续场景分块方案以“6–12轮对话切分、相邻块按对话轮重叠”解决这一问题。
5. **引用链路已具备**（好消息）：`rag_helper.build_citations` 已输出 `source_path/source_line/source_event_ids/source_lineage/kb_revision`；`generate.py` 已实现 abstained 拒答且 `modelInvoked=false`；seed documents 已示范 `source_lineage` 结构。P9 集成的工作量主要是把游戏 RAG 的证据字段映射进现有 citations，而非新建管线。
6. **评估基线注意事项**：`compute_confidence` 取绝对分数（0.7×top + 0.3×top3 均值）、阈值 0.3。embedding 模型随部署环境而变（见 §4.2），切换或跨环境对比（如本地 vs 服务器 BGE-M3）后分数分布会变，后续检索模型与索引消融（P10 正式评测前完成）时阈值需随基线重新标定，不能沿用 0.3 直接比较，且必须在 run manifest 中记录实际模型。

---

## 6. 统计核对总表（规划预期 vs 实测）

| 规划预期 | 实测 | 一致 |
|---|---|---|
| 17 个文件 | 17 | ✓ |
| 约 42,364 行 | 42,364 物理行 | ✓ |
| 1,598 条妃台词 | 1,598（raw.jsonl/manifest/独立复算三方一致） | ✓ |
| 标签统计 17,530 | 17,530（行首=任意位置=后跟引号三口径一致） | ✓ |
| 解析 17,523 | 17,523 | ✓ |
| 缺失 7 条 | 7 条全部定位，根因=7 处原文未闭合开引号（§2.2） | ✓ |

---

## 7. 下一阶段（P1）建议

1. **可以进入 P1**：基线数字全部核实，缺失 7 条定位正确，无阻塞项。
2. P1 数据模型采用中性结构，不预设平行/分支关系；字段按语义维度分离，**内容类型、时间状态、文件间结构关系是三个独立维度，不得混用**：
   ```text
   # —— 文件间结构关系维度（谁和谁是同一故事、同一连续性）——
   volume_number        # 卷号（1–13、日后谈）；同卷多文件的关系仅由此 + story_unit_id 表达
   story_unit_id        # 故事单元 id（以文件为初始单元，待审核后可细分/合并）
   story_title          # 故事标题（文件名去扩展名）
   continuity_id        # 连续性组 id：表达不同故事单元是否处于同一连续性，剧情结构审核后填写
   sequence_order       # 组内顺序，剧情结构审核后填写

   # —— 内容类型维度（这段文本属于哪类内容）——
   content_scope        # main_story（正篇）/ bonus_story（追加剧本，日后谈 39 行起）/
                        # promotional_meta（宣传元叙事，日后谈 1–37 行）/ unknown

   # —— 时间状态维度（叙述所处的叙事时间层）——
   temporal_scope       # current（主线当前）/ flashback（回忆）/ reconstruction（魔法重现）/
                        # hypothetical（假设/想象）/ unknown
                        # 注意：这是场景/事实的时间状态，与 route 无关

   # —— 路线维度（文件间结构关系的定性，审核后才有值）——
   route                # 可空：仅允许 main / branch / parallel / unknown（不含 flashback，
                        # 回忆属于 temporal_scope）；剧情结构审核完成后填写，禁止按文件编号自动推断

   # —— 其他可观察事实 ——
   viewpoint            # 视角（如琉璃第一人称），从文本可观察特征记录
   speakers             # 该场景中有台词的说话人（可观察事实）
   mentioned_characters # 该场景叙述/台词中被提及的人物（可观察事实）
   present_characters   # 该场景中实际在场的人物（含无台词者，按叙述证据判定）
   ```
   人物缺席状态（如"妃在该卷零台词"）**不在数据层存储**，由 `speakers/mentioned_characters/present_characters` 在运行时推导——"没有妃发言"不等于"场景没有叙述妃"，也不等于"该事实不可回答"。
3. P2 解析器验收应包含：逐行状态机实现（见 §2.2 修正后的 4 条规则）、7 处未闭合引号告警清单、9 条合法跨行台词保留、被吞台词独立成事件（可按本审计 §2.2 的行号做回归断言）。
4. P5 审核机制可直接复用 V4/V5 数据管线已验证的 approved/rejected/needs_review 模式与 `source_lineage` 结构。
5. P8 修复优先级建议：先分离查询扩展器（问题 1，会静默筛空结果），再修中文重写失效（问题 2，影响拒答质量）；两者都有明确的代码位置（§5.1/§5.2）。

---

## 8. 本阶段产物与边界

- 新增：`docs/research/KISAKI_GAME_RAG_BASELINE_AUDIT.md`（本文档）
- 修改：无（业务代码、数据文件、配置均未改动）
- 临时只读分析脚本（`_p0_audit_tmp*.py`、`_p0_audit_result.json`）已删除
- 未执行：Git 提交、推送、服务器同步

## 修订记录

### P0.1（2026-08-24，按人工复核意见修正，仅改动本文档）

1. §3.1：删除"6/8/9 已确认属于平行路线/分支路线"的结论（证据不足，开头视点差异不足以判定连续性关系），改为"同编号多故事单元，连续性关系尚待剧情结构人工审核，禁止按文件编号自动推断"。
2. §7.2：P1 建模不再预设 `6A/6B/8A/8B/9A/9B` 路线编号，改用 `volume_number + story_unit_id + continuity_id + sequence_order` 中性结构；`route` 保留为可空字段，仅剧情结构审核完成后填写。
3. §4.2：修正"运行时 embedding 基线为 MiniLM"的不准确表述——代码按环境变量和本地路径动态解析（优先本地 MiniLM，可回退 ModelScope 中文模型）；当前本地环境未配置可直接加载的运行时 RAG embedding；`KISAKI_EMBEDDING_MODEL_PATH` 只服务数据审核流程；服务器 R2 脚本（`scripts/lab-run-kisaki-r2.sh`）默认使用 BGE-M3。实际模型必须由运行环境记录（run manifest）。
4. §7.2：删除 `speaker_absent` 存储建议，改为存储 `speakers` / `mentioned_characters` / `present_characters` 三个可观察字段，缺席状态运行时推导。
5. §2.2：P2 解析建议改为逐行状态机（遇到下一条说话人标签前仍未闭合则告警并截断；正常跨行内容持续读取到闭引号），废弃"同一段落内找闭引号"（全库仅 32 个空行，段落边界基本不存在）。
6. §2.2 影响评估：补充人工复核结论——现有两个月社妃 SFT JSON 训练文件均未发现内嵌 `[说话人]「台词」` 污染。

### P0.2（2026-08-24，按人工复核意见修正，仅改动本文档）

1. §7.2：P1 字段按语义维度拆分——`content_scope`（内容类型：main_story/bonus_story/promotional_meta/unknown）、`temporal_scope`（时间状态：current/flashback/reconstruction/hypothetical/unknown）、`route`（文件间结构关系定性：main/branch/parallel/unknown）、`continuity_id`（连续性分组）分离为独立维度；删除 `content_scope` 中的"同编号其他单元"（同编号关系由 `volume_number + story_unit_id` 表达）；`route` 移除 `flashback`（回忆属于时间状态维度，与路线是两个维度）。§3.1 同步修正。
2. §4.2/§5.6："P8 模型实验/模型对比"改为"后续检索模型与索引消融"（P10 正式评测前完成）；明确 P8 只负责查询扩展、Corrective RAG 和置信度相关修复。
3. §2.2：SFT 核验表述改为准确文件与方法——扫描 `backend/data/character_dialogues/tsukiyashiro_kisaki_sft.json` 与 `backend/data/character_dialogues/tsukiyashiro_kisaki_sft_full.json`，使用内嵌说话人标签模式扫描，未发现 `[说话人]「台词」` 污染。
4. §4.2："当前本地环境未配置"改为"P0 审计时的本地环境状态"（时点快照，后续配置变化以 run manifest 为准）。
