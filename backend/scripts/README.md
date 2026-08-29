# Backend maintenance scripts

本目录保存 FastAPI 服务和生产知识数据所需的维护入口。角色研究与训练实验脚本位于仓库根目录的 [`../../scripts/`](../../scripts/README.md)。脚本应从仓库根目录或 `backend/` 目录运行，并通过参数或项目相对路径读取输入。

| 脚本 | 用途 |
| --- | --- |
| `local_smoke.py` | mock 模式 API、鉴权与 AstrBot 事件冒烟验证 |
| `launch_vllm.py` | 独立启动 vLLM OpenAI 兼容服务 |
| `train_intent_classifier.py` | 训练轻量 RAG 意图分类器 |
| `run_scene_metadata_candidates.py` | 生成 P4 场景元数据候选与审核材料 |
| `finalize_scene_metadata.py` | 应用 P4 人工决定并生成定稿材料 |
| `approve_scene_metadata.py` | 在显式批准后晋升场景元数据 |
| `run_knowledge_candidates.py` | 从批准场景生成 P5 知识卡候选 |
| `finalize_knowledge_review.py` | 应用批准的知识卡审核决定 |
| `build_character_rag_index.py` | 构建生产多粒度角色知识索引 |

角色知识索引构建说明见
[`../../docs/architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md`](../../docs/architecture/CHARACTER_KNOWLEDGE_RETRIEVAL.md)。历史实验脚本位于 `archive/`，不属于受支持的生产入口。

脚本不得写入个人机器绝对路径。覆盖已有索引或审核产物的操作必须要求显式确认参数。
