# Backend scripts

本目录只保存与 FastAPI 服务和当前 RAG 数据链直接相关的维护入口。角色研究与训练实验脚本位于仓库根目录的 [`../../scripts/`](../../scripts/README.md)；旧胡桃/原神缓存工具已移至 `scripts/archive/legacy_backend_tools/`。

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
| `build_knowledge_index.py` | 构建 P6 统一知识索引 |
| `evaluate_rag_retrieval.py` | 离线评估 P6 检索效果 |

所有脚本从仓库或 `backend/` 根目录运行，输入路径必须显式或相对项目解析，不得写入个人机器绝对路径。
