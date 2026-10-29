"""知识库种子文档导入器 - 将 kb_seed_documents.json 导入向量数据库。

导入后 FAISS 索引和 BM25 索引均可用，ID 保留为字面字符串（如 hutao_skill）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def main():
    parser = argparse.ArgumentParser(description="导入知识库种子文档")
    parser.add_argument(
        "--seed-file",
        type=str,
        default=str(_BACKEND_DIR / "data" / "kb_seed_documents.json"),
    )
    parser.add_argument("--verify", action="store_true", help="导入后验证搜索结果")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    seed_path = Path(args.seed_file)
    if not seed_path.exists():
        logger.error(f"种子文档不存在: {seed_path}")
        sys.exit(1)

    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data.get("documents", [])
    logger.info(f"加载 {len(documents)} 条种子文档 from {seed_path}")

    from knowledge.vector_db import get_vector_db

    vdb = get_vector_db()

    stats_before = vdb.get_stats()
    logger.info(f"导入前统计: {stats_before}")

    vdb.add_documents(documents)
    logger.info("文档已写入向量数据库")

    stats_after = vdb.get_stats()
    logger.info(f"导入后统计: {stats_after}")

    if args.verify:
        logger.info("\n=== 验证搜索结果 ===")
        test_queries = [
            ("胡桃的元素战技是什么？", "hutao_skill"),
            ("钟离的护盾怎么算？", "zhongli_shield"),
            ("七七的治疗量受什么影响？", "qiqi_heal"),
            ("元素反应有哪些类型？", "elemental_reactions"),
            ("树脂怎么恢复？", "resin_system"),
        ]
        hits = 0
        for query, expected_id in test_queries:
            results = vdb.search(query, top_k=5, threshold=0.0)
            found_ids = [str(r.get("id", "")) for r in results]
            hit = expected_id in found_ids
            hits += 1 if hit else 0
            status = "✅" if hit else "❌"
            logger.info(f"  {status} query='{query}' expected={expected_id} found={found_ids[:3]}")
        logger.info(f"\n验证结果: {hits}/{len(test_queries)} 命中")
        if hits != len(test_queries):
            sys.exit(1)


if __name__ == "__main__":
    main()
