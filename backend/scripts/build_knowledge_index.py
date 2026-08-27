"""P6 统一知识索引构建入口。

用法：
    python scripts/build_knowledge_index.py [--domain DOMAIN] [--output-dir DIR]
        [--full | --force] [--dry-run] [--limit N]

行为：
    1. 加载 source documents（approved 门禁：非 approved 直接报错）
    2. 转换为 canonical documents
    3. 计算内容/文本指纹
    4. 复用未变化文档的 embedding（缓存命中）
    5. 只重算新增或变化文档
    6. 删除已不存在的旧缓存项（增量模式自动清理）
    7. 写入索引（documents.jsonl + faiss.bin + manifest，原子切换）
    8. 写入/更新 embedding 缓存
    9. 打印统计 + 重载校验

--full/--force：全量重算 embedding（忽略缓存复用）；
默认增量模式。--dry-run 只统计不落盘。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from knowledge.rag_pipeline.embedding import (  # noqa: E402
    EmbeddingCache,
    SentenceTransformerEmbeddingProvider,
    embedding_cache_key,
)
from knowledge.rag_pipeline.index import build_domain_index  # noqa: E402
from knowledge.rag_pipeline.loaders import UnapprovedDataError  # noqa: E402
from knowledge.rag_pipeline.registry import get_default_registry  # noqa: E402


def source_fingerprint(documents) -> str:
    """approved 源的确定性指纹（manifest 记录，供版本追踪）。"""
    import hashlib

    digest = hashlib.sha256()
    for doc in documents:
        digest.update(doc.id.encode())
        digest.update(doc.embedding_text_fingerprint.encode())
    return digest.hexdigest()[:16]


def build_domain(config, provider: SentenceTransformerEmbeddingProvider, args) -> dict:
    index_root = Path(args.output_dir) if args.output_dir else config.resolve_index_root()
    print(f"== 域 {config.domain_id}（{config.display_name or config.domain_id}）==")
    print(f"源目录: {config.source_root}")
    print(f"索引目录: {index_root}")

    # 1-2. 加载与转换（approved 门禁在 loader 内强制）
    documents = config.loader(config.source_root)
    if args.limit:
        documents = documents[: args.limit]
    if not documents:
        print("  没有可索引的 approved 文档，跳过")
        return {"domain_id": config.domain_id, "status": "skipped", "documents": 0}

    type_counts: dict = {}
    for doc in documents:
        type_counts[doc.document_type] = type_counts.get(doc.document_type, 0) + 1
    print(f"  canonical 文档: {len(documents)}（{type_counts}）")

    # 3-6. embedding 计算（含缓存复用与增量）
    cache = EmbeddingCache(index_root / "embedding_cache.npz")
    cache.load()

    import numpy as np

    force = bool(args.force or args.full)
    fingerprint = provider.model_fingerprint
    vectors = []
    reused = 0
    keys = []
    todo: list = []
    for doc in documents:
        key = embedding_cache_key(doc.domain_id, doc.id, doc.embedding_text_fingerprint, provider.model_id, fingerprint)
        keys.append(key)
        vector = cache.get(key) if not force else None
        if vector is not None:
            vectors.append(vector)
            reused += 1
        else:
            vectors.append(None)
            todo.append(doc)

    if args.dry_run:
        print(f"  [dry-run] 文档 {len(documents)}，缓存复用 {reused} 条，需计算 {len(todo)} 条")
        return {
            "domain_id": config.domain_id,
            "status": "dry_run",
            "documents": len(documents),
            "reused_embeddings": reused,
            "computed_embeddings": len(todo),
        }

    if todo:
        print(f"  计算 embedding: {len(todo)} 条（缓存复用 {reused} 条）")
        start = time.time()
        matrix = provider.embed_texts([doc.embedding_text for doc in todo])
        elapsed = time.time() - start
        print(f"  embedding 完成: {matrix.shape}，耗时 {elapsed:.1f}s")
        cursor = 0
        for i, vector in enumerate(vectors):
            if vector is None:
                vectors[i] = matrix[cursor]
                cursor += 1

    embeddings = np.vstack([np.asarray(v, dtype=np.float32).reshape(1, -1) for v in vectors])

    # 7-8. 写入索引 + 缓存
    for key, vector in zip(keys, vectors):
        cache.put(key, np.asarray(vector, dtype=np.float32))
    valid_keys = set(keys)
    pruned = cache.prune_to(valid_keys)
    cache.save()

    index = build_domain_index(
        config,
        documents,
        embeddings,
        provider,
        source_fingerprint=source_fingerprint(documents),
        build_params={
            "mode": "full" if force else "incremental",
            "force": force,
            "limit": args.limit,
            "cached_embeddings_reused": reused,
            "embeddings_computed": len(todo),
        },
    )

    # 9. 统计
    stats = index.stats()
    print(
        f"  索引写入完成: {stats['document_count']} 文档，"
        f"维度 {stats['vector_dimension']}，BM25 词表 {stats['bm25']['vocab']}"
    )
    if pruned:
        print(f"  清理过期缓存项: {pruned}")
    # 验证：重新加载
    reloaded = type(index)(config.domain_id, index_root, index.dimension)
    ok = reloaded.load()
    if not ok or reloaded.count() != len(documents):
        raise RuntimeError(f"索引重载校验失败: domain={config.domain_id}")
    print(f"  重载校验通过: {reloaded.count()} 文档")

    return {
        "domain_id": config.domain_id,
        "status": "ok",
        "documents": len(documents),
        "document_type_counts": stats["document_type_counts"],
        "reused_embeddings": reused,
        "computed_embeddings": len(todo),
        "pruned_cache_entries": pruned,
        "index_root": str(index_root),
        "manifest": index.manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P6 统一知识索引构建")
    parser.add_argument("--domain", default="", help="只构建指定域（缺省全部启用域）")
    parser.add_argument("--output-dir", default="", help="覆盖索引输出目录（所有域）")
    parser.add_argument("--full", action="store_true", help="全量重算 embedding（同 --force）")
    parser.add_argument("--force", action="store_true", help="强制重建（忽略缓存复用）")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写索引")
    parser.add_argument("--limit", type=int, default=0, help="限制文档数（调试）")
    parser.add_argument("--embedding-model-path", default="", help="覆盖 embedding 模型路径")
    args = parser.parse_args()

    if args.embedding_model_path:
        os.environ["EMBEDDING_MODEL_PATH"] = args.embedding_model_path

    registry = get_default_registry()
    domains = registry.list_domains(enabled_only=True)
    if args.domain:
        domains = [registry.require(args.domain)]
    if not domains:
        print("没有启用的知识域")
        return 1

    provider = SentenceTransformerEmbeddingProvider()
    print(f"embedding 模型: {provider.model_id}，维度 {provider.dimension}")
    print(f"模型路径: {provider.model_path}")
    print(f"模型指纹: {provider.model_fingerprint}")
    print()

    results = []
    for config in domains:
        try:
            results.append(build_domain(config, provider, args))
        except UnapprovedDataError as e:
            print(f"  [拒绝] 未批准数据: {e}", file=sys.stderr)
            return 2
        print()

    ok = [r for r in results if r.get("status") == "ok"]
    print("== 构建汇总 ==")
    for r in results:
        print(
            f"  {r['domain_id']}: status={r['status']}, documents={r.get('documents', 0)}, "
            f"reused={r.get('reused_embeddings', 0)}, computed={r.get('computed_embeddings', 0)}"
        )
    if args.dry_run:
        print("（dry-run，未写盘）")
    else:
        print(f"成功域: {len(ok)}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
