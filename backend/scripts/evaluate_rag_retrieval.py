"""P6 离线检索评估（通用框架 + 域评估集）。

评估集 JSONL 契约（backend/data/eval/rag_retrieval/<domain>.jsonl）：
    {"id", "query", "domain_id", "category",
     "expected": {"criteria": {...}} 或 {"expected_document_ids": [...]},
     "filters"?, "notes"?}

criteria 为结构化 payload 条件（document_type/subject/predicate/
value_contains/relation_type/target/title_contains/participants_contain/
entities_contain/reality_status/temporal_scope/volume/either_of），
由本脚本在 canonical 文档上解析为期望文档 ID 集合——
正确性通过预先指定的文档 ID 或结构化 payload 判断，
不调用生成模型自动打分。

阶段对比：sparse_only / vector_only / hybrid / hybrid_rerank
指标：Hit@1 / Hit@3 / Hit@5 / MRR / 分类别 / 失败样本列表

用法：
    python scripts/evaluate_rag_retrieval.py --domain tsukiyashiro_kisaki
        [--eval-file PATH] [--top-k 5] [--report] [--no-rerank]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from knowledge.rag_pipeline.embedding import SentenceTransformerEmbeddingProvider  # noqa: E402
from knowledge.rag_pipeline.pipeline import RagPipeline  # noqa: E402
from knowledge.rag_pipeline.registry import get_default_registry  # noqa: E402

if TYPE_CHECKING:
    from knowledge.rag_pipeline.documents import KnowledgeIndexDocument

EVAL_DIR = BACKEND_DIR / "data" / "eval" / "rag_retrieval"
REPORT_DIR = EVAL_DIR / "reports"

STAGES = [
    ("sparse_only", {"mode": "sparse", "use_rerank": False}),
    ("vector_only", {"mode": "vector", "use_rerank": False}),
    ("hybrid", {"mode": "hybrid", "use_rerank": False}),
    ("hybrid_rerank", {"mode": "hybrid", "use_rerank": True}),
]


# ---------------------------------------------------------------------------
# 期望解析（结构化 criteria → 文档 ID 集合）
# ---------------------------------------------------------------------------


def _matches_criteria(doc: KnowledgeIndexDocument, criteria: dict[str, Any]) -> bool:
    meta = doc.metadata or {}
    doc_type = criteria.get("document_type")
    if doc_type and doc.document_type != doc_type:
        return False
    subject = criteria.get("subject")
    if subject and meta.get("subject") != subject:
        return False
    predicate = criteria.get("predicate")
    if predicate and meta.get("predicate") != predicate:
        return False
    value_contains = criteria.get("value_contains")
    if value_contains and value_contains not in str(meta.get("value", "")):
        return False
    relation_type = criteria.get("relation_type")
    if relation_type and meta.get("relation") != relation_type:
        return False
    target = criteria.get("target")
    if target and meta.get("target") != target:
        return False
    title_contains = criteria.get("title_contains")
    if title_contains and title_contains not in doc.title:
        return False
    participant = criteria.get("participants_contain")
    if participant and participant not in (meta.get("participants") or []):
        return False
    entity = criteria.get("entities_contain")
    if entity and entity not in doc.entities:
        return False
    reality = criteria.get("reality_status")
    if reality and doc.reality_status != reality:
        return False
    temporal = criteria.get("temporal_scope")
    if temporal and doc.temporal_scope != temporal:
        return False
    volume = criteria.get("volume")
    return volume is None or meta.get("volume_number") == volume


def resolve_expected_ids(
    documents: Sequence[KnowledgeIndexDocument],
    criteria: dict[str, Any] | None,
) -> set[str]:
    if not criteria:
        return set()
    either_of = criteria.get("either_of")
    if either_of:
        matched: set[str] = set()
        for sub in either_of:
            matched |= resolve_expected_ids(documents, sub)
        return matched
    return {doc.id for doc in documents if _matches_criteria(doc, criteria)}


# ---------------------------------------------------------------------------
# 评估执行
# ---------------------------------------------------------------------------


def run_stage(
    pipeline: RagPipeline,
    eval_entries: list[dict[str, Any]],
    expected_map: dict[str, set[str]],
    stage_name: str,
    stage_kwargs: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    hits_at = {1: 0, 3: 0, 5: 0}
    mrr_total = 0.0
    category_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "hit1": 0, "hit5": 0, "mrr": 0.0})
    failures: list[dict[str, Any]] = []
    no_answer_total = 0

    for entry in eval_entries:
        expected = expected_map[entry["id"]]
        is_no_answer = not expected
        if is_no_answer:
            no_answer_total += 1

        start = time.time()
        # no_answer 类不指定 domain：测试的是自动域门控本身；
        # 其他条目显式指定 domain，隔离检索质量与门控行为
        bundle = pipeline.retrieve(
            entry["query"],
            domain_id=entry.get("domain_id") if entry.get("category") != "no_answer" else None,
            top_k=top_k,
            filters=entry.get("filters"),
            **stage_kwargs,
        )
        elapsed = time.time() - start

        results = (bundle or {}).get("results", [])
        result_ids = [str(r.get("id")) for r in results]

        if is_no_answer:
            # 无答案：正确行为 = 未命中域（None）或返回空结果
            success = bundle is None or not result_ids
            category_stats[entry["category"]]["n"] += 1
            if success:
                category_stats[entry["category"]]["hit1"] += 1
                category_stats[entry["category"]]["hit5"] += 1
                category_stats[entry["category"]]["mrr"] += 1.0
                hits_at[1] += 1
                hits_at[3] += 1
                hits_at[5] += 1
                mrr_total += 1.0
            else:
                failures.append(
                    {
                        "id": entry["id"],
                        "category": entry["category"],
                        "query": entry["query"],
                        "reason": f"no_answer 但返回 {len(result_ids)} 条结果（首条 {result_ids[0]}）",
                    }
                )
            continue

        rank = None
        for idx, rid in enumerate(result_ids, 1):
            if rid in expected:
                rank = idx
                break

        category = entry["category"]
        category_stats[category]["n"] += 1
        if rank is not None:
            mrr_total += 1.0 / rank
            category_stats[category]["mrr"] += 1.0 / rank
            for k in (1, 3, 5):
                if rank <= k:
                    hits_at[k] += 1
            category_stats[category]["hit1"] += 1 if rank <= 1 else 0
            category_stats[category]["hit5"] += 1 if rank <= 5 else 0
        else:
            failures.append(
                {
                    "id": entry["id"],
                    "category": category,
                    "query": entry["query"],
                    "expected_ids": sorted(expected)[:8],
                    "expected_count": len(expected),
                    "returned_ids": result_ids[:8],
                    "elapsed_ms": round(elapsed * 1000, 1),
                }
            )

    n = len(eval_entries)
    return {
        "stage": stage_name,
        "n": n,
        "hit@1": round(hits_at[1] / n, 4) if n else 0.0,
        "hit@3": round(hits_at[3] / n, 4) if n else 0.0,
        "hit@5": round(hits_at[5] / n, 4) if n else 0.0,
        "mrr": round(mrr_total / n, 4) if n else 0.0,
        "categories": {
            k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
            for k, v in category_stats.items()
        },
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P6 离线检索评估")
    parser.add_argument("--domain", default="tsukiyashiro_kisaki", help="评估的域")
    parser.add_argument("--eval-file", default="", help="覆盖评估集路径")
    parser.add_argument("--top-k", type=int, default=5, help="每阶段返回条数")
    parser.add_argument("--report", action="store_true", help="写 JSON 报告到 reports/")
    args = parser.parse_args()

    eval_path = Path(args.eval_file) if args.eval_file else EVAL_DIR / f"{args.domain}.jsonl"
    if not eval_path.exists():
        print(f"评估集不存在: {eval_path}")
        return 1

    eval_entries: list[dict[str, Any]] = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_entries.append(json.loads(line))

    registry = get_default_registry()
    config = registry.require(args.domain)
    documents = config.loader(config.source_root)

    expected_map: dict[str, set[str]] = {}
    invalid: list[str] = []
    for entry in eval_entries:
        expected_spec = entry.get("expected") or {}
        explicit_ids = expected_spec.get("expected_document_ids")
        if explicit_ids:
            expected_map[entry["id"]] = set(str(i) for i in explicit_ids)
        else:
            expected_map[entry["id"]] = resolve_expected_ids(documents, expected_spec.get("criteria"))
        if entry["category"] != "no_answer" and not expected_map[entry["id"]]:
            invalid.append(entry["id"])

    if invalid:
        print(f"[警告] {len(invalid)} 条评估期望解析为空（criteria 未命中任何文档）: {invalid[:5]}")
        for entry in eval_entries:
            if entry["id"] in invalid:
                print(f"  - {entry['id']}: {entry['query']}")

    provider = SentenceTransformerEmbeddingProvider()
    pipeline = RagPipeline(registry=registry, embedding_provider=provider)
    if not pipeline.load_indexes():
        print("索引不可用，请先运行 build_knowledge_index.py")
        return 1

    print(f"评估集: {eval_path}（{len(eval_entries)} 条）")
    print(f"索引文档: {pipeline.domain_stats()[args.domain]['document_count']}")
    print()

    stage_results = []
    for stage_name, stage_kwargs in STAGES:
        result = run_stage(pipeline, eval_entries, expected_map, stage_name, stage_kwargs, args.top_k)
        stage_results.append(result)
        print(
            f"{stage_name:>16}: Hit@1={result['hit@1']:.3f} Hit@3={result['hit@3']:.3f} "
            f"Hit@5={result['hit@5']:.3f} MRR={result['mrr']:.3f} 失败={len(result['failures'])}"
        )

    print()
    print("== 分类别（hybrid_rerank） ==")
    final = stage_results[-1]
    for category in sorted(final["categories"]):
        stats = final["categories"][category]
        print(
            f"  {category:<20}: hit@1={stats['hit1']}/{stats['n']} hit@5={stats['hit5']}/{stats['n']} mrr={stats['mrr']:.3f}"
        )

    if final["failures"]:
        print()
        print("== 失败样本（hybrid_rerank） ==")
        for failure in final["failures"][:20]:
            print(f"  [{failure['id']}] {failure['query']}")
            if "reason" in failure:
                print(f"    原因: {failure['reason']}")
            else:
                print(f"    期望: {failure.get('expected_ids', [])[:4]}（共 {failure['expected_count']}）")
                print(f"    返回: {failure.get('returned_ids', [])[:4]}")

    if args.report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        report_path = REPORT_DIR / f"{args.domain}_{ts}.json"
        payload = {
            "domain": args.domain,
            "eval_file": str(eval_path),
            "entries": len(eval_entries),
            "top_k": args.top_k,
            "index_stats": pipeline.domain_stats(),
            "embedding_model": provider.model_id,
            "stages": stage_results,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n报告已写入: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
