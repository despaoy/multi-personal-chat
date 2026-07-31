"""检索评估指标 - recall@k / MRR / nDCG / faithfulness / answer_correctness。

遵循路线图 guardrail：
- faithfulness 和 answer_correctness 用规则版（关键词覆盖），不依赖 LLM judge
- evaluate_dataset 接受可注入的 retrieve_fn，便于测试和 ablation
"""
from __future__ import annotations

import json
import math
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """简易分词：中文按字符 + 英文按空格。"""
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())


class RetrievalMetrics:
    """检索质量评估指标。"""

    def recall_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
        """Recall@k：前 k 个检索结果中命中期望文档的比例。"""
        if not expected_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = sum(1 for eid in expected_ids if eid in top_k)
        return round(hits / len(expected_ids), 4)

    def mrr(self, retrieved_ids: List[str], expected_ids: List[str]) -> float:
        """MRR：第一个命中期望结果的倒数排名。"""
        for i, rid in enumerate(retrieved_ids, 1):
            if rid in expected_ids:
                return round(1.0 / i, 4)
        return 0.0

    def ndcg(self, retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
        """nDCG@k：归一化折损累积增益。"""
        def dcg(rels: List[float]) -> float:
            return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

        rels = [1.0 if rid in expected_ids else 0.0 for rid in retrieved_ids[:k]]
        ideal_rels = sorted(rels, reverse=True)
        idcg = dcg(ideal_rels)
        if idcg == 0:
            return 0.0
        return round(dcg(rels) / idcg, 4)

    def faithfulness(self, answer: str, citations: List[Dict[str, Any]]) -> float:
        """规则版 faithfulness：答案关键词被引用覆盖的比例。"""
        if not answer or not citations:
            return 0.0
        answer_tokens = set(_tokenize(answer))
        if not answer_tokens:
            return 0.0
        citation_tokens: set = set()
        for c in citations:
            excerpt = c.get("evidence_excerpt", "") or c.get("content", "")
            citation_tokens.update(_tokenize(excerpt))
        if not citation_tokens:
            return 0.0
        covered = answer_tokens & citation_tokens
        return round(len(covered) / len(answer_tokens), 4)

    def answer_correctness(self, answer: str, gold_answer: str) -> float:
        """规则版 answer_correctness：gold 答案关键词重合度。"""
        if not gold_answer or not answer:
            return 0.0
        gold_tokens = set(_tokenize(gold_answer))
        answer_tokens = set(_tokenize(answer))
        if not gold_tokens:
            return 0.0
        overlap = gold_tokens & answer_tokens
        return round(len(overlap) / len(gold_tokens), 4)

    def evaluate_dataset(
        self,
        dataset_path: str,
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
        k: int = 5,
    ) -> Dict[str, Any]:
        """Load a versioned dataset file and evaluate its question records."""
        with open(dataset_path, "r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        questions = dataset.get("questions", []) if isinstance(dataset, dict) else dataset
        if not isinstance(questions, list):
            raise ValueError("retrieval evaluation dataset must contain a questions list")
        return self.evaluate_questions(questions, retrieve_fn, k=k)

    def evaluate_questions(
        self,
        questions: List[Dict[str, Any]],
        retrieve_fn: Callable[[str], List[Dict[str, Any]]],
        k: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate in-memory questions against structured retrieval results."""
        per_question: List[Dict[str, Any]] = []
        recall_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        evaluated = 0

        for index, question in enumerate(questions):
            qid = str(question.get("id", f"q{index}"))
            query = str(question.get("question", ""))
            expected_ids = list(dict.fromkeys(
                str(value).strip()
                for value in question.get("expected_doc_ids", [])
                if str(value).strip()
            ))
            expected_titles = list(dict.fromkeys(
                str(value).strip().casefold()
                for value in question.get("expected_doc_titles", [])
                if str(value).strip()
            ))

            error_type = ""
            try:
                results = retrieve_fn(query) or []
            except Exception as exc:
                logger.warning("检索失败 (%s): %s", qid, exc)
                results = []
                error_type = type(exc).__name__

            if expected_ids:
                expected = expected_ids
                retrieved: List[str] = []
                for rank, result in enumerate(results):
                    aliases = {
                        str(result.get(key)).strip()
                        for key in ("document_id", "documentId", "id", "chunk_id")
                        if result.get(key) is not None and str(result.get(key)).strip()
                    }
                    matched = next((item for item in expected if item in aliases), None)
                    retrieved.append(matched or f"__unmatched_id_{rank}")
                match_basis = "document_id"
            elif expected_titles:
                expected = expected_titles
                retrieved = [
                    str(result.get("title", result.get("original_title", "")))
                    .strip()
                    .casefold()
                    for result in results
                ]
                match_basis = "title"
            else:
                expected = []
                retrieved = []
                match_basis = "none"

            if expected:
                recall_value = self.recall_at_k(retrieved, expected, k)
                mrr_value = self.mrr(retrieved, expected)
                ndcg_value = self.ndcg(retrieved, expected, k)
                recall_sum += recall_value
                mrr_sum += mrr_value
                ndcg_sum += ndcg_value
                evaluated += 1
            else:
                recall_value = 0.0
                mrr_value = 0.0
                ndcg_value = 0.0

            per_question.append({
                "id": qid,
                "question": query,
                "recall_at_k": recall_value,
                "mrr": mrr_value,
                "ndcg_at_k": ndcg_value,
                "retrieved_count": len(results),
                "match_basis": match_basis,
                "evaluable": bool(expected),
                "error_type": error_type,
            })

        return {
            "total": len(questions),
            "evaluated": evaluated,
            "avg_recall_at_k": round(recall_sum / max(evaluated, 1), 4),
            "avg_mrr": round(mrr_sum / max(evaluated, 1), 4),
            "avg_ndcg_at_k": round(ndcg_sum / max(evaluated, 1), 4),
            "k": k,
            "mock": False,
            "per_question": per_question,
        }
    def evaluate_mock(self) -> Dict[str, Any]:
        """Mock 模式：返回预置结果用于 CPU 验证。"""
        return {
            "total": 50,
            "avg_recall_at_k": 0.72,
            "avg_mrr": 0.58,
            "avg_ndcg_at_k": 0.65,
            "k": 5,
            "mock": True,
            "per_question": [
                {"id": "q001", "question": "mock", "recall_at_k": 0.8, "mrr": 0.5, "ndcg_at_k": 0.6, "retrieved_count": 5},
            ],
        }
