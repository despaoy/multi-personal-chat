"""Evaluate the balanced CAHM memory lifecycle and retrieval design.

The evaluator deliberately keeps model-backed relation judgement separate from
retrieval.  Relation metrics are reported as N/A when no reachable Memory LLM
has been configured; they are never replaced with rule-based proxy numbers.
Retrieval always compares the legacy fixed-score hybrid path with the balanced
RRF/query-expansion/version-filter/evidence path over the same Gold records.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from character.memory_extractor import extract_memories  # noqa: E402
from character.memory_llm import (  # noqa: E402
    MAX_EXISTING_MEMORIES,
    MemoryCompletion,
    MemoryLlmConfig,
    OpenAICompatibleMemoryCompletion,
    ValidatedMemoryProposal,
    build_memory_llm_messages,
    classify_memory_write_mode,
    parse_llm_proposals,
)
from character.memory_service import (  # noqa: E402
    MIN_HYBRID_MEMORY_SCORE,
    CharacterMemoryService,
)
from character.models import UserScope  # noqa: E402
from knowledge.retrieval_core.embedding import (  # noqa: E402
    EmbeddingProvider,
    get_default_embedding_provider,
)

DEFAULT_DATASET = BACKEND_ROOT / "evaluation" / "data" / "cahm_balanced_gold.jsonl"
DEFAULT_OUTPUT = BACKEND_ROOT / "evaluation" / "results" / "cahm_balanced_results.json"

_TARGET_REQUIRED_OPERATIONS = frozenset(("MERGE", "SUPERSEDE", "COEXIST", "RETRACT", "ERASE"))
_LEAKAGE_STATUSES = ("pending", "superseded", "retracted", "archived")


@dataclass(frozen=True)
class RetrievalVariant:
    name: str
    description: str
    rrf_enabled: bool
    query_expansion_enabled: bool
    version_filter_enabled: bool
    evidence_enabled: bool


RETRIEVAL_VARIANTS = (
    RetrievalVariant(
        name="legacy_hybrid",
        description="Legacy fixed weighted hybrid score without expansion, lifecycle filtering, or evidence packets",
        rrf_enabled=False,
        query_expansion_enabled=False,
        version_filter_enabled=False,
        evidence_enabled=False,
    ),
    RetrievalVariant(
        name="balanced_default",
        description="Balanced RRF retrieval with query expansion, lifecycle filtering, and evidence packets",
        rrf_enabled=True,
        query_expansion_enabled=True,
        version_filter_enabled=True,
        evidence_enabled=True,
    ),
)


class _CaseRepository:
    """Small repository adapter that accepts both legacy and versioned rows."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self.set_records(records or [])

    def set_records(self, records: list[dict[str, Any]]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        normalized: list[dict[str, Any]] = []
        for source in records:
            row = dict(source)
            # Old experiment fixtures used memory_id while the current service
            # returns MemoryItem IDs from id.  Normalize only at the adapter
            # boundary and leave lifecycle aliases for the service to decode.
            if row.get("id") in (None, "") and row.get("memory_id") not in (None, ""):
                row["id"] = row["memory_id"]
            row.setdefault("updated_at", now)
            normalized.append(row)
        self.records = normalized

    async def list_memory_records(self, character_id, user_scope, limit=100):
        return self.records[:limit]


class _AuditedEmbeddingProvider:
    """Expose service-level lexical fallback instead of silently mislabelling it."""

    def __init__(self, delegate: EmbeddingProvider) -> None:
        self._delegate = delegate
        self.failures: list[str] = []

    @property
    def model_id(self) -> str:
        return str(getattr(self._delegate, "model_id", type(self._delegate).__name__))

    @property
    def dimension(self) -> int:
        return int(self._delegate.dimension)

    def embed_texts(self, texts: list[str]):
        try:
            return self._delegate.embed_texts(texts)
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise

    def embed_query(self, query: str):
        try:
            return self._delegate.embed_query(query)
        except Exception as exc:
            self.failures.append(f"{type(exc).__name__}: {exc}")
            raise


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL object at {path}:{line_number}")
        if not str(row.get("id") or "").strip():
            raise ValueError(f"missing case id at {path}:{line_number}")
        if not str(row.get("task") or "").strip():
            raise ValueError(f"missing task at {path}:{line_number}")
        rows.append(row)
    return rows


def _record_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("memory_id") or "").strip()


def _record_status(row: dict[str, Any]) -> str:
    return str(row.get("status") or row.get("memory_status") or "active").strip().lower()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "average_ms": statistics.mean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
    }


def _classification_metrics(gold: list[str], predicted: list[str]) -> tuple[float, dict[str, Any]]:
    labels = sorted(set(gold) | set(predicted))
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted, strict=True))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted, strict=True))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "support": sum(item == label for item in gold),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return (statistics.mean(f1_values) if f1_values else 0.0), per_label


def _predicted_status(operation: str) -> str | None:
    if operation == "PENDING":
        return "pending"
    if operation == "RETRACT":
        return "retracted"
    if operation == "ERASE":
        return "erased"
    if operation in {"ADD", "MERGE", "SUPERSEDE", "COEXIST"}:
        return "active"
    return None


def _primary_relation(proposals: list[ValidatedMemoryProposal]) -> tuple[str, str, str, str | None]:
    if not proposals:
        return "NOOP", "", "", None
    primary = proposals[0]
    return (
        primary.operation,
        primary.target_memory_id,
        primary.target_memory_key,
        _predicted_status(primary.operation),
    )


def _relation_not_evaluated(case_count: int) -> dict[str, Any]:
    return {
        "evaluated": False,
        "reason": "memory_llm_not_configured",
        "cases": case_count,
        "successfully_processed_cases": 0,
        "operation_accuracy": None,
        "operation_macro_f1": None,
        "target_accuracy": None,
        "target_evaluated_cases": 0,
        "status_accuracy": None,
        "status_evaluated_cases": 0,
        "per_operation": {},
        "latency_ms": None,
        "failures": [],
    }


async def _evaluate_relations(
    cases: list[dict[str, Any]],
    config: MemoryLlmConfig | None,
    *,
    completion: MemoryCompletion | None = None,
) -> dict[str, Any]:
    if config is None or not config.enabled:
        return _relation_not_evaluated(len(cases))

    model_completion = completion or OpenAICompatibleMemoryCompletion(config)
    operation_gold: list[str] = []
    operation_predictions: list[str] = []
    operation_correct = 0
    target_correct = target_total = 0
    status_correct = status_total = 0
    successfully_processed = 0
    failures: list[dict[str, Any]] = []
    latencies: list[float] = []

    try:
        for case in cases:
            message = str(case.get("message") or "")
            history = tuple(case.get("history") or ())[-4:]
            existing = tuple(case.get("existing_memories") or ())[:MAX_EXISTING_MEMORIES]
            feedback_target_ids = tuple(str(item) for item in case.get("preferred_memory_ids") or ())
            raw_output = ""
            error = ""
            proposals: list[ValidatedMemoryProposal] = []
            started = time.perf_counter()
            try:
                raw_output = await model_completion.complete(
                    build_memory_llm_messages(
                        message,
                        tuple(extract_memories(message)),
                        history,
                        existing,
                        config.max_input_chars,
                        config.confidence_threshold,
                        feedback_target_ids=feedback_target_ids,
                        write_mode=classify_memory_write_mode(message),
                    )
                )
                proposals = parse_llm_proposals(
                    raw_output,
                    source_message=message,
                    history=history,
                    existing_memories=existing,
                    confidence_threshold=config.confidence_threshold,
                    feedback_target_ids=feedback_target_ids,
                )
                successfully_processed += 1
                predicted_operation, predicted_target_id, predicted_target_key, predicted_status = _primary_relation(
                    proposals
                )
            except Exception as exc:  # keep partial, auditable results instead of inventing predictions
                error = f"{type(exc).__name__}: {exc}"
                predicted_operation = "ERROR"
                predicted_target_id = ""
                predicted_target_key = ""
                predicted_status = None
            latencies.append((time.perf_counter() - started) * 1000.0)

            gold_operation = str(case.get("gold_operation") or "NOOP").strip().upper()
            gold_status = str(case.get("gold_status") or "").strip().lower() or None
            gold_target_id = str(case.get("gold_target_memory_id") or "").strip()
            gold_target_key = str(case.get("gold_target_memory_key") or "").strip()
            operation_gold.append(gold_operation)
            operation_predictions.append(predicted_operation)
            operation_match = predicted_operation == gold_operation
            operation_correct += int(operation_match)

            # NOOP may mention an unchanged claim in the Gold for audit purposes,
            # but the production protocol only requires targets for mutating/linking operations.
            target_evaluated = gold_operation in _TARGET_REQUIRED_OPERATIONS and bool(gold_target_id or gold_target_key)
            target_match: bool | None = None
            if target_evaluated:
                target_total += 1
                target_match = (
                    predicted_target_id == gold_target_id if gold_target_id else predicted_target_key == gold_target_key
                )
                target_correct += int(target_match)

            status_match: bool | None = None
            if gold_status is not None:
                status_total += 1
                status_match = predicted_status == gold_status
                status_correct += int(status_match)

            failed = bool(error or not operation_match or (target_match is False) or (status_match is False))
            if failed:
                failures.append(
                    {
                        "id": str(case.get("id") or ""),
                        "category": str(case.get("category") or ""),
                        "message": message,
                        "gold": {
                            "operation": gold_operation,
                            "target_memory_id": gold_target_id,
                            "target_memory_key": gold_target_key,
                            "status": gold_status,
                        },
                        "predicted": {
                            "operation": predicted_operation,
                            "target_memory_id": predicted_target_id,
                            "target_memory_key": predicted_target_key,
                            "status": predicted_status,
                            "all_operations": [item.operation for item in proposals],
                        },
                        "error": error,
                        "raw_llm_output": raw_output,
                    }
                )
    finally:
        await model_completion.close()

    macro_f1, per_operation = _classification_metrics(operation_gold, operation_predictions)
    return {
        "evaluated": True,
        "reason": "",
        "cases": len(cases),
        "successfully_processed_cases": successfully_processed,
        "operation_accuracy": _safe_ratio(operation_correct, len(cases)),
        "operation_macro_f1": macro_f1,
        "target_accuracy": _safe_ratio(target_correct, target_total),
        "target_evaluated_cases": target_total,
        "status_accuracy": _safe_ratio(status_correct, status_total),
        "status_evaluated_cases": status_total,
        "per_operation": per_operation,
        "latency_ms": _latency_summary(latencies),
        "failures": failures,
    }


def _service_for_variant(
    repository: _CaseRepository,
    variant: RetrievalVariant,
    provider: EmbeddingProvider,
    *,
    min_hybrid_score: float,
    candidate_limit: int,
) -> tuple[CharacterMemoryService, dict[str, Any], list[str]]:
    requested: dict[str, Any] = {
        "embedding_provider": provider,
        "semantic_enabled": True,
        "gate_enabled": True,
        "min_hybrid_score": min_hybrid_score,
        "candidate_limit": candidate_limit,
        "include_pending": False,
        "rrf_enabled": variant.rrf_enabled,
        "query_expansion_enabled": variant.query_expansion_enabled,
        "version_filter_enabled": variant.version_filter_enabled,
        "evidence_enabled": variant.evidence_enabled,
    }
    parameters = inspect.signature(CharacterMemoryService.__init__).parameters
    effective = {key: value for key, value in requested.items() if key in parameters}
    unsupported = sorted(set(requested) - set(effective))
    return CharacterMemoryService(repository, **effective), effective, unsupported


def _selected_evidence_complete(item: Any) -> bool:
    evidence = tuple(getattr(item, "evidence", ()) or ())
    source_ids = tuple(getattr(item, "source_message_ids", ()) or getattr(item, "source_ids", ()) or ())
    return bool(evidence and source_ids)


async def _evaluate_retrieval_variant(
    cases: list[dict[str, Any]],
    variant: RetrievalVariant,
    provider: EmbeddingProvider,
    *,
    min_hybrid_score: float = MIN_HYBRID_MEMORY_SCORE,
    candidate_limit: int = 100,
) -> dict[str, Any]:
    repository = _CaseRepository()
    audited_provider = _AuditedEmbeddingProvider(provider)
    service, effective_config, unsupported_switches = _service_for_variant(
        repository,
        variant,
        audited_provider,
        min_hybrid_score=min_hybrid_score,
        candidate_limit=candidate_limit,
    )
    load_parameters = inspect.signature(service.load_relevant_memories).parameters
    historical_query_control_supported = "include_historical" in load_parameters

    relevant_cases = 0
    recall_at_1_values: list[float] = []
    recall_at_5_values: list[float] = []
    reciprocal_ranks: list[float] = []
    total_injected = wrong_injected = 0
    leakage_counts = {status: 0 for status in _LEAKAGE_STATUSES}
    invalid_time_leakage = 0
    evidence_required = evidence_complete = 0
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    completed_cases = 0

    for case in cases:
        records = list(case.get("records") or ())
        repository.set_records(records)
        rows_by_id = {_record_id(row): row for row in records if _record_id(row)}
        query = str(case.get("query") or "")
        call_kwargs: dict[str, Any] = {}
        if historical_query_control_supported:
            call_kwargs["include_historical"] = bool(case.get("include_historical"))
        started = time.perf_counter()
        error = ""
        try:
            selected, _candidate_count = await service.load_relevant_memories(
                "cahm-balanced-eval",
                UserScope("evaluation", "cahm", "synthetic-user", "synthetic-user", "private"),
                query,
                **call_kwargs,
            )
            completed_cases += 1
        except Exception as exc:  # preserve the failure in the report; an error is not a successful retrieval
            selected = ()
            error = f"{type(exc).__name__}: {exc}"
        latencies.append((time.perf_counter() - started) * 1000.0)

        selected_ids = [str(getattr(item, "memory_id", "")) for item in selected]
        selected_by_id = {str(getattr(item, "memory_id", "")): item for item in selected}
        gold_ids = {str(item) for item in case.get("gold_ids") or ()}
        if gold_ids:
            relevant_cases += 1
            recall_at_1_values.append(len(gold_ids.intersection(selected_ids[:1])) / len(gold_ids))
            recall_at_5_values.append(len(gold_ids.intersection(selected_ids[:5])) / len(gold_ids))
            rank = next((index for index, item_id in enumerate(selected_ids, start=1) if item_id in gold_ids), None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        wrong_ids = [item_id for item_id in selected_ids if item_id not in gold_ids]
        total_injected += len(selected_ids)
        wrong_injected += len(wrong_ids)

        leaked_ids = {status: [] for status in _LEAKAGE_STATUSES}
        invalid_time_ids: list[str] = []
        now = datetime.now(timezone.utc)
        for item_id in selected_ids:
            row = rows_by_id.get(item_id, {})
            status = _record_status(row)
            # A historical Gold claim is intentional retrieval, not leakage.
            # Lifecycle leakage is a selected non-Gold row whose source status
            # should have excluded it from this answer.
            if status in leakage_counts and item_id not in gold_ids:
                leakage_counts[status] += 1
                leaked_ids[status].append(item_id)
            valid_from = _parse_timestamp(row.get("valid_from") or row.get("valid_at"))
            valid_to = _parse_timestamp(row.get("valid_to") or row.get("invalid_at"))
            if item_id not in gold_ids and (
                (valid_from is not None and valid_from > now) or (valid_to is not None and valid_to <= now)
            ):
                invalid_time_leakage += 1
                invalid_time_ids.append(item_id)

        missing_evidence_ids: list[str] = []
        if bool(case.get("require_evidence")):
            for item_id in sorted(gold_ids):
                evidence_required += 1
                item = selected_by_id.get(item_id)
                if item is not None and _selected_evidence_complete(item):
                    evidence_complete += 1
                else:
                    missing_evidence_ids.append(item_id)

        misses = sorted(gold_ids.difference(selected_ids[:5]))
        if error or misses or wrong_ids or any(leaked_ids.values()) or invalid_time_ids or missing_evidence_ids:
            failures.append(
                {
                    "id": str(case.get("id") or ""),
                    "category": str(case.get("category") or ""),
                    "query": query,
                    "gold_ids": sorted(gold_ids),
                    "selected_ids": selected_ids,
                    "missed_gold_ids_at_5": misses,
                    "wrong_ids": wrong_ids,
                    "leaked_ids": leaked_ids,
                    "invalid_time_ids": invalid_time_ids,
                    "missing_evidence_ids": missing_evidence_ids,
                    "error": error,
                }
            )

    latency = _latency_summary(latencies)
    leakage = {f"{status}_count": leakage_counts[status] for status in _LEAKAGE_STATUSES}
    leakage.update(
        {
            f"{status}_rate": leakage_counts[status] / total_injected if total_injected else 0.0
            for status in _LEAKAGE_STATUSES
        }
    )
    leakage["invalid_time_count"] = invalid_time_leakage
    leakage["invalid_time_rate"] = invalid_time_leakage / total_injected if total_injected else 0.0

    return {
        "name": variant.name,
        "description": variant.description,
        "requested_configuration": {
            **asdict(variant),
            "semantic_enabled": True,
            "gate_enabled": True,
            "min_hybrid_score": min_hybrid_score,
            "candidate_limit": candidate_limit,
        },
        "effective_configuration": {
            key: value for key, value in effective_config.items() if key != "embedding_provider"
        },
        "unsupported_service_switches": unsupported_switches,
        "historical_query_control_supported": historical_query_control_supported,
        "cases": len(cases),
        "successfully_processed_cases": completed_cases,
        "relevant_cases": relevant_cases,
        "recall_at_1": statistics.mean(recall_at_1_values) if recall_at_1_values else 0.0,
        "recall_at_5": statistics.mean(recall_at_5_values) if recall_at_5_values else 0.0,
        "mrr": statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "wrong_memory_injection_rate": wrong_injected / total_injected if total_injected else 0.0,
        "injected_memories": total_injected,
        "wrong_injected_memories": wrong_injected,
        "lifecycle_leakage": leakage,
        "evidence_completeness": _safe_ratio(evidence_complete, evidence_required),
        "evidence_complete_claims": evidence_complete,
        "evidence_required_claims": evidence_required,
        "average_retrieval_latency_ms": latency["average_ms"],
        "latency_ms": latency,
        "embedding_failure_count": len(audited_provider.failures),
        "embedding_failures": audited_provider.failures[:20],
        "failures": failures,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _memory_llm_config_from_args(args: Any) -> MemoryLlmConfig | None:
    base_url = str(getattr(args, "memory_llm_base_url", "") or "").strip()
    model = str(getattr(args, "memory_llm_model", "") or "").strip()
    if bool(base_url) != bool(model):
        raise ValueError("--memory-llm-base-url and --memory-llm-model must be provided together")
    if base_url and model:
        return MemoryLlmConfig(
            enabled=True,
            base_url=base_url,
            model=model,
            api_key=str(getattr(args, "memory_llm_api_key", "") or ""),
            timeout_seconds=float(getattr(args, "memory_llm_timeout", 30.0)),
            confidence_threshold=float(getattr(args, "memory_llm_confidence_threshold", 0.85)),
        )
    env_config = MemoryLlmConfig.from_env()
    return env_config if env_config.enabled else None


async def evaluate(
    args: Any,
    *,
    embedding_provider: EmbeddingProvider | None = None,
    memory_completion: MemoryCompletion | None = None,
) -> dict[str, Any]:
    dataset = Path(args.dataset)
    rows = _load_jsonl(dataset)
    relation_cases = [row for row in rows if str(row.get("task") or "").startswith("relation")]
    retrieval_cases = [row for row in rows if str(row.get("task") or "").startswith("retrieval")]

    llm_config = _memory_llm_config_from_args(args)
    relation = await _evaluate_relations(relation_cases, llm_config, completion=memory_completion)

    provider = embedding_provider or get_default_embedding_provider()
    # Fail explicitly before producing a report if the configured embedding is
    # unavailable.  Silent lexical fallback would mislabel the comparison.
    provider.embed_texts(["CAHM balanced evaluation warmup"])
    min_hybrid_score = float(getattr(args, "min_hybrid_score", MIN_HYBRID_MEMORY_SCORE))
    candidate_limit = max(1, int(getattr(args, "candidate_limit", 100)))
    retrieval: dict[str, Any] = {}
    for variant in RETRIEVAL_VARIANTS:
        retrieval[variant.name] = await _evaluate_retrieval_variant(
            retrieval_cases,
            variant,
            provider,
            min_hybrid_score=min_hybrid_score,
            candidate_limit=candidate_limit,
        )

    return {
        "method": "CAHM recommended balanced memory evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(dataset),
            "total": len(rows),
            "relation": len(relation_cases),
            "retrieval": len(retrieval_cases),
        },
        "configuration": {
            "embedding_model": str(getattr(provider, "model_id", type(provider).__name__)),
            "memory_llm_evaluated": relation["evaluated"],
            "memory_llm_model": llm_config.model if llm_config is not None else "",
            "metric_definitions": {
                "target_accuracy": "Only Gold operations that require a target are counted",
                "wrong_memory_injection_rate": "Selected non-Gold memory items divided by all selected memory items",
                "lifecycle_leakage_rate": "Selected rows of that original status divided by all selected memory items",
                "evidence_completeness": "Required Gold claims retrieved with both evidence text and source message IDs",
                "latency": "Wall-clock retrieval latency after one provider warmup; no values are synthesized",
            },
        },
        "relation": relation,
        "retrieval": retrieval,
    }


def _fmt(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _render_markdown(report: dict[str, Any]) -> str:
    relation = report["relation"]
    lines = [
        "# CAHM Recommended Balanced Evaluation",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Gold Set: {report['dataset']['total']} cases "
        f"({report['dataset']['relation']} relation, {report['dataset']['retrieval']} retrieval)",
        f"- Embedding: {report['configuration']['embedding_model']}",
        "",
        "## Relation judgement",
        "",
        "| Evaluated | Operation accuracy | Operation macro-F1 | Target accuracy | Status accuracy | Processed |",
        "|---|---:|---:|---:|---:|---:|",
        f"| {relation['evaluated']} | {_fmt(relation['operation_accuracy'])} | "
        f"{_fmt(relation['operation_macro_f1'])} | {_fmt(relation['target_accuracy'])} | "
        f"{_fmt(relation['status_accuracy'])} | "
        f"{relation['successfully_processed_cases']}/{relation['cases']} |",
        "",
    ]
    if not relation["evaluated"]:
        lines.extend(
            [
                "Memory LLM was not configured, so relation operation/target/status accuracy and macro-F1 are N/A. "
                "No rule-based proxy was substituted.",
                "",
            ]
        )

    lines.extend(
        [
            "## Retrieval comparison",
            "",
            "| Variant | R@1 | R@5 | MRR | Wrong injection | Pending leak | Superseded leak | Retracted leak | Evidence | Avg ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in report["retrieval"].values():
        leakage = result["lifecycle_leakage"]
        lines.append(
            f"| {result['name']} | {_fmt(result['recall_at_1'])} | {_fmt(result['recall_at_5'])} | "
            f"{_fmt(result['mrr'])} | {_fmt(result['wrong_memory_injection_rate'])} | "
            f"{_fmt(leakage['pending_rate'])} | {_fmt(leakage['superseded_rate'])} | "
            f"{_fmt(leakage['retracted_rate'])} | {_fmt(result['evidence_completeness'])} | "
            f"{_fmt(result['average_retrieval_latency_ms'], digits=2)} |"
        )

    lines.extend(
        [
            "",
            "Leakage rates use each selected row's original lifecycle status. Evidence completeness requires both "
            "evidence text and a source message ID; a missed required claim is incomplete.",
            "",
            "## Failure cases",
            "",
            f"### Relation ({len(relation['failures'])} total)",
            "",
        ]
    )
    if not relation["evaluated"]:
        lines.append("N/A because no Memory LLM was configured.")
    elif not relation["failures"]:
        lines.append("No relation failures under the Gold definition.")
    else:
        for item in relation["failures"][:20]:
            lines.append(
                f"- `{item['id']}` gold={item['gold']['operation']}/"
                f"{item['gold']['target_memory_id'] or '-'}; predicted={item['predicted']['operation']}/"
                f"{item['predicted']['target_memory_id'] or '-'}"
                + (f"; error={item['error']}" if item["error"] else "")
            )

    for result in report["retrieval"].values():
        lines.extend(["", f"### {result['name']} retrieval ({len(result['failures'])} total)", ""])
        if not result["failures"]:
            lines.append("No retrieval failures under the Gold definition.")
            continue
        for item in result["failures"][:20]:
            lines.append(
                f"- `{item['id']}` gold={item['gold_ids']}; selected={item['selected_ids']}; "
                f"wrong={item['wrong_ids']}; leaked={item['leaked_ids']}; "
                f"missing_evidence={item['missing_evidence_ids']}"
                + (f"; error={item['error']}" if item["error"] else "")
            )

    lines.extend(
        [
            "",
            "## Effective ablation controls",
            "",
        ]
    )
    for result in report["retrieval"].values():
        lines.append(
            f"- `{result['name']}`: {json.dumps(result['effective_configuration'], ensure_ascii=False, sort_keys=True)}; "
            f"unsupported={result['unsupported_service_switches']}; "
            f"historical_query_control_supported={result['historical_query_control_supported']}; "
            f"embedding_failures={result['embedding_failure_count']}"
        )
    lines.append("")
    return "\n".join(lines)


def _write_report(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    return output, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--memory-llm-base-url", default="")
    parser.add_argument("--memory-llm-model", default="")
    parser.add_argument("--memory-llm-api-key", default="")
    parser.add_argument("--memory-llm-timeout", type=float, default=30.0)
    parser.add_argument("--memory-llm-confidence-threshold", type=float, default=0.85)
    parser.add_argument("--min-hybrid-score", type=float, default=MIN_HYBRID_MEMORY_SCORE)
    parser.add_argument("--candidate-limit", type=int, default=100)
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    json_path, markdown_path = _write_report(report, args.output)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
