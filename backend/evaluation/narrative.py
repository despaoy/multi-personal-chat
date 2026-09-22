"""Small auditable counterfactual benchmark. All default stories are synthetic."""

from __future__ import annotations

from dataclasses import replace

from character.models import CompiledCharacterContext
from inference.generation_request import GenerationRequest, RetrievalResult, build_generation_request

CASES = (
    {
        "id": "meeting",
        "canonical": "林与岚是兄妹，原作中二人已经见面。",
        "a": "林与岚从未见面，血缘关系保持不变。",
        "b": "林与岚昨天已经重逢。",
        "question": "林与岚见过面吗？请说明依据属于哪个世界。",
        "expected": {"canonical": "见过面，兄妹关系不变", "A": "没有见面，兄妹关系不变", "B": "已经重逢"},
    },
    {
        "id": "library",
        "canonical": "原作中图书馆于周一开放，周日关闭。",
        "a": "图书馆周一关闭，其他日期不变。",
        "b": "图书馆周一全天开放。",
        "question": "图书馆周一开放吗？请说明依据属于哪个世界。",
        "expected": {"canonical": "开放", "A": "关闭", "B": "开放"},
    },
)


def packets(turns=10, arms=("A", "B", "C", "D")):
    """A/B shared history; C/D isolated history; D-labels is attribution ablation.

    Teacher-forced dialogue is shared across arms, never a free-running model result.
    Synthetic evidence is identified as such in every packet.
    """
    for case in CASES:
        trajectory = []
        for scope in ("A", "B"):
            for index in range(turns):
                trajectory.extend(
                    [
                        {
                            "scope": scope,
                            "role": "user",
                            "content": f"假想分支{scope}第{index + 1}轮：{case[scope.lower()]}继续讨论这一前提。",
                        },
                        {
                            "scope": scope,
                            "role": "assistant",
                            "content": f"我们正在讨论假想分支{scope}：{case[scope.lower()]}",
                        },
                    ]
                )
        for scope in ("A", "canonical", "B"):
            for arm in arms:
                history = [
                    {"role": h["role"], "content": h["content"]}
                    for h in trajectory
                    if arm in {"A", "B"} or h["scope"] == scope
                ]
                policy = "扮演冷静、简洁且重视证据的图书管理员，直接回答问题。"
                if arm == "B":
                    policy += "区分原作和假设，不要把假想内容当作原作事实。"
                context = CompiledCharacterContext("", "", "")
                message = case["question"]
                if scope != "canonical":
                    message = f"当前是假想分支{scope}，前提：{case[scope.lower()]}\n" + message
                    if arm == "D":
                        context = replace(
                            context, branch_context=f"用户假设（已生效，仅本分支）：{case[scope.lower()]}"
                        )
                    elif arm == "D-labels":
                        # Same isolated history and premise without explicit source labeling.
                        context = replace(context, reference_context=case[scope.lower()])
                else:
                    message = "已切回原作世界。" + message
                plan = build_generation_request(
                    GenerationRequest(
                        message=message,
                        persona_prompt=policy,
                        history=history,
                        retrieval=RetrievalResult(status="ok", evidence="自编测试故事证据：" + case["canonical"]),
                        character_context=context,
                        temperature=0.2,
                        max_tokens=256,
                        context_window_tokens=24576,
                    )
                )
                yield {
                    "id": f"{case['id']}:{turns}:{scope}:{arm}",
                    "case": case["id"],
                    "turns": turns,
                    "scope": scope,
                    "arm": arm,
                    "synthetic": True,
                    "split": "development",
                    "expected": case["expected"][scope],
                    "messages": list(plan.messages),
                    "input_chars": sum(len(m["content"]) for m in plan.messages),
                }


def score(rows):
    """Human-reviewed labels; refusal counts against correctness, never as success.

    Missing labels remain unknown and are reported with coverage. Metrics divide by
    eligible adjudicated probes (including abstentions), not just answered probes.
    """
    groups = {}
    for row in rows:
        groups.setdefault(row["arm"], []).append(row)
    results = {}
    metrics = {
        "canonical_contamination_rate": ("contaminated", lambda r: r["scope"] == "canonical"),
        "canonical_accuracy": ("correct", lambda r: r["scope"] == "canonical"),
        "branch_consistency_rate": ("correct", lambda r: r["scope"] != "canonical"),
        "cross_branch_leakage_rate": ("cross_branch_leak", lambda r: r["scope"] != "canonical"),
        "provenance_accuracy": ("provenance_correct", lambda r: True),
        "abstention_rate": ("abstained", lambda r: True),
    }
    for arm, items in groups.items():
        output = {"total": len(items)}
        for name, (field, eligible) in metrics.items():
            probes = [r for r in items if eligible(r)]
            labelled = [r for r in probes if isinstance(r.get("labels", {}).get(field), bool)]
            successes = sum(
                bool(r["labels"][field])
                and not (field in {"correct", "provenance_correct"} and r["labels"].get("abstained") is True)
                for r in labelled
            )
            output[name] = {
                "value": successes / len(labelled) if labelled else None,
                "numerator": successes,
                "denominator": len(labelled),
                "eligible": len(probes),
                "unknown": len(probes) - len(labelled),
            }
        results[arm] = output
    return results
