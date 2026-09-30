"""Build review documents and JSONL from individually authored dialogue cases.

This is an offline document builder, not an API or model test runner.
Run from any directory: python docs/testing/dialogue_suite_v1/build_suite.py
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]

# Ordered candidate investigation points, not assertions of confirmed defects.
GROUPS = {
    1: ("P1", "quality", ["backend/character/profile_registry.py", "backend/character/decision_policy.py", "backend/inference/generation_request.py"], "角色ID、画像版本、最终人物提示、实际模型及LoRA、三轮回复"),
    2: ("P1", "mixed", ["backend/services/character_context.py", "backend/character/conversation_flow.py", "backend/inference/generation_request.py"], "请求现场history、数据库历史、24条窗口、便签、裁剪后的最终提示"),
    3: ("P1", "quality", ["backend/character/situation_analyzer.py", "backend/character/decision_policy.py", "backend/character/output_guard.py"], "规则状态、用户行为标签、任务约束、策略、守卫诊断和生成调用数"),
    4: ("P0", "mixed", ["backend/character/situation_analyzer.py", "backend/character/natural_relationship.py", "backend/character/output_guard.py"], "本轮明确边界、实际注入边界、守卫违规类型、纠正次数、最终回复"),
    5: ("P1", "mixed", ["backend/character/memory_extractor.py", "backend/character/rule_memory_writer.py", "backend/character/memory_service.py"], "提取项、原句evidence、source_message_ids、作用域、写入完成时刻、检索与注入内容"),
    6: ("P1", "boundary_probe", ["backend/character/memory_extractor.py", "backend/character/rule_memory_writer.py", "backend/character/memory_service.py"], "极性、qualifiers、active与pending、完整条件和尾部否定、候选与实际注入差异"),
    7: ("P0", "mixed", ["backend/character/rule_memory_writer.py", "backend/character/memory_llm.py", "backend/repositories/character_memory.py"], "逻辑条目ID、版本链、observed_at、valid_from、生命周期、删除保护、后台任务与提交顺序"),
    8: ("P0", "contract", ["backend/api/generate.py", "backend/character/context_builder.py", "backend/services/character_context.py", "backend/api/characters.py"], "可信登录身份、规范化platform/adapter/sender/conversation、scope_level、授权原文、查询过滤条件"),
    9: ("P1", "mixed", ["backend/character/event_memory.py", "backend/character/rule_memory_writer.py", "backend/character/memory_service.py"], "metadata.event、完整事项名、state、日期、匹配候选数、原句和替代版本"),
    10: ("P1", "boundary_probe", ["backend/character/event_memory.py", "backend/character/memory_service.py", "backend/services/character_context.py"], "固定UTC/+08时钟、事件日期与状态、同名候选数、结果未知呈现、未支持表达原文"),
    11: ("P1", "contract", ["backend/character/natural_relationship.py", "backend/services/character_context.py", "backend/api/characters.py"], "关系命令解析、relationship键、候选覆盖、成功后提交、手动起点、计数和备忘录版本"),
    12: ("P1", "mixed", ["backend/character/natural_relationship.py", "backend/character/conversation_flow.py", "backend/character/output_guard.py"], "valid_to、筛选前后备忘录、8条/300字符预算、便签180字符、实际调用数和模式"),
    13: ("P1", "mixed", ["backend/knowledge/multiscale_rag/runtime.py", "backend/knowledge/multiscale_rag/service.py", "backend/knowledge/multiscale_rag/source_text.py", "backend/knowledge/grounded_answer/validator.py", "backend/api/generate.py"], "域与粒度、query_analysis、召回ID与分数、父场景、证据包、引用、abstained、answerMode"),
    14: ("P0", "mixed", ["backend/api/knowledge.py", "backend/knowledge/rag_helper.py", "backend/knowledge/vector_db.py", "backend/cache/response_cache.py"], "knowledge_base_id、所有权、文档和索引版本、分块、引用、更新完成状态、缓存键"),
    15: ("P0", "contract", ["backend/services/chat_generation.py", "backend/infra/input_validator.py", "backend/inference/generation_request.py", "backend/knowledge/multiscale_rag/source_text.py", "src/components/dashboard/TestChatDialog.tsx"], "输入校验、鉴权、转义、消息role、系统/参考区分隔、网络状态、浏览器脚本标记"),
    16: ("P0", "contract", ["backend/api/narrative.py", "backend/services/narrative.py", "backend/api/generate.py"], "branchId、归属、revision、假设/提议/确认事实、依赖与来源、CAS结果、事务前后快照"),
    17: ("P0", "contract", ["astrbot_plugins/multipersonal_gateway/main.py", "backend/api/integrations.py", "backend/db/integration_receipts.py", "backend/services/character_context.py"], "规范化事件、复合幂等键、traceId、模型次数、send结果、delivery确认、状态提交次数"),
    18: ("P0", "mixed", ["backend/services/chat_generation.py", "backend/infra/concurrency_control.py", "backend/infra/circuit_breaker.py", "backend/cache/response_cache.py", "backend/inference/generation_request.py"], "队列入出时间、锁键、限流层、故障注入点、预算、缓存命中、重试次数、事务结果"),
    19: ("P1", "mixed", ["src/components/dashboard/TestChatDialog.tsx", "src/components/dashboard/SessionManagerDialog.tsx", "src/components/history/HistoryMessagesCard.tsx", "src/lib/api.ts", "backend/api/messages.py"], "界面选择、请求payload、响应、loading/error、history来源、消息分页和过滤、身份清理"),
    20: ("P1", "mixed", ["backend/api/generate.py", "backend/inference/lora_router.py", "backend/inference/generation_request.py", "backend/character/semantic_state_estimator.py", "backend/character/evidence_selector.py", "backend/api/preferences.py"], "characterId与LoRA映射、画像/模型/配置版本、实验开关、实际证据ID、调用数、导出来源及评测分母"),
}


def load_cases():
    sections: dict[int, tuple[str, list[list[str]]]] = {}
    current = None
    for number, raw in enumerate((ROOT / "authoring.txt").read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.startswith("# "):
            continue
        if raw.startswith("## "):
            match = re.fullmatch(r"## (\d{2}) (.+)", raw)
            if not match:
                raise ValueError(f"Bad heading on line {number}")
            current = int(match[1])
            if current in sections:
                raise ValueError(f"Duplicate group {current}")
            sections[current] = (match[2], [])
            continue
        parts = raw.split("|")
        if current is None or len(parts) != 8 or any(not p.strip() for p in parts):
            raise ValueError(f"Line {number}: expected 8 nonempty fields, got {len(parts)}")
        sections[current][1].append(parts)
    if set(sections) != set(GROUPS):
        raise ValueError("Expected exactly groups 01..20")
    cases = []
    for group in sorted(sections):
        name, rows = sections[group]
        if len(rows) != 15:
            raise ValueError(f"Group {group}: expected 15 cases, got {len(rows)}")
        priority, basis, paths, observations = GROUPS[group]
        for path in paths:
            if not (REPO / path).exists():
                raise ValueError(f"Unknown investigation source {path}")
        for row in rows:
            title, setup, *pairs = row
            steps = []
            for step in range(3):
                raw_input, expected = pairs[2 * step:2 * step + 2]
                actions = []
                remaining = raw_input
                while remaining.startswith("【"):
                    action, remaining = remaining[1:].split("】", 1)
                    actions.append(action)
                # Some steps intentionally inspect or replay a transport event instead of
                # appending a new utterance. This is a human execution specification.
                steps.append({
                    "step": step + 1,
                    "operator_actions": actions,
                    "text": remaining.replace("\\n", "\n"),
                    "expected_checks": expected,
                    "execution_mode": "operator_assisted" if actions else "send_user_message",
                    "actual_response": None,
                    "verdict": "not_run",
                })
            case_id = f"D{len(cases) + 1:03d}"
            cases.append({
                "schema_version": "1.0",
                "id": case_id,
                "group_id": f"G{group:02d}",
                "group": name,
                "title": title,
                "priority": priority,
                "assessment_track": basis,
                "setup": setup,
                "default_setup_ref": "README.md#执行约定",
                "fixture_ref": "fixtures.md",
                "steps": steps,
                "observe": observations.split("、"),
                "investigation_paths": paths,
                "diagnosis_rule": "按README从入口到最终交付找最早偏离；候选路径不是已确认根因。",
                "design_status": "authored",
                "execution_status": "not_run",
            })
    return cases


def write_outputs(cases):
    out = ROOT / "batches"
    out.mkdir(exist_ok=True)
    index = ["# 300条对话测试索引", "", "设计稿；全部未执行。每个编号是独立场景，每个场景有3个检查步骤。", "", "先读 [执行约定与判分](README.md) 和 [测试夹具](fixtures.md)。", "", "| 批次 | 编号 | 场景数 | 内容 |", "| --- | --- | ---: | --- |"]
    for group in range(1, 21):
        subset = [c for c in cases if c["group_id"] == f"G{group:02d}"]
        filename = f"{group:02d}_{subset[0]['id']}_{subset[-1]['id']}.md"
        name = subset[0]["group"]
        index.append(f"| {group:02d} | {subset[0]['id']}–{subset[-1]['id']} | 15 | [{name}](batches/{filename}) |")
        lines = [f"# {name}（{subset[0]['id']}—{subset[-1]['id']}）", "", "所有场景状态：未执行。预期检查不是模型实际回答。", "", "适用共同约定见 [README](../README.md)，固定事实见 [fixtures](../fixtures.md)。", ""]
        for case in subset:
            lines += [f"## {case['id']} {case['title']}", "", f"- 优先级：{case['priority']}；评估轨道：`{case['assessment_track']}`。", f"- 前置条件：{case['setup']}。", "- 每例重置：专用测试用户、会话与夹具；仅本例内保持连续。", "", "| 步骤 | 发送前操作 | 用户输入或操作核对提示 | 预期检查 |", "| --- | --- | --- | --- |"]
            for step in case["steps"]:
                action = "；".join(step["operator_actions"]) or "正常发送并等待本轮完成"
                text = step["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                lines.append(f"| {step['step']} | {action} | {text} | {step['expected_checks']} |")
            links = " → ".join(f"[{Path(p).name}](../../../../{p})" for p in case["investigation_paths"])
            lines += ["", f"观察证据：{'、'.join(case['observe'])}。", "", f"候选排查入口：{links}。", "", "记录：`未执行`；后续填写逐步结果、原始证据路径、最早偏离环节和复测结论。", ""]
        (out / filename).write_text("\n".join(lines), encoding="utf-8")
    index += ["", "## 逐项标题", ""]
    for c in cases:
        index.append(f"- {c['id']} · {c['title']}")
    (ROOT / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    with (ROOT / "cases.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        for c in cases:
            file.write(json.dumps(c, ensure_ascii=False) + "\n")
    template_path = ROOT / "results.template.jsonl"
    with template_path.open("w", encoding="utf-8", newline="\n") as file:
        for c in cases:
            file.write(json.dumps({"case_id": c["id"], "run_id": None, "status": "not_run", "runtime_manifest": None, "steps": [{"step": s["step"], "request_id": None, "trace_id": None, "response": None, "evidence_paths": [], "mechanism_verdict": "not_run", "semantic_verdict": "not_run", "quality_score": None} for s in c["steps"]], "first_divergence": None, "candidate_cause": None, "confirmed_cause": None, "issue_id": None, "retest_of": None}, ensure_ascii=False) + "\n")
    titles = [c["title"] for c in cases]
    triples = [tuple(s["text"] for s in c["steps"]) for c in cases]
    repeated = [list(t) for t, count in Counter(triples).items() if count > 1]
    if len(set(titles)) != len(cases):
        raise ValueError("Duplicate case titles")
    # D179/D180 intentionally share input as an explicit lightweight/strict control.
    for triple in repeated:
        matched = [c for c in cases if [s["text"] for s in c["steps"]] == triple]
        if {c["title"] for c in matched} != {"轻量模式风格仅诊断", "严格模式对照"}:
            raise ValueError(f"Unexplained identical dialogues: {[c['id'] for c in matched]}")
    checked_links = 0
    for document in ROOT.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            local_target = target.split("#", 1)[0]
            if not (document.parent / local_target).exists():
                raise ValueError(f"Broken local link in {document.name}: {target}")
            checked_links += 1
    fixture_files = sorted((ROOT / "fixture_data").glob("*.txt"))
    lighthouse_lines = (ROOT / "fixture_data/lighthouse_source.txt").read_text(encoding="utf-8").splitlines()
    if len(lighthouse_lines) != 12:
        raise ValueError("Lighthouse source must retain its 12 oracle lines")
    for name in ("cases.jsonl", "results.template.jsonl"):
        loaded = [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]
        if len(loaded) != 300:
            raise ValueError(f"Expected 300 JSON records in {name}")
    report = {
        "validation_scope": "offline_design_integrity_only",
        "cases": len(cases),
        "groups": len(GROUPS),
        "steps": sum(len(c["steps"]) for c in cases),
        "ids_contiguous": [c["id"] for c in cases] == [f"D{i:03d}" for i in range(1, 301)],
        "unique_titles": len(set(titles)),
        "group_counts": dict(sorted(Counter(c["group_id"] for c in cases).items())),
        "intentional_duplicate_input_groups": len(repeated),
        "investigation_paths_exist": True,
        "local_markdown_links_checked": checked_links,
        "fixture_files": len(fixture_files),
        "fixture_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in fixture_files},
        "rag_oracle_source_lines": len(lighthouse_lines),
        "executed_cases": 0,
        "software_pass_count": None,
        "software_fail_count": None,
        "authoring_sha256": hashlib.sha256((ROOT / "authoring.txt").read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256((ROOT / "cases.jsonl").read_bytes()).hexdigest(),
    }
    if len(cases) != 300 or not report["ids_contiguous"]:
        raise ValueError("Expected exactly D001..D300")
    (ROOT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_outputs(load_cases())
