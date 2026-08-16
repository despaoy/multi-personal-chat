#!/usr/bin/env python3
"""Build a self-contained browser review page for the canonical Kisaki train set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "backend/data/character_dialogues/experiments/v4/train.jsonl"
DEFAULT_OUTPUT = (
    ROOT / "docs/research/review_packets/kisaki_v4/ALL_TRAINING_DATA_REVIEW.html"
)
DEFAULT_FULL_REVIEW = (
    ROOT
    / "backend/data/character_dialogues/experiments/v4/augmentation_candidates"
    / "llm_full_dialogue_review_20260816/record_reviews.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("training data must contain one JSON object per line")
    return rows


def browser_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def build_html(rows: list[dict], source_path: Path, reviews: list[dict] | None = None) -> str:
    sources = Counter(row.get("metadata", {}).get("data_source", "unknown") for row in rows)
    tasks = Counter(row.get("metadata", {}).get("task_type", "未分类") for row in rows)
    assistant_turns = sum(
        message.get("role") == "assistant"
        for row in rows
        for message in row.get("messages", [])
    )
    assistant_chars = sum(
        len(message.get("content", ""))
        for row in rows
        for message in row.get("messages", [])
        if message.get("role") == "assistant"
    )
    summary = {
        "records": len(rows),
        "assistantTurns": assistant_turns,
        "assistantChars": assistant_chars,
        "sources": dict(sorted(sources.items())),
        "tasks": dict(sorted(tasks.items())),
        "sourcePath": source_path.relative_to(ROOT).as_posix(),
    }

    review_map = {review["record_id"]: review for review in (reviews or [])}

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>月社妃 V4.1 全部训练数据审阅</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3f5f7;
      --surface: #ffffff;
      --line: #d8dde3;
      --text: #18202a;
      --muted: #65707e;
      --user: #eef5ff;
      --assistant: #f7f8fa;
      --accent: #1d5da8;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.55 system-ui, "Microsoft YaHei", sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 10; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.97); }}
    .header-inner, main {{ width: min(1440px, calc(100% - 32px)); margin: 0 auto; }}
    .header-inner {{ padding: 14px 0 12px; }}
    h1 {{ margin: 0 0 8px; font-size: 20px; font-weight: 680; letter-spacing: 0; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 8px 18px; color: var(--muted); font-size: 13px; }}
    .controls {{ display: grid; grid-template-columns: minmax(240px, 1fr) 240px 220px auto auto; gap: 8px; margin-top: 12px; }}
    input, select, button {{ min-height: 36px; border: 1px solid #bfc7d1; border-radius: 6px; background: #fff; color: var(--text); font: inherit; }}
    input, select {{ padding: 6px 10px; }}
    button {{ padding: 6px 12px; cursor: pointer; }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    main {{ padding: 16px 0 40px; }}
    .result-line {{ margin-bottom: 10px; color: var(--muted); }}
    .record {{ margin-bottom: 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); overflow: hidden; }}
    .record > summary {{ display: grid; grid-template-columns: 64px minmax(260px, 1fr) 210px 180px 90px; gap: 10px; align-items: center; padding: 10px 12px; cursor: pointer; list-style: none; }}
    .record > summary::-webkit-details-marker {{ display: none; }}
    .record > summary:hover {{ background: #f8fafc; }}
    .index {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
    .record-id {{ overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }}
    .tag {{ overflow: hidden; color: var(--muted); text-overflow: ellipsis; white-space: nowrap; }}
    .turn-count {{ text-align: right; color: var(--muted); }}
    .record-body {{ border-top: 1px solid var(--line); padding: 12px; }}
    .scene {{ margin: 0 0 10px; font-weight: 650; }}
    .messages {{ display: grid; gap: 8px; }}
    .message {{ display: grid; grid-template-columns: 88px minmax(0, 1fr); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }}
    .message.user {{ background: var(--user); }}
    .message.assistant {{ background: var(--assistant); }}
    .role {{ padding: 9px 10px; border-right: 1px solid var(--line); color: var(--muted); font-weight: 650; }}
    .content {{ margin: 0; padding: 9px 11px; overflow-wrap: anywhere; white-space: pre-wrap; font: inherit; }}
    .metadata {{ margin-top: 10px; }}
    .metadata summary {{ cursor: pointer; color: var(--accent); }}
    .metadata pre {{ max-height: 360px; overflow: auto; padding: 10px; border: 1px solid var(--line); border-radius: 6px; background: #f8fafc; white-space: pre-wrap; font-size: 12px; }}
    .review {{ margin-top: 12px; border: 1px solid #b9cbe0; border-radius: 6px; background: #f5f9fe; }}
    .review > summary {{ padding: 9px 10px; cursor: pointer; color: var(--accent); font-weight: 650; }}
    .review-body {{ padding: 0 10px 10px; }}
    .review-summary {{ margin: 0 0 8px; color: var(--muted); }}
    .turn-review {{ margin-top: 8px; padding: 9px 10px; border: 1px solid #d4dfeb; border-radius: 6px; background: #fff; }}
    .turn-review h4 {{ margin: 0 0 6px; font-size: 13px; }}
    .turn-review dl {{ display: grid; grid-template-columns: 190px minmax(0, 1fr); gap: 4px 10px; margin: 0; }}
    .turn-review dt {{ color: var(--muted); }}
    .turn-review dd {{ margin: 0; overflow-wrap: anywhere; }}
    .empty {{ padding: 32px; border: 1px dashed #bfc7d1; border-radius: 6px; background: var(--surface); text-align: center; color: var(--muted); }}
    @media (max-width: 900px) {{
      .header-inner, main {{ width: min(100% - 20px, 1440px); }}
      header {{ position: static; }}
      .controls {{ grid-template-columns: 1fr 1fr; }}
      .controls input {{ grid-column: 1 / -1; }}
      .record > summary {{ grid-template-columns: 48px minmax(0, 1fr); }}
      .record > summary .tag, .record > summary .turn-count {{ grid-column: 2; }}
      .turn-count {{ text-align: left; }}
      .message {{ grid-template-columns: 72px minmax(0, 1fr); }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>月社妃 V4.1 全部训练数据审阅</h1>
      <div class="summary" id="summary"></div>
      <div class="controls">
        <input id="search" type="search" placeholder="搜索 ID、场景或对话正文">
        <select id="source"><option value="">全部来源</option></select>
        <select id="task"><option value="">全部任务类型</option></select>
        <button id="openAll" type="button">展开当前</button>
        <button id="closeAll" type="button">全部收起</button>
      </div>
    </div>
  </header>
  <main>
    <div class="result-line" id="resultLine"></div>
    <div id="records"></div>
  </main>
  <script>
    const records = {browser_json(rows)};
    const stats = {browser_json(summary)};
    const reviews = {browser_json(review_map)};
    const state = {{ query: "", source: "", task: "" }};
    const recordsNode = document.getElementById("records");
    const resultLine = document.getElementById("resultLine");

    function sourceOf(record) {{ return record.metadata?.data_source || "unknown"; }}
    function taskOf(record) {{ return record.metadata?.task_type || "未分类"; }}
    function sceneOf(record) {{ return record.metadata?.scene || record.metadata?.source || "未标注场景"; }}
    function searchText(record) {{
      const review = reviews[record.id];
      return [record.id, sourceOf(record), taskOf(record), sceneOf(record), ...(record.messages || []).map(m => m.content || ""), review ? JSON.stringify(review) : ""]
        .join("\\n").toLocaleLowerCase();
    }}
    records.forEach(record => record.__search = searchText(record));

    function addOptions(selectId, counts) {{
      const select = document.getElementById(selectId);
      Object.entries(counts).sort((a, b) => a[0].localeCompare(b[0], "zh-CN")).forEach(([name, count]) => {{
        const option = document.createElement("option");
        option.value = name;
        option.textContent = `${{name}} (${{count}})`;
        select.append(option);
      }});
    }}

    function messageNode(message, index) {{
      const row = document.createElement("div");
      row.className = `message ${{message.role || "unknown"}}`;
      const role = document.createElement("div");
      role.className = "role";
      role.textContent = `${{index + 1}} · ${{message.role || "unknown"}}`;
      const content = document.createElement("pre");
      content.className = "content";
      content.textContent = message.content || "";
      row.append(role, content);
      return row;
    }}

    function recordNode(record, absoluteIndex) {{
      const details = document.createElement("details");
      details.className = "record";
      const summary = document.createElement("summary");
      const values = [
        ["index", String(absoluteIndex + 1).padStart(4, "0")],
        ["record-id", record.id || "无 ID"],
        ["tag", sourceOf(record)],
        ["tag", taskOf(record)],
        ["turn-count", `${{(record.messages || []).filter(m => m.role === "assistant").length}} 轮`],
      ];
      values.forEach(([className, text]) => {{
        const node = document.createElement("span");
        node.className = className;
        node.textContent = text;
        summary.append(node);
      }});
      const body = document.createElement("div");
      body.className = "record-body";
      const scene = document.createElement("p");
      scene.className = "scene";
      scene.textContent = sceneOf(record);
      const messages = document.createElement("div");
      messages.className = "messages";
      (record.messages || []).forEach((message, index) => messages.append(messageNode(message, index)));
      const metadata = document.createElement("details");
      metadata.className = "metadata";
      const metadataSummary = document.createElement("summary");
      metadataSummary.textContent = "查看完整 metadata";
      const metadataText = document.createElement("pre");
      metadataText.textContent = JSON.stringify(record.metadata || {{}}, null, 2);
      metadata.append(metadataSummary, metadataText);
      body.append(scene, messages);
      const review = reviews[record.id];
      if (review) {{
        const reviewDetails = document.createElement("details");
        reviewDetails.className = "review";
        const reviewHeading = document.createElement("summary");
        const revised = review.revised_assistant_turns?.length
          ? `已优化第 ${{review.revised_assistant_turns.join("、")}} 个 Assistant 回复`
          : "原文通过";
        reviewHeading.textContent = `逐轮综合评价 · ${{revised}}`;
        const reviewBody = document.createElement("div");
        reviewBody.className = "review-body";
        const reviewSummary = document.createElement("p");
        reviewSummary.className = "review-summary";
        reviewSummary.textContent = review.evaluation_summary || "";
        reviewBody.append(reviewSummary);
        (review.turn_reviews || []).forEach(turn => {{
          const turnNode = document.createElement("section");
          turnNode.className = "turn-review";
          const title = document.createElement("h4");
          title.textContent = `Assistant 第 ${{turn.turn}} 轮 · ${{turn.decision === "pass" ? "通过" : "修订后通过"}}`;
          const list = document.createElement("dl");
          Object.entries(turn.analysis || {{}}).forEach(([name, value]) => {{
            if (value === null || value === "") return;
            const term = document.createElement("dt");
            term.textContent = name;
            const description = document.createElement("dd");
            description.textContent = String(value);
            list.append(term, description);
          }});
          turnNode.append(title, list);
          reviewBody.append(turnNode);
        }});
        reviewDetails.append(reviewHeading, reviewBody);
        body.append(reviewDetails);
      }}
      body.append(metadata);
      details.append(summary, body);
      return details;
    }}

    function render() {{
      const query = state.query.trim().toLocaleLowerCase();
      const visible = records
        .map((record, index) => [record, index])
        .filter(([record]) => (!query || record.__search.includes(query)) && (!state.source || sourceOf(record) === state.source) && (!state.task || taskOf(record) === state.task));
      recordsNode.replaceChildren();
      resultLine.textContent = `显示 ${{visible.length}} / ${{records.length}} 条`;
      if (!visible.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "没有符合当前条件的训练记录";
        recordsNode.append(empty);
        return;
      }}
      const fragment = document.createDocumentFragment();
      visible.forEach(([record, index]) => fragment.append(recordNode(record, index)));
      recordsNode.append(fragment);
    }}

    document.getElementById("summary").textContent = `记录 ${{stats.records}} · Assistant 轮次 ${{stats.assistantTurns}} · Assistant 字符 ${{stats.assistantChars}} · ${{stats.sourcePath}}`;
    addOptions("source", stats.sources);
    addOptions("task", stats.tasks);
    let searchTimer;
    document.getElementById("search").addEventListener("input", event => {{
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {{ state.query = event.target.value; render(); }}, 120);
    }});
    document.getElementById("source").addEventListener("change", event => {{ state.source = event.target.value; render(); }});
    document.getElementById("task").addEventListener("change", event => {{ state.task = event.target.value; render(); }});
    document.getElementById("openAll").addEventListener("click", () => document.querySelectorAll(".record").forEach(node => node.open = true));
    document.getElementById("closeAll").addEventListener("click", () => document.querySelectorAll(".record").forEach(node => node.open = false));
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--full-review", type=Path, default=DEFAULT_FULL_REVIEW)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    reviews = load_jsonl(args.full_review) if args.full_review.is_file() else []
    html = build_html(rows, args.input.resolve(), reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "records": len(rows), "reviewed_records": len(reviews), "bytes": len(html.encode("utf-8"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
