"""Offline, descriptive audit report; never turns generated replies into passes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def latency_summary(rows: list[dict], key: str) -> dict:
    # Missing measurements stay missing, never get counted as zero latency.
    values = sorted(float(row[key]) for row in rows
                    if isinstance(row.get(key), (int, float)) and math.isfinite(row[key]))
    if not values:
        return {'samples': 0, 'p50_seconds': None, 'p95_seconds': None}
    return {'samples': len(values), 'p50_seconds': values[math.ceil(.5 * len(values)) - 1],
            'p95_seconds': values[math.ceil(.95 * len(values)) - 1]}


def summarize(rows: list[dict]) -> dict:
    budgets = [row['prepared']['memory_budget'] for row in rows
               if isinstance(row.get('prepared', {}).get('memory_budget'), dict)]
    recalls = [row['prepared']['memory_recall'] for row in rows
               if isinstance(row.get('prepared', {}).get('memory_recall'), dict)]
    model_totals = []
    for row in rows:
        calls = row.get('model_calls')
        known_no_model = row.get('generation', {}).get('model_invoked') is False
        if (isinstance(calls, list) and (calls or known_no_model)
                and all(isinstance(call.get('seconds'), (int, float)) for call in calls)):
            total = sum(call['seconds'] for call in calls)
            observation = {'model_seconds': total}
            if isinstance(row.get('seconds'), (int, float)) and row['seconds'] >= total:
                # Residual includes routing, RAG, preparation, writes, and overhead;
                # it is not a direct measurement of retrieval latency.
                observation['non_model_seconds'] = row['seconds'] - total
            model_totals.append(observation)
    return {
        'turns': len(rows),
        'cases': len({r['case_id'] for r in rows}),
        'status': dict(Counter(r.get('status', 'unknown') for r in rows)),
        'retrieval': dict(Counter(r.get('generation', {}).get('plan', {}).get('retrieval', {}).get('status', 'no_model_trace') for r in rows)),
        'guard_retries': sum(bool(r.get('generation', {}).get('guard_retried')) for r in rows),
        'guard_fallbacks': dict(Counter(r['generation']['guard_fallback'] for r in rows if r.get('generation', {}).get('guard_fallback'))),
        'observer_errors': sum(bool(r.get('observer_error')) for r in rows),
        'latency': {key: latency_summary(rows, key) for key in ('seconds', 'prepare_seconds', 'write_seconds')},
        'model_calls': sum(len(r.get('model_calls', [])) for r in rows),
        'response_modes': dict(Counter(r.get('generation', {}).get('response_mode', 'unobserved') for r in rows)),
        'generation_latency': {key: latency_summary(model_totals, key)
                               for key in ('model_seconds', 'non_model_seconds')},
        'memory_recall_observations': len(recalls),
        'memory_recall_status': dict(Counter(r.get('status', 'unknown') for r in recalls)),
        'memory_semantic_status': dict(Counter(r.get('semantic_status', 'unknown') for r in recalls)),
        'memory_budget_observations': len(budgets),
        'memory_budget_skipped': sum(b.get('budget_skipped', 0) for b in budgets) if budgets else None,
        'similarity_score': None,
        'assessment': 'Descriptive traces only; generated_not_yet_reviewed is not a passing grade.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, flat = {}, []
    lines = ['# 无 LoRA 逐轮链路审计', '',
             '本文件只整理真实调用证据，不自动评定人物相似度。生成成功不等于回答合格。', '']
    for source in args.trace:
        rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
        summaries[source.name] = summarize(rows)
        lines.extend([f'## {source.name}', '', f'完整原始追踪：{source.resolve()}', ''])
        for row in rows:
            generation = row.get('generation', {})
            compiled = row.get('prepared', {}).get('compiled', {})
            retrieval = generation.get('plan', {}).get('retrieval', {})
            response = row.get('response', {})
            record = dict(run=source.name, case_id=row['case_id'], title=row.get('title', ''),
                turn=row.get('turn', 'memory-only'), status=row.get('status'),
                history=len(row.get('prepared', {}).get('history', [])),
                memory_status=compiled.get('memory_status'), memory_ids=','.join(compiled.get('used_memory_ids', [])),
                retrieval=retrieval.get('status'), citations=len(response.get('citations') or []),
                model_calls=len(row.get('model_calls', [])), fallback=generation.get('guard_fallback', ''),
                message=row.get('message', ''), reply=response.get('reply', ''), rubric=row.get('rubric', ''),
                error=row.get('error', ''))
            flat.append(record)
            lines.extend([f"### {record['case_id']} / {record['title']} / {record['turn']}", '',
                f"用户：{record['message']}", '', f"回复：{record['reply']}", '',
                f"检查要求：{record['rubric'] or '移除历史后的证据使用；不得猜测。'}", '',
                f"链路：历史 {record['history']} 条 → 记忆 {record['memory_status']} ({record['memory_ids']})"
                f" → RAG {record['retrieval']} / 引用 {record['citations']} → 模型调用 {record['model_calls']}"
                f" / 回退 {record['fallback'] or '无'} → {record['status']}", '',
                *([f"错误：{record['error']}", ''] if record['error'] else [])])
    (args.output / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'review.md').write_text('\n'.join(lines), encoding='utf-8')
    if flat:
        with (args.output / 'turns.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
