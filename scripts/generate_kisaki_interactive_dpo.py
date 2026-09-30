"""Generate real candidates for reviewed short interactions; resumable, no auto approval."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from training.persona_sampling import chat_completion, digest

BASE = ROOT / 'backend/data/character_dialogues/experiments/interactive_source_v2_20260926'
MODEL = {'name':'deepseek_general','model':'deepseek-chat','revision':'deepseek-chat-unpinned-api-alias','base_url':'https://api.deepseek.com/v1','api_key_env':'DEEPSEEK_API_KEY'}
GEN = {'temperature':0.7,'top_p':0.95,'max_tokens':384,'seed':42}


def readl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []


def generate(r):
    result = {'id':r['id'],'prompt':r['prompt'],'input_sha256':digest(r['prompt']),
        'source_record_sha256':r['record_sha256'],'model':MODEL,'generation':GEN,
        'human_preference_approved':False,'original':r['original']}
    try:
        result['generated'] = chat_completion(MODEL,r['prompt'],GEN,timeout=90)
        result['generated_sha256'] = digest(result['generated'])
        result['status'] = 'generated'
    except (ValueError,RuntimeError,OSError) as exc:
        result['status'] = 'error'
        result['error_type'] = type(exc).__name__
        # Transport messages have been sanitized, never retain API response bodies.
        result['error'] = str(exc) if str(exc).startswith('model endpoint returned HTTP ') else 'generation failed; no response substituted'
    result['record_sha256'] = digest(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if not os.environ.get('DEEPSEEK_API_KEY'):
        raise SystemExit('DEEPSEEK_API_KEY is unset')
    rows=[r for r in readl(BASE/'interactive_source.pending.jsonl') if r['review_status']=='pending']
    journal=BASE/'deepseek_interactive_results.jsonl'
    old=readl(journal)
    for r in old:
        if r['record_sha256']!=digest({k:v for k,v in r.items() if k!='record_sha256'}):
            raise ValueError('journal hash mismatch')
    done={r['id']:r for r in old if r['status']=='generated'}
    todo=[]
    for r in rows:
        if r['id'] in done:
            if done[r['id']]['source_record_sha256']!=r['record_sha256'] or done[r['id']]['prompt']!=r['prompt']:
                raise ValueError('source changed: use a new generation journal')
        else: todo.append(r)
    if a.limit:todo=todo[:a.limit]
    with journal.open('a',encoding='utf-8') as f, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(generate,todo):
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({'id':r['id'],'status':r['status'],'error':r.get('error')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
