"""Resume real multi-candidate sampling on semantically reviewed source interactions."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import chat_completion,digest
from inference.prompt_policy import build_grounded_user_message

BASE=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
MODEL={'name':'deepseek_general','model':'deepseek-chat','revision':'deepseek-chat-unpinned-api-alias','base_url':'https://api.deepseek.com/v1','api_key_env':'DEEPSEEK_API_KEY'}
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def readl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []

def materialize():
    inventory={r['number']:r for r in load(BASE/'inventory.json')}
    decisions=load(BASE/'source_review.json')
    system=(ROOT/'backend/data/character_dialogues/kisaki_system_prompt_v3.txt').read_text(encoding='utf-8').strip()
    rows=[]
    for d in decisions:
        if d['decision']!='keep':continue
        r=inventory[d['number']];source=r['source'];start=d.get('context_start',r['context_start']);end=source['source_line_start']-1
        f=ROOT/'gametext/纸上魔法使'/source['source_file'];lines=f.read_text(encoding='utf-8-sig').splitlines()
        events=[e for e in r['history_events'] if e['line_start']>=start]
        if not events:raise ValueError('missing interlocutor')
        current=events[-1]
        reference='当前场景：'+d['scene']+'\n妃的关系与状态：'+d['state']+'\n回复前原文（旁白可能是他人内心，不自动等于妃知道的事）：\n'+'\n'.join(f'L{i+1}: {lines[i]}' for i in range(start-1,end))
        user=build_grounded_user_message(current['text'],reference,max_chars=len(reference),speaker=current['speaker'])
        row={'number':d['number'],'id':f'kisaki_expand_{d["number"]:03d}_{digest(source)[:12]}',
            'prompt':[{'role':'system','content':system},{'role':'user','content':user}],
            'original':r['original'],'source':source,'context_start':start,'context_end':end,
            'source_file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'source_review':d,'human_source_admission_authorized':True}
        row['record_sha256']=digest(row);rows.append(row)
    return rows

def generate(job):
    row,index=job;params={'temperature':[0.7,0.95,1.1,1.2,1.0,1.15][index%6],'top_p':0.95,'max_tokens':384,'seed':1042+index}
    result={'number':row['number'],'scene_id':row['id'],'sample_index':index,'id':row['id']+f'_sample_{index}',
        'prompt':row['prompt'],'input_sha256':digest(row['prompt']),'source_record_sha256':row['record_sha256'],'model':MODEL,'generation':params}
    try:
        result['generated']=chat_completion(MODEL,row['prompt'],params,timeout=90)
        result['status']='generated'
    except (ValueError,RuntimeError,OSError) as exc:
        result.update(status='error',error_type=type(exc).__name__)
    result['record_sha256']=digest(result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=int,default=4);p.add_argument('--through',type=int,required=True);p.add_argument('--numbers',default='');a=p.parse_args()
    if not 1<=a.samples<=12:raise ValueError('samples must be 1..12')
    if not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('DEEPSEEK_API_KEY unset')
    rows=materialize();snapshot=BASE/'source_ready.jsonl'
    old={r['id']:r for r in readl(snapshot)}
    for r in rows:
        if r['id'] in old and old[r['id']]!=r:raise ValueError('source changed after materialization')
    snapshot.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    path=BASE/'generations.jsonl';prior=readl(path)
    for r in prior:
        if r['record_sha256']!=digest({k:v for k,v in r.items() if k!='record_sha256'}):raise ValueError('journal integrity')
    done={(r['scene_id'],r['sample_index']) for r in prior if r['status']=='generated'}
    selected={int(n) for n in a.numbers.split(',') if n} if a.numbers else None
    jobs=[(r,i) for r in rows if r['number']<=a.through and (selected is None or r['number'] in selected) and r['number'] not in {50,100,171} for i in range(a.samples) if (r['id'],i) not in done]
    with path.open('a',encoding='utf-8') as f,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(generate,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            r=future.result();f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({k:r[k] for k in ('number','sample_index','status')}),flush=True)

if __name__=='__main__':main()
