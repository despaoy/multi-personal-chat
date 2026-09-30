"""Resample the original approved 001–038 interactions under their final prompt."""
import concurrent.futures,json,sys,hashlib
from pathlib import Path
from expand_kisaki_dpo_sample import generate,readl,BASE,ROOT
from training.persona_sampling import digest
from training.chat_dataset import normalize_chat_record
old=BASE.parent/'v4_interactive_20260926'
original={r['id']:r for r in readl(BASE.parent/'interactive_source_v2_20260926/interactive_source.pending.jsonl')}
system=(ROOT/'backend/data/character_dialogues/kisaki_system_prompt_v3.txt').read_text(encoding='utf-8').strip()
rows=[]
for s in readl(old/'sft.interactive.approved.jsonl'):
 m=s['metadata'];r=original[s['id']];src=r['source'];f=ROOT/'gametext/纸上魔法使'/src['source_file']
 row=dict(number=m['final_review_number'],id=s['id'],prompt=normalize_chat_record(s,default_system_prompt=system,system_prompt_policy='replace')[:-1],original=s['messages'][-1]['content'],source=src,context_start=m['source_line_start'],context_end=src['source_line_start']-1,source_file_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),source_review={'decision':'keep','reason':'原001–038已完成关系/语气/情景终审，本次按终审输入重新采样。'},prior_sft_id=s['id'])
 row['record_sha256']=digest(row);rows.append(row)
snapshot=BASE/'legacy_source_ready.jsonl'
if snapshot.exists() and readl(snapshot)!=rows:raise ValueError('legacy prompt changed')
snapshot.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
path=BASE/'legacy_generations.jsonl';done={(r['scene_id'],r['sample_index']) for r in readl(path) if r['status']=='generated'}
jobs=[(r,i) for r in rows for i in range(8) if (r['id'],i) not in done]
with path.open('a',encoding='utf-8') as f,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for r in pool.map(generate,jobs):
  f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
  print(json.dumps({k:r[k] for k in ('number','sample_index','status')}),flush=True)
