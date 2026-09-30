"""Append shorter-answer interactions omitted by the first mechanical filter."""
import json,re
from pathlib import Path
from expand_kisaki_dpo_inventory import ROOT,DATA,OUT,readl
from extract_character_dialogues import read_script_events,dialogue_link_reason,reliable_speaker_label
from build_kisaki_interactive_review import heldout
raw=readl(DATA/'tsukiyashiro_kisaki_raw.jsonl');by={(r['source_file'],r['source_line_start']):r for r in raw}
blocked,ranges=heldout(raw,readl(DATA/'experiments/v4/validation.jsonl'),json.loads((ROOT/'backend/evaluation/kisaki_gold_set_v3.json').read_text(encoding='utf-8')))
m=json.loads((DATA/'experiments/v4/canonical_dataset_manifest.json').read_text(encoding='utf-8'))
ids=set(re.findall(r'tsukiyashiro_kisaki_raw_[0-9a-f]+',json.dumps(m.get('rag_holdout',{}))))
for r in raw:
 if r['id'] in ids:blocked.add(r['scene_block_id']);ranges.setdefault(r['source_file'],[]).append((r['source_line_start'],r['source_line_end']))
rows=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
used={(r['source']['source_file'],r['source']['source_line_start']) for r in rows+readl(DATA/'experiments/interactive_source_v2_20260926/interactive_source.pending.jsonl')}
for f in sorted((ROOT/'gametext/纸上魔法使').glob('*.txt')):
 es=read_script_events(f);lines=f.read_text(encoding='utf-8-sig').splitlines()
 for i,e in enumerate(es):
  if e['speaker']!='妃' or (f.name,e['line_start']) in used or e['scene_block_id'] in blocked or i==0:continue
  prev=es[i-1]
  if prev['speaker']=='妃' or dialogue_link_reason(prev,e) or not reliable_speaker_label(prev['speaker']):continue
  if not 5<=len(e['text'])<=85 or not 2<=len(prev['text'])<=180 or not re.search(r'[\u4e00-\u9fff]',prev['text']):continue
  if e['text'].endswith(('——','，','：')):continue
  start=max(int(e['scene_block_id'].rsplit(':',1)[-1]),e['line_start']-28)
  history=[x for x in es[:i] if x['line_start']>=start]
  if not history:continue
  start=history[0]['line_start']
  if any(a<=e['line_end'] and b>=start for a,b in ranges.get(f.name,[])):continue
  rows.append(dict(number=39+len(rows),source=by[(f.name,e['line_start'])],history_events=history,context_start=start,context_end=e['line_start']-1,original=e['text'],context='\n'.join(f'L{x+1}: {lines[x]}' for x in range(start-1,e['line_end'])),status='pending_semantic_context_review'))
(OUT/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('last number',rows[-1]['number'])
