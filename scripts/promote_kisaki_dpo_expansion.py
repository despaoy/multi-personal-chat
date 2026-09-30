"""Audit and export an ordered, provenance-bound successor training snapshot."""
import copy,difflib,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'scripts')]
from training.persona_sampling import digest
from training.preference_validation import fingerprint,reviewed_pair_digest,validate_pairs
from training.chat_dataset import normalize_chat_record
from build_kisaki_interactive_review import heldout
from build_kisaki_gold_v3 import contamination_audit
from extract_character_dialogues import read_script_events
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
OLD=P.parent/'v4_interactive_20260926';V4=P.parent/'v4';OUT=P.parent/'v4_dpo100_20260926'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def readl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def writel(p,x):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x),encoding='utf-8',newline='\n')
def clean(s):return re.sub(r'[^\w\u4e00-\u9fff]','',s).casefold()
def main():
 frozen=[V4/'train.jsonl',V4/'validation.jsonl',V4/'canonical_dataset_manifest.json',ROOT/'backend/evaluation/kisaki_gold_set_v3.json',OLD/'train.jsonl',OLD/'dpo.train.jsonl']
 before={str(p.relative_to(ROOT)):sha(p) for p in frozen}
 raw=readl(P.parent.parent/'tsukiyashiro_kisaki_raw.jsonl');manifest=load(V4/'canonical_dataset_manifest.json');gold=load(ROOT/'backend/evaluation/kisaki_gold_set_v3.json')
 blocked,ranges=heldout(raw,readl(V4/'validation.jsonl'),gold)
 ragids=set(re.findall(r'tsukiyashiro_kisaki_raw_[0-9a-f]+',json.dumps(manifest.get('rag_holdout',{}))))
 for r in raw:
  if r['id'] in ragids:blocked.add(r['scene_block_id']);ranges.setdefault(r['source_file'],[]).append((r['source_line_start'],r['source_line_end']))
 sources=readl(P/'legacy_source_ready.jsonl')+readl(P/'source_ready.jsonl')
 byid={r['id']:r for r in sources};valid={};audit=[];sft=[]
 reviews=load(P/'candidate_review.json');reviewby={r['candidate_id']:r for r in reviews}
 assert len(reviewby)==len(reviews)
 generations={}
 for path in ['generations.jsonl','general_generations.jsonl','legacy_generations.jsonl']:
  for r in readl(P/path):
   assert r['record_sha256']==digest({k:v for k,v in r.items() if k!='record_sha256'}),'generation hash'
   if r['status']=='generated':generations[r['id']]=r
 system=(P.parent.parent/'kisaki_system_prompt_v3.txt').read_text(encoding='utf-8').strip()
 cached={}
 for r in sorted(sources,key=lambda r:r['number']):
  n=r['number'];src=r['source'];f=ROOT/'gametext/纸上魔法使'/src['source_file']
  assert r['record_sha256']==digest({k:v for k,v in r.items() if k!='record_sha256'})
  assert sha(f)==r['source_file_sha256'],'source changed'
  if f not in cached:cached[f]=read_script_events(f)
  matches=[e for e in cached[f] if e['line_start']==src['source_line_start'] and e['speaker']=='妃']
  assert len(matches)==1 and matches[0]['text']==r['original'],'literal target'
  hold=None
  if n in {50,100}:hold='场景注释待更正并重新生成，不入库。'
  if n in {170,171}:hold='前情不足，需完善汀的请求和回应边界。'
  if src['scene_block_id'] in blocked or any(a<=src['source_line_end'] and b>=r['context_start'] for a,b in ranges.get(f.name,[])):hold='heldout scene or context window'
  audit.append(dict(number=n,status='hold' if hold else 'approved',reason=hold or 'literal target, past context, speaker and held-out checks passed'))
  if hold:continue
  valid[r['id']]=r
  if n<=38:continue
  m=dict(character='月社妃',persona='tsukiyashiro_kisaki',data_source='dpo_expansion_reviewed_20260926',source_file=f.name,source_group=src['scene_block_id'],scene_block_id=src['scene_block_id'],source_ids=src.get('event_ids',[src.get('id')]),target_event_ids=src.get('event_ids',[src.get('id')]),source_line_start=r['context_start'],source_line_end=src['source_line_end'],response_line_start=src['source_line_start'],response_line_end=src['source_line_end'],assistant_supervision='last',review_status='approved',human_final_approved=True,review_method='project_owner_authorized_assistant_final_review',review_sha256=digest(r['source_review']),final_review_number=n,split='train')
  rec=dict(id=r['id'],messages=r['prompt']+[{'role':'assistant','content':r['original']}],metadata=m,review_status='approved')
  normalized=normalize_chat_record(rec,default_system_prompt=system,system_prompt_policy='replace')
  assert normalized==rec['messages'],'canonical input drift'
  sft.append(rec)
 prior=readl(OLD/'dpo.train.jsonl');oldby={r['metadata']['final_review_number']:r for r in prior}
 pairs=[];dropped=[];buckets={}
 for r in prior:
  r=copy.deepcopy(r);n=r['metadata']['final_review_number'];buckets[n]=[r];pairs.append(r)
 for v in sorted(reviews,key=lambda r:(r['number'],r.get('sampling_profile',''),r['sample_index'])):
  assert v['review_sha256']==digest({k:x for k,x in v.items() if k!='review_sha256'}),'review binding'
  if v['verdict']!='prefer_original':continue
  g=generations[v['candidate_id']];assert g['record_sha256']==v['generation_sha256'];r=valid.get(g['scene_id'])
  if r is None:continue
  assert g['source_record_sha256']==r['record_sha256'],'source/generation mismatch'
  assert g['input_sha256']==digest(g['prompt'])
  n=r['number'];priorpairs=buckets.setdefault(n,[]);answer=g['generated']
  if len(priorpairs)>=4: dropped.append(dict(number=n,candidate_id=g['id'],reason='每个交互最多4个负例，避免过度加权'));continue
  if any(difflib.SequenceMatcher(None,clean(answer),clean(x['rejected'][0]['content'])).ratio()>=.82 for x in priorpairs):
   dropped.append(dict(number=n,candidate_id=g['id'],reason='同场景负例近重复，未计入数量'));continue
  src=r['source'];prompt=r['prompt']
  if n in oldby:
   m=copy.deepcopy(oldby[n]['metadata']);assert prompt==oldby[n]['prompt'];assert r['original']==oldby[n]['chosen'][0]['content']
  else:
   m=dict(persona='tsukiyashiro_kisaki',source_group=src['scene_block_id'],source_ids=src.get('event_ids',[src.get('id')]),evidence_text_sha256=[digest(prompt[-1])],chosen_candidate_id=src.get('event_ids',[src.get('id')])[0],source_file=src['source_file'],source_line_start=r['context_start'],source_line_end=src['source_line_end'],target_event_ids=src.get('event_ids',[src.get('id')]))
  m.update(schema_version='persona-preference-v1',final_review_number=n,review_status='approved',human_final_approved=True,feedback_source='human_confirmed_ai_assisted',review_method='project_owner_authorized_assistant_final_review',review_sha256=v['review_sha256'],rejected_candidate_id=g['id'],generation_record_sha256=g['record_sha256'],rejected_generation_prompt_sha256=digest(g['prompt']),training_prompt_sha256=digest(prompt),rejected_regenerated_for_training_prompt=g['prompt']==prompt,prompt_adaptation='exact canonical input' if g['prompt']==prompt else '普通模型使用简短角色指令，未给专用人物卡；真实离策略候选在正式人物输入下重审。',reason=v['reason'],split='train')
  pair=dict(id=g['id']+'_approved',prompt=prompt,chosen=[{'role':'assistant','content':r['original']}],rejected=[{'role':'assistant','content':answer}],review_status='approved',metadata=m)
  priorpairs.append(pair);pairs.append(pair)
 for n,group in buckets.items():
  for row in group:
   row['metadata']['preference_family']={'id':f'kisaki-review-{n:03d}','max_negatives':4,'review_method':'individually_reviewed_distinct_negatives'}
   row['metadata']['pair_content_sha256']=reviewed_pair_digest(row)
 pairs.sort(key=lambda r:(r['metadata']['final_review_number'],r['id']))
 validate_pairs(pairs,split='train')
 previous=readl(OLD/'train.jsonl');eventids={i for r in sft for i in r['metadata']['target_event_ids']}
 replaced=[r for r in previous if eventids & set(r.get('metadata',{}).get('target_event_ids',[]))]
 merged=[r for r in previous if r not in replaced]+sft
 assert len({r['id'] for r in merged})==len(merged)
 for r in merged:normalize_chat_record(r,default_system_prompt=system,system_prompt_policy='replace')
 OUT.mkdir(exist_ok=True);writel(OUT/'train.jsonl',merged);writel(OUT/'sft.expansion.approved.jsonl',sft);writel(OUT/'replaced_parent_records.jsonl',replaced)
 writel(OUT/'dpo.approved.pending.jsonl',pairs);write(OUT/'source_admission.json',audit);write(OUT/'duplicate_and_balance_exclusions.json',dropped)
 summary=dict(dataset_id='KISAKI-DPO100-20260926',status='pending_minimum' if len(pairs)<100 else 'pending_verification',train={'path':str((OUT/'train.jsonl').relative_to(ROOT)).replace('\\','/'),'count':len(merged),'sha256':sha(OUT/'train.jsonl')},validation=manifest['validation'],rag_holdout=manifest.get('rag_holdout',{}),dpo_pairs=len(pairs),unique_interactions=len({r['metadata']['final_review_number'] for r in pairs}),source_scene_groups=len({r['metadata']['source_group'] for r in pairs}),new_sft=len(sft),reviewed_candidates=len(reviews),generated_candidates=len(generations),prior_pairs=4,max_negatives_per_interaction=4,near_duplicate_threshold=.82,review_order='001–038保持，新增039及后续按清单稳定编号；组内逐候选显示。',approval_method='project_owner_authorized_assistant_final_review',training_run_authorized=True,frozen_before=before)
 write(OUT/'dataset_manifest.json',summary)
 auditgold=contamination_audit(gold['prompts'],train_path=OUT/'train.jsonl',validation_path=V4/'validation.jsonl',manifest_path=OUT/'dataset_manifest.json');write(OUT/'gold_contamination_audit.json',auditgold)
 assert auditgold['status']=='clean','gold contamination'
 assert before=={str(p.relative_to(ROOT)):sha(p) for p in frozen}
 summary['verification']={'gold':'clean','source_integrity':'pass','heldout_windows':'pass','trainer_schema':'pass','frozen_unchanged':True}
 if len(pairs)>=100:
  writel(OUT/'dpo.train.jsonl',pairs);summary['status']='approved_ready_for_training';summary['dpo_sha256']=sha(OUT/'dpo.train.jsonl')
 write(OUT/'dataset_manifest.json',summary)
 print(json.dumps({k:summary[k] for k in ['status','dpo_pairs','unique_interactions','source_scene_groups','new_sft','reviewed_candidates','generated_candidates']},ensure_ascii=False))
if __name__=='__main__':main()
