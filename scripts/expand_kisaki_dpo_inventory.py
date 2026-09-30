"""Inventory additional short canonical interactions without touching held-out data."""
import json
import re
from pathlib import Path
from extract_character_dialogues import read_script_events, dialogue_link_reason, reliable_speaker_label
from build_kisaki_interactive_review import heldout

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'backend/data/character_dialogues'
OUT=DATA/'experiments/dpo_expansion_100_20260926'
def readl(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    raw=readl(DATA/'tsukiyashiro_kisaki_raw.jsonl');by={(r['source_file'],r['source_line_start']):r for r in raw}
    val=readl(DATA/'experiments/v4/validation.jsonl');gold=json.loads((ROOT/'backend/evaluation/kisaki_gold_set_v3.json').read_text(encoding='utf-8'))
    manifest=json.loads((DATA/'experiments/v4/canonical_dataset_manifest.json').read_text(encoding='utf-8'))
    blocked,ranges=heldout(raw,val,gold)
    ragids=set(re.findall(r'tsukiyashiro_kisaki_raw_[0-9a-f]+',json.dumps(manifest.get('rag_holdout',{}))))
    for r in raw:
        if r['id'] in ragids:
            blocked.add(r['scene_block_id']);ranges.setdefault(r['source_file'],[]).append((r['source_line_start'],r['source_line_end']))
    existing=readl(DATA/'experiments/interactive_source_v2_20260926/interactive_source.pending.jsonl')
    used={(r['source']['source_file'],r['source']['source_line_start']) for r in existing}
    accepted=[];excluded=[]
    for f in sorted((ROOT/'gametext/纸上魔法使').glob('*.txt')):
        es=read_script_events(f);lines=f.read_text(encoding='utf-8-sig').splitlines()
        for i,e in enumerate(es):
            if e['speaker']!='妃' or (f.name,e['line_start']) in used:continue
            reason=None
            if e['scene_block_id'] in blocked:reason='heldout_scene'
            elif i==0 or es[i-1]['speaker']=='妃' or dialogue_link_reason(es[i-1],e):reason='not_direct_reply'
            elif not reliable_speaker_label(es[i-1]['speaker']):reason='uncertain_speaker'
            elif not 5<=len(e['text'])<=65 or not 4<=len(es[i-1]['text'])<=140:reason='not_short_interaction'
            elif e['text'].endswith(('——','，','：')):reason='incomplete_reply'
            if reason:
                excluded.append({'file':f.name,'line':e['line_start'],'reason':reason});continue
            start=max(int(e['scene_block_id'].rsplit(':',1)[-1]),e['line_start']-28)
            previous=[p for p in es[:i] if p['line_start']>=start]
            if not previous:continue
            start=previous[0]['line_start']
            if any(a<=e['line_end'] and b>=start for a,b in ranges.get(f.name,[])):continue
            accepted.append({'number':39+len(accepted),'source':by[(f.name,e['line_start'])],'history_events':previous,
                'context_start':start,'context_end':e['line_start']-1,'original':e['text'],
                'context':'\n'.join(f'L{x+1}: {lines[x]}' for x in range(start-1,e['line_end'])),
                'status':'pending_semantic_context_review'})
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'inventory.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=accepted:raise ValueError('inventory changed; preserve existing numbering')
    else:path.write_text(json.dumps(accepted,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'inventory_exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'new_inventory':len(accepted),'numbers':[39,38+len(accepted)],'existing_dpo':4},ensure_ascii=False))

if __name__=='__main__':main()
