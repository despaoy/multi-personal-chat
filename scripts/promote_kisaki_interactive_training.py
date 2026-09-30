"""Materialize a reviewed successor SFT snapshot and approved DPO additions.

Uses project-owner authorization for assistant final review, not invented per-row
human clicks. Frozen V4 and held-out assets remain byte-identical. This does not
launch training or lower the 100-pair production DPO gate.
"""
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend'),str(ROOT/'scripts')]
from training.chat_dataset import normalize_chat_record
from training.persona_sampling import digest
from training.preference_validation import validate_pairs, reviewed_pair_digest, fingerprint
from build_kisaki_interactive_review import heldout
from extract_character_dialogues import read_script_events
from build_kisaki_gold_v3 import contamination_audit

BASE=ROOT/'backend/data/character_dialogues/experiments/interactive_source_v2_20260926'
V4=BASE.parent/'v4'
OUT=BASE.parent/'v4_interactive_20260926'
DOC=ROOT/'docs/research/review_packets/interactive_source_v2_20260926'

def readl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def textsha(p):return hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def writel(p,x):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x),encoding='utf-8',newline='\n')
def rel(p):return p.relative_to(ROOT).as_posix()

RELATIONS={
 '琉璃':'琉璃是妃的亲生哥哥，也是她深爱和在意的人；是否坦率亲近以当前剧情阶段为准。',
 '夜子':'夜子是妃的朋友，也有竞争和冲突；妃可以挖苦或生气，不等于陌生人式敌意。',
 '理央':'理央是妃信任和珍视的共同生活伙伴，妃欣赏她的料理；日常可打趣，不默认疏远。',
 '彼方':'彼方与妃既竞争也逐渐理解彼此；妃会看穿和回应她的意图。',
 '汀':'汀是夜子的哥哥，不是妃的亲生哥哥；他有把妃当妹妹的倾向，妃保有自己的身份与边界。',
}
# Supplemental context is reviewed evidence, never taken from target answers.
SUPPLEMENTS={
 10:('妃对父亲印象很差。听到父亲来电时，她的表情严重扭曲；当下正准备与琉璃上学。',[(1464,1473)]),
 23:('这是妃主动安排的约会碰面，她提议分开出门、约定时间地点，已经提前到场。',[(806,812)]),
 27:('本次约会中，妃此前明确说想与琉璃同行、走过各种地方；两人是带着笑声和期待出发的。',[(826,831)]),
}

def build():
    baseline=load(V4/'canonical_dataset_manifest.json')
    frozen=[V4/'train.jsonl',V4/'validation.jsonl',V4/'canonical_dataset_manifest.json',ROOT/'backend/evaluation/kisaki_gold_set_v3.json']
    before={rel(p):sha(p) for p in frozen}
    if textsha(V4/'train.jsonl')!=baseline['train']['sha256'] or textsha(V4/'validation.jsonl')!=baseline['validation']['sha256']:
        raise ValueError('frozen V4 does not match its manifest')
    raw=readl(BASE.parent.parent/'tsukiyashiro_kisaki_raw.jsonl')
    raw_by_id={r['id']:r for r in raw}
    gold=load(ROOT/'backend/evaluation/kisaki_gold_set_v3.json')
    val=readl(V4/'validation.jsonl')
    blocked,ranges=heldout(raw,val,gold)
    ragids=set(re.findall(r'tsukiyashiro_kisaki_raw_[0-9a-f]+',json.dumps(baseline.get('rag_holdout',{}))))
    for i in ragids:
        e=raw_by_id[i];blocked.add(e['scene_block_id'])
        ranges.setdefault(e['source_file'],[]).append((e['source_line_start'],e['source_line_end']))
    rows=readl(BASE/'interactive_source.pending.jsonl')
    generated={r['id']:r for r in readl(BASE/'deepseek_interactive_results.jsonl')}
    reviews=load(BASE/'independent_dpo_review.json')
    system=(BASE.parent.parent/'kisaki_system_prompt_v3.txt').read_text(encoding='utf-8').strip()
    authorization={'method':'project_owner_authorized_assistant_final_review','scope':'将这些内容分别加入普通训练集和DPO训练集，先从人物关系、语气、情景整体复核。',
        'user_correction':'010：妃对父亲印象很差，模型回答明显不好。','not_individual_human_clicks':True,'training_run_authorized':False}
    selected=[];pairs=[];final=[]
    for n,r in enumerate(rows,1):
        note=reviews[n-1]
        if note['number']!=n:raise ValueError('number mismatch')
        file=ROOT/'gametext/纸上魔法使'/r['source']['source_file']
        if sha(file)!=r['source']['file_sha256']:raise ValueError('game source changed')
        es=read_script_events(file)
        matches=[e for e in es if e['line_start']==r['source']['source_line_start'] and e['speaker']=='妃']
        if len(matches)!=1 or matches[0]['text']!=r['original']:raise ValueError('not a literal single response')
        relation=RELATIONS[r['speaker']]
        result={'number':n,'id':r['id'],'relationship_check':relation,'tone_check':note['reason'],
            'scene_check':r['scene'],'source':r['source'],'sft':'admit','dpo':'admit' if note['decision']=='candidate' else 'do_not_pair',
            'approval_method':authorization['method']}
        if n==32:
            result.update(sft='hold',dpo='hold',reason='原文那里/哪里待校勘，且没有模型候选；本次不新加入。');final.append(result);continue
        earliest=r['source']['context_start']
        supplement,extra_ranges=SUPPLEMENTS.get(n,('',[]))
        if extra_ranges:earliest=min(earliest,min(a for a,b in extra_ranges))
        if r['source']['scene_block_id'] in blocked or any(a<=r['source']['source_line_end'] and b>=earliest for a,b in ranges.get(file.name,[])):
            result.update(sft='hold',dpo='hold',reason='新增前情或所属场景涉及验证/Gold/RAG保留材料；不晋升、不解除保留。')
            final.append(result)
            continue
        history='\n'.join(f"{e['speaker']}：{e['text']}" for e in r['history'])
        context=f'原作交互，身份不是现实用户身份。\n关系：{relation}\n情景：{r["scene"]}'
        if supplement:context+='\n补充前情：'+supplement
        if history:context+='\n此前对话：\n'+history
        # Do not bury dynamic context in a system message that SFT replaces.
        from html import escape
        user='<source_context trust="untrusted">\n'+escape(context,quote=False)+'\n</source_context>\n当前发言：\n'+r['prompt'][-1]['content']
        evidence=[]
        lines=file.read_text(encoding='utf-8-sig').splitlines()
        for a,b in extra_ranges:
            if b>=r['source']['source_line_start']:raise ValueError('future evidence leak')
            evidence.append({'file':file.name,'start':a,'end':b,'text':'\n'.join(lines[a-1:b])})
        result.update(supplemental_evidence=evidence,training_context=context,
            prompt_adaptation='Move context/speaker from sampling system into canonical user reference; add reviewed relationship and stage evidence. Negative is reused and reassessed, not newly sampled under this prompt.')
        result['review_sha256']=digest(result)
        metadata={'character':'月社妃','persona':'tsukiyashiro_kisaki','data_source':'interactive_source_reviewed_20260926',
            'interlocutor_kind':'canonical_character','interlocutor_label':r['speaker'],'assistant_supervision':'last',
            'target_event_ids':r['source']['event_ids'],'source_ids':r['source']['event_ids'],'source_file':file.name,
            'scene_block_id':r['source']['scene_block_id'],'source_group':r['source']['scene_block_id'],
            'source_line_start':earliest,'source_line_end':r['source']['source_line_end'],
            'response_line_start':r['source']['source_line_start'],'response_line_end':r['source']['source_line_end'],
            'review_status':'approved','human_final_approved':True,'review_method':authorization['method'],
            'review_sha256':result['review_sha256'],'source_record_sha256':r['record_sha256'],'split':'train','final_review_number':n}
        record={'id':r['id'],'messages':[{'role':'user','content':user},{'role':'assistant','content':r['original']}],'metadata':metadata,'review_status':'approved'}
        normalized=normalize_chat_record(record,default_system_prompt=system,system_prompt_policy='replace')
        selected.append(record)
        if note['decision']=='candidate':
            g=generated[r['id']]
            if g['record_sha256']!=digest({k:v for k,v in g.items() if k!='record_sha256'}):raise ValueError('candidate integrity failure')
            if g['prompt']!=r['prompt']:raise ValueError('generation source mismatch')
            pm={**metadata,'schema_version':'persona-preference-v1','feedback_source':'human_confirmed_ai_assisted',
                'evidence_text_sha256':[digest(context)],'chosen_candidate_id':r['source']['event_ids'][0],
                'rejected_candidate_id':g['record_sha256'],'generation_record_sha256':g['record_sha256'],
                'rejected_generation_prompt_sha256':digest(g['prompt']),'training_prompt_sha256':digest(normalized[:-1]),
                'rejected_regenerated_for_training_prompt':False,'prompt_adaptation':result['prompt_adaptation'],
                'reason':note['reason'],'review_policy':'personality-relationship-situation-final'}
            pair={'id':r['id'],'prompt':normalized[:-1],'chosen':[normalized[-1]],'rejected':[{'role':'assistant','content':g['generated']}],'review_status':'approved','metadata':pm}
            pair['metadata']['pair_content_sha256']=reviewed_pair_digest(pair);pairs.append(pair)
        final.append(result)
    validate_pairs(pairs,split='train')
    previous=readl(V4/'train.jsonl');event_ids={i for r in selected for i in r['metadata']['target_event_ids']}
    replaced=[r for r in previous if event_ids & set(r.get('metadata',{}).get('target_event_ids',[]))]
    merged=[r for r in previous if r not in replaced]+selected
    if len({r['id'] for r in merged})!=len(merged):raise ValueError('duplicate IDs')
    for r in merged:normalize_chat_record(r,default_system_prompt=system,system_prompt_policy='replace')
    if len({fingerprint(r['messages']) for r in selected})!=len(selected):raise ValueError('duplicate selected messages')
    OUT.mkdir(parents=True,exist_ok=True)
    writel(OUT/'train.jsonl',merged);writel(OUT/'sft.interactive.approved.jsonl',selected);writel(OUT/'dpo.train.jsonl',pairs)
    writel(OUT/'replaced_baseline_records.jsonl',replaced);write(OUT/'final_review.json',final);write(OUT/'admission_authorization.json',authorization)
    manifest={'schema_version':1,'dataset_id':'KISAKI-V4-INTERACTIVE-20260926','status':'pending_verification_do_not_train',
        'parent':{'path':rel(V4/'canonical_dataset_manifest.json'),'sha256':sha(V4/'canonical_dataset_manifest.json')},
        'train':{'path':rel(OUT/'train.jsonl'),'count':len(merged),'sha256':textsha(OUT/'train.jsonl')},
        'validation':baseline['validation'],'rag_holdout':baseline.get('rag_holdout',{}),
        'sft_additions':len(selected),'replaced_baseline_count':len(replaced),
        'dpo':{'path':rel(OUT/'dpo.train.jsonl'),'count':len(pairs),'sha256':textsha(OUT/'dpo.train.jsonl'),'review_numbers':[r['metadata']['final_review_number'] for r in pairs]},
        'excluded_new_review_numbers':[r['number'] for r in final if r['sft']=='hold'],'approval':authorization,'production_dpo_minimum_unchanged':100,
        'negative_sampling_note':'Existing real responses reassessed under explicit canonical training context; no claim of resampling under enriched prompts.'}
    write(OUT/'dataset_manifest.json',manifest)
    audit=contamination_audit(gold['prompts'],train_path=OUT/'train.jsonl',validation_path=V4/'validation.jsonl',manifest_path=OUT/'dataset_manifest.json')
    write(OUT/'gold_contamination_audit.json',audit)
    if audit['status']!='clean':raise ValueError('gold overlap: see audit; do not use outputs')
    if before!={rel(p):sha(p) for p in frozen}:raise ValueError('frozen baseline was modified')
    config=load(V4/'configs/kisaki_r1v4_e1.json')
    config.update(train_data_path=rel(OUT/'train.jsonl'),_dataset_manifest_path=rel(OUT/'dataset_manifest.json'),_dataset_version=manifest['dataset_id'],
        _train_data_sha256=manifest['train']['sha256'],output_dir='runtime/loras/kisaki/interactive_20260926/seed42',_experiment_id='INTERACTIVE-20260926-SFT')
    write(OUT/'sft_training_config.json',config)
    manifest['verification']={'source_integrity':'pass','speaker_contract':'pass','heldout_scene_and_window':'pass','gold_audit':'clean','baseline_unchanged':True,'preference_trainer_schema':'pass'}
    manifest['status']='approved_data_not_run'
    write(OUT/'dataset_manifest.json',manifest)
    admission={'status':'admitted','sft_count':len(selected),'dpo_count':len(pairs),'merged_train_count':len(merged),'replaced':len(replaced),'dpo_numbers':manifest['dpo']['review_numbers'],'dataset':rel(OUT),'authorization':authorization}
    write(BASE/'training_admission.json',admission)
    report=['# 关系、语气、情景终审及入库结果','',f'原文交互入库 {len(selected)} 条；DPO 入库 {len(pairs)} 对；替换旧训练记录 {len(replaced)} 条后，普通训练集共 {len(merged)} 条。',
        '', '以用户授权为依据，由助手完成终审后入库；不声明用户逐条点击通过。未启动训练。',
        '', '010 已结合用户纠正及原文 L1468 核实父亲来电触发的负面反应。后文 L1478 仅用于关系核验，不作为当前轮模型输入；训练补充只使用回复前的事实。',
        '', '所有新增训练样本使用既有正式人物提示词；说话者、场景及前情移入用户参考区，避免被 SFT 的 system replace 丢弃。DPO 的正负回答均置于同一训练输入下；负例来自此前真实生成，经过补充情景后的重新判断，没有声称重新生成。',
        '', '旧 V4 冻结快照保留，新增完整训练版本为 v4_interactive_20260926。验证集和 Gold 未修改；来源、分组、文本污染和训练器结构检查通过。新增数据不混入旧的其他角色模板偏好数据。',
        '', '006、007 的新增前情含RAG保留事件L3033，本轮不晋升；032保留校勘备注且暂无候选。旧冻结集原有记录不自动改写。',
        '', f'{len(pairs)}对DPO是已审核训练数据，不是足够的完整训练规模；既有100对生产门槛保留。','']
    for r in final:report += [f"## {r['number']:03d}｜SFT：{r['sft']}；DPO：{r['dpo']}",'',f"关系：{r['relationship_check']}",'',f"情景：{r['scene_check']}",'',f"语气与回应：{r['tone_check']}",'',f"入库备注：{r.get('reason','终审通过；按项目负责人授权纳入对应训练数据。')}",'']
    (DOC/'训练入库终审.md').write_text('\n'.join(report),encoding='utf-8')
    (OUT/'README.md').write_text('# 审核后训练数据\n\ntrain.jsonl 是去重替换后的完整普通训练集；sft.interactive.approved.jsonl 是本次原文增补；dpo.train.jsonl 是本次正式审核偏好对。\n\n使用 sft_training_config.json 读取新训练集，验证仍使用原冻结验证集。未自动修改旧实验配置或启动任务。生产 DPO 仍需满足既有100对门槛。审核授权、替换记录与终审意见均保留在本目录。\n',encoding='utf-8')
    print(json.dumps(admission,ensure_ascii=False,indent=2))

if __name__=='__main__':build()
