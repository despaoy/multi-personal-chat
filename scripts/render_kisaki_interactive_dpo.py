"""Render bound independent reviews and pending DPO pairs for interaction v2."""
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import digest

BASE=ROOT/'backend/data/character_dialogues/experiments/interactive_source_v2_20260926'
DOC=ROOT/'docs/research/review_packets/interactive_source_v2_20260926'


def readl(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    rows=readl(BASE/'interactive_source.pending.jsonl')
    journal=readl(BASE/'deepseek_interactive_results.jsonl')
    for r in journal:
        assert r['record_sha256']==digest({k:v for k,v in r.items() if k!='record_sha256'})
    generated={r['id']:r for r in journal}
    notes=json.loads((BASE/'independent_dpo_review.json').read_text(encoding='utf-8'))
    notes={r['number']:r for r in notes}
    policy_path=BASE/'review_policy.json'
    policy=json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.exists() else {'id':'original-review'}
    policy_text='本轮优先判断人物性格、关系态度、语气和当下回应。自然动作、调侃和小幅生活发挥不单独判负；仍关注认错对象、明确冲突及不合情境的反应。'
    admission_path=BASE/'training_admission.json'
    admission=json.loads(admission_path.read_text(encoding='utf-8')) if admission_path.exists() else None
    final_rows={}
    if admission:
        final_rows={r['number']:r for r in json.loads((ROOT/admission['dataset']/'final_review.json').read_text(encoding='utf-8'))}
    assert set(notes)==set(range(1,len(rows)+1))
    pairs=[];audits=[];cards=[];blocks=[]
    labels={'candidate':'建议组成DPO对，待确认','tie':'人物回应可接受，不配对','weak':'较偏好原文，暂不配对','resample':'重采样或补充输入','blocked':'暂无模型候选，原文有校勘备注'}
    for n,r in enumerate(rows,1):
        g=generated.get(r['id']);note=notes[n]
        assert note['decision'] in labels
        if r['review_status']=='pending':
            assert g and g['status']=='generated'
            assert g['prompt']==r['prompt'] and g['source_record_sha256']==r['record_sha256']
        else:assert note['decision']=='blocked'
        if g:assert g['input_sha256']==digest(r['prompt'])
        audit={'number':n,'id':r['id'],'source_record_sha256':r['record_sha256'],
            'generation_record_sha256':g['record_sha256'] if g else None,'prompt':r['prompt'],
            'original':r['original'],'generated':g.get('generated') if g else None,
            'review':note,'review_policy':policy,'human_preference_approved':False,'source':r['source']}
        audit['record_sha256']=digest(audit);audits.append(audit)
        if note['decision']=='candidate':
            assert g['generated'].strip()!=r['original'].strip()
            pairs.append({'id':r['id'],'prompt':r['prompt'],
                'chosen':[{'role':'assistant','content':r['original']}],
                'rejected':[{'role':'assistant','content':g['generated']}],
                'review_status':'pending','metadata':{'schema_version':'interactive-source-dpo-v2',
                    'feedback_source':'ai','human_final_approved':False,'source_group':r['source']['scene_block_id'],
                    'source_ids':r['source']['event_ids'],'source':r['source'],
                    'audit_record_sha256':audit['record_sha256'],'generation_record_sha256':g['record_sha256'],
                    'reason':note['reason'],'risk':note['risk'],'review_policy':policy['id']}})
        esc=html.escape;label=labels[note['decision']]
        final_note=final_rows.get(n)
        admission_html=''
        if final_note:
            status='；'.join([('普通训练：已入库' if final_note['sft']=='admit' else '普通训练：本轮留出'),('DPO：已入库' if final_note['dpo']=='admit' else 'DPO：未入库')])
            admission_html=f'<p><b>终审入库结果：{esc(status)}</b> {esc(final_note.get("reason",""))}</p>'
            if final_note.get('training_context'):
                admission_html+='<details><summary>最终训练使用的关系与情景（与原采样输入区分）</summary><pre>'+esc(final_note['training_context'])+'</pre></details>'
        model_text=g.get('generated','生成失败') if g else '本条原文待校勘，未请求模型生成。'
        context='\n'.join(f"{e['speaker']}：{e['text']}" for e in r['history']) or '无'
        cards.append(f'''<article data-id="{r['id']}" data-decision="{note['decision']}"><h2>{n:03d} · {esc(label)}</h2><p>{esc(r['scene'])}</p><details><summary>前几句对话</summary><pre>{esc(context)}</pre></details><p class="incoming"><b>{esc(r['speaker'])}：</b>{esc(r['prompt'][-1]['content'])}</p>
<div class="compare"><section><h3>游戏原文{' · 建议 chosen' if note['decision']=='candidate' else ''}</h3><pre>{esc(r['original'])}</pre></section><section><h3>DeepSeek 实际回复{' · 建议 rejected' if note['decision']=='candidate' else ''}</h3><pre>{esc(model_text)}</pre></section></div>
<p><b>人物倾向：</b>{esc(note['tendency'])}</p><p><b>独立判断：</b>{esc(note['reason'])}</p><p><b>保留意见：</b>{esc(note['risk'])}</p>{admission_html}<details><summary>核对原文与实际模型输入</summary><pre>{esc(r['context_excerpt'])}</pre><pre>{esc(json.dumps(r['prompt'],ensure_ascii=False,indent=2))}</pre></details><label>你的配对决定 <select><option>待定</option><option>认可原文优于模型</option><option>两者均可</option><option>模型更好</option><option>重采样</option><option>补前情或校勘</option></select></label><textarea placeholder="你的理由；与AI建议分开保存"></textarea></article>''')
        blocks.append(f"## {n:03d} {label}\n\n场景：{r['scene']}\n\n前文：\n{context}\n\n**{r['speaker']}：** {r['prompt'][-1]['content']}\n\n**原文：** {r['original']}\n\n**模型：** {model_text}\n\n人物倾向：{note['tendency']}\n\n判断：{note['reason']}\n\n保留意见：{note['risk']}\n\n来源：{r['source']['source_file']} L{r['source']['source_line_start']}\n")
    counts=Counter(n['decision'] for n in notes.values())
    summary={'total':len(rows),'generated':sum(g['status']=='generated' for g in generated.values()),'counts':dict(counts),'proposed_pairs':len(pairs),'human_preference_approved':False,'source_confirmation':'用户：我看来是可以的，继续；认可继续生成，不等于批准未见过的偏好对。','model':next(iter(generated.values()))['model'],'independent_reviewer':'assistant; no DeepSeek self-judgment used'}
    summary['review_policy']=policy
    if admission:summary['training_admission']=admission
    for name,value in [('dpo.interactive.ai-reviewed.pending.jsonl',pairs),('dpo_audit.bound.jsonl',audits)]:
        (BASE/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in value),encoding='utf-8')
    write(BASE/'dpo_summary.json',summary)
    intro='# 新版交互DPO审核\n\n实际生成37条，另1条原文待校勘。所有模型回复均来自新版相同输入；原文答案未发送给生成模型。\n\n'+'\n'.join(f'- {labels[k]}：{counts[k]}条' for k in labels)+'\n\n配对是AI建议，不是你的最终批准；未启动训练。生成使用deepseek-chat在线别名，服务端版本未固定。\n\n'
    intro+='审核标准：'+policy_text+'\n\n旧审核已归档；以下为重新逐条判断的当前意见。\n\n'
    if admission:
        intro=intro.replace('配对是AI建议，不是你的最终批准；未启动训练。',f'项目负责人已授权终审后入库：普通训练{admission["sft_count"]}条、DPO {admission["dpo_count"]}对；下列AI候选与最终入库资格不同，详见[训练入库终审](训练入库终审.md)。未启动训练。')
    (DOC/'DPO完整审核.md').write_text(intro+'\n\n'.join(blocks),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>妃 · 新版DPO对照审核</title><style>body{font:16px/1.8 system-ui;background:#f4f6f8;color:#243345;max-width:1150px;margin:auto;padding:24px}article{padding:24px;border-radius:14px;background:#fff;margin:24px 0}.compare{display:grid;grid-template-columns:1fr 1fr;gap:16px}.compare section{background:#edf4f1;padding:16px}.compare section+section{background:#f0f2f7}pre{font:inherit;white-space:pre-wrap;overflow-wrap:anywhere}.incoming{background:#eaf1ff;padding:16px}summary{cursor:pointer}nav{position:sticky;top:0;background:#f4f6f8;padding:12px;z-index:2}textarea{display:block;width:95%;height:60px;margin-top:12px}select,button{font:inherit;padding:5px}article[hidden]{display:none}@media(max-width:720px){.compare{grid-template-columns:1fr}}</style><h1>妃 · 新版DPO对照审核</h1><p>原文 × DeepSeek真实回复 × 人物倾向 × 配对判断</p><p>SUMMARY</p><p>37条实际生成，1条原文待校勘。当前仅AI建议，未批准入训。<a href="DPO完整审核.md">完整文字版</a> · <a href="交互原文审核.html">原文选择页</a></p><nav><select id="filter"><option value="all">全部38条</option>OPTIONS</select> <button id="save">导出我的DPO审核草稿</button><span id="status"></span></nav>CARDS<script>
const metadata=META,key='kisaki-interactive-dpo-human-review-v1',cards=[...document.querySelectorAll('article')];let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
cards.forEach(c=>{let s=state[c.dataset.id]||{};c.querySelector('select').value=s.decision||'待定';c.querySelector('textarea').value=s.reason||'';c.addEventListener('input',()=>{state[c.dataset.id]={decision:c.querySelector('select').value,reason:c.querySelector('textarea').value};try{localStorage.setItem(key,JSON.stringify(state));document.querySelector('#status').textContent='草稿已保存'}catch(e){document.querySelector('#status').textContent='请导出备份'}})});
document.querySelector('#filter').onchange=e=>cards.forEach(c=>c.hidden=e.target.value!=='all'&&c.dataset.decision!==e.target.value);
document.querySelector('#save').onclick=()=>{const blob=new Blob([JSON.stringify({schema:'interactive-dpo-human-review-draft-v1',status:'draft',decisions:metadata.map(m=>({...m,...(state[m.id]||{decision:'待定',reason:''})}))},null,2)],{type:'application/json'}),u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='新版DPO审核草稿.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
</script></html>'''
    page=page.replace('SUMMARY','；'.join(f'{labels[k]} {counts[k]} 条' for k in labels))
    if admission:
        page=page.replace('当前仅AI建议，未批准入训。',f'已按授权完成终审：普通训练入库{admission["sft_count"]}条，DPO入库{admission["dpo_count"]}对。006、007因RAG保留前情留出，032暂缓。<a href="训练入库终审.md">查看终审报告</a>。未启动训练。')
    page=page.replace('<p>原文 × DeepSeek真实回复 × 人物倾向 × 配对判断</p>', '<p>人物性格与回应优先 · 全部38条重新审核</p><p>'+html.escape(policy_text)+'</p><p>本次仅更新AI判断，原文、模型回复和你的草稿保留。旧草稿不代表你认可新的AI建议。</p>')
    page=page.replace('OPTIONS',''.join(f'<option value="{k}">{labels[k]}（{counts[k]}）</option>' for k in labels))
    page=page.replace('CARDS',''.join(cards)).replace('META',json.dumps([{'number':r['number'],'id':r['id'],'audit_record_sha256':r['record_sha256']} for r in audits],ensure_ascii=False).replace('<','\\u003c'))
    (DOC/'DPO对照审核.html').write_text(page,encoding='utf-8')
    # Keep the already-open source page useful without touching its saved decisions.
    source_page=DOC/'交互原文审核.html'
    if source_page.exists():
        source_html=source_page.read_text(encoding='utf-8')
        notice='<p id="dpo-ready"><b>新版DPO已生成：</b><a href="DPO对照审核.html">查看37条真实模型回复与逐条配对审核</a>。原文选择与DPO审核的草稿分别保存。</p>'
        if 'id="dpo-ready"' not in source_html:
            source_html=source_html.replace('<nav>',notice+'<nav>',1)
        source_html=source_html.replace('新输入尚无模型候选，不沿用旧负样本。','模型回复与配对结论请查看新版DPO审核页；校勘项暂缓。')
        source_page.write_text(source_html,encoding='utf-8')
    for filename in ('调整说明.md','完整交互审核.md'):
        path=DOC/filename
        if path.exists():
            text=path.read_text(encoding='utf-8')
            marker='> 后续进度：'
            if marker not in text:
                text='> 后续进度：新版已实际生成37条回复，详见[DPO完整审核](DPO完整审核.md)。下文是生成前的原文筛选记录，其中“尚未生成”描述的是当时状态。\n\n'+text
                path.write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
