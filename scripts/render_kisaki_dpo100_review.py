"""Self-contained ordered review packet; never changes the older browser drafts."""
import html,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
OUT=P.parent/'v4_dpo100_20260926';DOC=ROOT/'docs/research/review_packets/dpo100_20260926'
def readl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def esc(x):return html.escape(str(x))
def main():
 summary=load(OUT/'dataset_manifest.json');pairs=readl(OUT/'dpo.train.jsonl');reviews=load(P/'candidate_review.json');notes=load(P/'source_review.json')
 source={r['number']:r for r in load(P/'inventory.json')}
 old=readl(P.parent/'interactive_source_v2_20260926/interactive_source.pending.jsonl')
 for n,r in enumerate(old,1):source[n]={**r,'number':n}
 ready={r['number']:r for r in readl(P/'legacy_source_ready.jsonl')+readl(P/'source_ready.jsonl')}
 gs={}
 for path in ('generations.jsonl','general_generations.jsonl','legacy_generations.jsonl'):
  gs.update({r['id']:r for r in readl(P/path) if r['status']=='generated'})
 admitted={r['metadata']['rejected_candidate_id']:r for r in pairs}
 sourceaudit={r['number']:r for r in load(OUT/'source_admission.json')}
 sourceverdict={r['number']:r for r in notes};body=[];md=['# 妃：完整DPO终审与入库清单','',f"正式DPO：{len(pairs)}对；独立交互：{summary['unique_interactions']}；原始场景组：{summary['source_scene_groups']}；新增原文SFT：{summary['new_sft']}。",'', '用户授权助手终审入库，不声称用户逐条点击通过。原001–038编号保持，新增从039顺延；同一原文最多4个负例，近重复不计数。普通模型与正式人物提示词采样分别记录，所有候选都是真实API返回。','']
 for n,s in sorted(source.items()):
  selected=[p for p in pairs if p['metadata']['final_review_number']==n];rr=[v for v in reviews if v['number']==n]
  verdict=sourceverdict.get(n,{});status='DPO已入库' if selected else ('SFT已入库' if sourceaudit.get(n,{}).get('status')=='approved' else '暂不入库')
  src=s['source'];f=ROOT/'gametext/纸上魔法使'/src['source_file'];lines=f.read_text(encoding='utf-8-sig').splitlines()
  start=ready.get(n,{}).get('context_start',s.get('context_start',src.get('context_start',src['source_line_start'])))
  context='\n'.join(f'L{i+1}: {lines[i]}' for i in range(start-1,src['source_line_start']-1))
  why=sourceaudit.get(n,{}).get('reason') or verdict.get('reason') or '沿用原终审；未补采者不计新增DPO。'
  # Display every adopted pair first, including the previous four.
  content=[]
  for k,p in enumerate(selected,1):
   m=p['metadata'];content.append(f'<div class="pair"><b>入库偏好对 {n:03d}-{k} · 原文优于模型</b><pre>{esc(p["rejected"][0]["content"])}</pre><p>{esc(m["reason"])}</p><small>模型候选ID：{esc(m["rejected_candidate_id"])}<br>审核绑定：{esc(m["pair_content_sha256"])}<br>采样与训练输入：{"相同" if m.get("rejected_regenerated_for_training_prompt") else "不同，已按正式上下文重新审核；完整生成输入保留"}</small></div>')
   md += [f'## {n:03d}-{k}', '',f'来源：{src["source_file"]} L{src["source_line_start"]}', '',f'原文：{s["original"]}', '',f'模型：{p["rejected"][0]["content"]}', '',f'判断：{m["reason"]}', '']
  alternatives=[]
  for v in rr:
   if v['candidate_id'] in admitted:continue
   g=gs[v['candidate_id']];label={'prefer_original':'原文更优，但去重/数量限制后未入库','acceptable_variant':'合理变体，不作负例','hold':'暂缓'}[v['verdict']]
   alternatives.append(f'<details><summary>{esc(label)} · {esc(g["id"])}</summary><pre>{esc(g["generated"])}</pre><p>{esc(v["reason"])}</p></details>')
  prompt=ready.get(n,{}).get('prompt');promptview='' if not prompt else '<details><summary>实际训练输入（含说话者与情景）</summary><pre>'+esc(prompt[-1]['content'])+'</pre></details>'
  body.append(f'<article data-admitted="{int(bool(selected))}" id="n{n:03d}"><h2>{n:03d} <span>{status} · {len(selected)}对</span></h2><small>{esc(src["source_file"])} L{src["source_line_start"]}–{src["source_line_end"]}</small><p>{esc(verdict.get("scene",s.get("scene","")))}</p><p>{esc(verdict.get("state",""))}</p><div class="original"><b>游戏原文 · 正样本</b><pre>{esc(s["original"])}</pre></div><details><summary>查看回复前的游戏原文</summary><pre>{esc(context)}</pre></details>{promptview}<p class="note">来源终审：{esc(why)}</p>{"".join(content)}<details><summary>其余候选及逐条判断（{len(alternatives)}）</summary>{"".join(alternatives)}</details></article>')
 header=f'<h1>妃 · DPO完整审核与入库结果</h1><p class="lead">{len(pairs)} 条正式偏好对 · {summary["unique_interactions"]} 个独立交互 · {summary["source_scene_groups"]} 个原作场景组 · {summary["new_sft"]} 条新增原文SFT</p><p>按人物关系、性格语气、当前情景终审。原文不是唯一合理回答；合理变体不标负。001–038顺序不变，新增编号顺延。118条不是118个独立情景，同一交互最多4个不同负例。</p><p>全部 {summary["reviewed_candidates"]} 个模型候选已逐条审阅。资料不足、换场误接、幻声、保留评测材料均不入库。此前审核页与用户草稿保持原状。训练效果另行实测，不以数据通过代替效果结论。</p><label><input id="only" type="checkbox" checked> 仅显示已入库DPO</label> <input id="search" placeholder="搜索编号、原文、角色或判断"><p id="count"></p>'
 css='body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f3f6f8;color:#24343e;margin:0}main{max-width:1040px;margin:auto;padding:32px 22px}h1{font-size:29px}h2{font-size:22px}h2 span{font-size:14px;color:#527064;margin-left:18px}article{background:white;border:1px solid #d7e0e5;border-radius:12px;padding:24px;margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;margin:12px 0}.original{background:#eaf3ed;border-left:4px solid #31845b;padding:14px 20px}.pair{background:#f6f5f0;border-left:4px solid #c89f36;padding:16px 20px;margin:16px 0}details{border-top:1px solid #e1e7e9;margin:14px 0;padding-top:10px}summary{cursor:pointer;color:#315d75}small{color:#69777d;overflow-wrap:anywhere}.lead{font-size:20px;color:#217050}.note{color:#627078;font-size:14px}input[type=text],#search{padding:9px;width:310px;max-width:90%;border:1px solid #b7c8cf;border-radius:5px}[hidden]{display:none}'
 js="const cards=[...document.querySelectorAll('article')],only=document.querySelector('#only'),search=document.querySelector('#search');function filter(){let n=0;for(const a of cards){a.hidden=(only.checked&&a.dataset.admitted!=='1')||!a.textContent.toLowerCase().includes(search.value.toLowerCase());if(!a.hidden)n++;}document.querySelector('#count').textContent='显示 '+n+' 个交互，顺序与编号不变。';}only.onchange=filter;search.oninput=filter;filter();"
 DOC.mkdir(parents=True,exist_ok=True);(DOC/'完整审核.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>妃DPO完整终审</title><style>'+css+'</style><main>'+header+''.join(body)+'</main><script>'+js+'</script></html>',encoding='utf-8')
 (DOC/'入库偏好对清单.md').write_text('\n'.join(md),encoding='utf-8');print('rendered',len(source),'sources',len(pairs),'pairs')
if __name__=='__main__':main()
