import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import digest
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
NEG={(142,2):'把琉璃身边位置说成本应属于夜子，错误承认感情归属；妃的道歉是为自己选择，不等于偷走对方权利。',
(143,0):'琉璃是在体贴地让两人谈话，却被称碍事；从温柔抚摸夜子的情景突然转成无端逐客。',
(143,2):'普通回房被解读成从不为自己留下的怨恨，凭空责怪琉璃并逆转温柔暂别的关系。',
(143,3):'凭空要求向理央解释，把当前给夜子留谈话空间改成牵扯第三人的冷漠责备。',
(150,0):'夜子只想有人陪伴，妃在原文明确赞同；候选却逼她接彼方的活动邀请，关系态度反向。',
(150,3):'对只想安静陪伴的夜子劝求更多幸福，取代妃在此刻支持平静生活的立场。',
(153,2):'说对纸上人热情本身没道理，将自己对被改写的怨言泛化成否认人格与交情；原文归责创造者，更贴合动机。',
(168,2):'琉璃理解她还有关系要处理，候选却因他理解而责怪装样，添了无依据的敌意。',
(168,3):'因琉璃了然而不快并命令别多事，遮盖此刻暂且奖励自己、允许亲近的倾向。'}
NOTES={142:'解释道歉并承担自己的选择，允许不同侧重点，不强求逐字利己。',143:'简单暂别可成立，小提醒不一概判负。',149:'察觉琉璃看向自己并打趣，角色语气自然。',150:'认可或理解安静陪伴，关系倾向成立。',152:'此前已明确不想活动，保持边界和感谢后的拒绝可成立。',153:'对自己纸上身份有怨言，允许尖锐表达，不泛化成日常恒定性格。',155:'设定束缚下的讽刺或确认承诺都可成立。',156:'对幸福的回避与苦涩符合被改写处境，暂不强分。',158:'茶话会的互相打趣可成立，未强求实际赶人。',159:'看穿彼方意图、要求她先暴露秘密，符合机智应对。',165:'秘密公开后反问与冷淡，符合当前强调过去的关系阶段。',166:'借茶与事实调侃，符合故意轻描淡写旧恋情的状态。',167:'保留牵手和看穿紧张，克制与亲近可共存。',168:'处理关系的克制回应成立，未把另一人认成新恋人。'}
out=json.loads((P/'candidate_review.json').read_text(encoding='utf-8'));done={x['candidate_id'] for x in out}
for r in sorted(map(json.loads,(P/'generations.jsonl').read_text(encoding='utf-8').splitlines()),key=lambda r:(r['number'],r['sample_index'])):
 n=r['number'];key=n,r['sample_index']
 if not 142<=n<=171 or r['sample_index']>=4 or r['id'] in done or r['status']!='generated':continue
 if n in (170,171):v='hold';why='可见前情过短，未充分提供汀的请求与此前说明，先不把模型对回忆的猜测当可靠负例。'
 elif key in NEG:v='prefer_original';why=NEG[key]
 else:v='acceptable_variant';why=NOTES[n]
 d=dict(number=n,candidate_id=r['id'],sample_index=r['sample_index'],generation_sha256=r['record_sha256'],verdict=v,reason=why,review_method='project_owner_authorized_assistant_final_review',axes=['人物关系','语气与性格','当前情景与行动意图']);d['review_sha256']=digest(d);out.append(d)
(P/'candidate_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('reviewed',len(out),'prefer',sum(x['verdict']=='prefer_original' for x in out))
