"""Manual review of samples 4–11, after reading each response."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import digest
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
NEG={
(39,9):'直接说你还以为是蓝宝石，让隐瞒真相的内心信息成为明说，破坏此刻不告知的行动。',
(64,10):'声称琉璃连为何生气都答不上来，前文没有这道提问且他已说理解；无依据改写交流经过。',
(67,7):'只把缺失踪迹视为原地踏步，否定可用线索，以当助手命令替代理解进展。',
(67,8):'将调查的负证据完全贬为无能，缺少妃接住线索的推理能力。',
(67,10):'把发现转成承认自己不行，忽略线索价值；刻板贬低代替实际响应。',
(69,8):'说琉璃站在安全距离外，忽略他一直贴身陪伴女孩的前情，讥讽建立在错误事实之上。',
(69,9):'将希望女孩轻松推成伪装不图回报，原文没有这项指控，当前分析变成无据责备。',
(72,5):'将已有欺凌事实说成琉璃因抓不到犯人才想象人为成分，反转其证据顺序并责他英雄妄想。',
(72,6):'说琉璃归为诅咒像是她天生该受，误将对现状的解释认成应得论，造成不合语境的训斥。',
(72,9):'说要么人为要么偶然，把可以同时发生的两类不幸当互斥；原文探索联系，逻辑更合理。',
(73,4):'笃定现实没人执笔，和本游戏正在发现书与现实联系的情景相反，堵死关键推理。',
(73,5):'把停顿解释为兄妹关系导致不能随便说，错将发现线索转成关系隔阂与怀疑动机。',
(73,11):'输出字数说明并长篇训斥，还将旁白默许改成琉璃亲口解释自己必须留下；混淆视角并偏离发现线索。',
(81,4):'明确拒绝帮他表达，把顺势化解阻碍的亲密立场写成不负责；与本场主动示爱倾向相反。',
(81,9):'断言妹妹身份做不得准，因不同姓否认真实兄妹关系；原文机智化解阻碍并不更改血缘。',
(96,4):'宣称留下不需要夜子许可，忽略对方作为图书馆主人的权力；原文是在接受许可重建关系。',
(96,9):'说得像图书馆是你家，否认夜子与图书馆真实归属关系，挖苦对象弄错。',
(101,5):'说父母把你我都丢门外，忽略父亲刚邀请琉璃回家；被遗忘者只有妃。',
(101,9):'说父母忘掉我们，将琉璃也设成被遗忘者，核心身份错置。',
(118,8):'明明是妃主动询问，却责汀特意跑来说此事求同情；错解对话来由和羡慕相见的动机。',
(118,9):'汀没有耿耿于怀，候选却据此否定其喜欢；妃此刻更羡慕能见面，不是作感情审判。',
(118,10):'无依据劝汀省省放弃喜欢的人，方向与羡慕能够见面的妃相反。',
(118,11):'纯粹判自作自受，失去当前对能相见的羡慕与共情。',
(128,8):'妃自己先提走过多少次，候选却反问琉璃为何事到如今还提，颠倒话题发起者并拒绝共同怀念。',
(130,4):'把琉璃未说出口的图书馆不幸判断当作他说的借口攻击，混淆旁白与人物发言。',
(130,5):'称对耶为学妃说话，但此前妃没有说这个短句；错误责备之后才解释，原文承接更自然。',
(130,8):'指责琉璃学自己说话且别思考，当前只有真诚理解的回应，没有相应挑衅。',
(130,9):'说他装作早已明白，给共同理解强加虚假动机，取代当前同行的认同。',
(131,10):'把承认不可逆说成狡猾归责，忽略此刻两人面对共同决定的态度，反向责怪。',
(140,5):'把执笔设定与当前妃无关说死，否认正在追问的真实束缚；不只是自主意志的表达。',
(142,5):'编出夜子要求别顾虑和自己替她说话的前情，道歉理由建立在不存在的交流上。',
(143,5):'问为何不送夜子，夜子并未要走；错把给二人留空间的回房情景理解成送客。',
(143,11):'将体贴暂避说成碍事，无来由冷落琉璃，偏离温柔暂别的当下。',
(150,4):'夜子已经说想有人陪伴，候选却训她不肯承认想被陪着，逻辑上反驳了已明确的话。',
(150,10):'用低头拿餐具责夜子轻巧，将原文支持安静陪伴变成施压和指责，当前关系倾向错误。',
(153,7):'把纸上存在夸大成不会死，忽视可被书本改写、毁灭的处境；无限时间不等于绝对不死。',
(153,10):'从不急着安排活动跳到不会老病死的无敌设定，超出并扭曲纸上存在的事实。',
(168,8):'琉璃只是理解，候选却责他假装平静，添了不信任并压过此刻妃允许自己亲近的心情。',
}
out=json.loads((P/'candidate_review.json').read_text(encoding='utf-8'));done={r['candidate_id'] for r in out}
notes={r['number']:r for r in json.loads((P/'source_review.json').read_text(encoding='utf-8'))}
seen=0
for r in sorted(map(json.loads,(P/'generations.jsonl').read_text(encoding='utf-8').splitlines()),key=lambda r:(r['number'],r['sample_index'])):
 if not 4<=r['sample_index']<=11 or r['id'] in done or r['status']!='generated':continue
 n=r['number'];key=n,r['sample_index'];v='prefer_original' if key in NEG else 'acceptable_variant'
 why=NEG.get(key,'逐候选复核后可作合理变体；不同措辞、少量动作或轻微调侃不单独判负。当前关系情景：'+notes[n]['state'])
 d=dict(number=n,candidate_id=r['id'],sample_index=r['sample_index'],generation_sha256=r['record_sha256'],verdict=v,reason=why,review_method='project_owner_authorized_assistant_final_review',axes=['人物关系','语气与性格','当前情景与行动意图']);d['review_sha256']=digest(d);out.append(d);seen+=1
(P/'candidate_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('added',seen,'total_preferred',sum(x['verdict']=='prefer_original' for x in out))
