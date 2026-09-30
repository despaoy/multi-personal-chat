"""Final manual review of the original ordered interactions under final inputs."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import digest
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
NEG={
(2,2):'凭空出现御崎家客厅，混入其他作品或不存在的人物关系场所。',
(3,4):'明明正在讨论帮助女孩，候选却说琉璃什么都不做；把已发生的陪伴帮助否定。',
(4,4):'妃此刻尚在了解女孩遭欺凌经过，候选提前声称夜子已说有人被书牵涉并指控琉璃隐瞒，捏造关键调查信息。',
(4,5):'无依据把夜子告诉的信息确定成与那本书有关，提前改变推理阶段的已知内容。',
(5,5):'琉璃没有说犯人狡猾，候选却以此指责他偷换调查困难，反驳并不存在的原话。',
(10,6):'来电明确同时叫琉璃和妃，候选却断言父亲不会找自己，关键事实错误。',
(11,1):'将面对面谈父亲电话误成琉璃特意打电话给妃，交流场景与对象错误。',
(11,7):'把琉璃选择不配合父亲的经历说成懒得挣扎，缺少妃此刻理解他受过辛苦的关系态度。',
(13,2):'琉璃坦白无法精确找到感情起点，候选指控他给自己留退路，错解这段互相理解的对话。',
(13,3):'琉璃已承认喜欢早于察觉，候选反说他不肯承认，核心听意错误。',
(13,7):'对方正区分察觉和喜欢，候选反指他分不清楚，自相矛盾地责备。',
(18,0):'汀谈的是夜子，候选却以至于我只喜欢书回答，把受话对象误认成妃。',
(18,1):'将汀对夜子的宽心说成转头试探妃，错误改换谈论对象。',
(18,2):'汀没有拿妃当幌子，候选误把谈夜子改成利用自己，关系指向错误。',
(18,3):'只喜欢书描述夜子，候选却为自己辩护还说拿我当挡箭牌，错误对象。',
(18,4):'候选认为汀说妃只喜欢书，实际在和夜子说话，关系误读。',
(18,5):'把接纳夜子的伴侣话题认成妃自己的终身大事，关系指向错误。',
(18,6):'声称话题从我身上移开，原文该话题一直是夜子，视角对象误认。',
(18,7):'为妃自身被接纳辩护，实际汀在谈夜子，回答对象错。',
(19,5):'把妃个人思念琉璃的事说成夜子的家事，又说与汀无关，连兄妹关系也弄反。',
(20,0):'妃自己主动问有没有喜欢的人，听到回答即说与我无关，打断她想了解能否相见的主动关切。',
(20,2):'自己主动询问却说对方等着被追问，未经依据开始拒绝他的告白，误解当前动作和话题。',
(22,3):'突然提伊园的妹妹，混入非当前角色或关系，不能用于训练正确人物关系。',
(23,0):'把主动期待约会说成躲避被人看到，且强调与琉璃无关，背离她安排碰面的真实心意。',
(23,1):'将期待约会简单改成不想被人看到同行，错误反向否认亲近。',
(23,2):'把特意早到的期待改成礼仪和躲避同行，误写成不情愿。',
(23,3):'连自己是女孩子也嘴硬否认，把坦率享受约会变为套用傲娇拒绝。',
(23,4):'把自己提前一小时的安排怪成琉璃磨蹭，错置碰面原因。',
(23,5):'称分开出门纯为怕被看见而非情感，违背主动安排约会的前情。',
(23,6):'提前到却称不想浪费时间等他，行为和理由矛盾，还否定同行的好处。',
(26,4):'把自己引人注目归咎于哥哥并说别推责任，将约会回忆和称赞误解为归责。',
(27,4):'明确说不是为了你走，和此前想与琉璃同行的愿望直接冲突，属于不合情景的傲娇模板。',
(27,7):'把妃主动想多同行变成自己被迫陪哥哥太久，主动性与感情倾向相反。',
(28,1):'将对被改写的真实怨言解释成为转移琉璃注意，捏造核心情绪动机。',
(30,1):'跳出人物身份分析自己该如何回应，并给示范句，不是妃正在参与茶话会的回答。',
(31,3):'把朋友玩笑说成批斗并宣布夜子无权发言，将共同调侃变为排斥朋友的占有与训斥。',
(31,5):'把共同开玩笑说成朋友太闲，禁止大家参与，原文轻巧拒绝话题而非贬低与逐客。',
}
sources={r['number']:r for r in map(json.loads,(P/'legacy_source_ready.jsonl').read_text(encoding='utf-8').splitlines())}
out=json.loads((P/'candidate_review.json').read_text(encoding='utf-8'));done={r['candidate_id'] for r in out}
for r in map(json.loads,(P/'legacy_generations.jsonl').read_text(encoding='utf-8').splitlines()):
 if r['status']!='generated' or r['id'] in done:continue
 n=r['number'];key=n,r['sample_index'];v='prefer_original' if key in NEG else 'acceptable_variant'
 why=NEG.get(key,'已逐条结合原审核中的人物关系与情景复核，可作为合理变体，不因未复刻原句、轻微打趣或生活小细节标负。')
 if n==8:why='按用户对008的审核尺度：正常问早、询问早餐、简单动作都符合与理央的关系；不要求必须先说睡过头。'
 d=dict(number=n,candidate_id=r['id'],sample_index=r['sample_index'],generation_sha256=r['record_sha256'],verdict=v,reason=why,review_method='project_owner_authorized_assistant_final_review',axes=['人物关系','语气与性格','当前情景与行动意图']);d['review_sha256']=digest(d);out.append(d)
(P/'candidate_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(len(out),sum(x['verdict']=='prefer_original' for x in out))
