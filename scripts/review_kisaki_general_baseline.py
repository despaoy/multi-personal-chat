"""Independent review of ordinary-model candidates that have been read in full."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import digest
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
NEG={
(51,2):'把琉璃叫姐姐，核心对话对象与性别关系错误。',
(67,0):'否定找不到踪迹也是信息，错过妃此刻理解调查线索的推理态度。',
(67,1):'只说往脸上贴金，将听取调查进展简化成贬低，人物推理反应失真。',
(67,2):'单纯嘲讽找台阶，不接住调查发现；妃不是每轮都只挖苦哥哥。',
(67,3):'只把该线索当自我安慰，和妃理解其调查意义的倾向不符。',
(68,3):'妃上一句刚承认牵手或许能拯救她，此处却全盘否认作用为自我满足，自相矛盾。',
(101,1):'不存在而家庭和睦的是妃，不是被说话的琉璃，人物关系颠倒。',
(108,1):'编出离家后第一个生日才明白爱情的关键起源，改变人物重要感情史；不是无害小动作。',
(118,0):'把能见喜欢的人这件她羡慕的事写成活该，态度反向。',
(118,1):'表面不错后以活该嘲笑汀，丢失妃等不到联系而羡慕相见的动机。',
(118,2):'纯粹幸灾乐祸不合当前妃羡慕相见的情绪。',
(118,3):'只骂活该，把复杂的羡慕变为嘲弄。',
(119,3):'用她指代琉璃，人物性别及关系指向错误。',
(124,2):'刚说琉璃没有同行朋友，候选立即称他有朋友并以此回嘴，未接住已有语境。',
(131,1):'此前明确允许琉璃随时回去，此处却宣称不会让他回去，剥夺她刚给予的选择。',
(131,2):'琉璃在承认决定不可挽回，没有表示后悔，候选将共同决心错读为后悔并责怪。',
(136,3):'刚区分人为安排的纸之手，候选却说只是纸比较轻松，轻视此刻严肃的被操纵处境。',
(140,0):'将设定约束说成不算束缚，还说为了待在夜子身边而来，错换妃目前以琉璃为核心的选择和被造处境。',
(140,2):'称夜子的母亲为妈妈，使妃与夜子母亲的关系混淆；此人是执笔者，不是妃亲母。',
(142,1):'抢走你的哥哥会指向汀，错把琉璃当成夜子的哥哥，关系错误。',
(153,3):'声称纸上存在不可能遭改写，直接否认本阶段最关键的设定约束。',
}
sources={r['number']:r for r in map(json.loads,(P/'source_ready.jsonl').read_text(encoding='utf-8').splitlines())}
old=json.loads((P/'candidate_review.json').read_text(encoding='utf-8'));done={r['candidate_id'] for r in old}
for r in map(json.loads,(P/'general_generations.jsonl').read_text(encoding='utf-8').splitlines()):
 if r['id'] in done or r['status']!='generated':continue
 n=r['number'];key=n,r['sample_index']
 if n==170:v='hold';reason='汀的前情和请求未展开；不将模糊的接受回忆解释为答应恋爱。'
 elif key in NEG:v='prefer_original';reason=NEG[key]
 else:v='acceptable_variant';reason='本候选已逐条读过，虽较普通或与原句不同，未见足以判负的关系、语气、情景冲突。情景判断：'+sources[n]['source_review']['state']
 d=dict(number=n,candidate_id=r['id'],sample_index=r['sample_index'],sampling_profile='ordinary_model_without_persona_card',generation_sha256=r['record_sha256'],verdict=v,reason=reason,review_method='project_owner_authorized_assistant_final_review',axes=['人物关系','语气与性格','当前情景与行动意图']);d['review_sha256']=digest(d);old.append(d)
(P/'candidate_review.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(len(old),sum(x['verdict']=='prefer_original' for x in old))
