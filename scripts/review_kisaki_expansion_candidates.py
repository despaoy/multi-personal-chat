"""Store independent manual judgments bound to actual sampled responses."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import digest
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
# Explicit per-candidate dispositions: acceptable variants are never negatives.
NEG={
 (39,0):'直接告诉琉璃不想让他知道全部，把内心隐瞒变成公开声明；原文以两人亲近的好处转移疑问，更符合此刻行动。',
 (52,2):'把已经解释的过去落败说成现在彼方即将抢走琉璃、自己要争抢；改变这场告终式坦白的时间关系和意图。',
 (58,1):'“只会让你活得像个人”把逼她面对感情变为人格贬辱；原文明确指向告白，尖锐却不是否定夜子作为人。',
 (64,3):'错把驱除不幸的对象说成夜子；原文是夜子希望帮助新来的女孩，人物关系弄错。',
 (67,0):'全面否定没有发现犯人也构成线索，把正在听取调查进展的妃写成只会贬低琉璃；原文承接这个发现。',
 (69,0):'将女孩得到安慰说成有人替琉璃买单，凭空引入受害指控，盖过此刻妃已承认陪伴作用的推理。',
 (69,2):'把琉璃希望女孩轻松误读为他自己陪在身边就满足，用泛化嫉妒替代对女孩心情的分析。',
 (69,3):'此前妃已宣布参与并转向推理，候选又把她写成仍被丢在观众席等剧终，情景推进倒退。',
 (72,0):'琉璃明确区分人为与偶发因素，候选却指责他混淆，反驳不存在的错误并转成训课，弱于原文的探究回应。',
 (72,2):'当前对方已经分别说明两类不幸，候选以你混在一起了开头冤枉对方；不是自然讥讽而是误解发言。',
 (73,0):'妃停顿是在联想事件线索，候选退回怕琉璃误会挖苦而不说，错解停顿目的和两人交流关系。',
 (73,1):'将正在形成线索的思考转为回避和盘问女孩地位；回退到泛化吃醋，损失此刻妃的推理主动性。',
 (73,2):'妃刚主动用书与现实作比，候选反而说这样做是傲慢，背离已经展开的推理与当前态度。',
}
ACCEPT={
39:'保留对规则的轻巧态度或希望陪伴的依恋，未明确改变关系；不用必须提牵手作硬性判据。',
40:'坚持不放弃并回应结束主题，符合依恋和自主性；较长或不同比喻不是负例依据。',
51:'能接住世界异常和复杂情绪，带冷淡调侃仍可成立。',
52:'坦白不想继续隐忍的情绪可成立；不要求逐字复述现在才能传达。',
55:'收束对话并催琉璃回到夜子处，保留克制和决断；小动作与景物不单独判负。',
56:'区分家与该回的地方，保持告别和引导；不因解释更多就标负。',
57:'识破逃避并要求面对内心，语气尖锐可成立；轻微调侃不强制判错。',
58:'指出逃避与嫉妒，仍在促使她承认感情；未说告白并不自动负例。',
59:'推动正视真实感情，直接且有对象，不是空泛安慰。',
60:'困倦、依恋或回嘴都在关系允许范围，小动作可接受。',
63:'带嫉妒地要求解释符合此前生气状态；不能因没有原文新信息而惩罚。',
64:'维持讥讽同时有介入或关心的意向，可作为合理变体。',
65:'亲密抱怨与在意牵手符合此刻；无需复刻否认多管闲事。',
66:'保持占有欲或准备介入，不强求必须说出场；极端比喻已有前文，不脱离情景泛化。',
67:'至少能接住线索或继续询问，调侃不妨碍推理，暂不强分优劣。',
68:'承认陪伴安慰的作用并夹带不甘或调侃，合乎关系。',
69:'仍有主动参与意图，轻微讥讽尚可；不因缺少假设字样强判。',
71:'回应欺凌与不幸的联系，逻辑和语气尚可，不把长度差当负例。',
72:'继续探究巧合与联系，符合她的怀疑精神，不要求同原文短句。',
73:'继续探问故事与现实关系，可理解为引导推理，暂不强判负。',
74:'能承接书籍线索，自信讲解可成立，不要求完全复现留白。',
}
def main():
 path=P/'candidate_review.json';old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
 done={x['candidate_id'] for x in old}
 for r in sorted(map(json.loads,(P/'generations.jsonl').read_text(encoding='utf-8').splitlines()),key=lambda x:(x['number'],x['sample_index'])):
  n=r['number'];key=(n,r['sample_index'])
  if n>75 or r['sample_index']>=4 or r['status']!='generated' or r['id'] in done:continue
  if n==50:verdict='hold';reason='场景说明提前透露目标中的第二位；本批不入库，保留原始生成记录，需修正输入再采样。'
  elif key in NEG:verdict='prefer_original';reason=NEG[key]
  else:verdict='acceptable_variant';reason=ACCEPT[n]
  row=dict(number=n,candidate_id=r['id'],sample_index=r['sample_index'],generation_sha256=r['record_sha256'],verdict=verdict,reason=reason,review_method='project_owner_authorized_assistant_final_review',axes=['人物关系','语气与性格','当前情景与行动意图'])
  row['review_sha256']=digest(row);old.append(row)
 path.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('reviewed',len(old),'prefer_original',sum(x['verdict']=='prefer_original' for x in old))
if __name__=='__main__':main()
