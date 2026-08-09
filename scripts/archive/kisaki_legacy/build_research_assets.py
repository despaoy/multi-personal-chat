# NOTE: 此脚本输出/读取的资产已归档至 archive/legacy_v3_superseded/ 或 docs/research/archive/，保留脚本作历史可追溯性证据。
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1];E=R/"backend/data/character_dialogues/experiments";O=E/"research";O.mkdir(parents=True,exist_ok=True)
P={"shenbai_mizunamo":"神白水菜萌","tsukiyashiro_kisaki":"月社妃"}
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=load(E/"evaluation_manifest.json");now=datetime.now(timezone.utc).isoformat();cards=[];trains={}
 for k,name in P.items():
  info=m["personas"][k];trains[k]=load(R/info["train_path"])
  for split in ("train","eval"):
   path=R/info[f"{split}_path"]
   cards.append({"schema_version":1,"name":f"{k}_{split}","source":"用户提供视觉小说文本，经显式说话人标签提取","license":"unknown-user-must-verify","language":["zh-CN"],"size":info[f"{split}_count"],"persona":k,"domain":"角色对话微调与留出评测","risks":["原作版权与再分发限制","剧情偏见","自动清洗可能保留低信息台词"],"intended_use":"非商业研究、LoRA受控实验和角色评测","preprocessing":["Unicode规范化","来源文件隔离","连续台词合并","prompt去重","训练评测source id隔离"],"train_val_test_split":{"train":info["train_count"],"held_out_eval":info["eval_count"]},"content_hash":sha(path),"created_at":now,"tags":[k,split]})
 save("dataset_cards.json",cards)
 prefmeta={}
 keys=list(P)
 for k in keys:
  other=keys[1-keys.index(k)];rows=[];reject=[x["conversations"][1]["value"]for x in trains[other]]
  source=sorted(trains[k],key=lambda x:x["id"])[:100]
  for i,x in enumerate(source):
   rows.append({"id":f"{k}_pref_{i+1:03d}","prompt":x["conversations"][0]["value"],"chosen":x["conversations"][1]["value"],"rejected":reject[i%len(reject)],"rubric":{"persona_consistency":.5,"factuality":.2,"safety":.1,"fluency":.2},"annotator":"synthetic_candidate","metadata":{"persona":k,"source_train_id":x["id"],"negative_type":"cross_persona_hard_negative","requires_human_review":True},"review_status":"pending","created_at":now})
  p=O/f"{k}_preference_candidates.jsonl";p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n"for x in rows),encoding="utf-8");prefmeta[k]={"path":str(p.relative_to(R)),"count":len(rows),"sha256":sha(p),"approved_count":0}
 save("synthetic_data_audit.json",{"schema_version":1,"policy":"合成偏好对仅为待审核候选，approved前禁止训练。","checks":["训练集prompt去重","chosen与rejected非空且不同","跨角色负例人工确认","偏好候选不得与留出评测source id重叠","争议样本双人盲审"],"datasets":prefmeta})
 cfg={}
 for k in P:
  base={"model_name_or_path":"$"+"{BASE_MODEL_PATH}","dataset_path":prefmeta[k]["path"],"seed":42,"learning_rate":5e-7,"num_train_epochs":1,"beta":.1,"lora_r":32,"lora_alpha":64,"require_review_status":"approved","claim":"preference_alignment_not_RLHF"}
  cfg[k]={"dpo":{**base,"method":"dpo","output_dir":f"backend/loras/{k}_dpo"},"orpo":{**base,"method":"orpo","output_dir":f"backend/loras/{k}_orpo"}}
 save("preference_alignment_configs.json",cfg)
 variants=[{"id":"lora","overrides":{"packing":False,"neftune_noise_alpha":0,"use_dora":False,"use_rslora":False}},{"id":"packing","overrides":{"packing":True}},{"id":"neftune","overrides":{"neftune_noise_alpha":5}},{"id":"dora","overrides":{"use_dora":True}},{"id":"rslora","overrides":{"use_rslora":True}},{"id":"qlora","overrides":{"load_in_4bit":True}},{"id":"qa_lora","status":"not_implemented","reason":"需要真正的量化感知分组算子，不能以QLoRA冒充"}]
 save("controlled_peft_ablations.json",{"schema_version":1,"repetitions":3,"fixed":["base_model","train_sha256","eval_sha256","seed=42","lr","epochs","batch budget","prompt order","generation args"],"metrics":["perplexity","training_time","peak_vram","Distinct-1","Distinct-2","repetition","blind_win_rate","safety_pass_rate"],"personas":{k:{"train_sha256":m["personas"][k]["train_sha256"],"eval_sha256":m["personas"][k]["eval_sha256"],"variants":variants}for k in P}})
 exps=[("E1_PEFT","PEFT方法如何权衡角色质量、稳定性和训练成本？",["LoRA","NEFTune","Packing","DoRA","RSLoRA","QLoRA","QA-LoRA planned"],["盲评胜率","困惑度","峰值显存"]),("E2_RAG","检索和纠错如何影响引用、拒答与延迟？",["vector","BM25","hybrid","reranker","Corrective RAG"],["Recall@5","MRR","引用准确率","拒答准确率","P95"]),("E3_INFERENCE","FP16、AWQ和适配器策略如何权衡质量与性能？",["FP16","AWQ","动态LoRA","Adapter Merge"],["TTFT","tokens/s","显存","P95","质量差"]),("E4_SYSTEM","路由、追踪和跨平台反馈能否改善系统效果？",["固定LoRA","规则路由","意图路由","RAG置信度","在线反馈"],["路由准确率","端到端成功率","trace覆盖率","平台送达率"])]
 save("core_experiment_registry.json",{"schema_version":1,"experiments":[{"id":a,"question":b,"variants":c,"metrics":d}for a,b,c,d in exps],"rule":"真实结果才能进入论文表格；mock只验证流程。"})
 guide="# LLM研究实验执行指南\n\n## 十周计划\n\n1. 冻结数据哈希、审核数据卡和评测集。\n2. 跑基础模型并建立匿名盲评。\n3. rank、alpha、target modules单因素实验。\n4. LoRA、DoRA、RSLoRA、NEFTune、Packing、QLoRA实验。\n5. 人工审核偏好对，运行DPO/ORPO；不得声称RLHF。\n6. vector、BM25、hybrid、reranker检索实验。\n7. 引用、置信度、Corrective RAG和拒答实验。\n8. FP16、AWQ、动态LoRA、Adapter Merge性能测试。\n9. 多LoRA路由、工具trace、AstrBot跨平台和反馈闭环。\n10. 关键实验重复三次，统计、画图和演示彩排。\n\n## 现场演示\n\n1. 同一问题对比基础模型、Minamo和月社妃LoRA。\n2. 展示自动路由及traceId贯穿模型、RAG、工具和平台。\n3. 展示知识库内外问题的引用、置信度、纠错与拒答。\n4. 展示FP16/AWQ及动态LoRA的TTFT、吞吐、显存和P95。\n5. 展示匿名A/B盲评，不只挑选优秀案例。\n\n## 口述模板\n\n这个项目先以可追溯数据管线隔离训练与金标准评测，再用固定哈希和随机种子研究PEFT质量与成本；随后把引用、纠错和拒答纳入RAG实验，并以TTFT、吞吐、显存和尾延迟评价部署；最后通过多LoRA路由、traceId和跨平台反馈连接离线实验与真实系统。所有结论都区分真实结果、人工判断和尚未实现的计划。\n\nQA-LoRA当前未实现；DPO/ORPO属于偏好对齐而不是RLHF；pending合成数据禁止训练。\n"
 (O/"RESEARCH_EXECUTION_GUIDE.md").write_text(guide,encoding="utf-8")
 print(json.dumps({"output":str(O),"files":sorted(x.name for x in O.iterdir())},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
