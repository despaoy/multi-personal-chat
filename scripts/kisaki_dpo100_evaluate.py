"""Deterministic held-out continuation evaluation, with no target in generation."""
import argparse,hashlib,json,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'backend'));RUN=CODE.parent
def readl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);p.add_argument('--adapter',default='');a=p.parse_args()
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,set_seed
 from peft import PeftModel
 from training.chat_dataset import normalize_chat_record,tokenize_assistant_turns
 set_seed(42);modelpath=RUN/'models/Qwen3-8B'
 tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
 if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
 model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16),torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
 if a.adapter:model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False)
 model.eval();system=(CODE/'backend/data/character_dialogues/kisaki_system_prompt_v3.txt').read_text(encoding='utf-8').strip()
 rows=readl(CODE/'backend/data/character_dialogues/experiments/v4/validation.jsonl')
 out=RUN/'evaluations';out.mkdir(exist_ok=True);path=out/(a.stage+'.jsonl')
 if path.exists():raise ValueError('evaluation artifact exists; use a new stage id')
 results=[]
 with path.open('w',encoding='utf-8') as f:
  for i,row in enumerate(rows):
   messages=normalize_chat_record(row,default_system_prompt=system,system_prompt_policy='replace')
   assert messages[-1]['role']=='assistant'
   prompt=messages[:-1];expected=messages[-1]['content'];start=time.time()
   rendered=tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   batch=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to('cuda')
   supervised=tokenize_assistant_turns(tokenizer,messages,max_length=4096,assistant_supervision='last',require_full_context=True)
   with torch.inference_mode():
    loss=float(model(**{k:torch.tensor([v],device='cuda') for k,v in supervised.items()}).loss)
    generated=model.generate(**batch,max_new_tokens=192,do_sample=False,pad_token_id=tokenizer.pad_token_id,eos_token_id=tokenizer.eos_token_id)
   tokens=generated[0,batch['input_ids'].shape[1]:].tolist();response=tokenizer.decode(tokens,skip_special_tokens=True).strip()
   result=dict(id=row['id'],stage=a.stage,prompt=prompt,expected=expected,generated=response,conditional_nll=loss,supervised_tokens=sum(x!=-100 for x in supervised['labels']),output_tokens=len(tokens),stopped_on_eos=bool(tokens and tokens[-1]==tokenizer.eos_token_id),duration_s=time.time()-start,prompt_sha256=hashlib.sha256(json.dumps(prompt,ensure_ascii=False,sort_keys=True).encode()).hexdigest())
   f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush();results.append(result)
   print(json.dumps({'stage':a.stage,'completed':i+1,'total':len(rows),'output_tokens':len(tokens)}),flush=True)
 summary=dict(stage=a.stage,count=len(results),mean_nll=sum(r['conditional_nll'] for r in results)/len(results),token_weighted_nll=sum(r['conditional_nll']*r['supervised_tokens'] for r in results)/sum(r['supervised_tokens'] for r in results),empty=sum(not r['generated'] for r in results),length_limit=sum(not r['stopped_on_eos'] for r in results),mean_output_tokens=sum(r['output_tokens'] for r in results)/len(results),model=str(modelpath),adapter=a.adapter,decoding={'do_sample':False,'max_new_tokens':192,'enable_thinking':False},evaluation_partition='unchanged_70_validation; not final Gold')
 (out/(a.stage+'.summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
