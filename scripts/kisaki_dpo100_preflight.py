"""CPU preflight and explicit configurations for the isolated server experiment."""
import hashlib,importlib.metadata,inspect,json,sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'backend'))
RUN=CODE.parent;DATA=CODE/'backend/data/character_dialogues/experiments/v4_dpo100_20260926'
MODEL=RUN/'models/Qwen3-8B';STATE=RUN/'state';STATE.mkdir(exist_ok=True)
def readl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 from transformers import AutoTokenizer
 from training.trainer import LoRATrainer,LoRATrainingConfig
 from training.chat_dataset import normalize_chat_record,tokenize_assistant_turns
 from training.preference_validation import validate_pairs,validate_token_budgets
 from training.preference_compat import resolve_preference_backend,reference_adapter_kwargs,render_preference_record
 from trl import SFTConfig
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 if tokenizer.pad_token is None:tokenizer.pad_token=tokenizer.eos_token
 system=(CODE/'backend/data/character_dialogues/kisaki_system_prompt_v3.txt').read_text(encoding='utf-8').strip()
 manifest=json.loads((DATA/'dataset_manifest.json').read_text(encoding='utf-8'))
 assert manifest['status']=='approved_ready_for_training'
 train=readl(DATA/'train.jsonl');valpath=CODE/'backend/data/character_dialogues/experiments/v4/validation.jsonl';val=readl(valpath);pairs=readl(DATA/'dpo.train.jsonl')
 assert len(pairs)>=100;validate_pairs(pairs,split='train')
 lengths={};datahash={}
 for key,rows in [('sft_train',train),('validation',val)]:
  sizes=[]
  for row in rows:
   m=normalize_chat_record(row,default_system_prompt=system,system_prompt_policy='replace')
   tokenized=tokenize_assistant_turns(tokenizer,m,max_length=32768,truncation_direction='left',use_chat_template=True,assistant_supervision=row.get('metadata',{}).get('assistant_supervision','all'),require_full_context=True)
   sizes.append(len(tokenized['input_ids']))
  lengths[key]={'max':max(sizes),'count':len(sizes)}
  datahash[key]=hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
 rendered=[render_preference_record(r,tokenizer) for r in pairs]
 plen=max(len(tokenizer(r['prompt'],add_special_tokens=False)['input_ids']) for r in rendered)
 flen=max(len(tokenizer(r['prompt']+r[k],add_special_tokens=False)['input_ids']) for r in rendered for k in ['chosen','rejected'])
 maxprompt=((plen+127)//128)*128+128;maxfull=max(((flen+127)//128)*128+128,maxprompt+128)
 validate_token_budgets(pairs,tokenizer,max_length=maxfull,max_prompt_length=maxprompt,render_conversation=render_preference_record)
 sftmax=((max(lengths['sft_train']['max'],lengths['validation']['max'])+127)//128)*128
 sft=LoRATrainingConfig(base_model_path=str(MODEL),train_data_path=str(DATA/'train.jsonl'),eval_data_path=str(valpath),output_dir=str(RUN/'outputs/sft_r1'),num_train_epochs=2,learning_rate=5e-5,lora_r=16,lora_alpha=32,lora_dropout=.05,load_in_4bit=True,load_in_8bit=False,bf16=True,fp16=False,max_seq_length=sftmax,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=8,gradient_checkpointing=True,packing=False,system_prompt=system,system_prompt_policy='replace',report_to='none',eval_steps=60,save_steps=60,logging_steps=10,save_total_limit=2,load_best_model_at_end=True,early_stopping_patience=3)
 sft.save(STATE/'sft_r1.json')
 # Instantiate TRL configuration on CPU to catch version contract failures first.
 from training.preference_compat import preference_length_kwargs
 _,dclass=resolve_preference_backend('dpo');ref=reference_adapter_kwargs(dclass,method='dpo',adapter_path='SFT-will-be-created')
 dclass(output_dir=str(RUN/'outputs/config_check'),use_cpu=True,report_to='none',**preference_length_kwargs(dclass,max_length=maxfull,max_prompt_length=maxprompt),**ref)
 dpo=dict(base_model_path=str(MODEL),adapter_path=str(RUN/'outputs/sft_r1/final'),output_dir=str(RUN/'outputs/dpo_r1'),method='dpo',beta=.1,learning_rate=2e-6,num_train_epochs=1,per_device_train_batch_size=1,gradient_accumulation_steps=4,max_length=maxfull,max_prompt_length=maxprompt,load_in_4bit=True,gradient_checkpointing=True,seed=42)
 write(STATE/'dpo_r1.json',dpo)
 report=dict(status='passed',data_hashes=datahash,lengths=lengths,dpo_prompt_tokens=plen,dpo_max_full_tokens=flen,sft_max_length=sftmax,dpo_max_length=maxfull,dpo_max_prompt_length=maxprompt,dpo_pairs=len(pairs),versions={x:importlib.metadata.version(x) for x in ['torch','transformers','trl','peft','datasets','accelerate','bitsandbytes']},sft_config_fields=list(inspect.signature(SFTConfig).parameters))
 write(STATE/'preflight.json',report);print(json.dumps({k:v for k,v in report.items() if k!='sft_config_fields'},ensure_ascii=False))
if __name__=='__main__':main()
