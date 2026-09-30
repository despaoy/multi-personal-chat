"""Independent ordinary-model baseline, without the specialist persona system card.

Off-policy candidates are judged under the canonical persona prompt; both prompts
are retained. No instruction asks the model to make errors or act out of character.
"""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from training.persona_sampling import chat_completion,digest
P=ROOT/'backend/data/character_dialogues/experiments/dpo_expansion_100_20260926'
MODEL={'name':'deepseek_general_baseline','model':'deepseek-chat','revision':'deepseek-chat-unpinned-api-alias','base_url':'https://api.deepseek.com/v1','api_key_env':'DEEPSEEK_API_KEY'}
def run(job):
 r,i=job
 prompt=[{'role':'system','content':'请根据提供的游戏对话上下文，为角色“妃”写出下一句自然、简短的回应。只输出她的回答，不要解释。'},r['prompt'][-1]]
 params={'temperature':[.7,.95,1.1,1.2][i],'top_p':.95,'max_tokens':256,'seed':2042+i}
 out=dict(number=r['number'],scene_id=r['id'],sample_index=i,id=r['id']+f'_general_{i}',prompt=prompt,training_prompt_sha256=digest(r['prompt']),input_sha256=digest(prompt),source_record_sha256=r['record_sha256'],model=MODEL,generation=params,sampling_profile='ordinary_model_without_persona_card')
 try:out.update(generated=chat_completion(MODEL,prompt,params,timeout=90),status='generated')
 except (ValueError,RuntimeError,OSError) as e:out.update(status='error',error_type=type(e).__name__)
 out['record_sha256']=digest(out);return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--samples',type=int,default=4);args=a.parse_args()
 if not os.environ.get('DEEPSEEK_API_KEY'):raise SystemExit('key unset')
 path=P/'general_generations.jsonl';old=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
 done={r['id'] for r in old if r['status']=='generated'}
 rows=[json.loads(s) for s in (P/'source_ready.jsonl').read_text(encoding='utf-8').splitlines()]
 # Inputs found needing a source-context correction are held, never sampled here.
 jobs=[(r,i) for r in rows if r['number'] not in {50,100,171} for i in range(args.samples) if r['id']+f'_general_{i}' not in done]
 with path.open('a',encoding='utf-8') as f,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  for r in pool.map(run,jobs):
   f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
   print(json.dumps({k:r[k] for k in ('number','sample_index','status')}),flush=True)
if __name__=='__main__':main()
