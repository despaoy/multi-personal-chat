"""Own one isolated GPU experiment and restore this project's inference service."""
import fcntl,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];RUN=CODE.parent;STATE=RUN/'state'
PY=str(RUN/'venv/bin/python');MANAGE='/home/boot/lhm/multipersonal-runtime/manage.py'
def record(**value):
 value['updated_at']=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
 (STATE/'queue.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
def main():
 lock=(STATE/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert json.loads((STATE/'preflight.json').read_text())['status']=='passed'
 assert json.loads((RUN/'model_download.json').read_text())['status']=='complete'
 status=subprocess.check_output([PY,MANAGE,'status','vllm'],text=True);restore=' running ' in status
 (STATE/'service_before.txt').write_text(status)
 child=None;stage='prepare';error=None;completed=False
 def interrupt(signum,frame):
  if child is not None and child.poll() is None:child.terminate()
  raise RuntimeError(f'interrupted by signal {signum}')
 signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
 steps=[('base_eval',[PY,str(CODE/'scripts/kisaki_dpo100_evaluate.py'),'--stage','base']),('sft_train',[PY,str(CODE/'scripts/kisaki_dpo100_train_stage.py'),'sft_r1']),('sft_eval',[PY,str(CODE/'scripts/kisaki_dpo100_evaluate.py'),'--stage','sft_r1','--adapter',str(RUN/'outputs/sft_r1/final')]),('dpo_train',[PY,str(CODE/'scripts/kisaki_dpo100_train_stage.py'),'dpo_r1']),('dpo_eval',[PY,str(CODE/'scripts/kisaki_dpo100_evaluate.py'),'--stage','dpo_r1','--adapter',str(RUN/'outputs/dpo_r1')])]
 try:
  if restore:subprocess.run([PY,MANAGE,'stop','vllm'],check=True)
  for _ in range(30):
   used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   if used<1500:break
   time.sleep(2)
  else:raise RuntimeError('GPU not free after stopping owned inference; refusing to affect other jobs')
  (RUN/'logs').mkdir(exist_ok=True)
  env={**os.environ,'PYTHONPATH':str(CODE/'backend'),'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
  for stage,cmd in steps:
   record(status='running',stage=stage,pid=os.getpid(),restore_inference=restore)
   with (RUN/'logs'/f'{stage}.log').open('ab') as log:
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=CODE,env=env)
    code=child.wait()
   if code:raise RuntimeError(f'{stage} exited {code}; inspect its log')
  completed=True
 except BaseException as exc:
  error=f'{type(exc).__name__}: {exc}'
  raise
 finally:
  if child is not None and child.poll() is None:
   child.terminate()
   try:child.wait(timeout=30)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  restart_code=subprocess.run([PY,MANAGE,'start','vllm']).returncode if restore else None
  healthy=None
  if restore and restart_code==0:
   healthy=False
   for _ in range(60):
    try:
     with urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=2) as response:
      healthy=response.status==200
     if healthy:break
    except (OSError,TimeoutError):pass
    time.sleep(2)
  record(status='complete' if completed and restart_code in (None,0) and healthy is not False else 'needs_attention',last_stage=stage,inference_restart_requested=restore,inference_restart_exit_code=restart_code,inference_healthy=healthy,error=error)
if __name__=='__main__':main()
