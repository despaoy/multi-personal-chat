"""Isolated DPO100 experiment transport. Credentials come only from LAB_* env."""
import argparse,hashlib,json,posixpath,shlex
from pathlib import Path
from remote_config import connect_ssh
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/boot/lhm/kisaki-dpo100-20260926'
PYTHON='/home/boot/lhm/multipersonal-runtime/venv/bin/python'
def mkdir(sftp,path):
 parts=path.strip('/').split('/');cur=''
 for part in parts:
  cur+='/'+part
  try:sftp.stat(cur)
  except OSError:sftp.mkdir(cur)
def command(c,cmd,timeout=60):
 _,o,e=c.exec_command(cmd,timeout=timeout);out=o.read().decode();err=e.read().decode();code=o.channel.recv_exit_status()
 print(out);print(err)
 if code:raise RuntimeError(f'remote command exit {code}')
def upload(c,files):
 s=c.open_sftp();hashes={}
 try:
  for path in files:
   rel=path.relative_to(ROOT).as_posix();dest=REMOTE+'/code/'+rel;mkdir(s,posixpath.dirname(dest));s.put(str(path),dest)
   h=hashlib.sha256(path.read_bytes()).hexdigest()
   with s.open(dest,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==h
   hashes[rel]=h
  mkdir(s,REMOTE+'/state')
  with s.open(REMOTE+'/state/upload_manifest.json','w') as f:f.write(json.dumps(hashes,indent=2))
 finally:s.close()
 print('uploaded',len(hashes),'verified files')
def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['download','status','upload']);a=p.parse_args();c=connect_ssh()
 try:
  if a.action=='upload':
   files=[]
   for folder in ['backend/training','backend/inference','backend/evaluation']:
    files += list((ROOT/folder).glob('*.py'))
   for folder in ['backend/data/character_dialogues/experiments/v4_dpo100_20260926']:
    files += [x for x in (ROOT/folder).iterdir() if x.suffix in {'.json','.jsonl'}]
   files += [ROOT/'backend/data/character_dialogues/experiments/v4/validation.jsonl',ROOT/'backend/data/character_dialogues/kisaki_system_prompt_v3.txt']
   files += list((ROOT/'scripts').glob('kisaki_dpo100_*.py'))
   upload(c,files)
  elif a.action=='download':
   script='''import json,os,time,traceback
from pathlib import Path
from huggingface_hub import snapshot_download,HfApi
root=Path("/home/boot/lhm/kisaki-dpo100-20260926");root.mkdir(exist_ok=True)
state=root/"model_download.json"
try:
 api=HfApi(endpoint="https://hf-mirror.com")
 info=api.model_info("Qwen/Qwen3-8B")
 state.write_text(json.dumps({"status":"downloading","repo":"Qwen/Qwen3-8B","revision":info.sha}),encoding="utf-8")
 path=snapshot_download("Qwen/Qwen3-8B",revision=info.sha,endpoint="https://hf-mirror.com",local_dir=root/"models/Qwen3-8B",allow_patterns=["*.json","*.safetensors","*.txt","*.jinja","LICENSE"],max_workers=4)
 state.write_text(json.dumps({"status":"complete","repo":"Qwen/Qwen3-8B","revision":info.sha,"path":path}),encoding="utf-8")
except Exception as e:
 state.write_text(json.dumps({"status":"failed","error_type":type(e).__name__,"error":str(e)[:300]}),encoding="utf-8");raise
'''
   s=c.open_sftp();mkdir(s,REMOTE)
   with s.open(REMOTE+'/download_model.py','w') as f:f.write(script)
   s.close()
   command(c,f"test ! -f {REMOTE}/download.pid || ! kill -0 $(cat {REMOTE}/download.pid) 2>/dev/null")
   command(c,f"nohup env HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 {PYTHON} -u {REMOTE}/download_model.py > {REMOTE}/download.log 2>&1 < /dev/null & echo $! > {REMOTE}/download.pid")
  else:
   command(c,f"cat {REMOTE}/model_download.json; tail -c 1500 {REMOTE}/download.log")
 finally:c.close()
if __name__=='__main__':main()
