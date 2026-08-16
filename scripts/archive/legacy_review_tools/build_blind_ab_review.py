"""Create a deterministic blinded A/B review package from two benchmark reports."""
import argparse,hashlib,json,random
from pathlib import Path
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--a",type=Path,required=True);p.add_argument("--b",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--seed",type=int,default=42);x=p.parse_args()
 a,b=load(x.a),load(x.b);am={z["id"]:z for z in a["samples"]};bm={z["id"]:z for z in b["samples"]};ids=sorted(set(am)&set(bm));rng=random.Random(x.seed);review,key=[],[]
 for i in ids:
  left_first=rng.choice([True,False]);left,right=(am[i],bm[i])if left_first else(bm[i],am[i])
  review.append({"id":i,"category":left["category"],"prompt":left["prompt"],"response_A":left["response"],"response_B":right["response"],"winner":"","reason":"","scores":{"A":{},"B":{}}})
  key.append({"id":i,"A_model":a["model"]if left_first else b["model"],"B_model":b["model"]if left_first else a["model"]})
 x.output_dir.mkdir(parents=True,exist_ok=True)
 (x.output_dir/"blind_review.json").write_text(json.dumps({"schema_version":1,"instructions":"winner填写A/B/tie/invalid；评审时不要打开blind_key.json","samples":review},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (x.output_dir/"blind_key.json").write_text(json.dumps({"schema_version":1,"seed":x.seed,"source_hashes":{"a":hashlib.sha256(x.a.read_bytes()).hexdigest(),"b":hashlib.sha256(x.b.read_bytes()).hexdigest()},"key":key},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print({"paired":len(ids),"review":str(x.output_dir/"blind_review.json"),"key":str(x.output_dir/"blind_key.json")})
if __name__=="__main__":main()
