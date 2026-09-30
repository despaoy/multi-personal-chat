"""Run only the named approved stage, refusing accidental output overwrite."""
import argparse,json,sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'backend'));RUN=CODE.parent
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['sft_r1','dpo_r1']);a=p.parse_args()
 config=json.loads((RUN/'state'/f'{a.stage}.json').read_text())
 output=Path(config['output_dir'])
 if output.exists() and any(output.iterdir()):raise ValueError('nonempty experiment output; inspect before resuming')
 if a.stage.startswith('sft'):
  from training.trainer import LoRATrainer,LoRATrainingConfig
  trainer=LoRATrainer(LoRATrainingConfig.from_dict(config));final=trainer.train()
  print('SFT complete',final,flush=True)
 else:
  from training.preference_trainer import PreferenceTrainer,PreferenceTrainingConfig
  from training.evidence_dataset import load_preference_training_rows
  rows=load_preference_training_rows(CODE/'backend/data/character_dialogues/experiments/v4_dpo100_20260926/dpo.train.jsonl')
  if len(rows)<100:raise ValueError('minimum 100 approved training pairs required')
  trainer=PreferenceTrainer(PreferenceTrainingConfig.from_dict(config));result=trainer.train(rows)
  trainer.save_report(result,output)
  if result.error:raise RuntimeError(result.error)
  print('DPO complete',result.to_dict(),flush=True)
if __name__=='__main__':main()
