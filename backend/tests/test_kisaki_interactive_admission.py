"""Integration checks on the admitted interaction snapshot and its provenance."""
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backend')]
import promote_kisaki_interactive_training as promotion
from training.preference_validation import validate_pairs
from training.chat_dataset import normalize_chat_record


class AdmissionTests(unittest.TestCase):
    def test_snapshot_and_frozen_baseline(self):
        m=promotion.load(promotion.OUT/'dataset_manifest.json')
        base=promotion.load(promotion.V4/'canonical_dataset_manifest.json')
        self.assertEqual(m['status'],'approved_data_not_run')
        self.assertEqual(promotion.textsha(promotion.V4/'train.jsonl'),base['train']['sha256'])
        self.assertEqual(promotion.textsha(promotion.V4/'validation.jsonl'),base['validation']['sha256'])
        self.assertEqual(promotion.textsha(promotion.OUT/'train.jsonl'),m['train']['sha256'])
        self.assertEqual(m['train']['count'],928)
        self.assertEqual(promotion.load(promotion.OUT/'gold_contamination_audit.json')['status'],'clean')

    def test_admission_excludes_reserved_context_and_copyedit(self):
        sft=promotion.readl(promotion.OUT/'sft.interactive.approved.jsonl')
        nums={r['metadata']['final_review_number'] for r in sft}
        self.assertEqual(nums,set(range(1,39))-{6,7,32})
        final=promotion.load(promotion.OUT/'final_review.json')
        self.assertEqual([r['number'] for r in final if r['sft']=='hold'],[6,7,32])
        for r in final:
            for e in r.get('supplemental_evidence',[]):
                self.assertLess(e['end'],r['source']['source_line_start'])

    def test_pair_content_binding_and_sft_prompt_agreement(self):
        pairs=promotion.readl(promotion.OUT/'dpo.train.jsonl')
        validate_pairs(pairs,split='train')
        self.assertEqual([r['metadata']['final_review_number'] for r in pairs],[10,18,23,27])
        sft={r['id']:r for r in promotion.readl(promotion.OUT/'sft.interactive.approved.jsonl')}
        system=(promotion.BASE.parent.parent/'kisaki_system_prompt_v3.txt').read_text(encoding='utf-8')
        for p in pairs:
            normalized=normalize_chat_record(sft[p['id']],default_system_prompt=system,system_prompt_policy='replace')
            self.assertEqual(p['prompt'],normalized[:-1])
            self.assertEqual(p['chosen'],normalized[-1:])
            self.assertFalse(p['metadata']['rejected_regenerated_for_training_prompt'])
        self.assertIn('妃对父亲印象很差',pairs[0]['prompt'][-1]['content'])
        self.assertNotIn('当前对话者',pairs[0]['prompt'][0]['content'])

    def test_same_source_is_replaced_not_oversampled(self):
        added=promotion.readl(promotion.OUT/'sft.interactive.approved.jsonl')
        merged=promotion.readl(promotion.OUT/'train.jsonl')
        added_ids={r['id'] for r in added}
        events={e for r in added for e in r['metadata']['target_event_ids']}
        self.assertEqual(len(added),35)
        self.assertEqual(len(promotion.readl(promotion.OUT/'replaced_baseline_records.jsonl')),33)
        for r in merged:
            if r['id'] not in added_ids:
                self.assertFalse(events & set(r.get('metadata',{}).get('target_event_ids',[])))


if __name__=='__main__':unittest.main()
