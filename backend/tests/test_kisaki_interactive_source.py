"""Source integrity and review-boundary tests for the new curated interaction set."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_kisaki_interactive_review import DATA, SELECTION, build_one, heldout, readl
from extract_character_dialogues import read_script_events


class InteractiveSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs = json.loads(SELECTION.read_text(encoding="utf-8"))
        cls.raw = readl(DATA / "tsukiyashiro_kisaki_raw.jsonl")
        cls.index = {(r['source_file'], r['source_line_start']): r for r in cls.raw}
        cls.blocked, cls.ranges = heldout(cls.raw, readl(DATA / "experiments/v4/validation.jsonl"), json.loads((ROOT / "backend/evaluation/kisaki_gold_set_v3.json").read_text(encoding="utf-8")))

    def args(self, spec):
        p = ROOT / 'gametext/纸上魔法使' / spec[0]
        return [spec, read_script_events(p), p.read_text(encoding='utf-8-sig').splitlines(), self.index, self.blocked, self.ranges]

    def test_all_selections_are_literal_single_events_without_target_in_history(self):
        ids = []
        for s in self.specs:
            r = build_one(*self.args(s))
            ids.append(r['id'])
            self.assertEqual(len(r['source']['event_ids']), 1)
            self.assertEqual(r['original'], self.index[(s[0],s[2])]['text'])
            self.assertLessEqual(len(r['original']),65)
            self.assertEqual(r['incoming'][-1]['speaker'],r['speaker'])
            self.assertTrue(all(e['line_end']<s[2] for e in r['history']+r['incoming']))
            self.assertNotIn(r['original'],r['prompt'][-1]['content'])
            self.assertFalse(r['human_approved'])
            self.assertIsNone(r['generated'])
        self.assertEqual(len(ids),len(set(ids)))

    def test_heldout_scene_and_overlapping_context_are_rejected(self):
        s = self.specs[0]
        args = self.args(s)
        args[4] = {self.index[(s[0],s[2])]['scene_block_id']}
        with self.assertRaisesRegex(ValueError,'heldout scene'):
            build_one(*args)
        args = self.args(s)
        args[5] = {s[0]:[(s[1],s[1])]}
        with self.assertRaisesRegex(ValueError,'heldout window'):
            build_one(*args)

    def test_source_mutation_and_interlocutor_mismatch_fail_closed(self):
        s = copy.deepcopy(self.specs[0]);s[3] = '汀'
        with self.assertRaisesRegex(ValueError,'interlocutor'):
            build_one(*self.args(s))
        args = self.args(self.specs[0])
        e = next(e for e in args[1] if e['line_start']==self.specs[0][2])
        e['text'] += '改写'
        with self.assertRaisesRegex(ValueError,'source mismatch'):
            build_one(*args)

    def test_copyedit_not_approved_and_room_scene_excludes_previous_speaker(self):
        s = next(s for s in self.specs if s[2]==1124)
        self.assertEqual(build_one(*self.args(s))['review_status'],'needs_copyedit')
        s = next(s for s in self.specs if s[2]==1720)
        r = build_one(*self.args(s))
        self.assertEqual(r['source']['context_start'],1717)
        self.assertEqual(r['speaker'],'琉璃')
        self.assertTrue(all(e['speaker']!='汀' for e in r['history']+r['incoming']))


if __name__ == '__main__':
    unittest.main()
