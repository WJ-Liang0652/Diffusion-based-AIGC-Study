import json
from pathlib import Path
import sys
import unittest

TASK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK/"src"))
from stage02b_relations import queue,next_candidate,interpretations


class QueueRules(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((TASK/"configs/stage02b.json").read_text())

    def test_all_p1_before_p2(self):
        expected=queue(self.cfg)
        self.assertEqual(len(expected),16)
        self.assertEqual(expected[0],("P1",7))
        rejected=[{"prompt_id":p,"seed":s,"qualified":False} for p,s in expected[:8]]
        self.assertEqual(next_candidate(self.cfg,rejected),("P2",7))

    def test_first_qualified_stops_queue(self):
        self.assertIsNone(next_candidate(self.cfg,[{"prompt_id":"P1","seed":7,"qualified":True}]))

    def test_missing_object_not_pure_degradation(self):
        self.assertEqual(interpretations({"objects_alive":False}),"D_object_missing_failure")


if __name__=="__main__":
    unittest.main()
