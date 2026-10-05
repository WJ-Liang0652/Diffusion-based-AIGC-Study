"""Critical decision rules, independent of GPU generation and attention proxies."""
import copy
import json
from pathlib import Path
import sys
import unittest

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK/"src"))
from stage02_relations import evaluate_annotation, relation


class AnnotationRules(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((TASK/"configs/stage02.json").read_text())
        self.annotation = {"objects": {
            "A": {"bbox_pixels": [100,500,300,700], "present": True, "unique": True, "identity_clear": True, "truncated": False},
            "B": {"bbox_pixels": [600,300,900,700], "present": True, "unique": True, "identity_clear": True, "truncated": False}}}

    def test_threshold_boundaries(self):
        self.assertEqual(relation(0.04,self.cfg), "clear_failure")
        self.assertEqual(relation(0.05,self.cfg), "uncertain")
        self.assertEqual(relation(0.06,self.cfg), "clear_success")

    def test_qualification_requires_new_target_unmet(self):
        result = evaluate_annotation(copy.deepcopy(self.annotation), self.cfg)
        self.assertTrue(result["baseline_qualified"])
        self.annotation["objects"]["A"]["bbox_pixels"] = [100,100,300,300]
        result = evaluate_annotation(copy.deepcopy(self.annotation),self.cfg)
        self.assertTrue(result["joint_success"])
        self.assertFalse(result["baseline_qualified"])

    def test_missing_object_is_failure(self):
        self.annotation["objects"]["A"].update(present=False,bbox_pixels=None)
        result=evaluate_annotation(copy.deepcopy(self.annotation),self.cfg)
        self.assertEqual(result["new_relation"],"failure_object_missing")
        self.assertFalse(result["joint_success"])
        self.assertFalse(result["baseline_qualified"])

    def test_multiple_instances_are_uncertain(self):
        self.annotation["objects"]["A"]["unique"] = False
        result=evaluate_annotation(copy.deepcopy(self.annotation),self.cfg)
        self.assertEqual(result["old_relation"],"uncertain")
        self.assertIsNotNone(result["dx"])
        self.assertFalse(result["baseline_qualified"])


if __name__ == "__main__":
    unittest.main()
