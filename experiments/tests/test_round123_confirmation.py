from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from ihc_feature_guidance import build_feature_guidance, relational_distance_loss
from train_unet import validate_feature_guidance_options
from cv_protocol import load_protocol, validate_protocol
from run_round123_confirmation import materialize, validate_frozen_recipe
from evaluate_distribution_gate import evaluate_gates
from test_feature_guidance import ToyTeacher


class Round123Tests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(ROOT / "configs/round123_fold1_relational_confirmation.json")
        self.original = load_protocol(ROOT / "configs/round122_cv2_encoder_relational_robust.json")
        self.plan = json.loads((ROOT / "configs/round123_teacher.json").read_text())
        self.original_pretraining = json.loads((ROOT / "configs/round96_cv2_multimarker_fcmae.json").read_text())["fcmae_pretraining"]

    def test_only_fold_and_teacher_binding_change(self):
        validate_frozen_recipe(self.protocol, self.original, self.plan, self.original_pretraining)
        changed = copy.deepcopy(self.protocol)
        changed["recipes"]["encoderrelational15"]["feature_guidance"]["weight"] = 0.1
        with self.assertRaisesRegex(ValueError, "trained recipe"):
            validate_frozen_recipe(changed, self.original, self.plan, self.original_pretraining)
        changed_plan = copy.deepcopy(self.plan)
        changed_plan["fold"]["val_rois"] = self.original["expected_rois"][:5]
        with self.assertRaisesRegex(ValueError, "Teacher changed"):
            validate_frozen_recipe(self.protocol, self.original, changed_plan, self.original_pretraining)

    def test_materialization_only_resolves_teacher_artifact(self):
        resolved = materialize(self.protocol, "/workspace/teacher.pt", "a" * 64)
        resolved["recipes"]["encoderrelational15"]["feature_guidance"]["teachers"] = copy.deepcopy(
            self.protocol["recipes"]["encoderrelational15"]["feature_guidance"]["teachers"])
        self.assertEqual(resolved, self.protocol)

    def test_confirmation_pass_is_not_submission_permission(self):
        p = self.protocol
        values = {c: {v: {"competition_score_official": score} for v in ("float", "jpeg_q95")}
                  for c, score in ((p["baseline_candidate"], 75), (p["primary_candidate"], 75.2))}
        report = {"protocol_name": p["name"], "marker": "CD68", "validation_rois": sorted(p["expected_rois"]),
                  "overall": values, "fold_metrics": {"fold1": values},
                  "roi_metrics": {c: {"jpeg_q95": {r: v["jpeg_q95"] for r in p["expected_rois"]}}
                                  for c, v in values.items()}}
        result = evaluate_gates(p, report)
        self.assertTrue(result["confirmation"]["passed"])
        self.assertFalse(result["promotion"]["allowed"])

    def test_robust_mode_rejects_microbatch_relations(self):
        p = copy.deepcopy(self.protocol)
        p["recipes"]["encoderrelational15"]["gradient_accumulation_steps"] = 2
        with self.assertRaises(ValueError):
            validate_protocol(p)
        args = SimpleNamespace(feature_guidance_mode="encoder-relational-robust", feature_guidance_weight=0.05,
                               feature_guidance_stage=4, feature_guidance_batch_samples=16, gradient_accumulation_steps=2)
        with self.assertRaisesRegex(ValueError, "one full batch"):
            validate_feature_guidance_options(args, ("CD68",))

    def test_perceptual_factory_remains_usable(self):
        loss = build_feature_guidance(ToyTeacher(), "perceptual", stage=4, crop_size=16, batch_samples=3)
        prediction = torch.rand(3, 1, 16, 16, requires_grad=True)
        loss(prediction, torch.rand_like(prediction)).backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_median_objective_and_gradient_unchanged(self):
        torch.manual_seed(4)
        s = torch.randn(16, 8, requires_grad=True)
        t = torch.randn(16, 8)
        distances = torch.pdist(s.float())
        reference = torch.pdist(t.float())
        expected = torch.nn.functional.smooth_l1_loss(
            distances / distances.median().clamp_min(1e-6), reference / reference.median().clamp_min(1e-6))
        actual = relational_distance_loss(s, t, robust_scale=True)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, s)[0], torch.autograd.grad(expected, s)[0], rtol=0, atol=0)
        module = build_feature_guidance(ToyTeacher(), "encoder-relational-robust", stage=4, crop_size=16, batch_samples=16)
        self.assertTrue(module.robust_scale)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
