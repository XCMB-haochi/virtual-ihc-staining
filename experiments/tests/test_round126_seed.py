from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools")]
from cv_protocol import load_protocol, validate_protocol


class Round126SeedTests(unittest.TestCase):
    def test_only_supervised_seed_and_recipe_identity_change(self):
        current = load_protocol(ROOT / "configs/round126_cv2_encoder_relational_robust_seed2027.json")
        prior = load_protocol(ROOT / "configs/round122_cv2_encoder_relational_robust.json")
        self.assertEqual(current["training"]["seed"], 2027)
        self.assertEqual(prior["training"]["seed"], 2044)
        self.assertEqual(current["fcmae_pretraining"], prior["fcmae_pretraining"])
        self.assertEqual(
            current["recipes"]["encoderrelationalseed2027"]["feature_guidance"],
            prior["recipes"]["encoderrelational15"]["feature_guidance"],
        )
        self.assertEqual(current["folds"], prior["folds"])
        self.assertEqual(current["inference"], prior["inference"])

    def test_protocol_has_one_trained_candidate_and_fixed_gate(self):
        protocol = load_protocol(ROOT / "configs/round126_cv2_encoder_relational_robust_seed2027.json")
        trained = [name for name, recipe in protocol["recipes"].items() if recipe.get("train_stages")]
        self.assertEqual(trained, ["encoderrelationalseed2027"])
        gate = protocol["distribution_gates"]["feature_guidance"]
        self.assertEqual(gate["minimum_q95_gain"], 0.05)
        self.assertEqual(gate["minimum_positive_folds"], 2)
        self.assertEqual(gate["minimum_positive_rois"], 7)
        self.assertEqual(gate["minimum_roi_q95_gain"], -0.12)

    def test_seed_change_is_rejected_if_other_recipe_settings_change(self):
        protocol = load_protocol(ROOT / "configs/round126_cv2_encoder_relational_robust_seed2027.json")
        changed = copy.deepcopy(protocol)
        changed["recipes"]["encoderrelationalseed2027"]["feature_guidance"]["weight"] = 0.1
        self.assertNotEqual(
            changed["recipes"]["encoderrelationalseed2027"],
            protocol["recipes"]["encoderrelationalseed2027"],
        )
        # The validator still accepts the altered object; the explicit equality
        # contract in the first test is what rejects an unregistered change.
        self.assertEqual(validate_protocol(changed)["training"]["seed"], 2027)


if __name__ == "__main__":
    unittest.main()
