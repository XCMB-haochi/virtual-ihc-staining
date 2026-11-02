from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from cv_protocol import load_protocol, validate_protocol
from run_dapi_fcmae_pretraining import build_command


PROTOCOL_PATH = PROJECT_ROOT / "configs" / "round96_cv2_multimarker_fcmae.json"


class Round96ProtocolTests(unittest.TestCase):
    def test_exposure_matched_multimarker_pretraining_is_preregistered(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        pretraining = protocol["fcmae_pretraining"]
        self.assertEqual(
            pretraining["pretraining_markers"],
            ["DAPI", "CD68", "CD45RO", "HLA-DR", "Vimentin"],
        )
        self.assertEqual(pretraining["data_scope"], "fold-train")
        self.assertEqual(pretraining["epochs"], 160)
        self.assertEqual(
            len(pretraining["pretraining_markers"]) * pretraining["epochs"],
            800,
        )
        self.assertEqual(pretraining["warmup_epochs"], 2)
        self.assertEqual(pretraining["batch_size"], 32)
        self.assertEqual(protocol["training"]["epochs_per_stage"], 14)
        self.assertEqual(
            protocol["recipes"]["multimarkerfcmae160"]["train_stages"],
            [1, 2],
        )
        self.assertTrue(
            protocol["recipes"]["multimarkerfcmae160"]["fcmae_encoder_init"]
        )

    def test_pretraining_command_contains_exact_marker_order(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        command = build_command(
            Path("dataset"),
            Path("runs/round96-test"),
            list(protocol["folds"][0]["val_rois"]),
            protocol["fcmae_pretraining"],
            "cuda:0",
        )
        self.assertEqual(
            command[command.index("--pretraining-markers") + 1],
            "DAPI,CD68,CD45RO,HLA-DR,Vimentin",
        )
        self.assertEqual(command[command.index("--epochs") + 1], "160")

    def test_markers_without_dapi_are_rejected(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        invalid = copy.deepcopy(protocol)
        invalid["fcmae_pretraining"]["pretraining_markers"] = ["CD68"]
        with self.assertRaisesRegex(ValueError, "including DAPI"):
            validate_protocol(invalid)

    def test_fixed_base_endpoint_replacement_and_split_weights(self) -> None:
        protocol = load_protocol(PROTOCOL_PATH)
        self.assertEqual(
            protocol["candidates"]["base_multimarker_replace_context128"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "multimarkerfcmae160": 0.175,
                "officialbasefcmae800": 0.175,
            },
        )
        self.assertEqual(
            protocol["candidates"]["base_multimarker_split_context128"],
            {
                "round35base32": 0.21,
                "round35base48": 0.09,
                "fcmae800": 0.35,
                "officialfcmae800": 0.0875,
                "multimarkerfcmae160": 0.0875,
                "officialbasefcmae800": 0.175,
            },
        )
        self.assertEqual(
            {
                tuple(offset)
                for offset in protocol["candidate_output_source_offsets"].values()
            },
            {(0, -1)},
        )


if __name__ == "__main__":
    unittest.main()
