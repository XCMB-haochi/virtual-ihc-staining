from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from ihc_spatial import (
    apply_complete_roi_conditioning, apply_input_conditioning,
    spatial_thumbnail_conditioning_specification,
)
from ihc_torch import load_context_gray, PairedPatchDataset, save_grayscale_rgb, shift_array_edge, shift_tensor_edge
from search_postprocessing import evaluate, jpeg_roundtrip
from train_unet import validate


class Identity(torch.nn.Module):
    def forward(self, source, condition=None):
        return source


class ReassessmentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_subsampling_predictions_preserves_full_roi_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = []
            for row in range(2):
                for col in range(3):
                    name = f"ROI000_{row:02d}_{col:02d}.jpg"
                    names.append(name)
                    Image.fromarray(np.full((256, 256), 10 + row * 70 + col * 20, np.uint8)).save(root / name)
            spec = spatial_thumbnail_conditioning_specification()
            all_conditions = apply_input_conditioning(root, names, spec)
            query = [names[0], names[-1]]
            old = apply_input_conditioning(root, query, spec)
            corrected = apply_complete_roi_conditioning(root, query, spec)
            self.assertFalse(np.array_equal(old[query[0]], all_conditions[query[0]]))
            self.assertEqual(set(corrected), set(query))
            for name in query:
                np.testing.assert_array_equal(corrected[name], all_conditions[name])
            with self.assertRaisesRegex(ValueError, "missing"):
                apply_complete_roi_conditioning(root, ["ROI000_99_99.jpg"], spec)

    def test_validator_scores_each_image_before_psnr_clipping(self):
        # One perfect image must not saturate the PSNR contribution of the other.
        prediction = torch.stack([torch.zeros(1, 32, 32), torch.full((1, 32, 32), .5)])
        target = torch.zeros_like(prediction)
        with np.errstate(divide="ignore"):
            actual = validate(Identity(), [(prediction, target, ["a", "b"])],
                              torch.device("cpu"), ("CD68",), 0)
            expected = evaluate(prediction[:, 0].numpy() * 255, target[:, 0].numpy() * 255, 0)
        self.assertAlmostEqual(actual["mean_competition_score"], expected["competition_score_official"], places=7)
        self.assertLess(actual["mean_competition_score"], 60)

    def test_context_shift_and_jpeg_paths_match_for_regular_and_missing_neighbors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in (root / "train/DAPI", root / "train/CD68"):
                directory.mkdir(parents=True)
                for row, col in ((0, 0), (0, 1), (1, 0)):
                    rng = np.random.default_rng(row * 3 + col)
                    Image.fromarray(rng.integers(0, 256, (256, 256), dtype=np.uint8)).save(directory / f"ROI000_{row:02d}_{col:02d}.jpg")
            name = "ROI000_00_00.jpg"
            for halo in (0, 64, 128):
                source, _, _ = PairedPatchDataset(root, [name], markers=("CD68",), context_halo=halo)[0]
                inference = torch.from_numpy(load_context_gray(root / "train/DAPI", name, halo)[None] / 255).float()
                self.assertTrue(torch.equal(source, inference))
            array = np.random.default_rng(4).random((256, 256), dtype=np.float32)
            np.testing.assert_array_equal(shift_array_edge(array, (0, -1)), shift_tensor_edge(torch.from_numpy(array)[None, None], (0, -1))[0, 0].numpy())
            output = root / "saved.jpg"
            save_grayscale_rgb(array, output, quality=95)
            with Image.open(output) as im:
                np.testing.assert_array_equal(np.asarray(im.convert("L")), jpeg_roundtrip(array * 255, 95))


if __name__ == "__main__":
    unittest.main()
