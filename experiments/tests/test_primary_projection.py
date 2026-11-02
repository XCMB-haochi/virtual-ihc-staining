from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]
from ihc_gradient_projection import PrimaryGradientProjection
from ihc_torch import CompetitionLoss
from train_unet import training_batch_losses
from run_training_screen import compute_busy_gpus, eligible_gpus
from cv_protocol import load_protocol, validate_protocol
from run_cv_training import build_train_command
import train_unet


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def project_linear(self, primary, auxiliary, scale=1.):
        p = torch.nn.Parameter(torch.zeros(len(primary)))
        proj = PrimaryGradientProjection([p])
        lp = (p * torch.tensor(primary)).sum() * scale
        la = (p * torch.tensor(auxiliary)).sum() * scale
        proj.accumulate(lp)
        (lp + la).backward()
        stats = proj.project()
        return p.grad / scale, stats

    def test_conflict_removed_but_orthogonal_auxiliary_component_preserved(self):
        for scale in (1., 65536.):
            value, stats = self.project_linear([2., 0.], [-1., 3.], scale)
            torch.testing.assert_close(value, torch.tensor([2., 3.]))
            self.assertEqual(stats["coefficient"], .5)

    def test_aligned_and_zero_primary_are_unchanged(self):
        value, stats = self.project_linear([2., 0.], [1., 3.])
        self.assertFalse(stats["conflict"])
        torch.testing.assert_close(value, torch.tensor([3., 3.]))
        value, _ = self.project_linear([0., 0.], [1., 3.])
        torch.testing.assert_close(value, torch.tensor([1., 3.]))

    def test_projection_occurs_after_accumulation_including_short_group(self):
        # Opposing microbatch gradients cancel before projection. Projecting
        # each microbatch would incorrectly increase the total primary update.
        p = torch.nn.Parameter(torch.zeros(2))
        proj = PrimaryGradientProjection([p])
        for weight, gp, ga in ((2/3, [1., 0.], [-1., 0.]), (1/3, [1., 0.], [3., 0.])):
            lp = (p * torch.tensor(gp)).sum() * weight
            la = (p * torch.tensor(ga)).sum() * weight
            proj.accumulate(lp)
            (lp + la).backward()
        self.assertFalse(proj.project()["conflict"])
        torch.testing.assert_close(p.grad, torch.tensor([4/3, 0.]))
        proj.reset()
        with self.assertRaises(RuntimeError):
            proj.project()

    def test_nonfinite_primary_is_visible_to_scaler(self):
        p = torch.nn.Parameter(torch.ones(2))
        proj = PrimaryGradientProjection([p])
        proj.primary = [torch.tensor([float("inf"), 1.])]
        p.grad = torch.ones(2)
        self.assertFalse(proj.project()["finite"])
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_training_loss_keeps_baseline_value_and_primary_gradient(self):
        model = torch.nn.Conv2d(1, 4, 1)
        class Wrapper(torch.nn.Module):
            def forward(self, x, condition=None):
                return model(x).sigmoid()
        x, y = torch.rand(2, 1, 16, 16), torch.rand(2, 4, 16, 16)
        criterion = CompetitionLoss(ssim_mode="uniform7", channel_weights=[.85, .05, .05, .05])
        primary = CompetitionLoss(ssim_mode="uniform7")
        output = []
        baseline = training_batch_losses(Wrapper(), x, y, None, criterion, 0)[0]
        total = training_batch_losses(Wrapper(), x, y, None, criterion, 0,
            primary_criterion=primary, primary_loss_output=output)[0]
        torch.testing.assert_close(total, baseline, rtol=0, atol=0)
        reference = .85 * primary(model(x).sigmoid()[:, :1], y[:, :1])
        torch.testing.assert_close(torch.autograd.grad(output[0], model.weight, retain_graph=True)[0],
                                   torch.autograd.grad(reference, model.weight)[0])
        proj = PrimaryGradientProjection(model.parameters())
        proj.accumulate(output[0])
        total.backward()
        self.assertTrue(proj.project()["finite"])

    def test_real_tiny_model_supports_two_accumulated_backward_paths(self):
        from ihc_unext2 import UNeXt2VirtualStainer
        from ihc_spatial import SPATIAL_THUMBNAIL_CONDITION_DIMENSION
        model = UNeXt2VirtualStainer(4, "spatial-thumbnail-film")
        projector = PrimaryGradientProjection(model.parameters())
        criterion = CompetitionLoss(ssim_mode="uniform7", channel_weights=[.85, .05, .05, .05])
        primary = CompetitionLoss(ssim_mode="uniform7")
        for _ in range(2):
            losses = []
            loss = training_batch_losses(model, torch.rand(1, 1, 32, 32), torch.rand(1, 4, 32, 32),
                torch.zeros(1, SPATIAL_THUMBNAIL_CONDITION_DIMENSION), criterion, 0,
                primary_criterion=primary, primary_loss_output=losses)[0]
            projector.accumulate(losses[0] * .5)
            (loss * .5).backward()
        self.assertTrue(projector.project()["finite"])
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_protocol_cli_and_commands_accept_only_the_registered_recipe(self):
        import copy
        p = load_protocol(ROOT / "configs/round118_cv2_primary_projection.json")
        recipe = p["recipes"]["primaryprojection"]
        command = build_train_command("python", Path("dataset"), Path("unused"),
            p["folds"][0]["val_rois"], "CD68", recipe, p["training"], 1, None, "cpu")
        self.assertIn("--primary-gradient-projection", command)
        with mock.patch.object(sys, "argv", command[1:]), mock.patch.object(train_unet, "prepare_output_dir"), \
             mock.patch.object(train_unet, "resolve_data_root", side_effect=RuntimeError("dataset boundary reached")):
            with self.assertRaisesRegex(RuntimeError, "dataset boundary reached"):
                train_unet.main()
        for field, value in (("d4_consistency_weight", .1), ("primary_gradient_projection", "true"),
                             ("marker_loss_weights", [1., 0., 0., 0.])):
            invalid = copy.deepcopy(p)
            invalid["recipes"]["primaryprojection"][field] = value
            with self.assertRaises(ValueError):
                validate_protocol(invalid)


class EmptyGpuTests(unittest.TestCase):
    @mock.patch("run_training_screen.subprocess.check_output")
    def test_busy_gpu0_does_not_block_idle_gpu1(self, query):
        query.side_effect = ["0, GPU-zero\n1, GPU-one\n", "GPU-zero\n"]
        self.assertEqual(compute_busy_gpus(), {0})

    @mock.patch("run_training_screen.time.sleep")
    @mock.patch("run_training_screen.compute_busy_gpus", return_value={1})
    @mock.patch("run_training_screen.subprocess.check_output", return_value="0, 0, 40427\n1, 0, 32000")
    def test_idle_utilization_does_not_steal_an_occupied_gpu(self, query, busy, sleep):
        self.assertEqual(eligible_gpus(24000, candidates=[1], require_empty=True), [])

    @mock.patch("run_training_screen.subprocess.check_output")
    def test_unknown_process_mapping_fails_closed(self, query):
        query.side_effect = ["0, GPU-zero\n1, GPU-one\n", "MIG-unknown\n"]
        self.assertEqual(compute_busy_gpus(), {0, 1})


if __name__ == "__main__":
    unittest.main()
