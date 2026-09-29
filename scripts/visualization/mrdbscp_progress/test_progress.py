"""CPU-only geometry, selection, output-safety and checkpoint contract tests."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEFAULT, FIXED50_CACHED_BATCH, candidate_overrides, checkpoint_steps, presentation
from .geometry import display_plane, reference_grid, sample_plane, slice_grid
from .preview import COLORS, validate_output, warm_rows


class ProgressTests(unittest.TestCase):
    def test_protocol(self):
        self.assertEqual(DEFAULT.features, ("cpbe",))
        self.assertEqual(DEFAULT.full_cps, (8, 4, 2))
        self.assertEqual(DEFAULT.axis, 2)
        self.assertEqual(DEFAULT.slice_index, 96)
        self.assertEqual(len(COLORS), 36)

    def test_fixed50_cached_batch_profile(self):
        value = presentation("fixed50_cached_batch")
        self.assertEqual(value, FIXED50_CACHED_BATCH)
        self.assertEqual(value.features, ("cache", "batch", "cpbe"))
        self.assertEqual(value.max_steps, (50, 50, 50))
        overrides = candidate_overrides(value)
        self.assertEqual(overrides["stage_epochs"], "50,50,50")
        self.assertEqual(overrides["epochs"], 150)
        self.assertEqual(overrides["early_stop_mode"], "none")

    def test_stage_end_and_no_duplicate(self):
        self.assertEqual(checkpoint_steps(25), [0, 10, 20, 25])
        self.assertEqual(checkpoint_steps(20), [0, 10, 20])
        self.assertEqual(checkpoint_steps(0), [0])
        with self.assertRaises(ValueError):
            checkpoint_steps(25, 0)

    def test_identity_all_planes(self):
        shape = (7, 9, 11)
        volume = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        for axis in range(3):
            index = shape[axis] // 2
            actual = sample_plane(volume, slice_grid(shape, axis, index), axis)
            np.testing.assert_allclose(actual, volume.numpy().take(index, axis), atol=.0001, rtol=0)

    def test_through_plane_motion(self):
        shape = (7, 9, 11)
        volume = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        coords = slice_grid(shape, 2, 5)
        coords[..., 0] += 2. / (shape[2] - 1)
        actual = sample_plane(volume, coords, 2)
        np.testing.assert_allclose(actual, volume[:, :, 6].numpy(), atol=.0001, rtol=0)

    def test_in_plane_translation_and_padding(self):
        shape = (7, 9, 11)
        volume = torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
        coords = slice_grid(shape, 2, 5)
        coords[..., 2] += 2. / (shape[0] - 1)
        actual = sample_plane(volume, coords, 2)
        np.testing.assert_allclose(actual[:-1], volume[1:, :, 5].numpy(), atol=.0001, rtol=0)
        np.testing.assert_allclose(actual[-1], 0., atol=.001)

    def test_nearest_labels_no_new_classes(self):
        shape = (7, 9, 11)
        volume = (torch.arange(np.prod(shape)).reshape(shape) % 4).float()
        coords = slice_grid(shape, 2, 4) + .025
        actual = sample_plane(volume, coords, 2, "nearest")
        self.assertTrue(set(np.unique(actual)).issubset({0., 1., 2., 3.}))

    def test_plane_matches_full_3d_warp(self):
        shape = (7, 9, 11)
        generator = torch.Generator().manual_seed(1)
        volume = torch.rand(shape, generator=generator)
        z, y, x = torch.meshgrid(*(torch.linspace(-1, 1, n) for n in shape), indexing="ij")
        displacement = torch.randn((*shape, 3), generator=generator) * .03
        full_grid = torch.stack((x, y, z), dim=-1) + displacement
        full = F.grid_sample(volume[None, None], full_grid[None], align_corners=True)[0, 0]
        plane = slice_grid(shape, 2, 5) + displacement[:, :, 5:6]
        np.testing.assert_array_equal(sample_plane(volume, plane, 2), full[:, :, 5].numpy())

    def test_reference_grid_and_orientation(self):
        grid = reference_grid((7, 9, 11), 2, 3)
        self.assertTrue(torch.equal(grid[:, :, 0], grid[:, :, 10]))
        self.assertEqual(grid[1, 1, 5], 0)
        self.assertEqual(grid[3, 1, 5], 1)
        self.assertEqual(grid[1, 3, 5], 1)
        # Increasing RAS axis0 must move leftward; increasing axis1 moves upward.
        original = np.arange(12).reshape(3, 4)
        shown = display_plane(original)
        self.assertEqual(shown[0, 0], original[-1, 0])
        self.assertEqual(shown[-1, -1], original[0, -1])

    def test_selection_rejects_duplicates_and_missing(self):
        rows = [dict(variant="cpbe", condition="warm", id=f"headline_cpbe_{i}", rank=str(i)) for i in range(10)]
        self.assertEqual([int(r["rank"]) for r in warm_rows(rows[::-1])], list(range(10)))
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                warm_rows(invalid)

    def test_output_is_external_new_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(validate_output(root / "new", root), root / "new")
            for path in (root, root.parent / "outside"):
                with self.assertRaises(ValueError):
                    validate_output(path, root)
            (root / "exists").mkdir()
            with self.assertRaises(FileExistsError):
                validate_output(root / "exists", root)

    def test_playback_real_time_and_frame_count(self):
        from .render import schedule
        snapshots = [dict(stage=s, epoch=e, final=(e == 25), elapsed_sec=s+e/30+.1)
                     for s in range(3) for e in (0, 10, 20, 25)]
        rows = schedule(dict(endpoint_sec=3., snapshots=snapshots))
        self.assertEqual(sum(r["video_frames"] for r in rows), 900)
        self.assertEqual(rows[0]["elapsed_sec"], 0.)
        self.assertEqual(rows[-1]["elapsed_sec"], 3.)
        self.assertTrue(all(r["video_frames"] >= 0 for r in rows))

    def test_completed_columns_stay_frozen(self):
        from .render import stage_indices
        data = dict(snapshots=[dict(stage=0, final=False), dict(stage=0, final=True),
                              dict(stage=1, final=False), dict(stage=1, final=True),
                              dict(stage=2, final=False)])
        self.assertEqual(stage_indices(data, 2), {0: 1, 1: 2})
        self.assertEqual(stage_indices(data, 4), {0: 1, 1: 3, 2: 4})

    def test_nonmonotonic_clock_rejected(self):
        from .render import schedule
        data = dict(endpoint_sec=3., snapshots=[dict(stage=0, final=False, elapsed_sec=2.),
                                               dict(stage=0, final=True, elapsed_sec=1.)])
        with self.assertRaises(ValueError):
            schedule(data)

    def test_loss_global_epoch_does_not_replace_stage_epoch(self):
        from .offline import loss_event_key
        self.assertEqual(loss_event_key(dict(stage_idx=1., epoch=126., global_epoch=126., stage_epoch=1.)),
                         (1, 1, False))


if __name__ == "__main__":
    unittest.main()
