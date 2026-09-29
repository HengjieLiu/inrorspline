"""Regression checks for the reported grid defects and loss replay cadence."""
import unittest
import matplotlib.pyplot as plt
import numpy as np
from .render import schedule
from .render_revision import coordinate_planes, draw_grid, playback_frames


class RevisionTests(unittest.TestCase):
    def test_endpoint_caption_keeps_field_and_file_times_separate(self):
        from .render import frame
        snapshots = [dict(stage=s, epoch=1, global_step=s+1, final=True,
                          elapsed_sec=s+1., dice=.8) for s in range(3)]
        history = [dict(stage_idx=s, global_epoch=s+1, data_loss=-.9,
                        weighted_be=.01, loss=-.89) for s in range(3)]
        data = dict(snapshots=snapshots, history=history, steps=[1,1,1],
                    fixed_id="fixed", moving_id="moving")
        static = {k: np.zeros((16,16)) for k in ("moving", "fixed", "moving_seg", "fixed_seg")}
        planes = dict(warped=static["moving"], warped_seg=static["moving_seg"],
                      **coordinate_planes(np.zeros((16,16,3))))
        state = dict(snapshot_index=2, initial=False, output_ready=True, elapsed_sec=3.2)
        fig = frame(data, static, [planes]*3, state, None, grid_drawer=draw_grid, keep_figure=True)
        caption = fig.texts[1].get_text()
        self.assertIn("Stage 3/3 · Step 1/1 @ 3.00 s", caption)
        self.assertIn("DVF file ready", caption)
        self.assertNotIn("DVF file ready @ 3.00", caption)
        self.assertIn("3.20 s", fig._elapsed_text.get_text())
        plt.close(fig)

    def test_identity_coordinate_spacing_and_display_direction(self):
        planes = coordinate_planes(np.zeros((160,224,3), dtype=np.float32))
        np.testing.assert_array_equal(planes["phi_z"][0], np.arange(159,-1,-1))
        np.testing.assert_array_equal(planes["phi_y"][:,0], np.arange(224))
        np.testing.assert_array_equal(np.diff(np.where(planes["phi_z"][0] % 8 == 0)[0]), 8)

    def test_normal_motion_cannot_fade_geometric_grid(self):
        field = np.zeros((160,224,3), dtype=np.float32)
        identity = coordinate_planes(field)
        field[...,0] = 10.
        translated = coordinate_planes(field)
        for key in identity:
            np.testing.assert_array_equal(identity[key], translated[key])

    def test_pull_translation_sign_and_units(self):
        field = np.zeros((160,224,3), dtype=np.float32)
        original = coordinate_planes(field)
        field[...,2] = 2 * 3.5 / 159
        field[...,1] = -2 * 2.25 / 223
        moved = coordinate_planes(field)
        np.testing.assert_allclose(moved["phi_z"] - original["phi_z"], 3.5, atol=1e-6)
        np.testing.assert_allclose(moved["phi_y"] - original["phi_y"], -2.25, atol=1e-6)

    def test_no_missing_lines_at_actual_panel_resolution(self):
        fig = plt.figure(figsize=(1.30,1.82), dpi=100)
        ax = fig.add_axes((0,0,1,1))
        draw_grid(ax, coordinate_planes(np.zeros((160,224,3),dtype=np.float32)))
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[...,:3]
        for x in range(7,160,8):
            center = (x+.5)*130/160
            region = rgb[178:180, max(0,int(center)-1):min(130,int(center)+2)]
            self.assertGreater(region.max(), 190, f"Missing vertical line x={x}")
        for y in range(0,224,8):
            center = 182-(y+.5)*182/224
            region = rgb[max(0,int(center)-1):min(182,int(center)+2), 2:4]
            self.assertGreater(region.max(), 190, f"Missing horizontal line y={y}")
        plt.close(fig)

    def test_loss_advances_independently_and_holds_are_real(self):
        snapshots = [dict(stage=s, epoch=e, final=e==25, elapsed_sec=s+e/30+.1)
                     for s in range(3) for e in (0,10,20,25)]
        history = [dict(stage_idx=s, global_epoch=s*25+e, elapsed_sec=s+e/30+.1)
                   for s in range(3) for e in range(1,26)]
        data = dict(endpoint_sec=3., snapshots=snapshots, history=history)
        frames = playback_frames(data, schedule(data))
        self.assertEqual(len(frames),900)
        self.assertEqual(frames[-1]["loss_step"],75)
        self.assertEqual(frames[-1]["elapsed_sec"],3.)
        self.assertTrue(all(f["elapsed_sec"]==0 for f in frames[:30]))
        self.assertTrue(any(b["loss_step"]>a["loss_step"] and a["snapshot_index"]==b["snapshot_index"]
                            for a,b in zip(frames,frames[1:])))
        for f in frames:
            reached = [h for h in history if h["elapsed_sec"]<=f["elapsed_sec"]]
            self.assertEqual(f["loss_rows_revealed"],len(reached))
        for a,b in zip(frames,frames[1:]):
            self.assertGreaterEqual(b["elapsed_sec"],a["elapsed_sec"])
            if a["holding"] and b["holding"] and a["state_index"]==b["state_index"]:
                self.assertEqual(a["elapsed_sec"],b["elapsed_sec"])


if __name__ == "__main__":
    unittest.main()
