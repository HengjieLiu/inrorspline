"""Contracts for shifted display losses and presentation typography."""
import unittest
import matplotlib.pyplot as plt
import numpy as np

from .render_compact import compact_frame, display_losses


class CompactRenderTests(unittest.TestCase):
    def test_display_shift_does_not_change_weighted_objective_relation(self):
        values=display_losses(dict(data_loss=-.9,weighted_be=.03,loss=-.87))
        self.assertAlmostEqual(values["similarity_display"],.1)
        self.assertAlmostEqual(values["total_display"],.13)
        self.assertAlmostEqual(values["total_display"],values["similarity_display"]+values["weighted_be"])

    def test_single_loss_axis_and_large_title(self):
        snapshots=[dict(stage=s,epoch=1,global_step=s+1,final=True,elapsed_sec=s+1.,dice=.8) for s in range(3)]
        history=[dict(stage_idx=s,global_epoch=s+1,data_loss=-.9,weighted_be=.01,loss=-.89) for s in range(3)]
        data=dict(snapshots=snapshots,history=history,steps=[1,1,1],fixed_id="f",moving_id="m",
                  presentation=dict(full_cps=[8,4,2],downsampling=[4,2,1]))
        static={k:np.zeros((32,24)) for k in ("moving","fixed","moving_seg","fixed_seg")}
        plane=dict(warped=np.zeros((32,24)),warped_seg=np.zeros((32,24)),
                   phi_z=np.indices((32,24))[0],phi_y=np.indices((32,24))[1])
        state=dict(snapshot_index=2,initial=False,output_ready=True,elapsed_sec=3.,loss_step=3,
                   status="Complete · DVF file ready",status_detail="")
        fig=compact_frame(data,static,[plane]*3,state,keep_figure=True)
        self.assertEqual(len(fig._loss_artists),3)
        self.assertGreaterEqual(fig.texts[0].get_fontsize(),30)
        fig.canvas.draw()
        self.assertFalse(fig.texts[0].get_window_extent().overlaps(
            fig._elapsed_text.get_window_extent()))
        self.assertTrue(any(text.get_text()=="Optimization progress" for text in fig.texts))
        self.assertEqual(fig._phase_text.get_text(), "Complete · DVF file ready")
        self.assertFalse(fig._loss_value_text.get_window_extent().overlaps(
            fig.axes[-1].get_window_extent()))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
