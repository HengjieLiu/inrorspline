from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/reports'))
import report_oasis_symmetric_svf_phase_b as report


class PhaseBReportTests(unittest.TestCase):
    def test_pareto_keeps_tradeoffs(self):
        rows=[{'run_name':'a','dice':.8,'hd95':2.,'asd':.4,'log_jdet_std_fg':.6,'jdet_le0_pct_fg':0.},
              {'run_name':'b','dice':.79,'hd95':2.1,'asd':.41,'log_jdet_std_fg':.61,'jdet_le0_pct_fg':.001},
              {'run_name':'c','dice':.78,'hd95':2.2,'asd':.42,'log_jdet_std_fg':.4,'jdet_le0_pct_fg':0.}]
        self.assertEqual(report.pareto(rows),['a','c'])

    def test_primary_excludes_diagnostics(self):
        rows=[{'spec':{'swapped':False,'repeat':False}},
              {'spec':{'swapped':True,'repeat':False}},
              {'spec':{'swapped':False,'repeat':True}}]
        self.assertEqual(report.primary_rows(rows),rows[:1])

    def test_incomplete_design_rejected(self):
        with self.assertRaises(RuntimeError):report.validate_design([],{},Path('/tmp'))

    def test_table_reference_and_weights(self):
        row={'arm':'amplitude','alpha':.001,'beta':0,'dice':.8,'hd95':2.,'asd':.4,
             'log_jdet_std_fg':.5,'jdet_le0_pct_fg':.00001}
        text='\n'.join(report.table([row,{**row,'method':'historical'}]))
        self.assertIn('1.000000e-05',text)
        self.assertIn('historical',text)
        self.assertIn('0.001',text)

    def test_complete_markdown_sections_with_fixture(self):
        row={'arm':'base','run_name':'fixture','alpha':0.,'beta':0.,'dice':.8,'hd95':2.,'asd':.4,
             'log_jdet_std_fg':.5,'jdet_le0_pct_fg':0.,'swapped':False,'repeat':False,
             'a1_mm':3.,'a2_mm':3.,'common_mode_rms_mm':2.,'opposing_mode_rms_mm':1.,
             'relative_rms_imbalance':0.,'velocity_cosine':.2,'midpoint_positive1':.3,
             'midpoint_positive2':.3,'optimizer_sec':1.,'worker_sec':2.,'audit_sec':1.,
             'peak_cuda_gib':4.,'peak_reserved_gib':5.,'stage_updates':'1/1/1','stage_best':'1/1/1','gpu':'0'}
        off={'cycle_all':{'mean_mm':.01,'p95_mm':.02},'inbounds_fraction':1.,
             'export_vs_retained':{'mean_mm':.001,'p95_mm':.002}}
        result={'selected_comparison':[row],'overall_winner':'fixture',
                'all_candidates_original_direction':[row],'selected_by_family':{'base':'fixture'},
                'discrepancies':{},'queue':{'elapsed_sec':3.},'pareto_run_names':['fixture']}
        with tempfile.TemporaryDirectory() as tmp:
            report.write_markdown(Path(tmp),result,{'fixture':row},
                                  {'fixture':{'evaluations':[{'offgrid':{'forward':off,'reverse':off}}]}})
            text=(Path(tmp)/'report.md').read_text()
            self.assertIn('Selected within-fit inverse',text)
            self.assertIn('0.010000/0.020000',text)


if __name__=='__main__':unittest.main()
