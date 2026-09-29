"""CPU regression tests for experiment controls; run with python -m unittest."""
import unittest
from unittest.mock import patch
import torch
from scripts.experiments.oasis_alpha_sweep import run_one
from .backends import Backend, LossBuffer, c
from .run_suite import Monitor, Worker, dump


class Contracts(unittest.TestCase):
    def test_command_timeout_is_host_only_and_default_preserved(self):
        import inspect
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        self.assertEqual(inspect.signature(Worker.execute).parameters["command_timeout"].default,3600)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = object.__new__(Worker)
            worker.mail,worker.index,worker.name = root/"mail",0,"test"
            worker.suite = SimpleNamespace(root=root,container=str,attempts=[],
                monitor=SimpleNamespace(sample=lambda:None,interval=lambda a,b:dict(clean=True)))
            observed = []
            def ready(path,timeout):
                observed.append(timeout)
                dump(path,dict(ok=True,result={}))
            worker.wait_file = ready
            result = worker.execute("evaluate","timeout_check",command_timeout=14400,items=[])
            self.assertEqual(observed,[14400])
            self.assertNotIn("command_timeout",result["command"])
            self.assertEqual(json.loads((worker.mail/"command_0000.json").read_text())["items"],[])

    def test_same_common_module(self):
        self.assertIs(c, run_one.c)
        original = c.conv1d
        with Backend(("batch",)).activate():
            self.assertEqual(run_one.c.conv1d.__name__, "batched_conv")
        self.assertIs(c.conv1d, original)

    def test_default_cli_preserved(self):
        with patch("sys.argv", ["run_one.py", "--method", run_one.METHODS[6], "--alpha", "1000"]):
            args = run_one.parse_args()
        self.assertEqual(args.loss_history_mode, "cpu")
        self.assertEqual(args.timing_detail, "off")
        self.assertTrue(args.cudnn_benchmark)
        self.assertFalse(args.cudnn_deterministic)

    def test_exact_nested_stage_shapes(self):
        shapes = c.build_nested_cp_shapes(c.IMAGE_SHAPE,base_spacing=8,spacings=(8,4,2))
        self.assertEqual([shapes[s] for s in (8,4,2)],[(24,32,28),(44,60,52),(84,116,100)])
        for loss_shape, spacing in zip(((40,56,48),(80,112,96),(160,224,192)),(8,4,2)):
            self.assertEqual(tuple(n//2+4 for n in loss_shape),shapes[spacing])

    def test_endpoints_no_nested_synchronizations(self):
        timer = run_one.TimingRecorder("endpoints")
        with patch.object(timer, "sync") as sync:
            self.assertIsNone(timer.start("stage_0"))
            sync.assert_not_called()
            token = timer.start("setup.device")
            self.assertIsNotNone(token)
            timer.stop(token)
            timer.mark("case_load_complete")
            timer.mark("dvf_file_ready", sync=False)
            self.assertEqual(sync.call_count, 3)

    def test_original_early_stop_checkpoints(self):
        for minimum, patience, expected in ((40,20,50),(80,35,100),(130,50,150)):
            first = None
            for epoch in range(1,200):
                rows = [dict(epoch=i+1,loss=-1.,data_loss=-1.) for i in range(epoch)]
                stop,_ = run_one.stage_stop_decision(rows,min_epochs=minimum,patience=patience,
                                                     check_interval=25,rel_delta_total=1e-4,rel_delta_data=5e-5)
                if stop:
                    first = epoch
                    break
            self.assertEqual(first,expected)

    def test_loss_buffer_exact_rows(self):
        torch.manual_seed(3)
        values = torch.randn(100,3)
        buffer = LossBuffer(100,"cpu")
        rows = []
        for i, triple in enumerate(values):
            rows.append(dict(epoch=i+1))
            buffer.append(i,*triple)
            if i % 25 == 0:
                buffer.flush(rows)
        buffer.flush(rows)
        self.assertEqual([[r[k] for k in ("loss","data_loss","reg_loss")] for r in rows],values.tolist())

    def test_monitor_fail_closed(self):
        import threading
        monitor = object.__new__(Monitor)
        monitor.lock = threading.Lock()
        monitor.records = []
        self.assertFalse(monitor.interval(0,2)["clean"])
        monitor.records = [dict(unix=1,foreign=[],error=None)]
        self.assertTrue(monitor.interval(0,2)["clean"])
        monitor.records[0]["foreign"] = [{"pid":999}]
        self.assertFalse(monitor.interval(0,2)["clean"])
        monitor.records[0].update(foreign=[],error="query failed")
        self.assertFalse(monitor.interval(0,2)["clean"])

    def test_coordinate_and_crop_mutations_detectable(self):
        cs, shape = (10,12,14),(12,16,20)
        values = torch.randn((*cs,3))*.01
        base = c.expand_bspline_controls(values,cs,shape,2)
        wrong_axis = c.expand_bspline_controls(values.flip(-1),cs,shape,2)
        self.assertGreater(float((base-wrong_axis).abs().max()),1e-4)
        self.assertGreater(float((base[:,1:]-base[:,:-1]).abs().max()),1e-4)
        with Backend(("cache",)).activate() as backend:
            k0 = c.cubic_bspline1d(2,0)
            k1 = c.cubic_bspline1d(2,1)
            self.assertIs(k0,c.cubic_bspline1d(2,0))
            self.assertIsNot(k0,k1)

    def test_mixed_derivative_factor_mutation_detectable(self):
        from . import backends
        torch.manual_seed(91)
        control = torch.randn((10,12,14))[:, :, :, None].expand(-1,-1,-1,3).contiguous()*.01
        good = Backend(()).be(control,(12,16,20),2)
        wrong = tuple((orders,1) for orders,_ in backends.TERMS)
        with patch.object(backends,"TERMS",wrong):
            bad = Backend(()).be(control,(12,16,20),2)
        self.assertGreater(float(abs(good-bad)),1e-4*float(good))


if __name__ == "__main__":
    unittest.main()
