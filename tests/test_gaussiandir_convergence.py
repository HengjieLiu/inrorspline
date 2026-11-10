"""Convergence/scheduling/resource contracts for the approved sampled-loss study."""
import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.experiments.gaussiandir.convergence import MonitoringBanks, Plateau, out_of_time, schedule
from scripts.experiments.gaussiandir.model import GaussianModel, adaptation_due, paper_ncc
from scripts.experiments.gaussiandir.resources import Resources
from scripts.experiments.gaussiandir.run import training_step

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    folder = ROOT/"experiments/2026-11-06_oasis_rank000_gaussiandir"
    return (json.loads((folder/"protocol_v1.json").read_text()),
            json.loads((folder/"protocol_convergence_v1.json").read_text())["refinement"])


def test_short_schedule_events_and_exact_anchor():
    base, _ = configuration()
    for budget, warmup in ((200, 5), (500, 13), (2000, 50)):
        cfg = schedule(base, budget)
        assert cfg["warmup_steps"] == warmup
        assert cfg["paper_reconstruction"]["stage_steps"] == [budget//2]*2
        for variant, maximum in (("released_control", budget), ("paper_reconstruction", budget//2)):
            actual = [i for i in range(1, maximum+1) if adaptation_due(variant, i, maximum, cfg)]
            expected = ([round(budget*x/100) for x in range(15, 61, 5)] if variant == "released_control"
                        else [round(maximum*x/10) for x in range(1, 10)])
            assert actual == expected
        if budget == 2000:
            assert cfg == base


def test_flat_loss_requires_two_reductions_and_another_window():
    _, settings = configuration()
    controller = Plateau(settings)
    probe = torch.zeros(8, 3)
    for step in range(0, 1501, 100):
        decision = controller.observe(step, [-.7, -.69], probe)
        assert decision["plateau"] == (step == 1500)
    assert controller.reductions == [500, 1000]
    assert controller.multiplier(2000) == pytest.approx(.025)
    restored = Plateau.restore(copy.deepcopy(controller.state()))
    assert restored.observe(1600, [-.7, -.69], probe)["plateau"]


def test_improving_or_moving_or_secondary_changing_is_unresolved():
    _, settings = configuration()
    for kind in ("loss", "field", "secondary"):
        controller = Plateau(settings)
        for step in range(0, 5100, 100):
            scores = [-.7-step*1e-5, -.69] if kind == "loss" else [-.7, -.69-step*1e-5 if kind == "secondary" else -.69]
            probe = torch.ones(8, 3)*step*.0001 if kind == "field" else torch.zeros(8, 3)
            decision = controller.observe(step, scores, probe)
            assert not decision["plateau"]
        if kind == "loss":
            assert not controller.reductions
        else:
            assert len(controller.reductions) == 4
    assert out_of_time(1190, 9, 1, 1200)
    assert not out_of_time(1189, 9, 1, 1200)


def test_monitor_banks_do_not_change_rng_or_adaptation_statistics():
    base, settings = configuration()
    base["knn_k"] = 3
    base["query_batch"] = 40
    settings.update(monitor_batches=2, probe_points=16)
    mask = torch.ones(24, 24, 24, dtype=torch.bool)
    model = GaussianModel("paper_reconstruction", base, mask, "cpu", backend="brute_test")
    rng = torch.get_rng_state().clone()
    stats = model.active.contribution.clone()
    banks = MonitoringBanks(mask, base, settings)
    image = torch.rand(mask.shape, generator=torch.Generator().manual_seed(2))
    inputs = dict(fixed=image, moving=image, affine=torch.eye(4))
    a, probe, _ = banks.evaluate(model, inputs, paper_ncc)
    b, probe2, _ = banks.evaluate(model, inputs, paper_ncc)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(model.active.contribution, stats, rtol=0, atol=0)
    torch.testing.assert_close(probe, probe2, rtol=0, atol=0)
    assert a == b
    assert not torch.equal(banks.banks[0][0], banks.banks[1][0])


def test_joint_continuation_frozen_populations_and_field_preservation():
    base, _ = configuration()
    base["knn_k"] = 3
    base["paper_reconstruction"]["sparsifications"] = [8, 4]
    mask = torch.ones(24, 24, 24, dtype=torch.bool)
    model = GaussianModel("paper_reconstruction", base, mask, "cpu", backend="brute_test")
    with torch.no_grad():
        model.active.translation[:] = .01
    queries = torch.rand(20, 3)*1.5-.75
    before = model(queries).detach().clone()
    model.add_stage(mask, "cpu")
    optimizer = model.optimizer(joint=True)
    torch.testing.assert_close(model(queries), before, rtol=0, atol=0)
    assert all(p.requires_grad for p in model.parameters())
    image = torch.rand(mask.shape)
    inputs = dict(fixed=image, moving=image.roll(1, 0), shape=mask.shape, affine=torch.eye(4))
    counts = model.counts()
    def update(m, opt):
        # Deliberately choose an adaptation-due step. The continuation must suppress it.
        return training_step(m, opt, queries, inputs, "unsupervised", None, 100, 1000,
                             base, paper_ncc, adaptation_enabled=False)
    first = model.layers[0].translation.detach().clone()
    values = update(model, optimizer)
    assert values["adaptation"] is None
    assert model.counts() == counts
    assert not torch.equal(first, model.layers[0].translation)
    assert all(torch.count_nonzero(l.contribution) == 0 for l in model.layers)
    payload = copy.deepcopy(model.payload())
    restored = GaussianModel.restore(payload, "cpu")
    restored_optimizer = restored.optimizer(joint=True)
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    update(model, optimizer)
    update(restored, restored_optimizer)
    torch.testing.assert_close(model(queries), restored(queries), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA phase accounting")
def test_phase_memory_peaks_and_nonoverlap():
    torch.cuda.empty_cache()
    inputs = torch.zeros(1024, device="cuda")
    resources = Resources()
    with resources.phase("optimization"):
        large = torch.zeros(1024*1024, device="cuda")
    del large
    with resources.phase("output_generation"):
        small = torch.zeros(8192, device="cuda")
    with resources.phase("analysis_storage"):
        with pytest.raises(RuntimeError, match="Nested"):
            with resources.phase("monitoring"):
                pass
    rows = resources.summary()["phases"]
    assert rows["optimization"]["incremental_peak_allocated_mib"] >= 4
    assert rows["output_generation"]["peak_allocated_mib"] < rows["optimization"]["peak_allocated_mib"]
    assert all(r["wall_seconds"] > 0 and r["cuda_event_seconds"] >= 0 for r in rows.values())
    del inputs, small
