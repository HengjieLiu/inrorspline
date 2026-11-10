"""Dense-reference loss/gradient parity, domains and sampling continuation."""
from unittest.mock import patch

import torch

from scripts.experiments.gaussiandir import lncc_diffusion as local
from scripts.experiments.gaussiandir.data import Sampler,false_to_true


def test_all_windows_match_historical_ncc_values_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(14)
    shape=(9,10,11)
    fixed=torch.rand(shape)
    moving=torch.rand(shape,requires_grad=True)
    indices,valid=local.window_indices(torch.arange(fixed.numel()),shape)
    actual=local.window_ncc(fixed.flatten()[indices]*valid,moving.flatten()[indices]*valid)
    expected=local.reference().NCC9(fixed[None,None])(moving[None,None])
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=0)
    ga,=torch.autograd.grad(actual,moving)
    ge,=torch.autograd.grad(expected,moving)
    torch.testing.assert_close(ga,ge,atol=2e-7,rtol=2e-4)


def test_complete_loss_and_voxel_gradients_match_p43_dense_objective():
    torch.manual_seed(4)
    shape=(6,6,6)
    fixed,moving=torch.rand(shape),torch.rand(shape)
    q=(torch.randn(*shape,3)*.07).requires_grad_()
    count=5*6*6
    objective=local.LocalObjective(fixed,moving,dict(edges_per_axis=count))
    original=local.valid_edge_indices
    with patch.object(local,"valid_edge_indices",lambda shape,axis,device,count:original(shape,axis,device)):
        batch=objective.prepare(torch.arange(fixed.numel()))
    data,reg=objective(q.reshape(-1,3)[batch["field_indices"]],batch)
    ref=local.reference()
    stage=ref.StageSpec(3,None,1,.001,1,1,100,2)
    total,reference_data,reference_reg=ref.StageObjective(fixed,moving,1.,stage)(false_to_true(q,shape))
    torch.testing.assert_close(data,reference_data,atol=2e-6,rtol=0)
    torch.testing.assert_close(reg,reference_reg,atol=1e-8,rtol=2e-6)
    ga,=torch.autograd.grad(data+reg,q)
    ge,=torch.autograd.grad(total,q)
    torch.testing.assert_close(ga,ge,atol=3e-7,rtol=3e-4)


def test_axis_edge_domains_are_complete_and_do_not_wrap():
    shape=(5,7,9)
    ids=torch.arange(5*7*9)
    coords=local.unravel(ids,shape)
    for axis in range(3):
        start,end=local.valid_edge_indices(shape,axis,"cpu")
        expected=ids[coords[:,axis]<shape[axis]-1]
        assert torch.equal(start,expected)
        delta=local.unravel(end,shape)-local.unravel(start,shape)
        unit=torch.eye(3,dtype=torch.long)[axis]
        assert torch.equal(delta,unit.expand_as(delta))


def test_zero_padding_fixed_divisor_epsilon_and_sign():
    shape=(5,6,7)
    indices,valid=local.window_indices(torch.tensor([0]),shape)
    assert int(valid.sum())==5*5*5
    fixed=torch.ones(shape)
    windows=fixed.flatten()[indices]*valid
    torch.testing.assert_close(windows.mean(),torch.tensor(125/729))
    zero=torch.zeros(3,729)
    assert float(local.window_ncc(zero,zero))==1.
    torch.testing.assert_close(local.window_ncc(windows,windows),local.window_ncc(windows,-windows))
    low=windows*1e-3
    assert float(local.window_ncc(low,low))>float(local.window_ncc(windows,windows))


def test_sampler_and_random_edges_resume_exactly():
    torch.manual_seed(18)
    fixed=torch.rand(12,13,14)
    objective=local.LocalObjective(fixed,fixed,dict(edges_per_axis=30))
    sampler=Sampler(torch.ones_like(fixed,dtype=torch.bool),8)
    objective.prepare(sampler.next()[0])
    state=sampler.state(); rng=torch.get_rng_state()
    expected=objective.prepare(sampler.next()[0])
    restored=Sampler(torch.ones_like(fixed,dtype=torch.bool),8,state=state)
    torch.set_rng_state(rng)
    actual=objective.prepare(restored.next()[0])
    for key in ("field_indices","valid","coordinates"):
        assert torch.equal(actual[key],expected[key])
