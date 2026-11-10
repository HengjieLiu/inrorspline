"""Mathematical, native-operator and continuation contracts for P83."""
import copy
import json
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from scripts.experiments.gaussiandir.data import (Sampler, false_to_mm, false_to_true,
    flat_coordinates, mm_to_false, sample_volume)
from scripts.experiments.gaussiandir.model import (GaussianModel, PaperLayer,
    adaptation_due, knn_indices, lr_multiplier, paper_ncc, quaternion_matrix,
    stable_release_weights, tv_loss, upstream)

ROOT=Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/"experiments/2026-11-06_oasis_rank000_gaussiandir/protocol_v1.json").read_text())


def fixture(dtype=torch.float64,k=3):
    centers=torch.tensor([[-.4,.1,0.],[.4,-.2,.3],[.1,.4,-.2]],dtype=dtype)
    return PaperLayer(centers,.7,k,20,backend="brute_test")


def test_paper_weights_motion_equation_independently():
    layer=fixture()
    with torch.no_grad():
        layer._node_scaling[:]=torch.tensor([[-.2,-.4,-.7],[-.1,-.5,-.3],[-.4,-.3,-.6]])
        layer.node_quaternion[:,1:]=.25
        layer.quaternion[:,1:]=torch.tensor([.1,-.2,.3])
        layer.translation[:]=torch.tensor([.1,-.2,.03])
    queries=torch.tensor([[.1,.2,.3],[-.1,.2,.1]],dtype=torch.float64)
    covrot=quaternion_matrix(layer.node_quaternion)
    cov=covrot @ torch.diag_embed(layer._node_scaling.exp().square()) @ covrot.transpose(-1,-2)
    motion=quaternion_matrix(layer.quaternion)
    expected=[]
    for x in queries:
        delta=x-layer.node_position
        density=torch.exp(-.5*torch.einsum("ni,nij,nj->n",delta,torch.linalg.inv(cov),delta))/torch.linalg.det(cov).sqrt()
        weights=density/density.sum()
        local=torch.einsum("nij,nj->ni",motion,delta)-delta+layer.translation
        expected.append((weights[:,None]*local).sum(0))
    torch.testing.assert_close(layer(queries),torch.stack(expected),atol=1e-12,rtol=1e-12)


def test_all_paper_parameters_have_correct_gradients():
    torch.manual_seed(2)
    layer=fixture()
    with torch.no_grad():
        layer._node_scaling.add_(torch.randn_like(layer._node_scaling)*.2)
        layer.node_quaternion.add_(torch.randn_like(layer.node_quaternion)*.15)
        layer.quaternion.add_(torch.randn_like(layer.quaternion)*.15)
        layer.translation.add_(torch.randn_like(layer.translation)*.1)
    x=torch.tensor([[.12,.22,.31],[-.13,.24,-.11]],dtype=torch.float64)
    names,values=zip(*layer.named_parameters())
    assert torch.autograd.gradcheck(lambda *p:torch.func.functional_call(layer,dict(zip(names,p)),(x,)),
                                   values,eps=1e-6,atol=1e-5,rtol=1e-3)


def test_identity_and_common_translation():
    layer=fixture()
    x=torch.randn(8,3,dtype=torch.float64)
    assert torch.count_nonzero(layer(x))==0
    with torch.no_grad():layer.translation[:]=torch.tensor([.3,-.1,.2])
    torch.testing.assert_close(layer(x),layer.translation[:1].expand(8,-1),atol=1e-12,rtol=1e-12)


def test_log_weights_handle_small_but_finite_scales():
    layer=fixture()
    with torch.no_grad():
        layer._node_scaling[:]=-12
        layer.translation[:]=torch.tensor([.3,-.1,.2])
    assert torch.isfinite(layer(torch.ones(2,3,dtype=torch.float64))).all()


def test_grid_physical_roundtrip_and_warp():
    shape=(9,11,13)
    indices=torch.arange(np.prod(shape))
    coords=flat_coordinates(indices,shape,dtype=torch.float64)
    volume=torch.arange(np.prod(shape),dtype=torch.float64).reshape(shape)
    torch.testing.assert_close(sample_volume(volume,coords),volume.reshape(-1),atol=1e-10,rtol=1e-12)
    affine=np.eye(4)
    affine[:3,:3]=[[0.,-1.5,0.],[.8,0.,0.],[0.,0.,2.1]]
    displacement=torch.randn(17,3,dtype=torch.float64)*.01
    physical=false_to_mm(displacement,shape,affine)
    torch.testing.assert_close(mm_to_false(physical,shape,affine),displacement,atol=1e-12,rtol=1e-12)
    n=displacement.new_tensor(shape[::-1])
    torch.testing.assert_close(false_to_true(displacement,shape)*(n-1)/2,displacement*n/2)


def test_density_uses_center_gradients_and_updates_adam_state():
    layer=fixture(k=1)
    optimizer=torch.optim.Adam([dict(params=[p],name=n,lr=.01) for n,p in layer.named_parameters()])
    for p in layer.parameters():p.grad=torch.ones_like(p)
    optimizer.step()
    before={name:p.detach().clone() for name,p in layer.named_parameters()}
    layer.contribution[:]=torch.tensor([.003,.001,0.])
    layer.counter[:]=1
    event=layer.densify_and_prune(optimizer)
    assert event==dict(before=3,pruned=1,cloned=1,after=3)
    for name,p in layer.named_parameters():
        torch.testing.assert_close(p[:2],before[name][:2])
        assert torch.count_nonzero(optimizer.state[p]["exp_avg"][-1])==0
        assert optimizer.state[p]["step"]==1
    layer.node_position.grad=torch.ones_like(layer.node_position)*.01
    layer.translation.grad=torch.zeros_like(layer.translation)
    layer.add_densification_stats()
    torch.testing.assert_close(layer.contribution,torch.full_like(layer.contribution,np.sqrt(3)*.01))


def test_population_guards_fail_without_silent_clamping():
    layer=fixture(k=3)
    optimizer=torch.optim.Adam([dict(params=[p],name=n) for n,p in layer.named_parameters()])
    layer.counter[:]=1
    with pytest.raises(RuntimeError,match="leave"):
        layer.densify_and_prune(optimizer)
    with pytest.raises(ValueError,match="at least"):
        knn_indices(torch.zeros(1,3),torch.zeros(2,3),3,"brute_test")


def test_multiscale_transition_and_serialization():
    cfg=config()
    cfg["knn_k"]=3
    cfg["paper_reconstruction"]["sparsifications"]=[4,2]
    mask=torch.ones(24,32,40,dtype=torch.bool)
    model=GaussianModel("paper_reconstruction",cfg,mask,"cpu",backend="brute_test")
    with torch.no_grad():model.active.translation[:]=torch.tensor([.1,-.2,.03])
    x=torch.rand(8,3)*2-1
    before=model(x).detach().clone()
    model.add_stage(mask,"cpu")
    torch.testing.assert_close(model(x),before,atol=0,rtol=0)
    assert all(not p.requires_grad for p in model.layers[0].parameters())
    restored=GaussianModel.restore(model.payload(),"cpu")
    torch.testing.assert_close(restored(x),model(x),atol=0,rtol=0)
    assert restored.counts()==model.counts()


def test_loss_reductions_and_scheduler():
    x=torch.arange(8,dtype=torch.float64)
    assert float(paper_ncc(x,x))==pytest.approx(-1,abs=1e-9)
    flow=torch.cat((torch.zeros(1,3),torch.eye(3)))
    assert tv_loss(flow,"released_control")==1
    assert tv_loss(flow,"paper_reconstruction")==3
    assert lr_multiplier(50,2000,50)==1
    cfg=config()
    assert adaptation_due("released_control",300,2000,cfg)
    assert not adaptation_due("released_control",1300,2000,cfg)
    assert adaptation_due("paper_reconstruction",900,1000,cfg)
    assert not adaptation_due("paper_reconstruction",1000,1000,cfg)


def test_sampler_checkpoint_exact_continuation():
    torch.manual_seed(5)
    mask=torch.ones(3,4,5,dtype=torch.bool)
    sampler=Sampler(mask,17)
    sampler.next()
    state=sampler.state()
    rng=torch.get_rng_state()
    expected=[sampler.next()[0] for _ in range(6)]
    restored=Sampler(mask,17,state=state)
    torch.set_rng_state(rng)
    for ids in expected:torch.testing.assert_close(restored.next()[0],ids,atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="native CUDA required")
def test_native_knn_cpu_gpu_brute_parity():
    torch.manual_seed(4)
    x,centers=torch.rand(29,3),torch.rand(41,3)
    expected=knn_indices(x,centers,20,"brute_test")
    torch.testing.assert_close(knn_indices(x,centers,20),expected)
    torch.testing.assert_close(knn_indices(x.cuda(),centers.cuda(),20).cpu(),expected)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="released CUDA constructor")
def test_released_stable_weight_forward_and_gradient_parity():
    torch.manual_seed(8)
    source,_=upstream()
    raw=source.GaussianWarp(K=20,node_shape=[10]*3,max_densify_num=5000,sparsification=4,
        max_contribution=.002,min_contribution=1e-7,with_quaternion=False,
        anisotropy=dict(scaling=False,quaternion=False))
    raw.reinitialize(torch.ones(24,32,40,dtype=torch.bool))
    raw.cuda()
    with torch.no_grad():raw.translation.normal_(std=.03)
    stable=copy.deepcopy(raw)
    stable.cal_nn_weight=MethodType(stable_release_weights,stable)
    x=torch.rand(97,3,device="cuda")*1.8-.9
    a,b=raw(x),stable(x)
    torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    for p,q in zip(raw.parameters(),stable.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-4)


def test_optimizer_checkpoint_continuation():
    cfg=config();cfg["knn_k"]=3
    model=GaussianModel("paper_reconstruction",cfg,torch.ones(24,32,40,dtype=torch.bool),"cpu",backend="brute_test")
    optimizer=model.optimizer()
    x=torch.tensor([[.2,.1,.4],[-.2,.1,-.1]])
    target=torch.tensor([[.02,.03,.04],[-.01,.02,.01]])
    def step(m,o):
        o.zero_grad();(m(x)-target).square().sum().backward();o.step()
    step(model,optimizer)
    restored=GaussianModel.restore(copy.deepcopy(model.payload()),"cpu")
    restored_optimizer=restored.optimizer()
    restored_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model,optimizer);step(restored,restored_optimizer)
    torch.testing.assert_close(model(x),restored(x),atol=0,rtol=0)
