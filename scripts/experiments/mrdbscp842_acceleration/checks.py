"""Independent algebra/oracle and actual PyTorch backend checks (no pytest needed)."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from scripts.experiments.oasis_alpha_sweep.run_one import stage_stop_decision, c
from .backends import Backend, LossBuffer, TERMS, VARIANTS


def oracle(controls, shape, spacing):
    total = controls.new_zeros(())
    for orders, weight in TERMS:
        value = controls
        for axis, q in enumerate(orders):
            t = (np.arange(shape[axis])[:, None] / spacing + 1
                 - np.arange(controls.shape[axis])[None, :])
            # Independent direct cubic derivative evaluation; enforce compact support.
            matrix = np.vectorize(lambda x: c.cubic_bspline_value(x, derivative=q)
                                  if abs(x) < 2 else 0.0)(t) / spacing**q
            matrix = torch.tensor(matrix, device=controls.device, dtype=controls.dtype)
            value = (value.movedim(axis, -1) @ matrix.T).movedim(-1, axis)
        total = total + weight * value.square().mean()
    return total


def value_grad(fn, x):
    y = fn(x)
    grad, = torch.autograd.grad(y, x)
    return y.detach(), grad.detach()


def error(a, b):
    delta = (a.double()-b.double())
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm()/b.double().norm().clamp_min(1e-30))}


def run_checks(device="cpu", real_shapes=False, only_real=False):
    torch.set_num_threads(6)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    records = []
    passed = {name: True for name in VARIANTS}
    shapes = [((9, 11, 13), 2), ((13, 17, 19), 4), ((21, 25, 27), 8)]
    if real_shapes:
        shapes += [((40, 56, 48), 2), ((80, 112, 96), 2), ((160, 224, 192), 2)]
    if only_real:
        shapes = shapes[3:]
    for shape, spacing in shapes:
        # MR uses CP_MARGIN=4 nested grids, not the separate single-stage
        # bspline_control_shape() helper (one fewer control per axis here).
        cs = (tuple(n//2+4 for n in shape) if max(shape) >= 40
              else c.bspline_control_shape(shape, spacing))
        coords = torch.stack(torch.meshgrid(*[torch.arange(n, device=device) for n in cs], indexing="ij"), -1)
        states = {"zero": torch.zeros((*cs, 3), device=device),
                  "constant": torch.full((*cs, 3), .1, device=device),
                  "affine": coords.float() * .001 + .03,
                  "near_affine": coords.float() * .001 + .03 + torch.randn((*cs, 3), device=device)*1e-5,
                  "random": torch.randn((*cs, 3), device=device)*.01,
                  "boundary": torch.zeros((*cs, 3), device=device)}
        states["boundary"][0, 1, 2, 0] = .1
        for name, state in states.items():
            x = state.detach().requires_grad_()
            reference = lambda v: c.analytic_bspline_sampled_be(v, shape, spacing)
            v0, g0 = value_grad(reference, x)
            be = Backend(("cpbe",))
            v1, g1 = value_grad(lambda v: be.be(v, shape, spacing), x)
            ev, eg = error(v1, v0), error(g1, g0)
            null = name in ("zero", "constant", "affine")
            ok = (abs(float(1000*(v1-v0))) <= 1e-7+3e-5*abs(float(1000*v0))
                  and (1000*eg["max_abs"] <= 1e-9 if null else eg["relative_l2"] <= 1e-4))
            passed["cpbe"] &= ok
            row = dict(test="BE", shape=shape, control_shape=cs, cps=spacing, state=name, reference=float(v0),
                       candidate=float(v1), value_error=ev, gradient_error=eg, passed=ok)
            if max(shape) < 30:
                xd = state.double().requires_grad_()
                vd, gd = value_grad(lambda v: oracle(v, shape, spacing), xd)
                vq, gq = value_grad(lambda v: be.be(v, shape, spacing), xd)
                fp64ok = torch.allclose(vd, vq, rtol=1e-10, atol=1e-12) and torch.allclose(gd, gq, rtol=1e-10, atol=1e-12)
                row.update(fp64_pass=fp64ok, fp64_gradient_error=error(gq, gd))
                passed["cpbe"] &= fp64ok
                # Regularizer-only multi-update stress; not a substitute for full-loss trajectory checks.
                params = [torch.nn.Parameter(state.clone()), torch.nn.Parameter(state.clone())]
                opts = [torch.optim.Adam([p], lr=.003) for p in params]
                for _ in range(25):
                    for j, (p, opt) in enumerate(zip(params, opts)):
                        opt.zero_grad(set_to_none=True)
                        val = reference(p) if j == 0 else be.be(p, shape, spacing)
                        (1000*val).backward()
                        opt.step()
                drift = (params[1]-params[0]).detach()*torch.tensor([(shape[2]-1)/2, (shape[1]-1)/2, (shape[0]-1)/2], device=device)
                row["adam25_max_voxel_delta"] = float(drift.norm(dim=-1).max())
                if null:
                    passed["cpbe"] &= row["adam25_max_voxel_delta"] <= 1e-4
            records.append(row)
        x = states["random"].detach().requires_grad_()
        projection = torch.randn((*shape, 3), device=device)
        out0 = c.expand_bspline_controls(x, cs, shape, spacing)
        grad0, = torch.autograd.grad((out0*projection).sum(), x)
        for variant in ("batch", "cache"):
            with Backend(VARIANTS[variant]).activate():
                out = c.expand_bspline_controls(x, cs, shape, spacing)
                grad, = torch.autograd.grad((out*projection).sum(), x)
                v, g = value_grad(lambda z: c.analytic_bspline_sampled_be(z, shape, spacing), x)
            vr, gr = value_grad(lambda z: c.analytic_bspline_sampled_be(z, shape, spacing), x)
            ok = (torch.allclose(out, out0, atol=3e-7, rtol=3e-5)
                  and error(grad, grad0)["relative_l2"] <= 1e-4
                  and error(g, gr)["relative_l2"] <= 1e-4)
            passed[variant] &= ok
            records.append(dict(test="expand_BE", variant=variant, shape=shape, cps=spacing,
                                passed=ok, output_error=error(out, out0), vjp_error=error(grad, grad0)))
        del states, x, out0, grad0, projection, out, grad, v, g, vr, gr
    for scale in (1., 1e-4, 1e-6):
        fixed = torch.randn((11, 13, 15), device=device)*scale
        moving = (torch.randn_like(fixed)*scale).requires_grad_()
        v, g = value_grad(lambda x: c.ncc_loss(fixed, x), moving)
        with Backend(("ncc",)).activate():
            v1, g1 = value_grad(lambda x: c.ncc_loss(fixed, x), moving)
        ok = torch.equal(v, v1) and torch.equal(g, g1)
        passed["ncc"] &= ok
        records.append(dict(test="NCC", scale=scale, passed=ok, gradient_error=error(g1, g)))
    rows = []
    buffer = LossBuffer(100, device)
    for i in range(100):
        loss = torch.tensor(-1., device=device)
        buffer.append(i, loss, loss, loss*0)
        rows.append(dict(epoch=i+1))
    buffer.flush(rows)
    ok = all(r["loss"] == -1 and r["reg_loss"] == 0 for r in rows)
    stop, reason = stage_stop_decision(rows, min_epochs=80, patience=35, check_interval=25,
                                     rel_delta_total=1e-4, rel_delta_data=5e-5)
    passed["buffer"] &= ok and stop
    records.append(dict(test="history_stop", passed=ok and stop, reason=reason))
    passed["combined"] = all(v for k, v in passed.items() if k != "combined")
    return dict(device=str(device), real_shapes=real_shapes, passed=passed, records=records,
                backend=c.cuda_backend_metadata() if str(device).startswith("cuda") else {"torch": torch.__version__})


def snapshot_checks(paths, oasis_root, pair_file, gpu, feature_sets):
    """Full-loss VJPs and fresh-Adam sensitivity at actual reference controls.

    Adam moments are deliberately fresh here; full-trajectory equivalence is
    established separately by the paired registration runs, not this probe.
    """
    device = c.set_cuda_device(str(gpu),cudnn_benchmark=False,cudnn_deterministic=True)
    batch = c.load_oasis_case(rank=0,device=device,oasis_root=Path(oasis_root),pair_file=Path(pair_file))
    rows = []
    pyramids = {}
    for path in paths:
        saved = np.load(path,allow_pickle=False)
        state = torch.tensor(saved["controls"],device=device)
        stage = int(saved["stage"])
        shape = tuple(map(int,saved["image_shape"]))
        cps = int(saved["cps"])
        if stage not in pyramids:
            down = (4,2,1)[stage]
            fixed = c.make_loss_volume(batch["fixed"],down,.5,3.)[0]
            moving = c.make_loss_volume(batch["moving"],down,.5,3.)[0]
            pyramids[stage] = (fixed,moving,c.make_coordinate_grid(shape,device))
        fixed,moving,identity = pyramids[stage]
        reference = None
        for name,features in {"reference":(), **feature_sets}.items():
            p = torch.nn.Parameter(state.clone())
            with Backend(features).activate():
                disp = c.expand_bspline_controls(p,p.shape[:3],shape,cps)
                warped = c.warp_full(moving,identity+disp)
                data = c.ncc_loss(fixed[None,None],warped[None,None])
                reg = c.analytic_bspline_sampled_be(p,shape,cps)
                total = data+1000*reg
                optimizer = torch.optim.Adam([p],lr=(.03,.01,.003)[stage])
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                grad = p.grad.detach().clone()
                optimizer.step()
                values = (float(data),float(reg),float(total))
                after = p.detach().clone()
            if name == "reference":
                reference = (values,grad,after)
            else:
                rv,rg,ra = reference
                delta = (after-ra)*torch.tensor([191/2,223/2,159/2],device=device)
                rows.append(dict(variant=name,features=list(features),source=str(path),stage=stage,
                                 epoch=int(saved["epoch"]),control_shape=list(state.shape[:3]),
                                 data_loss=values[0],reference_data_loss=rv[0],
                                 weighted_be_abs_delta=abs(1000*(values[1]-rv[1])),
                                 total_loss_abs_delta=abs(values[2]-rv[2]),gradient_error=error(grad,rg),
                                 fresh_adam_max_voxel_delta=float(delta.norm(dim=-1).max())))
    return {"rows":rows,"backend":c.cuda_backend_metadata(),"note":"Fresh Adam sensitivity; no transferred moments"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real-shapes", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_checks(args.device, args.real_shapes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["passed"]), flush=True)


if __name__ == "__main__":
    main()
