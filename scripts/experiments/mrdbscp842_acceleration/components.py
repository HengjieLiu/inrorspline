"""Opt-in nested host/CUDA-stream timings. Never sum overlapping clock domains.

Instrumentation preserves operation order; events resolve only at stage/output
boundaries. CUDA spans include stream idle/host gaps, not just kernel busy time.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import time
import numpy as np
import torch
import torch.nn.functional as F

ACTIVE = ContextVar("mr_components",default=None)


class Recorder:
    def __init__(self):
        self.origin = time.perf_counter()
        self.device = None
        self.stage = -1
        self.epoch = 0
        self.stack = []
        self.rows = []
        self.marks = {}
        self.counters = {}

    def count(self,name):
        key = f"stage{self.stage}.{name}"
        self.counters[key] = self.counters.get(key,0)+1

    def start(self,name,cuda=True):
        row = dict(id=len(self.rows),name=name,parent=self.stack[-1]["id"] if self.stack else None,
                   stage=self.stage,epoch=self.epoch,start_sec=time.perf_counter()-self.origin,
                   path="/".join([r["name"] for r in self.stack]+[name]))
        if cuda and self.device is not None and self.device.type == "cuda":
            row["events"] = (torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
            row["events"][0].record()
        self.rows.append(row)
        self.stack.append(row)
        return row

    def stop(self,row):
        if not self.stack or self.stack[-1] is not row:
            raise RuntimeError("Component nesting mismatch")
        if "events" in row:
            row["events"][1].record()
        row["end_sec"] = time.perf_counter()-self.origin
        row["wall_sec"] = row["end_sec"]-row["start_sec"]
        self.stack.pop()

    @contextmanager
    def scope(self,name,cuda=True):
        row = self.start(name,cuda)
        try:
            yield
        finally:
            self.stop(row)

    def resolve(self):
        pending = [r for r in self.rows if "events" in r and "end_sec" in r]
        if pending:
            torch.cuda.synchronize(self.device)
        for row in pending:
            a,b = row.pop("events")
            row["cuda_span_sec"] = a.elapsed_time(b)/1000

    def training_step(self,c,control,control_shape,full_cps,eval_shape,downsample,
                      moving,fixed,identity,domain_cps,alpha,optimizer):
        with self.scope("forward"):
            with self.scope("control_grid"):
                grid = c.flat_to_grid(control,control_shape)
            dense = c.expand_on_shape(control,control_shape,int(full_cps),eval_shape,int(downsample))
            with self.scope("warp_coordinates"):
                coordinates = identity+dense
            with self.scope("image_warp"):
                warped = c.warp_full(moving,coordinates,mode="bilinear")
            with self.scope("ncc"):
                data = c.ncc_loss(fixed[None,None],warped[None,None])
            with self.scope("be"):
                reg = c.analytic_bspline_sampled_be(grid,eval_shape,int(domain_cps),component_reduction="mean")
            with self.scope("loss_assembly"):
                loss = data+float(alpha)*reg
        with self.scope("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with self.scope("backward"):
            loss.backward()
        with self.scope("adam_step"):
            optimizer.step()
        return dense,data,reg,loss

    def payload(self):
        self.resolve()
        children = {}
        for row in self.rows:
            if row["parent"] is not None:
                children.setdefault(row["parent"],[]).append(row)
        groups = {}
        begin,end = self.marks.get("case_load_complete",float("inf")),self.marks.get("dvf_file_ready",-1)
        for row in self.rows:
            if "end_sec" not in row:
                raise RuntimeError("Unclosed component")
            child = children.get(row["id"],[])
            wall_self = row["wall_sec"]-sum(c["wall_sec"] for c in child)
            gpu = row.get("cuda_span_sec")
            gpu_self = None if gpu is None else gpu-sum(c.get("cuda_span_sec",0) for c in child)
            group = "first_iteration" if row["epoch"] == 1 else "remaining_iterations" if row["epoch"] > 1 else "outside_iterations"
            window = "headline" if row["start_sec"] >= begin and row["end_sec"] <= end else "outside_or_parent"
            key = (row["stage"],row["path"],group,window)
            out = groups.setdefault(key,dict(stage=row["stage"],path=row["path"],name=row["name"],
                iteration_group=group,window=window,calls=0,wall_inclusive_sec=0.,wall_exclusive_sec=0.,
                cuda_span_inclusive_sec=0.,cuda_span_exclusive_sec=0.,cuda_calls=0))
            out["calls"] += 1
            out["wall_inclusive_sec"] += row["wall_sec"]
            out["wall_exclusive_sec"] += wall_self
            if gpu is not None:
                out["cuda_calls"] += 1
                out["cuda_span_inclusive_sec"] += gpu
                out["cuda_span_exclusive_sec"] += gpu_self
        # Exact non-overlapping host coverage within the endpoint interval.
        intervals = sorted((max(begin,r["start_sec"]),min(end,r["end_sec"])) for r in self.rows
                           if r["start_sec"] < end and r["end_sec"] > begin)
        merged = []
        for a,b in intervals:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1],b)
            else:
                merged.append([a,b])
        return dict(schema_version=1,rows=list(groups.values()),counters=self.counters,
                    raw_event_count=len(self.rows),headline_sec=end-begin,
                    host_uncovered_sec=(end-begin)-sum(b-a for a,b in merged),
                    interpretation="Host dispatch/wait and CUDA stream spans overlap. Inclusive parents overlap children. CUDA spans are not kernel-only durations.")


@contextmanager
def instrument(c):
    """Wrap the active common module AFTER experimental backend activation."""
    from .backends import TERMS
    recorder = Recorder()
    token = ACTIVE.set(recorder)
    saved = []
    def replace(obj,name,value):
        saved.append((obj,name,getattr(obj,name)))
        setattr(obj,name,value)
    def wrap(obj,name,label,cuda=True):
        original = getattr(obj,name)
        def call(*args,**kwargs):
            with recorder.scope(label,cuda):
                return original(*args,**kwargs)
        replace(obj,name,call)
    original_kernel = c.cubic_bspline1d
    owner = getattr(original_kernel,"__self__",None)
    def kernel(stride,derivative=0,dtype=None,device=None):
        key = (int(stride),int(derivative),dtype,str(device))
        hit = owner is not None and key in owner.kernels
        recorder.count("kernel_hit" if hit else "kernel_build")
        with recorder.scope("kernel_cache_hit" if hit else "kernel_build"):
            return original_kernel(stride,derivative,dtype=dtype,device=device)
    def expansion(control,cs,shape,cps):
        with recorder.scope("bspline_field_expansion"):
            with recorder.scope("layout_input"):
                field = control.reshape(*cs,3).permute(3,0,1,2).unsqueeze(0)
            for dim in range(3):
                k = c.cubic_bspline1d(cps,derivative=0,dtype=field.dtype,device=field.device)
                with recorder.scope("axis_"+"zyx"[dim]):
                    field = c.conv1d(field,dim=dim+2,kernel=k,stride=int(cps),padding=(k.shape[0]-1)//2,transpose=True)
            with recorder.scope("crop_layout_output"):
                field = field[(slice(None),slice(None))+tuple(slice(cps,cps+shape[i]) for i in range(3))]
                return field.squeeze(0).permute(1,2,3,0).contiguous()
    def derivative(control,shape,cps,orders):
        with recorder.scope("derivative_"+"".join(map(str,orders))):
            with recorder.scope("layout_input"):
                field = c.dhw3_to_ncdhw(control).float()
            for dim,order in enumerate(orders):
                k = c.cubic_bspline1d(cps,derivative=int(order),dtype=field.dtype,device=field.device)
                if order:
                    with recorder.scope("derivative_kernel_scale"):
                        k = k*(1.0/float(cps))**int(order)
                with recorder.scope("axis_"+"zyx"[dim]):
                    field = c.conv1d(field,dim=dim+2,kernel=k,stride=int(cps),padding=(k.shape[0]-1)//2,transpose=True)
            with recorder.scope("crop_layout_output"):
                return field[(slice(None),slice(None))+tuple(slice(cps,cps+int(shape[i])) for i in range(3))].contiguous()
    original_be = c.analytic_bspline_sampled_be
    be_owner = getattr(original_be,"__self__",None)
    def cpbe(controls,image_shape,cps,mask=None,*,component_reduction="mean"):
        from .backends import axis_basis
        if mask is not None or component_reduction != "mean":
            raise ValueError("CP BE supports unmasked component mean only")
        total = controls.new_zeros(())
        for orders,weight in TERMS:
            with recorder.scope("term_"+"".join(map(str,orders))):
                value = controls
                with recorder.scope("control_differences"):
                    for axis,order in enumerate(orders):
                        if order:
                            value = torch.diff(value,n=order,dim=axis)
                for axis,order in enumerate(orders):
                    n,m = int(image_shape[axis]),controls.shape[axis]
                    key = (n,m,cps,order,value.dtype,str(value.device))
                    with recorder.scope("factor_lookup",cuda=False):
                        hit = key in be_owner.factors
                    recorder.count("factor_hit" if hit else "factor_build")
                    if not hit:
                        with recorder.scope("factor_basis_qr_cpu",cuda=False):
                            r = np.linalg.qr(axis_basis(n,m,cps,order),mode="reduced")[1]
                        with recorder.scope("factor_transfer"):
                            be_owner.factors[key] = torch.as_tensor(r.copy(),dtype=value.dtype,device=value.device)
                    r = be_owner.factors[key]
                    with recorder.scope("projection_"+"zyx"[axis]):
                        value = (value.movedim(axis,-1) @ r.T).movedim(-1,axis)
                with recorder.scope("energy_reduction"):
                    total = total+weight*value.square().sum()
        with recorder.scope("normalization"):
            return total/(3*int(np.prod(image_shape)))
    try:
        replace(c,"cubic_bspline1d",kernel)
        replace(c,"expand_bspline_controls",expansion)
        replace(c,"expand_cubic_bspline_derivative",derivative)
        wrap(c,"_density_from_terms","energy_density_reduction")
        wrap(c,"_masked_density_mean","energy_mean_reduction")
        if be_owner is not None and "cpbe" in be_owner.features:
            replace(c,"analytic_bspline_sampled_be",cpbe)
        wrap(c,"gaussian_blur3d","gaussian_smoothing")
        wrap(c,"make_loss_volume","loss_volume")
        wrap(c,"make_coordinate_grid","identity_grid")
        wrap(F,"interpolate","downsample_interpolate")
        wrap(F,"conv1d","conv1d_kernel")
        wrap(F,"conv_transpose1d","conv_transpose1d_kernel")
        yield recorder
    finally:
        for obj,name,value in reversed(saved):
            setattr(obj,name,value)
        ACTIVE.reset(token)
