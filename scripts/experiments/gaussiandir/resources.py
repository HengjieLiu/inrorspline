"""Synchronized phase accounting and NVML process-memory sampling (no dependencies)."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes as C
import os
import threading
import time

import torch


class ProcessInfo(C.Structure):
    _fields_ = [("pid", C.c_uint), ("usedGpuMemory", C.c_ulonglong),
                ("gpuInstanceId", C.c_uint), ("computeInstanceId", C.c_uint)]


class NvmlSampler:
    def __init__(self, uuid, interval=.1):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        self.check(self.lib.nvmlInit_v2())
        self.device = C.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), C.byref(self.device)))
        self.interval, self.phase, self.rows = interval, "setup", []
        self.pid = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error {code}")

    def sample(self):
        count = C.c_uint(64)
        processes = (ProcessInfo*64)()
        self.check(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(
            self.device, C.byref(count), processes))
        rows = [dict(pid=int(p.pid), mib=p.usedGpuMemory/2**20) for p in processes[:count.value]]
        # NVML reports host PIDs in this container. The calling process PID is
        # visible only in a shared PID namespace. When the GPU is exclusive,
        # its sole compute PID identifies this worker; otherwise leave ownership
        # unresolved and retain all observed sibling PIDs without guessing.
        if self.pid is None and len(rows) == 1:
            self.pid = rows[0]["pid"]
        self.rows.append(dict(time=time.monotonic(), phase=self.phase, processes=rows))

    def loop(self):
        try:
            while not self.stop_event.is_set():
                self.sample()
                self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = repr(exc)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.check(self.lib.nvmlShutdown())
        if self.error:
            raise RuntimeError(self.error)

    def summary(self, baseline=0.):
        phases = {}
        for row in self.rows:
            own = [p["mib"] for p in row["processes"] if p["pid"] == self.pid]
            if own:
                phases[row["phase"]] = max(phases.get(row["phase"], 0.), own[0])
        others = sorted({p["pid"] for r in self.rows for p in r["processes"] if p["pid"] != self.pid})
        return dict(host_pid=self.pid, interval_seconds=self.interval, samples=len(self.rows),
                    peak_process_mib_by_phase=phases,
                    increase_above_loaded_baseline_mib={k:max(0.,v-baseline) for k,v in phases.items()},
                    loaded_baseline_mib=baseline, competing_compute_pids=others,
                    note="Sampled process peaks are lower bounds; NVML includes context/non-PyTorch allocations.")


class Resources:
    def __init__(self, sampler=None):
        self.sampler = sampler
        self.phases = {}
        self.baseline = dict(allocated_mib=torch.cuda.memory_allocated()/2**20,
                             reserved_mib=torch.cuda.memory_reserved()/2**20)
        self.current = None

    @contextmanager
    def phase(self, name):
        if self.current is not None:
            raise RuntimeError("Nested phase accounting is invalid")
        self.current = name
        if self.sampler:
            self.sampler.phase = name
        began = time.monotonic()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            elapsed = time.monotonic()-began
            row = self.phases.setdefault(name, dict(wall_seconds=0., cuda_event_seconds=0., calls=0,
                peak_allocated_mib=0., peak_reserved_mib=0.))
            row["wall_seconds"] += elapsed
            row["cuda_event_seconds"] += start.elapsed_time(end)/1000
            row["calls"] += 1
            row["peak_allocated_mib"] = max(row["peak_allocated_mib"], torch.cuda.max_memory_allocated()/2**20)
            row["peak_reserved_mib"] = max(row["peak_reserved_mib"], torch.cuda.max_memory_reserved()/2**20)
            for key in ("allocated", "reserved"):
                row[f"incremental_peak_{key}_mib"] = max(0., row[f"peak_{key}_mib"]-self.baseline[f"{key}_mib"])
            self.current = None
            if self.sampler:
                self.sampler.phase = "instrumentation_and_control"

    def seconds(self, name):
        return self.phases.get(name, {}).get("wall_seconds", 0.)

    def summary(self):
        return dict(input_loaded_baseline=self.baseline, phases=self.phases,
                    cuda_note="Event intervals include GPU idle gaps during CPU work; synchronized wall is primary.")
