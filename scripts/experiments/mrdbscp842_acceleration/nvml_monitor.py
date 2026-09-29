"""Read-only NVML calls without a Python package or CUDA context.

ABI checked against the installed nvml.h, nvmlProcessInfo_v2_t (24 bytes).
API: https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html
"""
import ctypes as ct


class ProcessInfo(ct.Structure):
    _fields_ = [("pid", ct.c_uint), ("usedGpuMemory", ct.c_ulonglong),
                ("gpuInstanceId", ct.c_uint), ("computeInstanceId", ct.c_uint)]


class NVML:
    def __init__(self, index, expected_uuid):
        try:
            self.lib = ct.CDLL("libnvidia-ml.so.1")
        except OSError:
            # Linuxbrew Python does not necessarily search the host ldconfig path.
            self.lib = ct.CDLL("/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
        if ct.sizeof(ProcessInfo) != 24:
            raise RuntimeError("Unexpected NVML process ABI")
        self.check(self.lib.nvmlInit_v2())
        self.handle = ct.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ct.c_uint(index), ct.byref(self.handle)))
        uuid = ct.create_string_buffer(96)
        self.check(self.lib.nvmlDeviceGetUUID(self.handle, uuid, ct.c_uint(len(uuid))))
        if uuid.value.decode() != expected_uuid:
            raise RuntimeError("NVML/nvidia-smi GPU UUID mismatch")

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError(f"NVML return code {code}")

    def processes(self):
        # v3 function uses nvmlProcessInfo_t == nvmlProcessInfo_v2_t, not the
        # distinct confidential-compute process detail structure.
        count = ct.c_uint(64)
        infos = (ProcessInfo*64)()
        result = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.handle, ct.byref(count), infos)
        if result == 7:
            infos = (ProcessInfo*(count.value+16))()
            count.value += 16
            result = self.lib.nvmlDeviceGetComputeRunningProcesses_v3(self.handle, ct.byref(count), infos)
        self.check(result)
        return [dict(pid=p.pid, name="NVML_compute_process", memory_mib=p.usedGpuMemory/(1024**2))
                for p in infos[:count.value]]
