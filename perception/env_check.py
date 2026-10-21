"""GB10 environment check: torch/CUDA versions, device capability and a small GPU matmul checked against fp64.

Usage: uv run python perception/env_check.py
"""

import platform
import time

import torch

print("python", platform.python_version(), platform.machine())
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("cuda available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
print("arch list", torch.cuda.get_arch_list())
a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
c = a @ b; torch.cuda.synchronize()
ref = (a.double() @ b.double()).float()
print("fp32 matmul max rel err vs fp64:", ((c - ref).abs().max() / ref.abs().max()).item())
t = time.perf_counter(); n = 20
for _ in range(n):
    c = a @ b
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n
print(f"4096^3 fp32 matmul: {dt*1e3:.2f} ms  -> {2*4096**3/dt/1e12:.1f} TFLOP/s")
ah, bh = a.half(), b.half(); ah @ bh; torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(n):
    ch = ah @ bh
torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n
print(f"4096^3 fp16 matmul: {dt*1e3:.2f} ms  -> {2*4096**3/dt/1e12:.1f} TFLOP/s")
