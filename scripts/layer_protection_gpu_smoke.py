"""Exercise Triton's CUDA driver compilation and the RoPE-shaped batched matmul."""

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(source, target, SIZE: tl.constexpr):
    offsets = tl.arange(0, SIZE)
    tl.store(target + offsets, tl.load(source + offsets))


assert torch.cuda.is_available(), "CUDA unavailable"
x = torch.arange(128, device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
copy_kernel[(1,)](x, y, 128)
torch.testing.assert_close(x, y)
a = torch.randn(1, 64, 1, device="cuda")
b = torch.randn(1, 1, 32, device="cuda")
torch.testing.assert_close(torch.bmm(a, b), a * b)
torch.cuda.synchronize()
print(sys.executable, torch.__version__, torch.cuda.get_device_name(), "GPU smoke PASS", flush=True)
