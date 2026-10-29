"""Device compatibility patches for Baidu Unlimited-OCR-2.

Enables seamless execution on both CPU and GPU (CUDA):
- Overrides tensor.cuda() to safely return self when CUDA is not available or model is on CPU.
- Fixes masked_scatter_ to prevent device mismatch between CPU embeddings and CUDA masks.
- Overrides torch.cuda.synchronize() to no-op when running without a GPU.
"""

import sys
import torch


def apply_device_compat_patches():
    """Apply compatibility patches to PyTorch tensor operations."""
    # 1. Safe .cuda(): no-op if CUDA is not available or if called on CPU-only tensors
    orig_cuda = torch.Tensor.cuda

    def safe_cuda(self, *args, **kwargs):
        if not torch.cuda.is_available():
            return self
        try:
            return orig_cuda(self, *args, **kwargs)
        except (RuntimeError, AssertionError):
            return self

    torch.Tensor.cuda = safe_cuda

    # 2. Safe masked_scatter_: ensure mask and source align with self.device
    orig_masked_scatter_ = torch.Tensor.masked_scatter_

    def safe_masked_scatter_(self, mask, source):
        if mask.device != self.device:
            mask = mask.to(self.device)
        if source.device != self.device:
            source = source.to(self.device)
        return orig_masked_scatter_(self, mask, source)

    torch.Tensor.masked_scatter_ = safe_masked_scatter_

    # 3. Safe cuda.synchronize: no-op if CUDA is not available
    orig_cuda_sync = torch.cuda.synchronize

    def safe_cuda_sync(*args, **kwargs):
        if not torch.cuda.is_available():
            return
        try:
            return orig_cuda_sync(*args, **kwargs)
        except (RuntimeError, AssertionError):
            return

    torch.cuda.synchronize = safe_cuda_sync


# Apply automatically upon module import
apply_device_compat_patches()
