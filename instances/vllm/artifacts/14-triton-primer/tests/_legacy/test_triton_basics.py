"""Tests — Ch14 Triton Primer."""
import sys, torch, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

try:
    import triton
    from implementation.triton_basics import (
        vector_add, tiled_matmul,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(),
                    reason="Triton + CUDA required")
class TestVectorAdd:
    def test_correctness(self):
        x = torch.randn(10000, device='cuda')
        y = torch.randn(10000, device='cuda')
        out = vector_add(x, y)
        assert torch.allclose(out, x + y)

    def test_non_power_of_two(self):
        x = torch.randn(1234, device='cuda')
        y = torch.randn(1234, device='cuda')
        out = vector_add(x, y)
        assert torch.allclose(out, x + y)


@pytest.mark.skipif(not HAS_TRITON or not torch.cuda.is_available(),
                    reason="Triton + CUDA required")
class TestTiledMatmul:
    def test_small_shape(self):
        a = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        b = torch.randn(64, 128, device='cuda', dtype=torch.float16)
        c = tiled_matmul(a, b, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
        c_ref = torch.matmul(a.float(), b.float()).half()
        err = (c.float() - c_ref.float()).abs().max().item()
        assert err < 0.1  # fp16 tolerance

    def test_rectangular(self):
        a = torch.randn(256, 128, device='cuda', dtype=torch.float16)
        b = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        c = tiled_matmul(a, b)
        assert c.shape == (256, 64)


class TestCPUOnly:
    """These tests work without Triton."""
    def test_import_structure(self):
        from implementation import triton_basics
        assert hasattr(triton_basics, 'demonstrate')

    def test_tile_size_constants(self):
        """Verify block sizes used in vLLM Triton kernels are reasonable."""
        # From triton_unified_attention.py: BLOCK_Q=16, BLOCK_KV=32 (decode)
        BLOCK_Q, BLOCK_KV = 16, 32
        HEAD_DIM = 128
        fp16 = 2
        # Verify tile fits in L1 cache
        tile_bytes = (BLOCK_Q * HEAD_DIM + 2 * BLOCK_KV * HEAD_DIM) * fp16
        tile_bytes += BLOCK_Q * BLOCK_KV * 4 * 2  # S + P in fp32
        assert tile_bytes < 228 * 1024  # H100 L1 cache = 228 KB


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
