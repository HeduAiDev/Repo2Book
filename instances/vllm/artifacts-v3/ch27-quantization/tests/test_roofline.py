"""带宽账(roofline)—— arXiv:2306.00978 §4.1(4090:165 TFLOPS / 1TB/s,
算术强度 < 165 即 memory-bound;FP16 生成阶段强度 ≈ 1;W4 提到 ≈ 4 FLOPs/Byte
「AWQ reduces the weight memory by four times」);decode 逐 token = 矩阵-向量积
的算力/带宽口径同出 GPTQ §5("compute is dominated by matrix-vector products ...
primarily limited by memory bandwidth")。
测试先于实现书写(TDD),每条断言对应论文的一句可复现声明。"""
import numpy as np
import pytest

from roofline import (
    decode_arithmetic_intensity,
    is_memory_bound,
    matvec_flops,
    matvec_weight_bytes,
)


def test_decode_matvec_intensity_is_one_at_fp16_and_four_at_int4():
    # §4.1:batch-1 生成阶段 FP16 算术强度 ≈ 1;权重量到 4-bit 后
    # "approximately increase the arithmetic intensity to 4 FLOPs/Byte"。
    assert decode_arithmetic_intensity(weight_bits=16) == pytest.approx(1.0)
    assert decode_arithmetic_intensity(weight_bits=4) == pytest.approx(4.0)
    assert decode_arithmetic_intensity(weight_bits=8) == pytest.approx(2.0)


def test_matvec_flops_and_bytes_intensity_agree():
    # 强度 = FLOPs / 字节数:逐 token 的 y = W·x 是 d_out×d_in 矩阵-向量积,
    # FLOPs = 2·d_in·d_out,权重字节 = d_in·d_out·bits/8。
    d_in, d_out = 512, 256
    assert matvec_flops(d_in, d_out) == 2 * 512 * 256
    assert matvec_weight_bytes(d_in, d_out, weight_bits=4) == 512 * 256 // 2
    intensity = matvec_flops(d_in, d_out) / matvec_weight_bytes(
        d_in, d_out, weight_bits=4
    )
    assert intensity == pytest.approx(decode_arithmetic_intensity(weight_bits=4))


def test_4090_roofline_boundary_at_165():
    # §4.1:"any workload with arithmetic intensity less than 165 is memory
    # bounded on 4090 GPUs"(165 TFLOPS / 1 TB/s)。
    assert is_memory_bound(1.0, peak_flops=165e12, peak_bw=1e12)  # FP16 生成
    assert is_memory_bound(4.0, peak_flops=165e12, peak_bw=1e12)  # W4A16
    assert not is_memory_bound(165.0, peak_flops=165e12, peak_bw=1e12)
    assert not is_memory_bound(200.0, peak_flops=165e12, peak_bw=1e12)
    # prefill/大 batch 矩阵-矩阵积算术强度高 -> compute-bound(GPTQ A.2.2 的
    # 「先整体解压再普通 GEMM」对策的场景)。
    assert not is_memory_bound(300.0, peak_flops=165e12, peak_bw=1e12)
