"""带宽账(roofline)—— arXiv:2306.00978 §4.1(4090 峰值 165 TFLOPS /
带宽 1TB/s:算术强度 = FLOPs/访存字节,低于 165 即 memory-bound;FP16 逐
token 生成阶段算术强度 ≈ 1;权重量到 4-bit「approximately increase the
arithmetic intensity to 4 FLOPs/Byte」「AWQ reduces the weight memory by four
times」)。decode 逐 token = 矩阵-向量积的算力/带宽口径另出 GPTQ §5
("compute is dominated by matrix-vector products ... primarily limited by
memory bandwidth")与 §6(自认提速全部来自 memory movement 减少)。

口径:batch-1 decode 的 y = W·x,FLOPs = 2·d_in·d_out、权重字节 =
d_in·d_out·bits/8 —— 权重访问主导(§4.1 Figure 3 右面板),激活/向量
字节是加性小项,此参考口径不计(论文自己的「≈」即此意)。
"""


# PAPER: arXiv:2306.00978 §4.1 —— 逐 token 矩阵-向量积 y = W·x 的 FLOPs:
# 每输出元素 d_in 次乘加 = 2·d_in·d_out。
def matvec_flops(d_in, d_out):
    return 2 * d_in * d_out


# PAPER: arXiv:2306.00978 §4.1 —— 权重访存字节:每权重 bits/8 字节
# (W4 = 1/2 字节/权重,即「reduces the weight memory by four times」)。
def matvec_weight_bytes(d_in, d_out, weight_bits):
    return d_in * d_out * weight_bits // 8


# PAPER: arXiv:2306.00978 §4.1 —— decode 阶段算术强度 = FLOPs/权重字节 =
# 2/(bits/8) = 16/bits:FP16 → 1(论文实测 ≈ 1)、INT8 → 2、INT4 → 4
# (论文:"increase the arithmetic intensity to 4 FLOPs/Byte")。
def decode_arithmetic_intensity(weight_bits):
    return 2.0 / (weight_bits / 8.0)


# PAPER: arXiv:2306.00978 §4.1 —— roofline 判据:算术强度 < 峰值算力/峰值
# 带宽(4090:165 TFLOPS / 1 TB/s = 165)即 memory-bound;默认参数即 4090。
def is_memory_bound(intensity, peak_flops=165e12, peak_bw=1e12):
    return intensity < peak_flops / peak_bw
