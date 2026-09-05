# impl-notes — v3 ch27《【primer】量化》

本章 `kind: primer`——不是 vLLM 源码的减法精简版,而是**论文忠实的小型参考实现**
(NumPy,纯 CPU,host `python -m pytest` 小参数即可全量跑完,用于 explainer 产出
可示教的数值轨迹)。每个 `def`/`class` 用 `# PAPER: §x Eq.y` 锚定论文出处
(替代普通章节的 `# SOURCE:`);门禁为 `scripts/lint_paper_grounding.py`
(lint_fidelity 不跑)。

论文包(真相源):`instances/vllm/book/papers/ch26-primer-quantization/`——
`paper.md`(GPTQ, arXiv:2210.17323,主)、`paper-awq.md`(arXiv:2306.00978)、
`paper-smoothquant.md`(arXiv:2211.10438)。注意 lint 脚本按章目录名推导论文包
路径(`book/papers/ch27-quantization`)与实际包名(`ch26-primer-quantization`,
沿用 v2 章位)不一致,报 WARN 不阻断——与 ch20(包 `ch24-primer-flash-attention`)
同款既有约定。

## 文件清单

- `uniform_quant.py` —— 均匀量化底座(m01):SmoothQuant §2 Eq.1 对称式
  (Δ = max|X|/(2^(N-1)-1))+ GPTQ §5 Setup 的 per-row 非对称 min-max 网格
  (zero-point)+ §2 Figure 3 粒度谱(per-tensor/per-token/per-channel)+ §3
  obs.2 有效量化级数(RTN 之死 m03 的定量件)。
- `gptq.py` —— GPTQ 全推导件(m04):H=2XᵀX → dampening(§4 Step 3)→
  Cholesky(H^{-1})ᵀ(Algorithm 1 前置行)→ 主循环(lazy batch B 列分块);
  OBQ 贪心单行(§3 Eq.2-3)与「每列一次 Eq.2+Eq.3」的 Step 1 参考实现
  (`gptq_naive_inverse_updates`,与 Algorithm 1 数值等价的对账件);
  RTN 基线(§5 Baselines:同一副网格直接取整)+ 复杂度账(§4 Step 1)。
- `awq.py` —— AWQ(m05):Eq.1 组量化器(Δ = max|w|/2^(N-1),注意与
  SmoothQuant Eq.1 分母差 1:论文各自的约定,保留)、Eq.3 两条误差表达式与
  (Δ'/Δ)·(1/s) 误差比、Table 2 统计协议、Eq.4-Eq.5 搜索(s = s_X^α,
  grid 20)、§4.2 SIMD 打包(interleave [0,2,4,6,1,3,5,7],与 vLLM awq_pack
  同构)。
- `smoothquant.py` —— SmoothQuant(m06):Eq.3 等价平滑、Eq.4 α 配平、
  W8A8 per-tensor 静态模拟、α 消融(§5.5 Figure 10 协议)。
- `roofline.py` —— 论文侧带宽账(m10 的 §4.1/§5 件):算术强度 16/bits
  (FP16→1、INT4→4)、4090 roofline 判据(165 = 165TFLOPS/1TB/s)。

## 范围决定(不发明论文没有的机制)

- **FP8(e4m3)/e8m0/FP4 两级缩放(m07/m08)不在本实现**:格式数学出自 OCP
  MX / NVIDIA 规范,dossier 明记不在论文包;这两机制的正文素材走 vLLM 源码
  侧(`scaled_quantize` / `_quantize_group_native` / NVFP4 装载面,dossier
  embed_excerpts 已备),不是论文参考实现的对象。
- **布局遵循各论文自己的记法**:GPTQ 写层输出 WX(W 为 (out,in),X 为
  (d_col,m)——HF Linear 权重同构);AWQ/SmoothQuant 写 Y = XW(X (T,C_i)、
  W (C_i,C_o))。两套互为转置,模块 docstring 已声明;这本身是个教学点。
- **W4A16 kernel 不做 numpy 仿真**:GPTQ §5 的「动态反量化矩阵-向量积
  kernel」收益在带宽,NumPy 里无从示教;带宽账由 `roofline.py` 的公式化
  数字承担(FP16 强度 1 → W4 强度 4),kernel 谱系走 vLLM 源码侧(m10-m12)。

## 1:1 Paper Map(参考实现符号 ↔ 论文出处 ↔ 对应关系/取舍)

| 参考实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `uniform_quant.quantize_symmetric` / `dequantize_symmetric` | arXiv:2211.10438 §2 Eq.1 | 对称式逐字:Δ = max\|X\|/(2^(N-1)-1);clip 只防浮点越界 |
| `uniform_quant.quantize_per_tensor` / `per_token` / `per_channel` | arXiv:2211.10438 §2 Figure 3 | 粒度谱三档;per-channel 挂输入通道维(激活侧 GEMM 不可行、权重侧标准——§3 Table 1 的论断) |
| `uniform_quant.quantize_asymmetric` / `dequantize_asymmetric` | arXiv:2210.17323 §5 Setup + arXiv:2211.10438 §2 | min-max 非对称网格:xmin→qmin、xmax→qmax 精确落格;zp = qmin − round(xmin/scale);scale 下限 1e-12 为全常数向量的数值护栏 |
| `uniform_quant.effective_quant_levels` | arXiv:2211.10438 §3 obs.2 | 2^N·m_i/m 逐字——RTN 之死(m03)的定量件 |
| `gptq.layer_hessian` | arXiv:2210.17323 §3 Eq.2 | H = 2XᵀX(论文 X 为 (d_col,m),此处 X 为 (m,d_col),故为 XᵀX);只依赖层输入——Step 1 合法性来源 |
| `gptq.dampen_hessian` | arXiv:2210.17323 §4 Step 3 | λ = 1% 平均对角元,逐字 |
| `gptq.inverse_hessian_cholesky` | arXiv:2210.17323 §4 Algorithm 1 前置行 | H^{-1} ← Cholesky(H^{-1})ᵀ(上三角);NumPy 无 cholesky_inverse,H^{-1} 直接求逆(数学同值),算法内容(一次 Cholesky 预取全部行)不变 |
| `gptq.row_grid_params` / `quantize_with_grid` / `dequantize_with_grid` | arXiv:2210.17323 §5 Setup | per-row 非对称 min-max 网格——GPTQ 与 RTN 共用(§5 Baselines:"exactly the same ... grid") |
| `gptq.rtn_quantize` | arXiv:2210.17323 §5 Baselines | RTN:同网格直接取整,无补偿;分组网格取自原始权重 |
| `gptq.layer_output_error` | arXiv:2210.17323 §3 Eq.1 | ‖WX − ŴX‖²(转置不变,以 X@Wᵀ 计算);GPTQ 的 argmin 目标与记分板 |
| `gptq.obq_quantize_row` | arXiv:2210.17323 §3 Eq.2 + Eq.3 | OBQ 贪心单行逐字:逐候选算 (quant(w)−w)²/[H^{-1}]_qq、δ_F 补偿、高斯消元删行列;§3 的 OBQ 无 dampening(Step 3 才加),H 须正定——docstring 已注 |
| `gptq.gptq_quantize` | arXiv:2210.17323 §4 Algorithm 1 | 主循环逐字:Cholesky 上三角 U、块内「量化→记 E→即时补偿」、块末 E@U 一次性全局补偿;分组网格取当前最新权重(§5 Additional Tricks 原话);`trace` 只记算法自身循环变量(示教轨迹,非新机制) |
| `gptq.gptq_naive_inverse_updates` | arXiv:2210.17323 §4 Step 1 | 每列一次 Eq.2 + Eq.3 全矩阵消元(无 Cholesky、无 lazy batch)——与 Algorithm 1 精确算术等价(测试逐位对账),也是复杂度中间形态 |
| `gptq.hessian_update_flops` | arXiv:2210.17323 §4 Step 1 | O(d_row·d_col³) → O(max{d_row·d_col², d_col³}),提速 min{d_row,d_col} 逐字 |
| `awq.awq_group_quantize` / `awq_dequantize` | arXiv:2306.00978 §3.2 Eq.1 | Q(w) = Δ·Round(w/Δ)、Δ = max\|w\|/2^(N-1) 逐字;网格以 ±max 为界(±2^(N-1)·Δ 都可达,round-trip 误差恒 ≤ Δ/2)——分母与 SmoothQuant Eq.1(−1)不同是论文各自的约定,保留并示教 |
| `awq.awq_quantize_matrix` | arXiv:2306.00978 §3.2 + 附录(group 128) | 组 = 输入通道维连续 group_size 个权重、每输出通道一列 |
| `awq.round_err` | arXiv:2306.00978 §3.2 obs.(1) | Round(·)−(·);平均 0.25(误差均匀分布于 [0,0.5]) |
| `awq.err_Qwx` / `err_Qws_xs` | arXiv:2306.00978 §3.2 Eq.3 | 两条误差表达式逐字(带符号);比值 (Δ'/Δ)·(1/s) 由测试/worked example 验证 |
| `awq.channel_mean_activation` / `salient_channels` | arXiv:2306.00978 §3.1 + §3.2(s_X) | 显著通道按激活平均幅度选(top-k);「看激活不看权重」的操作化 |
| `awq.table2_statistics` | arXiv:2306.00978 §3.2 Table 2 | 协议复刻:显著通道乘 s 后逐组统计 Δ 变化率/平均 Δ'/Δ/平均 (Δ'/Δ)(1/s);PPL 不可测,三行数值列即为可示教替代 |
| `awq.awq_loss` | arXiv:2306.00978 §3.2 Eq.4 | L(s) = ‖(diag(s)^{-1}X)@Q(W·diag(s)) − XW‖_F(按本章 (in,out) 布局转写,数学同式) |
| `awq.awq_search_scale` | arXiv:2306.00978 §3.2 Eq.5 + 附录 | s = s_X^α、α ∈ [0,1] 均匀网格 20 点(附录 "grid size of 20")、取 L 最小;不做权重 clipping(论文一句带过的工程增强,非搜索机制本体) |
| `awq.awq_pack` / `awq_unpack` | arXiv:2306.00978 §4.2 SIMD-aware packing | 每 8 个权重按 [0,2,4,6,1,3,5,7] interleave 后逐 nibble 压 int32(第 i 个 nibble 放重排后第 i 个);与 vLLM `quant_utils.awq_pack`(列维同一 interleave)互为印证;`& 0xF` 先取 4-bit 二补码 nibble 再 OR(负 int64 直接 OR 会污染高位——实现细节);unpack 符号扩展回 [-8,7] |
| `smoothquant.smooth_scale` | arXiv:2211.10438 §4 Eq.4 | s_j = max\|X_j\|^α/max\|W_j\|^{1-α} 逐字(逐输入通道) |
| `smoothquant.apply_smoothing` | arXiv:2211.10438 §4 Eq.3 | X̂ = X/s(列除)、Ŵ = s·W(行乘);严格等价由测试浮点验证 |
| `smoothquant.w8a8_per_tensor_output` / `w8a8_output_error` | arXiv:2211.10438 §2 Eq.1 + §4 | 激活/权重同量化器(§4:"the same quantizer ... per-tensor, static")的 W8A8 模拟;误差记分板 ‖Y_sim − Y_fp16‖_F |
| `smoothquant.migration_ablation` | arXiv:2211.10438 §5.5 Figure 10 | α 0→1 扫描;两端崩、甜点中段(合成层上最优 α 落 [0.3,0.7]) |
| `roofline.matvec_flops` / `matvec_weight_bytes` | arXiv:2306.00978 §4.1(及 arXiv:2210.17323 §5 的矩阵-向量积口径) | batch-1 decode:FLOPs = 2·d_in·d_out、权重字节 = d_in·d_out·bits/8;激活字节为加性小项不计(论文自己的「≈」) |
| `roofline.decode_arithmetic_intensity` | arXiv:2306.00978 §4.1 | 16/bits:FP16→1(论文实测 ≈1)、INT8→2、INT4→4("4 FLOPs/Byte") |
| `roofline.is_memory_bound` | arXiv:2306.00978 §4.1 | 强度 < 峰值算力/带宽即 memory-bound;默认参数即 4090(165 TFLOPS/1TB/s=165) |

## 取舍与实现说明

- **两套量化分母约定并存示教**:SmoothQuant §2 Eq.1 用 (2^(N-1)−1)(INT8 的
  127)、AWQ §3.2 Eq.1 用 2^(N-1)(INT4 的 8)——各自论文原文如此,不统一、
  各自逐字。GPTQ 主线用 §5 Setup 的非对称 min-max 网格(与 RTN 同网格对账)。
- **Cholesky 路径与朴素路径互为对账件**:`gptq_naive_inverse_updates`(每列
  Eq.2+Eq.3)与 `gptq_quantize`(Cholesky + lazy batch)在精确算术下同结果,
  测试逐位断言相等——Step 2/3「只改执行方式不改数学」的直接证据,lazy batch
  换 block_size 不变结果的测试同理。
- **OBQ 与 GPTQ 的 dampening 不对称是论文史实**:§3 的 OBQ 没有 dampening,
  §4 Step 3 才引入(1% 平均对角元);故 `obq_quantize_row` 不内置 dampening、
  `gptq_quantize` 默认 0.01。校准样本少于特征数时 H=2XᵀX 奇异——这正是
  worked example 的教学点之一。
- **banker's rounding**:np.round 取偶(0.5→偶数),论文未指定舍入细节,
  性质(误差 ≤ Δ/2)不受影响;测试选值避开 0.5 平手点。
- **合成数据口径**(测试即样例):GPTQ 对照用特征强相关(corr=0.95)的 X
  造各向异性 H——3-bit 下 GPTQ/RTN 误差比 ~0.17-0.21(论文 Table 3 的
  定性关系);SmoothQuant 合成层用单通道 ~100× 离群(§3 obs.2 量级),
  线性层 Frobenius 口径平滑收益 ~3×、最优 α≈0.5(真实 PPL 口径收益更大,
  非线性放大,正文引用论文数字);AWQ 搜索在 ~5% 显著通道的合成层上呈
  U 形曲线(α=0 最差、内点最优、α=1 反弹)——Table 2/Table 3 的形状。

## 给下游(explainer/writer)的话

- worked example 弹药已在测试里:GPTQ/OBQ 的 1×4 手推(含 trace 的逐列
  (U_jj, q, err))、AWQ 的 w=0.9/组内 9.9/s=2 手算(误差 0.3375→0.28125、
  误差比 5/6 vs (Δ'/Δ)/s = 1/2)、SmoothQuant 的 α=0.5 精确配平
  (max|X̂_j| == max|Ŵ_j| == sqrt(max|X_j|·max|W_j|))、SIMD 打包的
  0x75316420 手算。测试文件即这些轨迹的可执行版。
- `table2_statistics` 的三行数值列(Δ 变化率/平均 Δ'/Δ/平均 (Δ'/Δ)(1/s))
  是 Table 2 在合成数据上的可跑对应物;PPL 列只能引论文数字。
