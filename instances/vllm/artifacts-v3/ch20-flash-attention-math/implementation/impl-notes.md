# impl-notes — v3 ch20《【primer】Flash-Attention 数学》

本章 `kind: primer`——不是 vLLM 源码的减法精简版,而是**论文忠实的小型参考实现**
(NumPy,纯 CPU,host `python -m pytest` 小参数即可全量跑完,用于 explainer 产出可示教
的数值轨迹)。每个 `def`/`class` 用 `# PAPER: §x Eq.y` 锚定论文出处(替代普通章节的
`# SOURCE:`);门禁为 `scripts/lint_paper_grounding.py`(lint_fidelity 不跑)。

论文包(真相源):`instances/vllm/book/papers/ch24-primer-flash-attention/`——
`paper.md`(FA, arXiv:2205.14135)、`paper-online-softmax.md`(arXiv:1805.02867)、
`paper-fa2.md`(arXiv:2307.08691)。

## 文件清单

- `online_softmax.py` —— arXiv:1805.02867 §2-§3:naive(两遍,溢出)→ safe(三遍,
  减 max)→ online(单遍,running (m,d) 递推)三版收敛,§3.1 Eq.(3)-(4) 的 ⊕ 算子
  (结合律+交换律)与分块归并。
- `flash_attention.py` —— arXiv:2205.14135 §2.2 Algorithm 0(标准注意力,物化两张
  N×N)、§3.1 Algorithm 1(FA tiling + online-softmax 递推,Theorem 1 精确性)、
  §3.2 Theorem 2(IO 复杂度账的元素级计数);arXiv:2307.08691 §3.1.1 Algorithm 1
  (FA-2 循环序对调 + 未归一化 O + logsumexp L)与 §3.1.1 Causal masking(块级跳过)。
- `lse_merge.py` —— arXiv:1805.02867 §3.1 Eq.(4)(⊕ 算子)+ arXiv:2307.08691
  §2.3.1(两块表)/§3.1.1 Tweak 2(L=m+log ℓ):⊕ 在 (lse, output) 表示上的落地,
  对应 vLLM `merge_attn_states` 的数学原型;含 cascade(共享前缀+私有后缀)worked
  example。

## 1:1 Paper Map(参考实现符号 ↔ 论文出处 ↔ 对应关系/取舍)

| 参考实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `online_softmax.naive_softmax` | arXiv:1805.02867 §2 Eq.(1), Algorithm 1 | 两遍扫描、无 max 平移,逐字实现;用于对照演示 e^1000 上溢 |
| `online_softmax.safe_softmax` | arXiv:1805.02867 §2 Eq.(2), Algorithm 2 | 三遍扫描(max→Σexp→y),主流框架版本 |
| `online_softmax.online_softmax_stats` / `online_softmax` | arXiv:1805.02867 §3 Algorithm 3 lines 1-9 | 单遍 running (m,d) 递推,line 5 的 rescale 项逐字;Theorem 1 末值恒等由测试验证;可选 `trace` 逐元素记 (m_j,d_j) 快照(示教轨迹) |
| `online_softmax.online_softmax_merge` | arXiv:1805.02867 §3.1 Eq.(4) | ⊕ 二元算子逐字实现;论文声明结合律/交换律(省略证明),测试用具体数值验证 |
| `online_softmax.combine_blocks_via_merge` | arXiv:1805.02867 §3.1 Eq.(3) | 分块局部 (m,d) 经 ⊕ 归并 == 一遍顺序遍历(worked example 核心断言) |
| `flash_attention._safe_exp` | arXiv:2205.14135 §3.1 (f(x):=e^{x-m(x)}) | 数值安全包装:causal 掩码的 -inf 列给 0、-inf-(-inf)=nan 归 0——IEEE 浮点边界处理,非算法机制 |
| `flash_attention.causal_keep_mask` | arXiv:2307.08691 §3.1.1 Causal masking (S_ij=-inf for j>i) | 因果 keep 掩码;`query_offset = n_k - n_q` 的右下对齐约定取自 vLLM flash_attn_varlen_func docstring 的两个掩码原例(seqlen_q=2/5 两例由测试逐格锁定) |
| `flash_attention.standard_attention` | arXiv:2205.14135 §2.2 Algorithm 0 | S=QK^T→P=softmax(S)→O=PV 三步,物化两张 N×N(`return_weights=True` 可拿 P);`softmax_scale` 默认 1/√d(Appendix B.3 的 τ);全零掩码行输出 0(接口语义) |
| `flash_attention.materialized_intermediate_elements` | arXiv:2205.14135 §2.2 ("O(N^2) memory") | 感受数字:8K 上下文一张表 8192² = 67,108,864 元素 ≈ 6700 万,fp16 134MB |
| `flash_attention.fa_block_sizes` | arXiv:2205.14135 §3.1 Algorithm 1 line 1 | Bc=⌈M/4d⌉、Br=min(⌈M/4d⌉,d) 逐字 |
| `flash_attention.flash_attention_forward` | arXiv:2205.14135 §3.1 Algorithm 1 lines 2-16 + Theorem 1 | FA 主算法:外层 KV 列块 j、内层 Q 行块 i,running (m,ℓ,O) 递推;line 12 每步写回「归一化到当前为止」的 O_i(`trace` 逐 (j,i) 记快照);S/P 只存在于局部变量(代表 SRAM),numpy 数组即代表 HBM;Theorem 1 精确性=测试对 standard_attention 逐位断言 |
| `flash_attention.flash_attention_2_forward` | arXiv:2307.08691 §3.1.1 Algorithm 1 lines 3-17 (Tweak 1/2 + Causal masking) | FA-2:外层改 Q 行块、O 累加器未归一化收尾只除一次(Tweak 1)、返回 (O, L=m+log ℓ)(Tweak 2,即 vLLM return_softmax_lse 的 L);causal 整块跳过+跨对角块才施掩码(§3.1.1 两条)。见下方「FA-2 line 10 方向修正」 |
| `flash_attention.hbm_accesses_standard` / `hbm_accesses_flash` | arXiv:2205.14135 §3.2 Theorem 2 证明梗概 | 按 Algorithm 0/1 每行**精确计数**元素级访存(非仅 Θ 记号),worked example 代入 N=1024/d=64/M≈100KB 算比值;flash 侧渐近项 Θ(N²d²/M) 与块大小 tradeoff(块越大遍数越少)都可由此复算 |
| `lse_merge.attention_with_lse` | arXiv:2307.08691 §3.1.1 Tweak 2 (L=m+log ℓ) + §2.3.1 两块表 | 一次性(非分块)算完一段的 (O, lse)——段即 §2.3.1 表的「块 1」;空段(行内无合法 key)取 lse=-inf、O=0 |
| `lse_merge.merge_lse_states` | arXiv:1805.02867 §3.1 Eq.(4) + arXiv:2307.08691 §2.3.1 | ⊕ 在 (lse,output) 表示上:max_lse 稳定化 → e^{lse-max} → out_se=Σ → 权重=占比 → 加权合并 → lse=log(out_se)+max_lse。vLLM 落地:`vllm/v1/attention/ops/triton_merge_attn_states.py:L259-L322`(变量名 p_se/s_se/out_se/p_scale/s_scale/out_lse 与 Triton kernel 逐一对应;docstring 自引的 arXiv:2501.01005 §2.2 不在论文包内,推导只从论文包两文出发);双空护栏与「先算 scale 再乘 output」(NOTE(woosuk))同款遵守 |

## 取舍与实现说明(不发明论文没有的机制)

- **物理存储层被抽象掉**:论文的 HBM/SRAM 在此即 numpy 数组/局部变量——数值递推
  逐字保留,"load/write HBM"退化为数组切片读写;IO 复杂度由 `hbm_accesses_*` 显式
  记账补偿(可示教、可复算),不是丢失。
- **FA-2 line 10 方向修正(重要)**:arXiv:2307.08691 Algorithm 1 line 10 印作
  `O^{(j)} = diag(e^{m^{(j-1)}-m^{(j)}})^{-1} O^{(j-1)} + P̃^{(j)}V_j`,即旧账乘
  e^{m^{(j)}-m^{(j-1)}}(放大)——与论文自己的不变式 `Õ = Σ_j e^{S^{(j)}-m}V^{(j)}`
  (§3.1.1 两块表最终恒等式)方向相反,逐字转写会算错。实现按不变式方向(旧账乘
  e^{m^{(j-1)}-m^{(j)}} 折算到新 max,与 FA Alg.1 line 12 的 e^{m_i-m_new} 同向),
  测试对标准注意力逐位验证。explainer/writer 引用 line 10 时须带此修正说明。
- **causal 掩码**:FA 主文 Algorithm 0/1 不含 causal,但 FA-2 §3.1.1 "Causal
  masking" 一节明确讨论(j>i 置 -inf、整块在上侧直接跳过);seqlen_q≠seqlen_k 的
  右下对齐约定取自 vLLM `flash_attn_varlen_func` docstring 的两个掩码原例——
  `query_offset` 只是把因果掩码行索引平移(等价于先算好全序列位置再切片),不引入
  新算法。FA 前向(`flash_attention_forward`)保留逐块施掩码的朴素形态、FA-2 前向
  实现「整块跳过 + 跨对角块才掩码」两条优化——两者对任意分块逐位相等(测试断言),
  恰是「同一份数学、两种调度」的示教对照。
- **空段/全遮行的 -inf 约定**:FA2 对空序列返回 lse=inf、FA3 返回 -inf(vLLM
  Triton merge kernel L270-L276 做 inf→-inf 归一);本实现统一采用 -inf 约定。双空
  (两侧都 -inf,权重 0/0=NaN)按 vLLM kernel L319-L322 同款护栏输出 0、合并 lse
  保持 -inf。
- **`trace` 可选参数**:三个算法模块均支持,只逐格记录算法自身循环变量的快照
  ((m_j,d_j) 递推、(j,i) 块的 (m,ℓ,O)、FA-2 的未归一化 O 累加器、merge 的
  p_se/s_scale 记账)——供 explainer 产可示教轨迹,不是论文之外的新机制。
- **不实现**:backward/recomputation(推理章只需「重算比读表快」的结论)、dropout/
  滑窗/alibi/block-sparse(FA Alg.1 的可选项,vLLM 调用面在 dossier 站 6 已列)、
  分页 block_table 寻址(ch13/ch22 的工程内容,论文假设 K/V 连续)——均 out-of-scope
  (见 dossier.scope_note)。

## 测试(TDD:测试先于实现书写)

`tests/test_online_softmax.py`(Theorem 1 三版恒等/溢出对照/轨迹逐格/1≤d_j≤j/
⊕ 结合律交换律/分块乱序归并)、`tests/test_flash_attention.py`(Algorithm 0 行分布/
8K 感受数字/右下对齐两原例/FA 与 FA-2 对标准注意力逐位相等·五种分块/causal/
FA-2 causal 整块跳过≈1.78x 落在论文 1.7-1.8x 口径内/
「O_i 每步都是至今为止的正确答案」/LSE 闭式/IO 账方向与单调性)、`tests/test_lse_merge.py`
(两段合并=一次性/合并记账快照 p_scale+s_scale=1/cascade 前缀+后缀=完整因果/
三段结合律/空段/双空护栏)。
`conftest.py` 把 `implementation/` 加入 `sys.path`。

跑法(host,纯 CPU,无需容器):
```
cd instances/vllm/artifacts-v3/ch20-flash-attention-math
python -m pytest tests/ -q        # 46 passed
```

## 门禁与已知 WARN

```
python scripts/lint_paper_grounding.py instances/vllm/artifacts-v3/ch20-flash-attention-math --expect-primer
```
无 BLOCKING(实现侧每个 def 的 `# PAPER:` 锚在定义行 ±3 行内——长注释块因此都在
def 紧邻处补了一行短锚,沿用 v2 ch24 primer 的既有模式)。两条已知 WARN:
narrative 尚不存在(写作前正常);**论文包目录名不匹配**——linter 按章目录名推导
`book/papers/ch20-flash-attention-math`,而论文包实际在 `book/papers/
ch24-primer-flash-attention`(dossier.papers 明确指向),仅为 linter 启发式限制,
后续 Write/Review 站如见同款 WARN 按此理解(不改名论文包:meta.json 的 key_figures
登记都在那里)。
