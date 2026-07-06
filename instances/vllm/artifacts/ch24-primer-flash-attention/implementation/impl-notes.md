# impl-notes — ch34 primer《FlashAttention:online-softmax 到 IO-aware 注意力》

本章 `kind: primer`——不是目标代码仓的减法精简版，而是**论文忠实的小型参考实现**
（NumPy，纯 CPU，小参数即可全量跑完，用于配合 explainer 产出可示教的数值轨迹）。
每个 `def`/`class` 用 `# PAPER: §x Eq.y` 锚定论文出处（替代普通章节的 `# SOURCE:`）。

## 文件清单

- `online_softmax.py` —— arXiv:1805.02867（online-softmax 论文）§2-3：naive → safe →
  online 三版 softmax 的收敛证明，以及 §3.1 Eq.4 的结合律 ⊕ 合并算子。
- `flash_attention.py` —— arXiv:2205.14135（FlashAttention）§2.2 Algorithm 0（标准注意力，
  物化 N×N）、§3.1 Algorithm 1（分块 tiling + online-softmax 递推）、§3.2 Theorem 2
  （IO 复杂度账）。
- `lse_merge.py` —— arXiv:1805.02867 §3.1 Eq.4（⊕ 算子）+ arXiv:2307.08691（FA-2）
  §3.1.1（logsumexp 定义 L=m+log(l)）：把 ⊕ 算子搬到 (lse, output) 表示上，对应 vLLM
  `merge_attn_states` 的数学原型；含 cascade attention（共享前缀+私有后缀两段合并）
  worked example，验证与"一次性整体因果注意力"逐位相等。

## 1:1 Paper Map（精简版 ↔ 论文出处 ↔ 说明）

| 参考实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `online_softmax.naive_softmax` | arXiv:1805.02867 §2 Eq.(1), Algorithm 1 | 两遍扫描，无 max 平移，逐字实现 Algorithm 1；用于对照证明会上溢/下溢 |
| `online_softmax.safe_softmax` | arXiv:1805.02867 §2 Eq.(2), Algorithm 2 | 三遍扫描（先 max 再 sum 再算 y），数值稳定；当前主流框架版本 |
| `online_softmax.online_softmax_stats` / `online_softmax` | arXiv:1805.02867 §3 Algorithm 3 lines 1-9 | 单遍融合 running (m,d) 递推；Theorem 1 保证末值与 safe softmax 恒等 |
| `online_softmax.online_softmax_merge` | arXiv:1805.02867 §3.1 Eq.(4) | ⊕ 二元算子逐字实现；FlashAttention tiling 与 vLLM LSE 合并的数学地基 |
| `online_softmax.combine_blocks_via_merge` | arXiv:1805.02867 §3.1 Eq.(3) | 分块局部 (m,d) 经 ⊕ 依次归并——worked example：分块合并==一遍遍历==三遍 safe softmax |
| `flash_attention.standard_attention` | arXiv:2205.14135 §2.2 Algorithm 0 | S=QK^T、P=softmax(S)、O=PV 三步，物化两张 N×N 中间矩阵；`causal`/`query_offset` 为 FA/FA-2 论文里讨论过的因果掩码约定的显式建模，非杜撰 |
| `flash_attention.fa_block_sizes` | arXiv:2205.14135 §3.1 Algorithm 1 line 1 | Bc=⌈M/4d⌉, Br=min(⌈M/4d⌉,d) 逐字实现 |
| `flash_attention.flash_attention_forward` | arXiv:2205.14135 §3.1 Algorithm 1 lines 2-16 | 外层 j 遍历 K,V 块、内层 i 遍历 Q 块，running (m,l,O) 递推；省略"HBM/SRAM"物理层（此处即 numpy 数组本身），保留全部数值递推；Theorem 1（正确性）用测试验证 |
| `flash_attention.hbm_accesses_standard` / `hbm_accesses_flash` | arXiv:2205.14135 §3.2 Theorem 2 证明梗概 | 按算法每行精确计数元素级访存次数（非仅渐近符号），用于 worked example 代入具体 N,d,M 算比值 |
| `lse_merge.attention_with_lse` | arXiv:2307.08691 §3.1.1（L=m+log(l)） | 一次性（非分块）算完一段注意力的 (O, lse)，供两段合并使用 |
| `lse_merge.merge_lse_states` | arXiv:1805.02867 §3.1 Eq.(4) | ⊕ 算子在 (lse,output) 表示下的实现，对应 vLLM `merge_attn_states_kernel`（`vllm/v1/attention/ops/triton_merge_attn_states.py:L118-L161`）落地面 |

## 取舍说明（不发明论文没有的机制）

- **不实现 backward / recomputation**：dossier/chapter 聚焦前向推理（vLLM 推理场景），
  两篇论文的 backward pass（Appendix B / §3.1.2）不在本章推导范围内。
- **不单独实现 FlashAttention-2 的分块前向**：`fa2-loop-order-warp` 机制在 dossier 里
  标 `needs_worked_example:false`、`difficulty:supporting`（"一节带过"），narrative 直接
  引用 dossier 里的论文摘录（`paper-fa2.md` Algorithm 1）讲解循环序对调/warp 分工，不需要
  额外参考实现；但 `lse_merge.py` 用到的 `L=m+log(l)` 定义（FA-2 §3.1.1 的第 2 条改进）
  确有落地在代码里，故单独起了 `attention_with_lse`。
- **causal 掩码的 `-inf` 边界**：整块被因果掩码全部遮住时 `-inf - (-inf) = nan`，用
  `_safe_exp`（IEEE 浮点安全包装，非算法机制）把它归零，两篇论文都未显式讨论这个数值边界，
  但这是忠实实现 Algorithm 1/§3.1.1 因果注意力时必然遇到的实现细节，不是杜撰。
- **`query_offset` 参数**：两篇论文都不含"两段拼接建模真实序列位置"这个概念本身，但
  cascade attention（`lse-merge`/`cascade-shared-prefix` 机制）需要它来在小参数下精确复现
  vLLM `flash_attn.py:L1145-L1236` 的前缀/后缀语义；`query_offset` 只是把因果掩码的行索引
  平移，数学上等价于"先算好整个序列的位置再切片"，不引入新算法。

## 测试

`tests/test_online_softmax.py`、`tests/test_flash_attention.py`、`tests/test_lse_merge.py`
（TDD：均先写断言目标行为，再补/验证实现）。全部由 `conftest.py` 把 `implementation/`
加入 `sys.path`，host `python3 -m pytest tests/` 即可跑（纯 CPU/NumPy，无需进容器）。

跑法：
```
cd instances/vllm/artifacts/ch34-primer-flash-attention
python3 -m pytest tests/ -q
```
28 passed（online_softmax 11 + flash_attention 12 + lse_merge 5，无 xfail/skip）。
