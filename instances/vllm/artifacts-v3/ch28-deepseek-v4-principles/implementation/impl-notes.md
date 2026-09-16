# impl-notes — v3 ch28《【primer】DeepSeek-V4 原理：七件套怎么还两本账》

本章 `kind: primer`——**不是 vLLM 源码的减法精简版**，而是**论文忠实的小型参考实现**
（NumPy、纯 CPU、host 直接跑，参数小到能心算）。每个 `def`/`class` 用 `# PAPER: §x Eq.y`
锚定论文出处（替代普通章的 `# SOURCE:`）；门禁 = `scripts/lint_paper_grounding.py`
（`lint_fidelity` 不跑，硬规则 2 的豁免仅限本章 kind）。

- **论文包（真相源）**：`instances/vllm/book/papers/ch28-deepseek-v4-principles/`（`paper.md`
  主料 arXiv:2606.19348 + V3.2/HC/mHC/DeepSeekMoE/HashLayers/V3 六篇谱系锚 + V4-Flash config 数字表）。
- **外部参考实现（只读蓝本，用户 2026-09-16 指定）**：**DeepSeek 官方的 DeepSeek-V4-Pro
  inference 代码**（`model.py` / `kernel.py` / `generate.py`）——论文作者自己的 Python 实现，
  且本来就是「小模型」形态（`n_layers=7`、`n_routed_experts=8`），正好当伪码蓝本。
  **引用纪律**（dossier `citation_discipline`）：注释里只写**标签形态**
  「官方参考实现（DeepSeek-V4-Pro inference/model.py:Lxxx）」（kernel.py / generate.py 同理），
  **绝不**写成 `vllm/...`（那是本仓 pin 的路径，官方代码不是它）、**绝不**写仓内脚手架路径或
  绝对盘符路径（脚手架泄漏）。引用 **pin** 时写规范 `vllm/...` 路径。与 pin 冲突处以 pin 为准
  （pin 是本书要读的代码；官方代码用于核对机制与做蓝本，两侧差异见 dossier difference_list）。
  **不得再引 HF transformers 的改写版实现**（用户明确要求以官方代码为准）。
- **运行**：`/d/Env/Miniconda/python -m pytest tests -q`（Windows Git Bash 下 `python3` 是
  WindowsApps 存根、exit 49，须用 Miniconda 的解释器）。当前：**96 passed**（78 条 implementer TDD + 16 条 tester 性质测试
  `test_paper_properties.py` + 本批回修新增 2 条：官方算子序列对账、c^Q 共用）。

## 文件清单

| 文件 | 覆盖机制（dossier id） | 内容 |
|---|---|---|
| `ledger.py` | m01 两本账 + m20 压缩坐标系（算术侧） | 584B/条、132B/条 IndexCache、656B(V3.2)、4096B(GQA8 基线) 的逐层加权账（三层型分布 + 滑窗常数窗 + 两个分母都在输出里）、每 query 实看条数 |
| `compressor.py` | m03/m04 CSA 软池化压缩 + m08 HCA 重压缩 + m20 坐标系 | Eq.(9)-(12) 四组投影 → 2m 窗 → 一次 softmax → 加权和（i=0 的 −∞/0 边界）；Eq.(20)-(23) HCA 单序列版；**官方参考实现两半区标签**的对账函数；跨 forward 续窗（`prior_a`）；攒批状态 `WindowAccumulator`（块尾 token 才产出条目） |
| `indexer.py` | m05 打分 + m06 块级 top-k 与因果 | Eq.(13)-(17)：低秩 q、逐头权重、ReLU 内积、top-k + −1 哨兵、短上下文全选；paper/impl 两种因果口径；K^IComp 复用同一套压缩 |
| `windowed_attention.py` | m07 共享 KV 的 MQA + m09 滑窗合成 + m10 RoPE 正/反旋 + m11 sink | Eq.(18) 的 `core_queries`（主 q 从**同一个** `c^Q` 升维，与索引器支路共用）+ Eq.(19)/(24)-(26) 的 CoreAttn（一条条目同时当 K 和 V，朴素 softmax + mask + sink）；滑窗范围与两路 KV 合成；RoPE 只动最后 rope_dim 维、输出侧反向旋转；`Exp(z)` 只进分母 |
| `mhc.py` | m12 更新式与双随机 + m13 动态参数化 + m14 Sinkhorn 与 t_max=20 + m15 出口 | Eq.(1)-(8)：展平（无权）RMSNorm → 一次 GEMM 出 Ã/B̃/C̃ → σ / 2σ / Sinkhorn-Knopp；双随机判定、谱范数、封闭性；逐轮收敛轨迹；`hc_head`（V4 自造出口） |
| `moe.py` | m16 两板斧 + m17 sqrtsoftplus/aux-loss-free + m18 hash 路由 | Eq.(3)-(11)：标准 MoE、细粒度计数、共享专家无条件项；Eq.(12)-(17) 两套均衡损失（谱系背景）；V4 的 sqrt(softplus) 亲和分、bias 只进选择、tid2eid 查表派单 |
| `mtp.py` | m19 MTP 模块与损失 | Eq.(21)-(25)：两个 RMSNorm 拼接 → 因果 TRM 块 → 共享输出头；下标对齐函数（`mtp_alignment`）、逐深度 CE、λ/D 加权（D=1、λ=0.3） |

测试：`tests/test_{ledger,compressor,indexer,windowed_attention,mhc,moe,mtp}.py` + `conftest.py`。

## TDD 记录（verification-before-completion）

1. **先写测试**（7 个模块的断言全部对齐论文断言的可观察后果）；跑 →
   `7 errors during collection`（`ModuleNotFoundError: compressor/indexer/...`）。
2. 逐个实现；中途 13 处红 → 逐条区分「实现错」还是「测试的期望值/口径错」：
   - **实现真错** 2 处：`mhc_update` 的 `C_l F_l(·)` 广播方向（改用 `einsum("i,tj->tij")`
     的外积，逐位对上 Eq.(1)）；RoPE 闭式例里 **测试**把首对频率写成 `1/θ`（应为
     `θ^0 = 1`，实现无误）。
   - 其余是**测试自身的口径/期望值**问题（如 softmax+sink 的期望值我只算了 6 位、
     `~valid[2:]` 切错段、`[0.9,0.2]/1.1` 列表除 float、Sinkhorn 的 eps 地板 ~1e-6 被我
     写成 `<1e-6`）——逐条改成实测值或改断言口径，**没有为了过测试而改实现**。
3. 终态：`78 passed`（本 implementer TDD 那一轮的计数；其后 tester 追加 16 条性质测试、
   本批回修再追加 2 条 ⇒ 全套 **96 passed**）。每处数值断言都先实测再落笔（Sinkhorn 收敛表、
   谱范数 1.009583、标签互换逐位等价、滑窗条数等）。

## 1:1 Paper Map（本实现符号 ↔ 论文出处 ↔ 对应关系/取舍）

| 本实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `compressor.csa_project` | arXiv:2606.19348 Eq.(9)(10) | `C^a=H·W^{aKV}`、`Z^b=H·W^{bZ}` 四式逐字；W 形状 (d,c) |
| `compressor.softmax_row` | arXiv:2606.19348 Eq.(11) | `Softmax_row` 跨 **2m** 个元素（逐列归一到 1）；数值上按列减最大值防溢出（等价变形） |
| `compressor.csa_entry_inputs` | arXiv:2606.19348 Eq.(11) | `[Z^a（当前窗）; Z^b（前一个窗）]` 堆叠 + 位置偏置；i=0 时 `Z^b→−∞`、`C^b→0` 逐字 |
| `compressor.csa_compress` | arXiv:2606.19348 Eq.(9)-(12) | 全式串起来；只产完整窗（不整除的尾巴不产条目，=pin 的 `(pos+1)%CR==0`） |
| `compressor.csa_entry_input_index_sets` | arXiv:2606.19348 §2.3.1 | 「C^b for i 与 C^a for i−1 重叠」⇒ 共享 m 个输入、条目数 n/m（不是 n/2m） |
| `compressor.csa_compress_reference_layout` | arXiv:2606.19348 Eq.(11)(12)（蓝本=官方参考实现 model.py 的 `Compressor`） | **两半区标签与论文相反**的对账件：前半=前一个窗、后半=当前窗（官方 L296/L302 的注释原话即此）；测试证明两套标签逐位等价 |
| `compressor.softpool_single_series` / `hca_compress` | arXiv:2606.19348 Eq.(22)(23) | 单序列软池化 = HCA：无重叠 ⇒ 条目数 n/m′；窗内 softmax 只跨 m′ 个元素 |
| `compressor.WindowAccumulator` | arXiv:2606.19348 §2.3.1（pin 形态） | 推理期压缩坐标系：`(pos+1)%m==0` 触发、条目序号 `pos//m`、未满窗的 token 留在 buffer |
| `indexer.indexer_queries` | arXiv:2606.19348 Eq.(13)(14) | `c^Q=h·W^{DQ}` → `q^I=c^Q·W^{IUQ}`；c^Q 与主注意力 Eq.(18) 共用（一次降维两处用） |
| `indexer.indexer_head_weights` | arXiv:2606.19348 Eq.(15) | 逐头标量权重 `w^I=h·W^w` |
| `indexer.index_scores` | arXiv:2606.19348 Eq.(16)（同形于 arXiv:2512.02556 Eq.(1)） | `Σ_h w_h·ReLU(q_h·k_s)` 逐字；**两个缩放默认关闭**（论文未写，两侧实现各补一个：官方 model.py:L395/L418、pin attention.py:L766；由调用方显式传） |
| `indexer.causal_candidate_count` | arXiv:2606.19348 Eq.(16) `s<Floor(t/m)` vs pin 口径 | 两种规则并存：`paper` = `t//m`、`impl` = `(t+1)//m`；只在块尾差一格（dossier m06），**不裁定谁对** |
| `indexer.select_topk_compressed` | arXiv:2606.19348 Eq.(17) | Top-k 选择 + 候选不足/past-causal 处填 **−1 哨兵**（与两侧实现同款） |
| `indexer.short_context_select_all` | arXiv:2606.19348 §2.3.1 + 两侧实现短路 | 论文无此条——候选不足时的必然形态；pin（attention.py:L842-L848）与官方（model.py:L409-L410）都显式短路 |
| `indexer.compressed_index_keys` | arXiv:2606.19348 §2.3.1 | 「same compression operation」⇒ 直接复用 CSA 压缩（列宽换成 c^I） |
| `windowed_attention.core_queries` | arXiv:2606.19348 Eq.(18) | 主注意力 q 从 `c^Q` 升维；**入参就是 `indexer.indexer_queries` 返回的同一个 `c^Q`**（论文的 shared latent，官方 model.py:L496-L499 传同一个 `qr` 进 L411）——一次降维两处用的显式落点 |
| `windowed_attention.core_attention_mqa` | arXiv:2606.19348 Eq.(19)(26) | 一条条目**同时当 K 和 V**（共享 KV 的字面含义）；mask 承载因果与选中集合；缩放默认 1/√d（论文只写 `CoreAttn(·)`，缩放是标准惯例、pin 同口径） |
| `windowed_attention.softmax_with_sink` | arXiv:2606.19348 §2.3.3 | `Exp(z′) 只进分母`：权重和 < 1、分子里没有对应 value；sink=−∞ 退化成普通 softmax（=pin 的初值语义） |
| `windowed_attention.sliding_window_slice` | arXiv:2606.19348 §2.3.3（式子来自 pin） | `start_pos=max(pos−n_win+1,0)`、含 query 自己；论文只给 n_win 机制不给数字（config 口径 128） |
| `windowed_attention.compose_attention_inputs` | arXiv:2606.19348 §2.3.3 | 滑窗条目 + 压缩条目并进同一次注意力；`has_compressor=False` 对应 ratio≤1 的纯滑窗层 |
| `windowed_attention.rope_forward/rope_inverse` | arXiv:2606.19348 §2.3.3 | 只动最后 rope_dim 维；输出侧 position −i 反旋（两个符号全翻，=pin 的 fused_inv_rope_fp8_quant、=官方 `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` + `freqs_cis.conj()`，官方参考实现 model.py:L534 与 L235-L237）；配对口径取 pin 的 `offsets^1`（相邻成对） |
| `windowed_attention.compressed_entry_positions` | arXiv:2606.19348 §2.3.3（pin 注释口径） | 条目带块级位置 `i·m`——条目代表整块，所以输出侧必须反旋回来 |
| `mhc.rms_norm_flat` | arXiv:2606.19348 Eq.(3) | `X̂=RMSNorm(vec(X))`；无权（官方 model.py:L676-L678 与 pin torch.py:L62-L65 都无增益）；两者把 rsqrt 折进 GEMM 之后（数学等价） |
| `mhc.mhc_raw_mappings` | arXiv:2606.19348 Eq.(3)(4)(5) | 一次 GEMM 出三组 raw 参数；**打包顺序是 (pre, post, res)**（官方 kernel.py:L391-L396 的切法），与论文列举顺序 (pre, res, post) 不同——测试里对账 |
| `mhc.sinkhorn_knopp` / `sinkhorn_trace` / `_start_matrix_and_ops` / `_normalize` | arXiv:2512.24880 Eq.(8)(9) / arXiv:2606.19348 Eq.(8) | `M^(0)=exp(H̃)`、`T_r`/`T_c` 交替、`t_max=20`；起点与轮内顺序都做成开关（`exp`/`softmax`/`raw` × `row-first`/`col-first`），因为论文书面顺序与两侧实现相反、而极限相同 |
| `mhc.mhc_gates` | arXiv:2606.19348 Eq.(6)(7)(8) | `A=σ(Ã)`、`C=2σ(C̃)`（2 = pin 硬编码 `hc_post_alpha`）、`B=Sinkhorn(B̃)`；σ 后加 eps = pin 的 hc_eps |
| `mhc.is_doubly_stochastic` / `spectral_norm` | arXiv:2512.24880 Eq.(6) + §4.1 | 三条件判定与「范数 ≤1」；反例（列和≠1 ⇒ 谱范数 >1）给出 1.009583 实测 |
| `mhc.doubly_stochastic_residual` / `mhc_layer_input` / `mhc_update` | arXiv:2606.19348 Eq.(1) | `X_{l+1}=B X + C·F(A X)` 三项各自成函数；**方向按论文写 `B X`**（两侧实现取 `combᵀ X`，注释已标） |
| `mhc.hc_head` | 无论文出处（V4 自造件） | sigmoid 门控加权求和压回单流；docstring 明写「mHC 论文没有这一步」，来源是官方 `ParallelHead.hc_head`（model.py:L728-L735，只有 σ）与 pin 的 `HCHeadOp` |
| `moe.moe_gate_scores` / `topk_gate` / `standard_moe_forward` | arXiv:2401.06066 Eq.(5)(4)(3) | 标准 MoE 三件套；`topk_gate(start=..)` 承载 Eq.(10) 的「只在共享专家之后挑」 |
| `moe.fine_grained_counts` / `segment_counts` | arXiv:2401.06066 Eq.(6)(7)(8) | (mN, mK) 与组合数（算力不变、组合更灵活） |
| `moe.shared_expert_output` / `deepseekmoe_forward` | arXiv:2401.06066 Eq.(9) | 第一项**没有 gate**（无条件）；完整式 = 共享项 + 路由项 + 残差 |
| `moe.expert_balance_loss` / `device_balance_loss` | arXiv:2401.06066 Eq.(12)-(17) | 谱系背景（V4 不再用它们作主力）；f 的归一化约定 Σf_i=N′ 有测试 |
| `moe.affinity_scores_v3` / `affinity_scores_v4` | arXiv:2606.19348 §2.1 | Sigmoid → **Sqrt(Softplus(·))** 的对照；softplus 不饱和、开根号拉平分布但不均匀 |
| `moe.topk_with_correction_bias` | arXiv:2606.19348 §2.1 + pin 注释 | bias 只进选择、权重从无偏分数 gather；`use_biased_weights=True` 是「注释点破的错误做法」对照组（熵更大=分布被压平） |
| `moe.hash_route` | arXiv:2106.04426 路由式 + arXiv:2606.19348 §2.1 | 按**原始输入 token id** 查表定专家（[vocab,topk] 整数表）；权重仍从无偏亲和分 gather |
| `mtp.rms_norm` / `mtp_input` | arXiv:2412.19437 Eq.(21) | 两个 RMSNorm **拼接**后过 `M_k`；docstring 点明 V4 把拼接换成两个投影相加（不同式子，别混讲） |
| `mtp.trm_block` | arXiv:2412.19437 Eq.(22) | 最小因果块（只保留能验证因果性的部分；论文把它写成黑盒，不发明内部结构） |
| `mtp.out_head` / `mtp_predict` | arXiv:2412.19437 Eq.(23) | 输出头与主模型共享 |
| `mtp.mtp_alignment` | arXiv:2412.19437 Eq.(23)(24) 的下标算术 | 显式化「模块算到 i+k≤T−1、损失只用到 i+k+1≤T−1」这一格边界 |
| `mtp.mtp_depth_loss` / `mtp_total_loss` | arXiv:2412.19437 Eq.(24)(25) | 逐深度 CE 取平均；总损失 `(λ/D)Σ`，默认 D=1、λ=0.3（论文超参） |
| `ledger.*` | arXiv:2606.19348 摘要 + §2.3.4（**算术件，不是机制实现**） | 把相对比例换成分母明确的绝对账；字节数与层分布来自 pin/config（docstring 已标），输出里同时给 `ratio_vs_gqa8` 与 `ratio_vs_v32` |

## 范围决定（不发明论文没有的机制）

- **不做工程件**：batch / TP / 分页 / KV cache / kernel / 量化（FP8·FP4 的字节数只作为
  ledger 的入参常数出现）。规模以「explainer 能跑出可示教轨迹」为度。
- **不做论文没写的机制**：mHC 的 `hc_head`（V4 自造件）与 `WindowAccumulator`（pin 的推理期
  形态）都不是新机制，注释里已标出处与性质；`sliding_window_slice` 的精确式子来自 pin，
  docstring 明写「论文只给机制不给数字/式子」。
- **不做完整模型装配**：不实现 43 层主干、不接 config、不加载权重（那是 ch29 的活）。
- **两处「论文未说明」保持诚实**：`compress_ratios` 为什么交替、`num_hash_layers` 为什么是 3 ——
  实现与注释都不给理由（只给结构性观察）。

## 与档案/蓝本对账时发现的差异（请 Lead/writer 知悉）

1. **dossier `implementation_blueprint` 的 MTP 例子有一格下标滑移**：蓝图写
   「P^1_2 = OutHead(h^1_1)」，而论文 Eq.(23) 是 `P^k_{i+k+1} = OutHead(h^k_i)` ⇒ k=1、i=1 给的是
   `P^1_3`（损失对齐 `t_{3:T+1}` = 蓝图自己后半句；蓝图前半句少算了一格）。**以论文为准**
   （`mtp_alignment` 已把这段下标算术显式化并有测试）。其余蓝本条目逐条核对**一致**。
2. **两本账的自算数与论文的 10% 不重合**（这是蓝图预期的口径差，不是错误）：默认口径（计
   IndexCache、不计滑窗）得 **89.5 B/token/层** ⇒ 对 GQA8 基线 **2.19%**（与论文 `~2%` 同量级 ✓）、
   对 V3.2 **13.6%**（论文说 10%）；不计 IndexCache 则对 V3.2 为 **11.2%**；把滑窗也计入是
   **92.7 B/token**（加账）。差异来源是账面口径（是否计 IndexCache/SWA 窗口、逐层算术平均 vs
   论文自己的分母），dossier m01 已预告「只作口径示范，结论数须复算」。**正文写这三个数时必须
   带分母并标「按 config 与代码字节数自算」**，不得当成论文原话。
3. **`attn_sink` 的默认行为取决于检查点**：pin 初值 −inf（`exp(−inf)=0`，等价无 sink，
   attention.py:L221-L224）；官方的 `nn.Parameter(torch.empty(...))`（model.py:L456）是
   **未初始化**的垃圾值，装载 checkpoint 后才有意义——**不是** 0、也不等价于"有 sink"。
   实现里两种都能表达（`sink=None`/`sink=-inf`），注释已标（dossier difference_list 第 2 条：
   不得把任一侧的初始化当语义）。

## 本批回修（Implement 站复核 —— 逐条对官方参考实现核锚点后修正）

复核方法：把每条「实现层」注释断言拿回**官方参考实现**（DeepSeek-V4-Pro inference/
model.py、kernel.py、generate.py）与本仓 pin 里逐条核，发现并修正下列问题（**机制与数值未动，
只修正注释/出处/命名**）：

1. **删掉全部 HF transformers 引用**（用户 2026-09-16 明确要求以官方代码为准、不得引 HF 改写
   版）。原注释把蓝本写成「HF transformers 的 DeepSeek-V4 实现」并抄了一批**两侧都不存在**的
   符号名——`overlap_kv`、`store_compression_weights`、`DeepseekV4UnweightedRMSNorm`、
   `DeepseekV4HyperHead`、`apply_rotary_pos_emb`、`collapsed`、`we drop the sink here`
   （逐个 grep 官方三件与本仓 pin：零命中）。已全部换成两侧真实符号与行号：`overlap_transform`
   + `kv_state`/`score_state`（official model.py:L302-L313）、`ape`（L294）、官方的 prefill
   余数切分（L327-L336）、`ParallelHead.hc_head`（L728-L735）、`layer_input`（pin
   mhc/torch.py:L84-L86）、`attn_sink` 进分母（official kernel.py:L345-L346）、`tid2eid`
   （official model.py:L559/L576-L577）。
2. **删掉一处伪造引文**：原 mhc.py 注释称「参考实现的注释原话 "equivalent to comb.T @
   residual"」——全仓 grep（pin）+ 官方三件 grep 均无此句。已改为陈述可核实的事实：两侧算的是
   `combᵀ X`（官方 L685 的 `sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)` 与 pin
   torch.py:L94-L106 的 einsum `"...ij,...ih->...jh"` 同一收缩；pin tilelang 核写成字面式
   `x[o,j] += comb[i,o] * res[i,j]`），而本实现按论文原式写 `B X`——双随机集合对转置封闭，
   两种读法都在 Birkhoff 多面体里。
3. **改掉一处事实错误**：原注释称「官方参考实现 `attn_sink` 初值 0 等价于有 sink」——官方是
   `nn.Parameter(torch.empty(...))`（model.py:L456）**未初始化**，不是 0。已改为 dossier
   difference_list 第 2 条的口径：谈默认行为只能写「取决于检查点权重」。
4. **补一条口径提醒**：`csa_entry_inputs(prior_a=...)` 收的是**未折位置偏置的原始投影**，而
   pin 的 state 在**写**的时候就折进了 `ape[position % compress_ratio]`
   （save_partial_states）——pin 的 state 不能原样当 `prior_a` 传（会重复加偏置）。原注释把两者
   说成同一件东西，已更正。
5. **补两条覆盖缺口（新增代码/测试，机制仍只来自论文）**：①`windowed_attention.core_queries`
   ——Eq.(18) 的 `q_t = c_t^Q·W^{UQ}` 原先只在测试里内联，现给它一个具名函数，好让「同一个
   `c^Q` 分叉两路」可被一行调用演示（m07 的核心论点）；②`tests/test_compressor.py` 新增
   `_official_compressor_prefill` + `test_official_operator_sequence_and_paper_form_agree_bitwise`
   ——把**官方 prefill 的算子序列**（unflatten + `overlap_transform` + `softmax(dim=1)` 一枪出）
   与论文逐式实现做逐位对账，覆盖「窗口↔条目对齐」这层风险（原有的标签对账只比了本实现的两个
   函数，属自洽而非独立）。
6. **补 MTP 的一手对照物**（dossier m19）：「直接丢弃」那条路在官方侧的字面证据是
   `n_mtp_layers=1`（model.py:L48）+ MTPBlock 存在（L738-L766）+ **推理脚本只调主干**
   （generate.py:L51）；「转投机解码」那条路在 pin 侧（nvidia/mtp.py + `_mtp_hidden_buffer`）。

## 给下游（explainer / writer / illustrator）的话

**可直接跑的示教产出（全部经实跑核过，见测试与 `explainer/traces/` 驱动脚本可复现）**：

| 机制 | 调用 | 实跑输出 |
|---|---|---|
| m03 CSA 压缩 | `csa_compress(H(12×3), …, m=4)` | 3 条条目；每条 8 个位置权重逐列和 = 1.0000；i=0 的 b 半区权重恰为 0 |
| m04 重叠窗 | `csa_entry_input_index_sets(4, 3)` | `[[0,1,2,3],[]] / [[4..7],[0..3]] / [[8..11],[4..7]]`——12 个 token ⇒ 3 条、相邻共享 4 个 |
| m05 索引器 | `index_scores(q(2×2), K(4×2), w=[0.7,0.3])` | `I=[1.55, 0.7, 0.0, 1.125]` ⇒ top-2 = 块 0、3（负相关被打成 0） |
| m06 因果 | `causal_candidate_count(7,4,'paper'/'impl')` | `1 / 2`（块尾差的那一格） |
| m07 共用 c^Q | `core_queries(*indexer_queries(h, W_DQ, W_IUQ, c_i), W_UQ, n_heads, c)` | 同一个 `c^Q` 分叉：索引器 `(n_h^I, c^I)`=（3,2）玩具 /（64,128）config、主注意力 `(n_h, c)`=（5,8）玩具 /（n_h,512）config |
| m08 HCA | `hca_compress(H(1024×4), …, m'=128)` | 8 条条目、每条只看自己窗内 128 个；每 query 全看 8 条（对照 CSA 的 min(512, n/4)） |
| m09 滑窗合成 | `compose_attention_inputs(pos, m, n_win)` | t=2/m=4/n_win=3 ⇒ 滑窗 3 条、压缩 0 条、KV 轴长 3；t=8 ⇒ 滑窗 3 + 压缩 2 = 5；n_win=128 时 t=8 ⇒ 9 + 2 = 11 |
| m10 RoPE | `rope_forward` → `rope_inverse` | 反旋后逐位回到原值（1e-10 内）；单独一步会改变数值；正反只差两个符号 |
| m11 sink | `softmax_with_sink([2.0,1.0], sink=1.5)` | 权重 `[0.5065, 0.1863]`、总质量 **0.6928** < 1（sink 拿走 0.307） |
| m12/m13 mHC | `is_doubly_stochastic` / `spectral_norm` / `mhc_raw_mappings` | `[[0.7,0.3],[0.3,0.7]]` ✓ 范数 **1.0000**；反例 `[[0.9,0.1],[0.2,0.8]]` ✗ 列和 1.1/0.9、范数 **1.009583**；封闭性 `B@B=[[0.58,0.42],[0.42,0.58]]` |
| m14 Sinkhorn | `sinkhorn_trace([[4,1],[1,3]], iters=3, start='raw')` | 逐轮 row_dev 0.02757 → 0.00840 → 0.00256、col_dev 恒 0（每轮末列归一）；20 轮后偏差 ~1e-6（= eps 地板）；条件数差的矩阵 20 轮仍有 6.4e-4、200 轮才到 1e-6 ⇒ **20 是实用值不是精确值** |
| m16 MoE | `deepseekmoe_forward(u, shared(2), routed(2), gates)` | 共享项与门控无关（换门控共享项一字不动）；V4 口径 256 挑 6 + 1 共享 |
| m17 路由 | `topk_with_correction_bias([.9,.8,.2,.1], [0,0,1,0], 2)` | 选择 = {0,2}（bias 把专家 3 推进 top-2）；无偏权重 `[0.818,0.182]` vs 带偏对照 `[0.571,0.429]`（**被压平**、熵更大） |
| m18 hash | `hash_route(table, tokens, scores)` | 选谁由 token id 钉死（换 hidden 不变）、权重现算 |
| m19 MTP | `mtp_alignment(T=5,k=1)` | hidden `[0,1,2,3]`、targets `[(0,2),(1,3),(2,4)]`（= 论文 1 基 `t_{3:T+1}`）；`mtp_total_loss([2.5]) = 0.75`（λ=0.3、D=1） |
| m01 两本账 | `kv_byte_account(V4_FLASH_COMPRESS_RATIOS[:43])` | 层分布 `{swaonly:2, c4a:21, c128a:20}`；逐型 `179 / 4.5625 / 0` B；**89.541 B/token/层**；对 GQA8 **2.186%**、对 V3.2 **13.65%**；计入滑窗 ⇒ 92.711 B/token（窗口总量 3,170,304 B 常数） |

**写作用到的三条口径纪律**（实现里已经落成参数/开关，正文引用时要带）：

1. **标签陷阱**：论文 a 半区=当前窗、b 半区=前一个窗；官方参考实现的两半区正好相反
   （前半=前一个窗、后半=当前窗，官方参考实现 model.py:L296/L302/L312-L313，pin 的压缩核
   `head_offset` 同序）。实现两者都给了（`csa_compress` / `csa_compress_reference_layout`），
   测试证明逐位等价——讲代码时**不要**把论文的 a/b 照抄到两半区上。
2. **因果差一格**：论文 `s<Floor(t/m)`、代码 `(t+1)//m`；差在「块尾 token 能否看自己那块」。
   两种都在实现里（`rule="paper"|"impl"`），正文按论文写公式、按代码讲口径，别断言等价。
3. **实现层补充要标注**：索引器的两个缩放（c^I^-1/2、n_h^I^-1/2）论文 Eq.(16) 未写；核心注意力
   的 1/√d 缩放、滑窗的精确式子、mHC 的 σ 后 eps、`hc_post_alpha=2.0`、raw 参数的 (pre,post,res)
   打包顺序、paged SparseMLA 独有的 `pos_after_compress%block_size` 槽位算术——这些都不是论文内容。

**给 explainer 的素材提示**：`sinkhorn_trace` 直接给 m14 收敛曲线的逐轮数据；`csa_compress`
返回的 `S` 就是 m03/m04 的「8 个位置 → softmax 权重 → 1 条 512 维」图的原始权重（玩具维度）；
`compose_attention_inputs` 的 dict 就是 m09 滑窗合成图的 kv_len 分解；`ledger.kv_byte_account`
的 `per_type` 是 m02 层堆叠图/分辨率三档图的账目来源。
