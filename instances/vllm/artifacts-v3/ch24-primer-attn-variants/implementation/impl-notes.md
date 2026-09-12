# impl-notes — v3 ch24《【primer】注意力变体数学》

本章 `kind: primer`——不是 vLLM 源码的减法精简版,而是**论文忠实的小型参考实现**
(NumPy,纯 CPU,host `python -m pytest` 小参数即可全量跑完,用于 explainer 产出可示教
的数值轨迹)。每个 `def`/`class` 用 `# PAPER: §x Eq.y` 锚定论文出处(替代普通章节的
`# SOURCE:`);门禁为 `scripts/lint_paper_grounding.py`(lint_fidelity 不跑)。

论文包(真相源):`instances/vllm/book/papers/ch24b-primer-attn-variants/`——
`paper-mla.md`(arXiv:2405.04434 §2.1 + §3.1.2 超参 + App B.1/C/D,分隔线后附
DeepSeek-V3 arXiv:2412.19437 §2.1.1 重述注记)、`paper-gqa.md`
(arXiv:2305.13245 全文)、`meta.json`(key_figures 策展)。
**注**:发车指令写的 `paper.md` 不存在——实际论文包就是上述双文件(dossier
`papers` 字段为准,已核实),不构成缺口。

## 续跑说明(本次 Implement 站为完成中断运行)

本目录 implementation/tests 由上一次中断的运行留下:3 个实现文件与测试已在,但
`mla.py` 的 `_causal_softmax_weighted_sum` 有一处 einsum 下标错误(probs 与 v 的
头下标写作 `h`/`i` 两个自由指标,`"hts,sid->tid"` 把头求和掉,n_h=2 时输出恰放大
2 倍——3 条一致性测试红),且多个 def 的 `# PAPER:` 长注释块超出 linter ±3 行窗口、
`impl-notes.md` 缺失。本次:**修复 einsum**(头下标同名 `i` 且出现在输出里,只对
key 位置 s 求和)、**补齐 12 处短锚**、**补 m08/m01 两个 dossier 点名的 worked
example**(2×2 不交换律演示、Llama-2-7B 0.5MB/token 算例)、**补写本文件**。实现
主体(论文映射与结构)为前次工作,经本次逐条对论文复核无误,未重写。

## 文件清单

- `mha_gqa.py` —— arXiv:2405.04434 §2.1.1 Eq.(1)-(8)(标准 MHA 基线)+
  arXiv:2305.13245 §2.2(GQA-g 分组插值:GQA-1=MQA、GQA-H=MHA)与 §2.1
  (mean-pool 转换,uptraining 第一步)。MHA/GQA/MQA 是同一前向在 `num_kv_heads`
  一根参数轴上的三个点(对照 vLLM `QKVParallelLinear` 的 `total_num_kv_heads`)。
- `mla.py` —— arXiv:2405.04434 §2.1.2 Eq.(9)-(13)(低秩联合压缩 + query 侧同构)、
  §2.1.3 Eq.(14)-(19)(解耦 RoPE + 不交换律论证)、§3.1.2(超参 + 潜向量后
  RMSNorm 注)、App B.1(V2-Lite 不压 query)、App C Eq.(37)-(47)(计算序全公式,
  naive 展开 / 离线吸收两形态)。V3 §2.1.1 重述与 V2 数学一字不差(V3-1..V3-11)。
- `kv_cache_table.py` —— §2.1.4 Table 1 四机制每 token 元素总账 + §3.1.2 超参
  (576=512+64, V3 §4.2 同值)+ GQA §2.2 factor-H 句 + 2.25 组换算。

## 1:1 Paper Map(参考实现符号 ↔ 论文出处 ↔ 对应关系/取舍)

| 参考实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `mha_gqa.softmax_lastdim` | arXiv:2405.04434 §2.1.1 Eq.(7) Softmax_j | 数值稳定版(减行 max),与 Softmax 定义逐点相等(测试对拍手算) |
| `mha_gqa.split_heads` | §2.1.1 Eq.(4)-(6) | 逐头切分等式 `[q_{t,1};…;q_{t,n_h}]=q_t` 的向量化:(T, n·d_h)→(T, n, d_h) |
| `mha_gqa.kv_head_of_query_head` | arXiv:2305.13245 §2.2 | "divides query heads into G groups" 的连续分组约定 `idx // (H/G)`(HF/vLLM `repeat_kv` 同款);端点 G=H 恒等(MHA)、G=1 全共享(MQA),测试逐例锁定 |
| `mha_gqa.attention_forward` | §2.1.1 Eq.(1)-(8) + §2.2 Eq.(1)-(3) 三投影 → Eq.(4)-(6) 切头(K/V 按 num_kv_heads 切、组内共享)→ Eq.(7) 逐头因果 softmax(qᵀk/√d_h)·v → Eq.(8) W^O 拼回。`num_kv_heads=None` 默认 = num_heads(=vLLM `QKVParallelLinear.total_num_kv_heads=None` 语义,vllm/model_executor/layers/linear.py:L1070-L1072);因果性 = Eq.(7) 求和上限 j≤t;√d_h 缩放 = llama.py `self.scaling=head_dim**-0.5` |
| `mha_gqa.mha_to_gqa_mean_pool` | arXiv:2305.13245 §2.1 + §2.2 | uptraining 第一步:组头 K/V 投影矩阵 = 组内原头投影的算术平均(§2.2 "construct each group key and value head by mean-pooling all the original heads within that group";优于选单头/随机初始化);线性 ⇒ 先投再平均 == 先平均再投,测试断言 |
| `mla.rms_norm` | arXiv:2405.04434 §3.1.2 | "additional RMS Norm layers after the compressed latent vectors":x/√(mean(x²)+eps);γ 取单位权重(可学 γ 的参考退化);作用位置 c^Q/c^KV 下投影之后、上投影/缓存之前(=vLLM `q_a_layernorm`/`kv_a_layernorm`,vllm/model_executor/models/deepseek_v2.py:L1035/L1051) |
| `mla.rope_rotate` | §2.1.3 Eq.(14)-(15) RoPE(·) | 论文引用的标准算子(§3.1.4 θ=10000;RoPE 本身出自 Su et al. 2022,本论文包未定义配对布局);取 vLLM/Llama 半分式配对(前半×cos−后半×sin);性质由测试锁定:位置 0 恒等、范数保持、同位旋转内积不变 |
| `mla.MLAWeights` / `make_mla_weights` | §2.1.2 Eq.(9)-(13) + App C Eq.(37)-(47) | 论文八矩阵形状账本 W^DQ/W^UQ/W^QR/W^DKV/W^UK/W^KR/W^UV/W^O,与 vllm/model_executor/layers/attention/mla_attention.py:L44-L63 的 W_DQ…W_O 逐一同名同向(权重方向 out×in,前向算 x@W.T);`q_lora_rank=None` → App B.1 的 V2-Lite 分支(W^DQ/W^UQ 缺席,W^QC 直投代替、W^QR 改吃 h) |
| `mla.mla_naive_forward` | App C Eq.(37)-(47) | naive/展开形态,计算序逐式照排:(37)(38) 双下投影+RMSNorm → (39)(40) RoPE 只旋 R 段 → (41) 蓝框两向量 (c^KV, k_R) 即缓存 → (42)-(44) 三上投影把 k^C/v^C 从 c^KV 恢复 → (45)(46) 拼接 [C;R](k^R 所有头共享)→ (47) 逐头 Softmax(qᵀk/√(d_h+d_h^R))·v^C + W^O;返回 `(u, (c_KV, k_R))`——后者即每 token 写进 KV cache 的全部内容 |
| `mla.absorb_weights` | App C 吸收句(结合律,"offline at once") | 纯参数侧变换:逐头 B_i=(W_UK_i)^T·W_UQ_i("W^UK absorbed into W^UQ",把 query 投进潜空间)、W_O_absorbed=W^O·blockdiag(W^UV_1..H)("W^UV absorbed into W^O");下投影/共享 RoPE 路不动。vLLM 脚印 = process_weights_after_loading 的 W_UK/W_UV split 与 W_UK_T/W_UV 转置副本(vllm/.../mla_attention.py:L1027-L1035/L1092-L1100;运行时消费归 ch25) |
| `mla.mla_absorbed_forward` | App C 吸收形态 | c^KV 直接当单头 K/V 参与 attention(MQA 形状、潜向量内容):分数=[q̂_i·c^KV + q^R_i·k^R]/√(d_h+d_h^R)(R 段旁路不参与吸收),注意力权重作用于潜向量 e_i=Σ_j p_j·c^KV_j,u=W_O_absorbed·[e_1;…;e_H];与 naive 逐位恒等(测试断言,结合律) |
| `mla.absorption_score_three_orders` | §2.1.2 吸收论断("we even do not need to compute keys and values out") | dossier m07 worked example:同一分数 q^Cᵀk^C 的三种结合次序(朴素物化/吸收进潜空间/离线折叠)逐位相等;2×2 手算三连 -137 |
| `mla.noncommutativity_and_rope_coupling` | §2.1.3 不交换律论证("matrix multiplication does not obey a commutative law") | dossier m08 worked example:① 具体 2×2 矩阵 AB≠BA(直感);② 灾难版——若对 k^C 施 RoPE,可离线折叠的固定 B=(W^UK)^T W^UQ 变成 F(m,t)=(W^UK)^T R(m)^T R(t) W^UQ,随位置对变化、无单一离线形态(吸收失效;R(0)=I 时才退化回 B)。这是论文把位置信息解耦成旁路共享 k^R 的动机 |
| `kv_cache_table.mha/mqa/gqa/mla_kv_cache_per_token` | §2.1.1 尾句(2n_h·d_h·l)、§2.1.4 Table 1 四行、§2.1.3 尾句((d_c+d_h^R)l) | 四机制每 token 元素账,按元素数、不分精度(Table 1 caption 口径);谱系端点在账上合拢(GQA-H=MHA、GQA-1=MQA) |
| `kv_cache_table.mha_to_mqa_cache_reduction_factor` | arXiv:2305.13245 §2.2 factor-H 句 | MHA→MQA 的 cache 与加载量缩减倍数就是头数 H |
| `kv_cache_table.equivalent_gqa_groups` | §2.1.4 Table 1 caption | (d_c+d_h^R)/(2·d_h)=(4d_h+d_h/2)/(2d_h)=2.25——MLA 放回 GQA 谱系的定位换算 |
| `kv_cache_table.deepseek_v2_mla_hyperparams` / `table1_kv_cache_per_token` | §3.1.2(+ V3 §4.2 同值) | n_h=128/d_h=128/d_c=512/d_c'=1536/d_h^R=64 → 576=512+64 跨代稳定;代超参得 MHA 32768 / MQA 256 / GQA-8 2048 / MLA 576 |

## 取舍与实现说明(不发明论文没有的机制)

- **权重方向按论文记号**:W∈R^{out×in}(与 `nn.Linear.weight` 同向),前向统一算
  `x @ W.T`——转置方向手滑是静默数值错误的常见来源,测试用三重循环论文直译对拍
  兜底(rtol=1e-12)。
- **RMSNorm 的 γ 与 eps**:论文只说"additional RMS Norm layers",未给 γ/eps;
  参考实现取 γ=1(单位)、eps=1e-6——RMSNorm 的标准退化形态,不引入新机制。
  位置与 vLLM 一致(c^Q/c^KV 下投影后、上投影与缓存前)。
- **RoPE 配对布局是约定不是机制**:两篇论文都只引用 RoPE(·) 算子本身;实现取
  vLLM 半分式配对,并用性质测试(位置 0 恒等/范数保持/同位内积不变)锁定——这些
  性质对任意正交配对布局成立,不绑死具体约定。
- **GQA 组映射取连续分组**(Fig.2 画法/`idx // 组宽`):论文 §2.2 只说"divides
  query heads into G groups"未定排列;连续分组是 HF/vLLM 的通用约定,端点语义
  (G=1=MQA、G=H=MHA)与论文严格一致。
- **吸收形态的两个参数件**:(a) 逐头 B_i=(W_UK_i)^T·W^UQ_i;(b) W^UV 按头
  blockdiag 折进 W^O——都是 App C"absorb
  W^UK into W^UQ, and W^UV into W^O"的直接矩阵化,论文附录 C 精确口径(V2 正文
  的"吸进 W^Q"是简写)。
- **`trace` 可选参数**:三个前向都支持,只逐格记录算法自身中间量快照(q/k/v、
  逐头分数、probs、蓝框两向量、吸收态 q̂/e 等)——供 explainer 产可示教轨迹,
  不是论文之外的新机制。
- **不实现**(dossier.scope_note 的 out-of-scope,全部留后续章):forward_mha/
  forward_mqa 双路调度与 Sq/Skv 比值分流、混批 num_mha_tokens/num_mqa_tokens、
  chunked prefill workspace 与 merge_attn_states(ch25);DSV4 第三代 MLA
  (统一 head_dim=512、fp8_ds_mla 584B、compress_ratio、o_lora/o_groups——本论文
  包止于 V3,以源码为准,ch25 末节/ch28);NSA/DSA 索引器(ch26);FlashAttention
  kernel 数学(√d 缩放的 softmax 侧,ch20);PagedAttention 分页/账本接入(ch13/
  ch14);KV cache 6-bit 量化(§3.2.3,ch27)。
- **einsum 头下标修复(重要)**:中断运行遗留的 `"hts,sid->tid"` 把 probs 的头
  下标 h 与 v 的头下标 i 当作两个自由指标、把 h 求和掉——n_h=2 时全部输出恰放大
  2 倍(t=0 处 softmax 单元素权重应为 1,实测 o=2v)。修复为头下标同名且出现在
  输出(`"its,sid->tid"`,与 mha_gqa.py 同款)。explainer/writer 引用时以当前
  版本为准(absorbed==naive 逐位恒等已由测试锁定)。

## 测试(TDD:测试先于实现书写)

`tests/test_mha_gqa.py`(向量化前向 == 论文三重循环直译/None 默认=MHA/
GQA-H==MHA/GQA-1==MQA/中间组数对拍/连续分组映射/mean-pool 组内平均+线性等价/
uptrained 链路/m02 worked example 形状账(T=3,H=2,d_h=4)/每 token 元素账/
因果无未来泄漏)、`tests/test_mla.py`(naive == App C 逐式循环直译/分母
√(d_h+d_h^R)/缓存恰为蓝框两向量/**蓝框充分性:仅凭缓存 (c^KV,k^R)+新 token
的 h_t 逐 token 解码 == 整段前向**(tester 补)/位置平移只动 R 段/吸收三序
2×2 手算/吸收参数形状/吸收==naive 逐位/V2-Lite 分支跑通且可吸/RoPE 三性质
+**范数/同位内积统计检验(512/256 抽样,固定种子)**(tester 补)/
不交换律+RoPE 耦合演示)、`tests/test_kv_cache_table.py`(Table 1 四行代 DSV2
超参/层数因子 l/谱系端点合拢/factor-H/2.25 组/≈1.75% 口径(与 93.3%、14%-4%
区分)/576 宽度/Llama-2-7B 0.5MB 算例)。
`conftest.py` 把 `implementation/` 加入 `sys.path`。

跑法(host,纯 CPU,无需容器):
```
cd instances/vllm/artifacts-v3/ch24-primer-attn-variants
python -m pytest tests/ -q        # 36 passed
```

## 门禁与已知 WARN

```
python scripts/lint_paper_grounding.py instances/vllm/artifacts-v3/ch24-primer-attn-variants --expect-primer
```
无 BLOCKING(每个 def/class 的 `# PAPER:` 锚在定义行 ±3 行内——长注释块都在 def
紧邻处补了一行短锚,沿用 ch20 primer 的既有模式)。两条已知 WARN:
① narrative 尚不存在(写作前正常);② **论文包目录名不匹配**——linter 按章目录名
推导 `book/papers/ch24-primer-attn-variants`,而论文包实际在
`book/papers/ch24b-primer-attn-variants`(dossier.papers 与 meta.json 的
key_figures 登记都在那里),系 linter 启发式限制,与 ch20 同款;后续 Write/Review
站见同款 WARN 按此理解(不改名论文包:key_figures 策展在那边)。
