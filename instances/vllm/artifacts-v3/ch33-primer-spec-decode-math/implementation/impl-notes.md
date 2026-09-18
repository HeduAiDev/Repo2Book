# impl-notes — v3 ch33《【primer】投机解码数学+DSpark》

本章 `kind: primer`——不是 vLLM 源码的减法精简版,而是**论文忠实的小型参考实现**
(NumPy,纯 CPU,host `python -m pytest` 小参数即可全量跑完,供 explainer 产出可示教
的数值轨迹)。每个 `def`/`class` 用 `# PAPER: §x Eq.y` 锚定论文出处(替代普通
章节的 `# SOURCE:`);门禁为 `scripts/lint_paper_grounding.py`(lint_fidelity 不跑)。

论文包(真相源):`instances/vllm/book/papers/ch41-primer-dspark/paper.md`
(arXiv:2607.05147,包名 ch41 是 v2 章号、本章为 v3 ch33)。**⚠ 论文包两处
LaTeXML 截断**(§3.1 Eq.(4) 的 softmax 形式、RNN 头门控公式;App A 尾句)——
涉及处一律按 dossier 口径重构并逐处标注(见下「重构口径」)。

## 文件清单

- `spec_decode.py` —— §1/§2.1 验证数学:接受准则 min(1, p_t/p_d)(u 形态/对数
  形态两副姿态)+ 残差 norm(max(0,p_t−p_d)) 的 Gumbel-max 免归一化采样 +
  verify_block(accepted/recovered/bonus)+ greedy one-hot 退化 + 无损的经验检验
  (单位置分布、整流生成流 bigram)+ TV 恒等式/Eq.(8) 解析接受率 + Eq.(1) 延迟账。
- `acceptance_length.py` —— τ 账:前缀存活 a=∏c(§3.2.2)、E[τ]=1+Σa(§4.2
  fn.4 含 bonus 口径)、恒 α 闭式 (1−α^{γ+1})/(1−α)、条件⇄无条件互推
  (vLLM unconditional_to_conditional_rates 同构)、蒙特卡洛对拍。
- `dspark.py` —— §2.2 Eq.(2)(3) 并行骨干(上下文投影/KV 注入/双向注意/共享
  embed+lm_head,预计算+单次前向两段式)+ §3.1 半自回归(anchor-as-first 布局、
  Markov 头 Eq.(5)、序列采样 Eq.(4) 重构口径、多模态碰撞 worked example、低秩
  参数账)+ §3.2.1 Eq.(7) 置信头/Eq.(8) 标签。
- `scheduler.py` —— §3.2.1 STS 逐位温度缩放校准(ECE/保序)+ §3.2.2
  Algorithm 1 硬件感知前缀调度器(逐行对齐,含早停)+ Appendix A 回顾式
  反例(全局搜索 + 输出分布偏差 0.85/0.15 经验模拟)。
- `training.py` —— §3.3 Eq.(9)-(12) 三件套 + 位置权重 w_k + 有限差分梯度下降
  示教轨迹(L_tv ↓ ⇒ 接受率 ↑)。

## 1:1 Paper Map(参考实现符号 ↔ 论文出处 ↔ 对应关系/取舍)

| 参考实现符号 | 论文出处 | 对应关系 / 取舍 |
|---|---|---|
| `spec_decode.acceptance_probability` | §2.1 | 论文原式 min(1, p_t(x_k)/p_d(x_k));手算组 (0.7,0.3)/(0.5,0.5) → A:1、B:0.6 |
| `spec_decode.accepted` | §2.1 | u < p_t/p_d(乘法形态 u·q<p,免除零);min(1,·) 由 u∈[0,1) 隐式——vLLM V1 kernel L829 同构(`target_prob/draft_prob >= uniform_prob`) |
| `spec_decode.accepted_logspace` | §2.1 | log p(x) > log u + log q——vLLM V2 kernel L622-L625 的对数空间姿态;与概率形态逐点等价(测试对拍) |
| `spec_decode.residual_distribution` / `sample_recovered` / `gumbel_max_argmax` | §2.1 | r(x)=max(0,p_t−p_d);从 norm(r) 采样 ≡ argmax r(x)/E_x(E~Exp(1))——Gumbel-max 免归一化(vLLM V1 L913-L952 score=prob·inv_q 同构;分母 Z 不用算,测试锁死缩放不变性) |
| `spec_decode.verify_block` | §2.1 | 左到右逐位验证、首拒即停截断、拒绝位恢复、全收追加 bonus(只从 target 行采);γ+1 行 target 条件分布=「一次前向验证所有位」的玩具形态;术语 accepted/recovered/bonus=vLLM L38-L59 |
| `spec_decode.verify_block_greedy` | §2.1+§1 | one-hot q 退化:接受⟺draft==target argmax、拒绝位/bonus 直接取 target argmax ⇒ 输出≡target greedy rollout(vLLM V2 L564-L586/V1 L756-L757 形态;m17) |
| `spec_decode.empirical_output_distribution` / `generate_stream` | §1 | 无损断言("preserves the target distribution exactly")的可运行检验:单位置经验分布==p_t;整流流 bigram 条件分布==target 转移行;经验 τ=发射/周期 |
| `spec_decode.acceptance_rate_sum_min` / `analytical_acceptance_rate` | §3.2.1 Eq.(8) | TV 恒等式 Σmin(p,q)=1−½‖p−q‖₁ 的两实现互为对拍(测试锁死恒等);(0.5,0.5)vs(0.7,0.3)→0.8 与 m02/m11 手算闭环 |
| `spec_decode.latency_per_token` / `speedup_vs_plain` | §2.1 Eq.(1) | L=(T_draft+T_verify)/τ;基线延迟=T_verify(每 token 一次 target 前向)是 Eq.(1) 的直接推论——m01 worked example 数字(1ms/8ms/τ=3→3ms、2.67x;τ=1.2→7.5ms、1.07x)由测试锁定 |
| `acceptance_length.prefix_survival_probs` | §3.2.2 (Alg.1 L2) | a_{r,j}=∏_{i≤j}c_{r,i};c=[0.8,0.9,0.5]→[0.8,0.72,0.36](m13 手算);单调不增⇒贪心准入天然尊重前缀依赖(测试验证单调性) |
| `acceptance_length.expected_accepted_length(_constant)` | §2.1+§4.2 fn.4 | E[τ]=1+Σ_k a_k(1=恢复/bonus 的口径展开);恒 α 闭式=(1−α^{γ+1})/(1−α),α=0.8/γ=3→2.952(m04 手算);α→1 极限=γ+1 |
| `acceptance_length.unconditional_to_conditional_rates` / `conditional_to_unconditional_rates` | §3.2.2 | 链式法则两向;c_i=p_i/p_{i−1}(p_0=1)与 vLLM v1/spec_decode/utils.py:L598-L601 逐字同构(含前位 0→0 的除零守卫);往返恒等由测试锁定 |
| `dspark.ParallelBackbone.precompute_context` | §2.2 Eq.(2)+Eq.(3) 左半 | H_ctx=RMSNorm(W_c·拼接(target 多层隐状态))→逐层 K/V_ctx 预计算(vLLM precompute_and_store_context_kv 同构:「prefill 一次」的两段式拆分);W_c∈R^{d×m·d} 按论文方向 out×in |
| `dspark.bidirectional_attention_with_context` | §2.2 Eq.(3) | K_i=[W_i^K H_ctx; W_i^K H_d]、V 同构——上下文与块用**同一** W^K/W^V 沿序列维拼接;块内双向(位 0 输出依赖位 1,测试验证);softmax(qKᵀ/√d) 标准缩放算子(论文未给式,通用形态) |
| `dspark.ParallelBackbone.forward` | §3.1 | 单次前向:γ 输入→h_1..h_γ 与 U=h@lm_headᵀ(共享 target 的 embed/lm_head、冻结——构造器直接持有同一对象,测试验证引用同一);γ 任意、形状 [γ,V](T_draft 与 γ 无关的结构性来源) |
| `dspark.anchor_as_first_inputs` / `fill_in_inputs` / `query_layout` / `num_lookahead_slots` | §3.1(§2.2 对照) | anchor-as-first("minor modification"):γ 输入(anchor+γ−1 mask)出 γ logits、sample_off=0、target=query_off+1;DFlash:1+γ 输入、sample_off=1、anchor 是 bonus 槽;lookahead N vs N+1(vLLM dflash speculator L574-L585 的 SAMPLE_FROM_ANCHOR 分叉 + scheduler.py:L261-L270 的槽位账;m09 的 N=5 小表由测试锁定) |
| `dspark.MarkovHead.embed` / `.bias` | §3.1 Eq.(5) | B(x_{k−1},·)=W_1[x_{k−1}]W_2∈R^V;W_1∈R^{V×r} 查表、W_2∈R^{r×V} 投影(embed/bias 两步拆开=vLLM DSparkMarkovHead 的两方法,qwen3_dspark.py:L36-L78) |
| `dspark.markov_head_param_account` | §3.1 Eq.(5)+§4.3.2 | V=129280、r=256:全表 V²≈1.671e10、低秩 2Vr=66,191,360(各≈33.1M)、省 V/(2r)=252.5x(论文口径≈253x,取整差异在 impl-notes 备案);每步 O(r) 查表+O(rV) GEMV |
| `dspark.sample_sequential` | §3.1 Eq.(4)(重构口径) | p_k(x)∝softmax(U_k(x)+B(x_{k−1},x)) 左到右采样;prev 链 x_0=anchor、x_k=已采——串行依赖的全部实现就是 prev 赋值(vLLM _sample_sequential L100-L149 同构;greedy=argmax/one-hot q) |
| `dspark.dspark_draft` | §3.1+§3.2.1 | 完整 draft:并行骨干→序列采样→置信度;返回 dict 含 base_logits/hiddens/steps/confidences 全中间量(示教轨迹) |
| `dspark.marginal_over_predecessors` / `multimodal_collision_demo` | §3.1 | 碰撞形式化 p(x_k)=Σp(x_{k−1})p(x_k\|x_{k−1});双 mode 例:'of'(0.6)/'no'(0.4),x_2 边缘 problem 0.56>course 0.368 → 并行逐位 argmax='of problem';rank-1 Markov 偏置(+3/−3)修正为 'of course';并行独立采样连贯率解析 0.4448 vs DSpark≈0.99(m06 worked example) |
| `dspark.ConfidenceHead.confidence` | §3.2.1 Eq.(7) | c_k=σ(w·[h_k; W_1[x_{k−1}]]);吃骨干隐状态+Markov 嵌入(prev 链与采样共用);vLLM 边界:confidence_head 未接线(qwen3_dspark.py:L184-L188 skip)——论文侧蓝图 |
| `dspark.analytical_confidence_label` | §3.2.1 Eq.(8) | c*=1−½‖p_d−p_t‖₁,委托 spec_decode.analytical_acceptance_rate(与验证准则同一 TV 恒等式,单一实现) |
| `scheduler.hardware_aware_prefix_scheduler` | §3.2.2 Algorithm 1 | 逐行:L1-3 a=∏c→L4 按 a 降序→L5-6 初始化(B=R、τ*=R、Θ_best=R·SPS(R))→L7-15 贪心准入+早停(Θ≤Θ_best 即 break=非前瞻)→L16 返回;sps 为可调用(包装 dict 成本表=「profile 一次的轻量表」);单峰 Θ 下与全枚举对拍(测试 10 例);App A 数字(R=1、c=[0.8]→Θ_0=1.0>Θ_1=0.9→ℓ*=0)由测试锁定 |
| `scheduler.retrospective_global_search` / `retrospective_output_distribution` | Appendix A | 无早停全局搜索:Θ_2=(1+a_1+a_1c_2)·SPS(3);c_2=0.9→ℓ=2(Θ_2=1.134 全局最大)、c_2=0→ℓ=0;x_1=A 必被接受(min(1,0.7/0.5)=1)、x_1=B 不准入→target 重采 ⇒ P(Y=A)=0.5+0.5×0.7=0.85(经验模拟对拍;对照:恒准入的标准验证≡p_t)——m15 反例全套 |
| `scheduler.expected_calibration_error` / `temperature_scaled_confidence` / `sequential_temperature_scaling` | §3.2.1 | ECE=Σ(n_b/N)\|acc_b−conf_b\|;STS:位 k 冻结已校前缀、1D 网格搜温度最小化**累积乘积** ECE、保序(测试:过自信 z=2·logit(rate)→T≈2、校准后均值对齐真实 a、排序不变) |
| `training.position_weights` / `cross_entropy_loss` / `tv_loss` / `confidence_loss` / `total_loss` | §3.3 Eq.(9)-(12) | w_k=exp(−(k−1)/γ);L_ce/L_tv/L_conf 按式;总目标默认 α=(0.1,0.9,1.0);手算组全部由测试锁定(m19) |
| `training.train_toy_drafter` | §3.3 | 目标分布冻结、有限差分梯度下降:轨迹逐项记录 loss/L_tv/逐位接受率/E[τ]——「最小化 L_tv 直接最大化期望接受率」(Eq.(10) 下文断言)的可运行版 |

## 重构口径(论文包截断处,dossier 已核)

- **Eq.(4)**:paper.md:L95 句中断(softmax 形式与 p_k 记号均截断)。
  `sample_sequential` 的 p_k(x)∝softmax(U_k(x)+B(x_{k−1},x)) 由 Eq.(5)(偏置以
  **加法**注入低秩 B=W_1W_2)与 vLLM 落地(dspark/speculator.py:L118-L148
  `logits_i = base_logits[:, i] + bias`)双向确认重构。**writer 引用须带此口径**。
- **RNN 头**:门控公式整体缺失(paper.md:L109-L111);§4.3.2 结论仅长块边际
  增益、默认 Markov 头,vLLM 也只落地 Markov 头——参考实现不实现 RNN 头。
- **App A 尾句**(L849「Since Θ_1 …」截断):内容(Θ_1<Θ_0 时早停在 ℓ=0 前即
  停)由上下文可推出,体现在 Algorithm 1 的 break 语义里,无需重构公式。

## 取舍与实现说明(不发明论文没有的机制)

- **权重方向按论文记号**:W∈R^{out×in}(如 W_c∈R^{d×m·d}),前向统一
  `x @ W.T`;Eq.(2)(3)(5)(7) 全部如此,测试手工复算对拍。
- **RMSNorm 取 γ=1、eps=1e-6 的退化形态**:论文只引用算子未给参数(ch24
  primer 同款口径);作用于 Eq.(2) 投影后与骨干层间。
- **骨干层内结构是最小形态**(单头注意+残差+层间 RMSNorm、无 FFN/无 RoPE):
  论文包未给块内细节(DFlash 内部在外部论文 arXiv:2602.06036,未收录);论文
  **写明**的部分——Eq.(2) 投影、Eq.(3) 同 W^K/W^V 序列维拼接、块内双向、共享
  冻结 embed/lm_head、γ 输入出 γ logits——逐条实现,不多不少。
- **mask 槽**:论文只说「mask token embeddings」;实现取独立 mask_embed 向量
  (输入串约定 MASK=−1)。vLLM 用词表行当 mask(qwen3_dspark L184 注释)是
  工程变体,不影响机制。
- **采样器**:draft/bonus 用 categorical(论文只说「samples left to right」/
  「sampled from the target probabilities」);**残差采样特意用 Gumbel-max 免
  归一化形态**(§2.1 残差+vLLM kernel 的数学点,与 ch30 的 Gumbel-max 定理
  接上);u 一律 float64(vLLM 口径)。
- **玩具训练的优化器论文未指定**(训练细节在 DeepSpec 仓):取有限差分梯度
  下降——对目标函数本身最少的额外假设;每步从冻结 target 采一个 γ-token 块作
  L_ce 的 x*(§3.3「randomly sample multiple anchor positions」的玩具化:
  条件抽象为位置独立、上下文固定)。
- **多模态碰撞 demo 的构造**:base logits=边缘分布的对数(并行 drafter 每位
  学到的就是边缘——§3.1「marginalizes over all possible predecessors」),
  Markov 偏置换成等价的 rank-1 因子(B=W_1W_2 的最小实例),数字手算可验。
- **不实现**(vLLM 落地侧机制,写作直接引真实源码,不进参考实现):
  m05 drafter 谱系的方法表/分发表(config/speculative.py:L67-L77、
  spec_decode/__init__.py:L8-L40)、m16 落地拓扑(checkpoint 卫兵/强制 V2/
  registry)、m18 一次前向验证的布局算术(gpu_model_runner.py:L2851-L2924,
  docstring 自带数值例)、m20 block verification 旁路(rejection_sampler_
  utils.py:L535-L561,一句话点到归 ch34)、V1/V2 kernel 的 Triton 数值细节
  (对读锚已在各模块 docstring 备好)。

## 测试(TDD:测试先于实现书写,先红后绿)

`tests/test_spec_decode.py`(softmax/接受准则 u 形态与对数形态等价/残差恒等式/
Gumbel-max 缩放不变+统计对拍/verify_block 全收-首拒-截断-确定性恢复/无损单位置
40000 样本/整流流 bigram+经验 τ=3.545(α=0.92 解析对拍)/greedy 流≡target
greedy rollout/TV 恒等式/Eq.(1) 数字)、`test_acceptance_length.py`(a 手算/
E[τ]=2.952/通式=闭式/条件⇄无条件往返+除零边界/蒙特卡洛 30000 周期对拍)、
`test_dspark.py`(RMSNorm/Eq.(2) 投影/Eq.(3) 手工对拍+双向依赖/骨干形状任意
γ 单次前向/上下文影响/共享 embed+lm_head 同一对象/Markov Eq.(5)/低秩账
252.5x/置信头 Eq.(7)/布局 N=5 小表/序列采样 prev 链手推+统计/端到端 draft
交叉复算/边缘化/碰撞 demo 'of problem'→'of course'+连贯率 0.4448 vs >0.95)、
`test_scheduler.py`(Algorithm 1 App A 早停/因果版 vs 回顾版同场景/轻载全准入/
单峰下与全枚举对拍 10 例/a 单调/App A 全局搜索与分布偏差 0.85 对照 0.7/ECE
手算 0.26/STS 校准+保序)、`test_training.py`(位置权重/三损失手算/总目标
组合+默认权重/下降轨迹 τ 上升)。`conftest.py` 把 `implementation/` 加入
`sys.path`。

跑法(host,纯 CPU,无需容器):
```
cd instances/vllm/artifacts-v3/ch33-primer-spec-decode-math
python -m pytest tests/ -q        # 64 passed(tester 站补 6 条论文性质闸门后)
```

## 回修记录

- **round 2(2026-09-18)**:修 tester 站 F1 —— `training.py::_loss_of` 的
  `c_star` 曾被构造成 shape (γ,1)(内层多套一层 `[]`),`c_star[None]`=(1,γ,1)
  与 `c[None]`=(1,γ) 在 `confidence_loss` 里广播成外积 (1,γ,γ)、`np.sum(...,axis=1)`
  沿错误轴求和——L_conf 被静默改写成「全位置标签之和牵引每位置信」的畸变目标
  (Eq.(11) 要求逐位 BCE;对照数字:_loss_of=4.4999 vs 正确 Eq.(12)=3.6880,
  差 0.81 全来自畸变 L_conf)。修法两处:①`c_star` 改扁平 (γ,)(去内层 `[]`);
  ②`confidence_loss` 入口加 `assert c.shape == c_star.shape` 防回归
  (回归锁=test_paper_properties.py::test_toy_trajectory_loss_follows_eq12,
  固定 z/zc/tokens 逐值对照正确形状的 Eq.(12))。修后 64/64 全绿,
  `lint_paper_grounding --expect-primer` 无 BLOCKING。

## 门禁

```
python scripts/lint_paper_grounding.py instances/vllm/artifacts-v3/ch33-primer-spec-decode-math --expect-primer
```
无 BLOCKING(每个 def/class 的 `# PAPER:` 锚在定义行 ±3 行内)。已知 WARN:
narrative 尚不存在(写作前正常)、dossier paper_origin.sections 的个别小节串
(如「§3.2.2 Algorithm 1」)在论文包里非连续子串——paper_ref 为 WARN 级、
dossier 侧账目,不在实现修改范围。
