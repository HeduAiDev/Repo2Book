# ch42《【原理篇】FlashAttention:在线 softmax 与分块遍历》定稿——Part IX 原理篇

- **Type**: delivery
- **Chapter**: ch42
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T21:52:37Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: triton, part-9, primer, flash-attention, online-softmax, tiling, rescale, lse, causal, fa2

## What happened

Part IX 原理篇(primer 原理章)。回答"注意力为什么能不物化 N×N 打分矩阵":(1)动机——朴素 attention 把 QKᵀ 的 N×N 打分矩阵整个摆上显存,O(N²)、访存受限、长序列爆;(2)在线 softmax——先讲标量版:一遍扫过序列维护 running max m_j 与 running sum l_j,见更大 max 就把旧分母乘降标度因子 e^{m_old−m_new} 补账,数值等价先求全局 max 的三遍法;再升级到 attention 版三件套 m_i/l_i/acc 联合更新——不只分母,加权 V 累加器 acc 也随 running max 刷新同乘 alpha=exp2(m_i−m_ij);(3)rescale 恒等性(exact)是命门:alpha 同乘 acc/l_i 令分块增量严格等于全矩阵 softmax,漏乘即错(worked example 偏 15.8%/25.3%);(4)分块遍历骨架:外层每 program 锁一块 Q 常驻 SRAM(回指 ch34)、内层逐块流过 K/V 增量更新,台面只一个 BLOCK_M×BLOCK_N,QKᵀ/PV 走 tl.dot(回指 ch27);(5)exp2+预乘 1/ln2 全程基-2,等价 exp 但命中 GPU 原生快指令;(6)epilogue 延迟一次归一化 acc/l_i + LSE(m_i+=log2(l_i))存标量给反向;(7)causal 分块 mask 两趟拆分(STAGE 3/1:off-band 整块跳过 mask、on-band 对角块逐元素判断);(8)为什么快+省:running 状态全 O(block) 与 N 无关、融一个 kernel 使 HBM IO O(N²)→O(N);(9)FA-2 并行改进(每 Q 块一 program、序列块维铺 grid)。A 档 tutorials/06-fused-attention 逐行对上(_attn_fwd/_attn_fwd_inner)。primer 章:skip_impl(无精简版接口)、走 lint_paper_grounding、embed_excerpts 双源(tutorials/06 逐字代码 + paper.md 公式,均带锚)。诚实边界:论文数学部分待核处按 arXiv:2205.14135/2307.08691 回指、不编造硬件坐标/下界公式。本章无伏笔埋/回收(bible.py due ch42 确为空)。逐机制(10 个 m01–m10)覆盖全绿,reviewer APPROVED。质量事件:归档前 Lead 修 dossier paper_origin.sections(占位→真 paper.md 小节)+ figure manifest 版本 25.2→25.3;writer 补 4 处(m03/m07 直觉说明 + §3.3 exp2 量纲 + §3.4 causal 施 mask 顺序)。archivist 为本轮唯一 bible 写入者、无竞态(ch39 workflow 在跑但其 archive 由 Lead 延后串行)。glossary +11(467→478),concepts +11(300→311),figures +5(71→76)。

## Why it matters

本章是 ch43 收官实战的原理地基——ch42(原理:数学恒等性 why)→ch43(实战:tutorials/06 逐行 how)是显式前向依赖,读者先在此吃透在线 softmax+rescale 恒等性,再去啃实战 kernel 才不硬啃。也是全书"不物化中间矩阵"这条省显存主线的收束点:把前面 tl.dot(ch27)/共享内存常驻(ch34)/softmax reduce(ch08)/流水线(ch29/ch30)几条线在一个真实高价值 kernel 里合流。primer 高密度、数学主角、点透 rescale 恒等性这一个命门。

## What to remember

Part IX 原理篇(primer 原理章)。回答"注意力为什么能不物化 N×N 打分矩阵":(1)动机——朴素 attention 把 QKᵀ 的 N×N 打分矩阵整个摆上显存,O(N²)、访存受限、长序列爆;(2)在线 softmax——先讲标量版:一遍扫过序列维护 running max m_j 与 running sum l_j,见更大 max 就把旧分母乘降标度因子 e^{m_old−...
