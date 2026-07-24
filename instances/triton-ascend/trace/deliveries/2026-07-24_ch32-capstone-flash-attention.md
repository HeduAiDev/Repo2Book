# ch32 capstone flash-attention CV 融合回望

- **Type**: delivery
- **Chapter**: ch32
- **Date**: 2026-07-24
- **Timestamp**: 2026-07-24T11:34:12Z
- **Agents involved**: writer, reviewer, illustrator, archivist
- **User present**: False
- **Tags**: capstone, flash-attention, online-softmax, cube-vector, 回望

## What happened

Part VII「度量·实战」第 1 章(deps=ch08,ch25)、kind=meta 回望章：不引入新机制，拿仓库自带 365 行真实 Flash Attention v2 实现(third_party/ascend/tutorials/06-fused-attention.py)当活体标本，把前 30 章讲过的每一层在同一个真核上从头串一遍。核心内容：①内循环 Cube→Vector→Cube 三段心跳——两处 tl.dot(QK^T 第90行/PV 第120行)落 cube、softmax(max/exp/sum/alpha 重标定)落 vector，判据即 ch16 核亲和分析；②在线 softmax 递推——三累加器 m_i/l_i/acc 靠重标定因子 alpha=exp(m_i-m_ij)把新旧两块换算到同基相加，峰值显存 O(N^2)→O(N·d)，附可心算小例子(块0/块1/收尾表)与归纳法正确性证明；③STAGE 位掩码——一个整数两个二进制位穷尽编码 off-band/on-band/全序列三种因果情形，4-STAGE 做外层到内层的转换；④block_ptr 六元组是语言层显式搬运在真核上的活例；⑤持久化网格——num_cores=20 钉死物理核数 + grid-stride 循环领逻辑块，09-persistent-matmul.py 印证非本例特例；⑥大 head 维分片(acc 搬出片上放 GM workspace 按 1/4 BLOCK_M 分片)与 tile_mix_cube_num 编译提示(CV 融合内存治理的工程折痕)；⑦全链回望——六层剖面(语言层/ttadapter/核亲和/HFusion/HIVM/AscendC)逐一对回同一段源码的具体行号；⑧诚实边界——host 无 NPU/CANN，正文数值用 numpy 严格复现源码算术并与一次性 softmax 对拍(最大差 0.0)，测试夹具全部 causal=False、因果两趟拆分尚未被真机对拍覆盖(新埋伏笔 f8→ch33)。5 张机制图(fig-cube-vector-heartbeat/fig-online-softmax-evolution/fig-causal-stage-tiling/fig-persistent-grid-stride/fig-full-descent-chain)+ roadmap + chapter-map，全部 blind review 通过，16 项 linter + algorithm-pedagogy 专项评审均绿，verdict APPROVED(reviewer 提 9 条非阻断建议：2 处代码块行号 off-by-one、1 处数量级近似值偏差、1 处缺配图建议、2 处公式密度警告、3 处 reader-comprehension 顺序/心算完备性/术语对应问题——均已记入 review-report.json 供后续可选定点小修)。Bible 回写：glossary 新增 9 条术语(Flash Attention v2/在线 softmax/running max·sum/alpha 重标定/logsumexp/因果注意力/STAGE 位掩码/持久化网格/tile_mix_cube_num/npu_fusion_attention)，concepts.json 新增 7 条机制登记 ch32，figures.json 新增 5 条机制图登记，arc-map.json 新增 f8(测试覆盖边界，plant ch32→payoff ch33)。

## Why it matters

capstone 章的价值不在引入新知，而在证明「30 章讲的机制协同起来是什么样」——用同一个读者熟悉的真实算子把语言层/分水岭下降/核亲和/HFusion/HIVM 串成一条看得见的因果链，是全书 Part VII 收官前的关键验证点。诚实边界与新埋伏笔(测试覆盖不完整)延续了全书「交叉验证靠什么、证明不了什么」的一贯纪律，为下一章(收官/边界讨论章)做铺垫。

## What to remember

ch32 capstone 回望章 APPROVED，无阻断问题；9 条非阻断评审建议待可选定点小修；新埋伏笔 f8(因果掩码测试覆盖边界)plant ch32→payoff ch33；Bible(glossary/concepts/figures/arc-map)已回写。
