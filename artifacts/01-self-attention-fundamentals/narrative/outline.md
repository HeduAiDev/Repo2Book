# Chapter 1 Outline — Self-Attention 算子深度解析

## Narrative Strategy
- 起点：0基础（"一个词的意思取决于上下文"——小学生都懂）
- 路径：QKV直觉 → 单头attention → 数学推导 → 方差分析 → Multi-Head → GQA → Triton kernel
- 终点：Triton fused attention kernel的shared memory分配策略 + 与FlashAttention的对比
- 节奏：前1.1让零基础读者不会掉队，1.3开始加速，1.4达到HPC深度

## Sections

### 1.1 Scaled Dot-Product Attention: 从"上下文"到数学
- Cell 2 (Hook): "Attention Is All You Need"这个论文标题为什么不是夸张
- Cell 3 (Problem): 没有attention的世界——每个词是孤岛
- Cell 4 (Theory Part 1): Q/K/V的直觉定义——开组会比喻
- Cell 4 (Theory Part 2): 数学推导——Step 1到Step 4
- Cell 4 (Theory Part 3): 方差分析——为什么是√d_k（运行variance_analysis.py）
- Cell 4 (Viz): Q·K方差随d_k增长的图表
- 关键公式：Attention(Q,K,V) = softmax(QK^T/√d_k)V

### 1.2 Multi-Head Attention: 线性代数的美
- Cell 5 (Walkthrough): 为什么要多头——一个头只能学一种关系
- Cell 5 (Math): 低秩分解视角——每个头是d_model空间的一个d_k维子空间投影
- Cell 6 (Implementation): MultiHeadAttention类完整代码走读
- 关键公式：head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

### 1.3 GQA/MQA: KV Cache的压缩哲学
- Cell 5 (Walkthrough): 从MHA到GQA到MQA——KV压缩的演进
- Cell 5 (Analysis): KV Cache的数学公式——GQA的节省量
- Cell 6 (Implementation): GroupedQueryAttention代码
- 量化对比：MHA vs GQA vs MQA的参数量、cache大小、精度

### 1.4 Triton Fused Attention Kernel: HPC入门
- Cell 5 (Walkthrough): 朴素attention的HBM读写问题——[seq²]矩阵不存在
- Cell 5 (Algorithm): Online Softmax的伪代码
- Cell 5 (Memory): Block划分的SRAM使用量计算
- Cell 6 (Implementation): _fused_attention_kernel完整走读——逐段解释
- Cell 7 (Benchmark): Triton kernel vs PyTorch vs FlashAttention性能对比

### 1.5 Attention Masks: 控制"谁能看谁"
- Cell 5: Causal/Padding/SlidingWindow三种mask的实现与适用场景
- Cell 6: mask代码
- Cell 7: 三种mask可视化

### Cell 9: Source Mapping Table
### Cell 10: 11个PASSED测试
### Cell 11: Summary → 引出Chapter 2 (KV Cache: 既然Attention每次重算QK^T太浪费，为什么不存起来？)
