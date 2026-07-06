FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
==============================================================================================================================================

 Tri Dao Department of Computer Science, Princeton University Department of Computer Science, Stanford University 

###### Abstract

Scaling Transformers to longer sequence lengths has been a major problem in the last several years, promising to improve performance in language modeling and high-resolution image understanding, as well as to unlock new applications in code, audio, and video generation. The attention layer is the main bottleneck in scaling to longer sequences, as its runtime and memory increase quadratically in the sequence length. FlashAttention [5] exploits the asymmetric GPU memory hierarchy to bring significant memory saving (linear instead of quadratic) and runtime speedup (2-4$\times$ compared to optimized baselines), with no approximation. However, FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes. We propose FlashAttention-2, with better work partitioning to address these issues. In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory. These yield around 2$\times$ speedup compared to FlashAttention, reaching 50-73% of the theoretical maximum FLOPs/s on A100 and getting close to the efficiency of GEMM operations. We empirically validate that when used end-to-end to train GPT-style models, FlashAttention-2 reaches training speed of up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization).111FlashAttention-2 is available at https://github.com/Dao-AILab/flash-attention

1 Introduction
-----------------------------------------------------------

Scaling up the context length of Transformers [18] is a challenge, since the attention layer at their heart has runtime and memory requirements quadratic in the input sequence length. Ideally, we would like to go beyond the standard 2k sequence length limit to train models to understand books, high resolution images, and long-form videos. Just within the last year, there have been several language models with much longer context than before: GPT-4 [12] with context length 32k, MosaicML’s MPT with context length 65k, and Anthropic’s Claude with context length 100k. Emerging use cases such as long document querying and story writing have demonstrated a need for models with such long context.

To reduce the computational requirement of attention on such long context, there have been numerous methods proposed to approximate attention [9, 14, 19, 8, 4, 2, 20, 3]. Though these methods have seen some use cases, as far as we know, most large-scale training runs still use standard attention. Motivated by this, Dao et al. [5] proposed to reorder the attention computation and leverages classical techniques (tiling, recomputation) to significantly speed it up and reduce memory usage from quadratic to linear in sequence length. This yields 2-4$\times$ wall-clock time speedup over optimized baselines, up to 10-20$\times$ memory saving, with no approximation, and as a result FlashAttention has seen wide adoption in large-scale training and inference of Transformers.

However, context length increases even more, FlashAttention is still not nearly as efficient as other primitives such as matrix-multiply (GEMM). In particular, while FlashAttention is already 2-4$\times$ faster than a standard attention implementation, the forward pass only reaches 30-50% of the theoretical maximum FLOPs/s of the device (Fig. 5), while the backward pass is even more challenging, reaching only 25-35% of maximum throughput on A100 GPU (Fig. 6). In contrast, optimized GEMM can reach up to 80-90% of the theoretical maximum device throughput. Through careful profiling, we observe that FlashAttention still has suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes.

Building on FlashAttention, we propose FlashAttention-2 with better parallelism and work partitioning to address these challenges.

1. 1.
    In Section 3.1, we tweak the algorithms to reduce the number of non-matmul FLOPs while not changing the output. While the non-matmul FLOPs only account for a small fraction of the total FLOPs, they take longer to perform as GPUs have specialized units for matrix multiply, and as a result the matmul throughput can be up to 16$\times$ higher than non-matmul throughput. It is thus important to reduce non-matmul FLOPs and spend as much time as possible doing matmul FLOPs.
2. 2.
    We propose to parallelize both the forward pass and backward pass along the sequence length dimension, in addition to the batch and number of heads dimension. This increases occupancy (utilization of GPU resources) in the case where the sequences are long (and hence batch size is often small).
3. 3.
    Even within one block of attention computation, we partition the work between different warps of a thread block to reduce communication and shared memory reads/writes.

In Section 4, we empirically validate that FlashAttention-2 yields significant speedup compared to even FlashAttention. Benchmarks on different settings (with or without causal mask, different head dimensions) show that FlashAttention-2 achieves around 2$\times$ speedup over FlashAttention, reaching up to 73% of the theoretical max throughput in the forward pass, and up to 63% of the theoretical max throughput in the backward pass. When used end-to-end to train GPT-style models, we reach training speed of up to 225 TFLOPs/s per A100 GPU.

2 Background
---------------------------------------------------------

We provide some background on the performance characteristics and execution model of GPUs. We also describe the standard implementation of attention, as well as FlashAttention.

### 2.1 Hardware characteristics

GPU performance characteristics. The GPU consists of compute elements (e.g., floating point arithmetic units) and a memory hierarchy. Most modern GPUs contain specialized units to accelerate matrix multiply in low-precision (e.g., Tensor Cores on Nvidia GPUs for FP16/BF16 matrix multiply). The memory hierarchy comprise of high bandwidth memory (HBM), and on-chip SRAM (aka shared memory). As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [7, 6]. As the L2 cache is not directly controllable by the programmer, we focus on the HBM and SRAM for the purpose of this discussion.

Execution Model. GPUs have a massive number of threads to execute an operation (called a kernel). Threads are organized into thread blocks, which are scheduled to run on streaming multiprocessors (SMs). Within each thread blocks, threads are grouped into warps (a group of 32 threads). Threads within a warp can communicate by fast shuffle instructions or cooperate to perform matrix multiply. Warps within a thread block can communicate by reading from / writing to shared memory. Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

### 2.2 Standard Attention Implementation

Given input sequences ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ where $N$ is the sequence length and $d$ is the head dimension, we want to compute the attention output $\mathbf{O} \in {\mathbb{R}}^{N \times d}$:

| | | |
|-----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| | $${{\mathbf{S} = {\mathbf{Q}\mathbf{K}}^{\top} \in {\mathbb{R}}^{N \times N}},{\mathbf{P} = {{softmax}\hspace{0pt}{(\mathbf{S})}} \in {\mathbb{R}}^{N \times N}},{\mathbf{O} = {\mathbf{P}\mathbf{V}} \in {\mathbb{R}}^{N \times d}}},$$ | |

where $softmax$ is applied row-wise.222For clarity of exposition, we omit the scaling of ${\mathbf{Q}\mathbf{K}}^{\top}$ (typically by $1/d$), and optionally elementwise masking on $\mathbf{S}$ and/or dropout applied to $\mathbf{P}$ For multi-head attention (MHA), this same computation is performed in parallel across many heads, and parallel over the batch dimension (number of input sequences in a batch).

The backward pass of attention proceeds as follows. Let ${\mathbf{d}\mathbf{O}} \in {\mathbb{R}}^{N \times d}$ be the gradient of $\mathbf{O}$ with respect to some loss function. Then by the chain rule (aka backpropagation):

| | | | |
|-----|------------------------|-----------------------------------------------------------------------------------------|-----|
| | $\mathbf{d}\mathbf{V}$ | $= {\mathbf{P}^{\top}\hspace{0pt}{\mathbf{d}\mathbf{O}}} \in {\mathbb{R}}^{N \times d}$ | |
| | $\mathbf{d}\mathbf{P}$ | $= {\mathbf{d}\mathbf{O}\mathbf{V}}^{\top} \in {\mathbb{R}}^{N \times N}$ | |
| | $\mathbf{d}\mathbf{S}$ | $= {{dsoftmax}\hspace{0pt}{({\mathbf{d}\mathbf{P}})}} \in {\mathbb{R}}^{N \times N}$ | |
| | $\mathbf{d}\mathbf{Q}$ | $= {\mathbf{d}\mathbf{S}\mathbf{K}} \in {\mathbb{R}}^{N \times d}$ | |
| | $\mathbf{d}\mathbf{K}$ | ${= {\mathbf{Q}\mathbf{d}\mathbf{S}}^{\top} \in {\mathbb{R}}^{N \times d}},$ | |

where $dsoftmax$ is the gradient (backward pass) of softmax applied row-wise. One can work out that if $p = {{softmax}\hspace{0pt}{(s)}}$ for some vector $s$ and $p$, then with output gradient $d\hspace{0pt}p$, the input gradient ${d\hspace{0pt}s} = {{({{{diag}\hspace{0pt}{(p)}} - {p\hspace{0pt}p^{\top}}})}\hspace{0pt}d\hspace{0pt}p}$.

Standard attention implementations materialize the matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM, which takes $O\hspace{0pt}{(N^{2})}$ memory. Often $N \gg d$ (typically $N$ is on the order of 1k–8k and $d$ is around 64–128). The standard attention implementation (1) calls the matrix multiply (GEMM) subroutine to multiply $\mathbf{S} = {\mathbf{Q}\mathbf{K}}^{\top}$, writes the result to HBM, then (2) loads $§$ from HBM to compute softmax and write the result $\mathbf{P}$ to HBM, and finally (3) calls GEMM to get $\mathbf{O} = {\mathbf{P}\mathbf{V}}$. As most of the operations are bounded by memory bandwidth, the large number of memory accesses translates to slow wall-clock time. Moreover, the required memory is $O\hspace{0pt}{(N^{2})}$ due to having to materialize $\mathbf{S}$ and $\mathbf{P}$. Moreover, one has to save $\mathbf{P} \in {\mathbb{R}}^{N \times N}$ for the backward pass to compute the gradients.

### 2.3 FlashAttention

To speed up attention on hardware accelerators such as GPU, [5] proposes an algorithm to reduce the memory reads/writes while maintaining the same output (without approximation).

#### 2.3.1 Forward pass

FlashAttention applies the classical technique of tiling to reduce memory IOs, by (1) loading blocks of inputs from HBM to SRAM, (2) computing attention with respect to that block, and then (3) updating the output without writing the large intermediate matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM. As the softmax couples entire rows or blocks of row, online softmax [11, 13] can split the attention computation into blocks, and rescale the output of each block to finally get the right result (with no approximation). By significantly reducing the amount of memory reads/writes, FlashAttention yields 2-4$\times$ wall-clock speedup over optimized baseline attention implementations.

We describe the online softmax technique [11] and how it is used in attention [13]. For simplicity, consider just one row block of the attention matrix $\mathbf{S}$, of the form $\begin{bmatrix}
\mathbf{S}^{(1)} & \mathbf{S}^{(2)} \\
\end{bmatrix}$ for some matrices ${\mathbf{S}^{(1)},\mathbf{S}^{(2)}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$, where $B_{r}$ and $B_{c}$ are the row and column block sizes. We want to compute softmax of this row block and multiply with the value, of the form $\begin{bmatrix}
\mathbf{V}^{(1)} \\
\mathbf{V}^{(2)} \\
\end{bmatrix}$ for some matrices ${\mathbf{V}^{(1)},\mathbf{V}^{(2)}} \in {\mathbb{R}}^{B_{c} \times d}$. Standard softmax would compute:

| | | | |
|-----|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| | $m$ | $= {\max{({{rowmax}\hspace{0pt}{(\mathbf{S}^{(1)})}},{{rowmax}\hspace{0pt}{(\mathbf{S}^{(2)})}})}} \in {\mathbb{R}}^{B_{r}}$ | |
| | $\ell$ | $= {{{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(1)} - m})}} + {{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(2)} - m})}}} \in {\mathbb{R}}^{B_{r}}$ | |
| | $\mathbf{P}$ | $= \begin{bmatrix}                                                                                                                                                                                                  
                      \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \\                                                                                                                                                                               
                      \end{bmatrix} = {{diag}\hspace{0pt}{(\ell)}^{- 1}\hspace{0pt}\begin{bmatrix}                                                                                                                                         
                      e^{\mathbf{S}^{(1)} - m} & e^{\mathbf{S}^{(2)} - m} \\                                                                                                                                                               
                      \end{bmatrix}} \in {\mathbb{R}}^{{B_{r} \times 2}\hspace{0pt}B_{c}}$ | |
| | $\mathbf{O}$ | ${= {\begin{bmatrix}                                                                                                                                                                                                
                      \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \\                                                                                                                                                                               
                      \end{bmatrix}\hspace{0pt}\begin{bmatrix}                                                                                                                                                                             
                      \mathbf{V}^{(1)} \\                                                                                                                                                                                                  
                      \mathbf{V}^{(2)} \\                                                                                                                                                                                                  
                      \end{bmatrix}} = {{{diag}\hspace{0pt}{(\ell)}^{- 1}\hspace{0pt}e^{\mathbf{S}^{(1)} - m}\hspace{0pt}\mathbf{V}^{(1)}} + {e^{\mathbf{S}^{(2)} - m}\hspace{0pt}\mathbf{V}^{(2)}}} \in {\mathbb{R}}^{B_{r} \times d}}.$ | |

Online softmax instead computes “local” softmax with respect to each block and rescale to get the right output at the end:

| | | | |
|-----|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| | $m^{(1)}$ | $= {{rowmax}\hspace{0pt}{(\mathbf{S}^{(1)})}} \in {\mathbb{R}}^{B_{r}}$ | |
| | $\ell^{(1)}$ | $= {{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(1)} - m^{(1)}})}} \in {\mathbb{R}}^{B_{r}}$ | |
| | ${\overset{\sim}{\mathbf{P}}}^{(1)}$ | $= {{diag}\hspace{0pt}{(\ell^{(1)})}^{- 1}\hspace{0pt}e^{\mathbf{S}^{(1)} - m^{(1)}}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$ | |
| | $\mathbf{O}^{(1)}$ | $= {{\overset{\sim}{\mathbf{P}}}^{(1)}\hspace{0pt}\mathbf{V}^{(1)}} = {{diag}\hspace{0pt}{(\ell^{(1)})}^{- 1}\hspace{0pt}e^{\mathbf{S}^{(1)} - m^{(1)}}\hspace{0pt}\mathbf{V}^{(1)}} \in {\mathbb{R}}^{B_{r} \times d}$ | |
| | $m^{(2)}$ | $= {\max{(m^{(1)},{{rowmax}\hspace{0pt}{(\mathbf{S}^{(2)})}})}} = m$ | |
| | $\ell^{(2)}$ | $= {{e^{m^{(1)} - m^{(2)}}\hspace{0pt}\ell^{(1)}} + {{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(2)} - m^{(2)}})}}} = {{{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(1)} - m})}} + {{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(2)} - m})}}} = \ell$ | |
| | ${\overset{\sim}{\mathbf{P}}}^{(2)}$ | $= {{diag}\hspace{0pt}{(\ell^{(2)})}^{- 1}\hspace{0pt}e^{\mathbf{S}^{(2)} - m^{(2)}}}$ | |
| | $\mathbf{O}^{(2)}$ | ${= {{{diag}\hspace{0pt}{({\ell^{(1)}/\ell^{(2)}})}^{- 1}\hspace{0pt}\mathbf{O}^{(1)}} + {{\overset{\sim}{\mathbf{P}}}^{(2)}\hspace{0pt}\mathbf{V}^{(2)}}} = {{{diag}\hspace{0pt}{(\ell^{(2)})}^{- 1}\hspace{0pt}e^{s^{(1)} - m}\hspace{0pt}\mathbf{V}^{(1)}} + {{diag}\hspace{0pt}{(\ell^{(2)})}^{- 1}\hspace{0pt}e^{s^{(2)} - m}\hspace{0pt}\mathbf{V}^{(2)}}} = \mathbf{O}}.$ | |

We show how FlashAttention uses online softmax to enable tiling (Fig. 1) to reduce memory reads/writes.

Figure 1: Diagram of how FlashAttention forward pass is performed, when the key K is partitioned into two blocks and the value V is also partitioned into two blocks. By computing attention with respect to each block and rescaling the output, we get the right answer at the end, while avoiding expensive memory reads/writes of the intermediate matrices S and P. We simplify the diagram, omitting the step in softmax that subtracts each element by the row-wise max.

#### 2.3.2 Backward pass

In the backward pass, by re-computing the values of the attention matrices $\mathbf{S}$ and $\mathbf{P}$ once blocks of inputs $\mathbf{Q},\mathbf{K},\mathbf{V}$ are already loaded to SRAM, FlashAttention avoids having to store large intermediate values. By not having to save the large matrices $\mathbf{S}$ and $\mathbf{P}$ of size $N \times N$, FlashAttention yields 10-20$\times$ memory saving depending on sequence length (memory required in linear in sequence length $N$ instead of quadratic). The backward pass also achieves 2-4$\times$ wall-clock speedup due to reduce memory reads/writes.

The backward pass applies tiling to the equations in Section 2.2. Though the backward pass is simpler than the forward pass conceptually (there is no softmax rescaling), the implementation is significantly more involved. This is because there are more values to be kept in SRAM to perform 5 matrix multiples in the backward pass, compared to just 2 matrix multiples in the forward pass.

3 FlashAttention-2: Algorithm, Parallelism, and Work Partitioning
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

We describe the FlashAttention-2 algorithm, which includes several tweaks to FlashAttention to reduce the number of non-matmul FLOPs. We then describe how to parallelize the computation on different thread blocks to make full use the GPU resources. Finally we describe we partition the work between different warps within one thread block to reduce the amount of shared memory access. These improvements lead to 2-3$\times$ speedup as validated in Section 4.

### 3.1 Algorithm

We tweak the algorithm from FlashAttention to reduce the number of non-matmul FLOPs. This is because modern GPUs have specialized compute units (e.g., Tensor Cores on Nvidia GPUs) that makes matmul much faster. As an example, the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32. Another way to think about this is that each non-matmul FLOP is 16$\times$ more expensive than a matmul FLOP. To maintain high throughput (e.g., more than 50% of the maximum theoretical TFLOPs/s), we want to spend as much time on matmul FLOPs as possible.

#### 3.1.1 Forward pass

We revisit the online softmax trick as shown in Section 2.3 and make two minor tweaks to reduce non-matmul FLOPs:

1. 1.
    We do not have to rescale both terms of the output update by ${diag}\hspace{0pt}{(\ell^{(2)})}^{- 1}$:

    | | | |
    |-----|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
    | | $${\mathbf{O}^{(2)} = {{{diag}\hspace{0pt}{({\ell^{(1)}/\ell^{(2)}})}^{- 1}\hspace{0pt}\mathbf{O}^{(1)}} + {{diag}\hspace{0pt}{(\ell^{(2)})}^{- 1}\hspace{0pt}e^{\mathbf{S}^{(2)} - m^{(2)}}\hspace{0pt}\mathbf{V}^{(2)}}}}.$$ | |

    We can instead maintain an “un-scaled” version of $\mathbf{O}^{(2)}$ and keep around the statistics $\ell^{(2)}$:

    | | | |
    |-----|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
    | | $${{\overset{\sim}{\mathbf{O}}}^{(2)} = {{{diag}\hspace{0pt}{(\ell^{(1)})}^{- 1}\hspace{0pt}\mathbf{O}^{(1)}} + {e^{\mathbf{S}^{(2)} - m^{(2)}}\hspace{0pt}\mathbf{V}^{(2)}}}}.$$ | |

    Only at the every end of the loop do we scale the final ${\overset{\sim}{\mathbf{O}}}^{({last})}$ by ${diag}\hspace{0pt}{(\ell^{({last})})}^{- 1}$ to get the right output.
2. 2.
    We do not have to save both the max $m^{(j)}$ and the sum of exponentials $\ell^{(j)}$ for the backward pass. We only need to store the logsumexp $L^{(j)} = {m^{(j)} + {\log{(\ell^{(j)})}}}$.

In the simple case of 2 blocks in Section 2.3, the online softmax trick now becomes:

| | | | |
|-----|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| | $m^{(1)}$ | $= {{rowmax}\hspace{0pt}{(\mathbf{S}^{(1)})}} \in {\mathbb{R}}^{B_{r}}$ | |
| | $\ell^{(1)}$ | $= {{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(1)} - m^{(1)}})}} \in {\mathbb{R}}^{B_{r}}$ | |
| | $\overset{\sim}{\mathbf{O}^{(1)}}$ | $= {e^{\mathbf{S}^{(1)} - m^{(1)}}\hspace{0pt}\mathbf{V}^{(1)}} \in {\mathbb{R}}^{B_{r} \times d}$ | |
| | $m^{(2)}$ | $= {\max{(m^{(1)},{{rowmax}\hspace{0pt}{(\mathbf{S}^{(2)})}})}} = m$ | |
| | $\ell^{(2)}$ | $= {{e^{m^{(1)} - m^{(2)}}\hspace{0pt}\ell^{(1)}} + {{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(2)} - m^{(2)}})}}} = {{{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(1)} - m})}} + {{rowsum}\hspace{0pt}{(e^{\mathbf{S}^{(2)} - m})}}} = \ell$ | |
| | ${\overset{\sim}{\mathbf{P}}}^{(2)}$ | $= {{diag}\hspace{0pt}{(\ell^{(2)})}^{- 1}\hspace{0pt}e^{\mathbf{S}^{(2)} - m^{(2)}}}$ | |
| | ${\overset{\sim}{\mathbf{O}}}^{(2)}$ | $= {{{diag}\hspace{0pt}{(e^{m^{(1)} - m^{(2)}})}^{- 1}\hspace{0pt}{\overset{\sim}{\mathbf{O}}}^{(1)}} + {e^{\mathbf{S}^{(2)} - m^{(2)}}\hspace{0pt}\mathbf{V}^{(2)}}} = {{e^{s^{(1)} - m}\hspace{0pt}\mathbf{V}^{(1)}} + {e^{s^{(2)} - m}\hspace{0pt}\mathbf{V}^{(2)}}}$ | |
| | $\mathbf{O}^{(2)}$ | ${= {{diag}\hspace{0pt}{(\ell^{(2)})}^{- 1}\hspace{0pt}{\overset{\sim}{\mathbf{O}}}^{(2)}} = \mathbf{O}}.$ | |

We describe the full FlashAttention-2 forward pass in Algorithm 1.

Algorithm 1 FlashAttention-2 forward pass

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V}} \in {\mathbb{R}}^{N \times d}$ in HBM, block sizes $B_{c}$, $B_{r}$.

1:   Divide $\mathbf{Q}$ into $T_{r} = \left\lceil \frac{N}{B_{r}} \right\rceil$ blocks $\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}$ of size $B_{r} \times d$ each, and divide $\mathbf{K},\mathbf{V}$ in to $T_{c} = \left\lceil \frac{N}{B_{c}} \right\rceil$ blocks $\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}$ and $\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}$, of size $B_{c} \times d$ each.

2:  Divide the output $\mathbf{O} \in {\mathbb{R}}^{N \times d}$ into $T_{r}$ blocks $\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}$ of size $B_{r} \times d$ each, and divide the logsumexp $L$ into $T_{r}$ blocks $L_{i},\ldots,L_{T_{r}}$ of size $B_{r}$ each.

3:  for $1 \leq i \leq T_{r}$ do

4:      Load $\mathbf{Q}_{i}$ from HBM to on-chip SRAM.

5:      On chip, initialize ${\mathbf{O}_{i}^{(0)} = {(0)}_{B_{r} \times d} \in {\mathbb{R}}^{B_{r} \times d}},{\ell_{i}^{(0)} = {(0)}_{B_{r}} \in {\mathbb{R}}^{B_{r}}},{m_{i}^{(0)} = {({- \infty})}_{B_{r}} \in {\mathbb{R}}^{B_{r}}}$.

6:     for $1 \leq j \leq T_{c}$ do

7:         Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.

8:         On chip, compute $\mathbf{S}_{i}^{(j)} = {\mathbf{Q}_{i}\hspace{0pt}\mathbf{K}_{j}^{T}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

9:         On chip, compute $m_{i}^{(j)} = {\max\hspace{0pt}{(m_{i}^{({j - 1})},{{rowmax}\hspace{0pt}{(\mathbf{S}_{i}^{(j)})}})}} \in {\mathbb{R}}^{B_{r}}$, ${\overset{\sim}{\mathbf{P}}}_{i}^{(j)} = {\exp{({\mathbf{S}_{i}^{(j)} - m_{i}^{(j)}})}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$ (pointwise), $\ell_{i}^{(j)} = {{e^{m_{i}^{j - 1} - m_{i}^{(j)}}\hspace{0pt}\ell_{i}^{({j - 1})}} + {{rowsum}\hspace{0pt}{({\overset{\sim}{\mathbf{P}}}_{i}^{(j)})}}} \in {\mathbb{R}}^{B_{r}}$.

10:         On chip, compute $\mathbf{O}_{i}^{(j)} = {{{diag}\hspace{0pt}{(e^{m_{i}^{({j - 1})} - m_{i}^{(j)}})}^{- 1}\hspace{0pt}\mathbf{O}_{i}^{({j - 1})}} + {{\overset{\sim}{\mathbf{P}}}_{i}^{(j)}\hspace{0pt}\mathbf{V}_{j}}}$.

11:     end for

12:     On chip, compute $\mathbf{O}_{i} = {{diag}\hspace{0pt}{(\ell_{i}^{(T_{c})})}^{- 1}\hspace{0pt}\mathbf{O}_{i}^{(T_{c})}}$.

13:     On chip, compute $L_{i} = {m_{i}^{(T_{c})} + {\log{(\ell_{i}^{(T_{c})})}}}$.

14:     Write $\mathbf{O}_{i}$ to HBM as the $i$-th block of $\mathbf{O}$.

15:     Write $L_{i}$ to HBM as the $i$-th block of $L$.

16:  end for

17:  Return the output $\mathbf{O}$ and the logsumexp $L$.

##### Causal masking.

One common use case of attention is in auto-regressive language modeling, where we need to apply a causal mask to the attention matrix $\mathbf{S}$ (i.e., any entry $\mathbf{S}_{i\hspace{0pt}j}$ with $j > i$ is set to $- \infty$).

1. 1.
    As FlashAttention and FlashAttention-2 already operate by blocks, for any blocks where all the column indices are more than the row indices (approximately half of the blocks for large sequence length), we can skip the computation of that block. This leads to around 1.7-1.8$\times$ speedup compared to attention without the causal mask.
2. 2.
    We do not need to apply the causal mask for blocks whose row indices are guaranteed to be strictly less than the column indices. This means that for each row, we only need apply causal mask to 1 block (assuming square block).

##### Correctness, runtime, and memory requirement.

As with FlashAttention, Algorithm 1 returns the correct output $\mathbf{O} = {{softmax}\hspace{0pt}{({\mathbf{Q}\mathbf{K}}^{\top})}\hspace{0pt}\mathbf{V}}$ (with no approximation), using $O\hspace{0pt}{({N^{2}\hspace{0pt}d})}$ FLOPs and requires $O\hspace{0pt}{(N)}$ additional memory beyond inputs and output (to store the logsumexp $L$). The proof is almost the same as the proof of Dao et al. [5, Theorem 1], so we omit it here.

#### 3.1.2 Backward pass

The backward pass of FlashAttention-2 is almost the same as that of FlashAttention. We make a minor tweak to only use the row-wise logsumexp $L$ instead of both the row-wise max and row-wise sum of exponentials in the softmax. We include the backward pass description in Algorithm 2 for completeness.

Algorithm 2 FlashAttention-2 Backward Pass

0:  Matrices ${\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},{\mathbf{d}\mathbf{O}}} \in {\mathbb{R}}^{N \times d}$ in HBM, vector $L \in {\mathbb{R}}^{N}$ in HBM, block sizes $B_{c}$, $B_{r}$.

1:  Divide $\mathbf{Q}$ into $T_{r} = \left\lceil \frac{N}{B_{r}} \right\rceil$ blocks $\mathbf{Q}_{1},\ldots,\mathbf{Q}_{T_{r}}$ of size $B_{r} \times d$ each, and divide $\mathbf{K},\mathbf{V}$ in to $T_{c} = \left\lceil \frac{N}{B_{c}} \right\rceil$ blocks $\mathbf{K}_{1},\ldots,\mathbf{K}_{T_{c}}$ and $\mathbf{V}_{1},\ldots,\mathbf{V}_{T_{c}}$, of size $B_{c} \times d$ each.

2:  Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\ldots,\mathbf{O}_{T_{r}}$ of size $B_{r} \times d$ each, divide $\mathbf{d}\mathbf{O}$ into $T_{r}$ blocks ${\mathbf{d}\mathbf{O}}_{i},\ldots,{\mathbf{d}\mathbf{O}}_{T_{r}}$ of size $B_{r} \times d$ each, and divide $L$ into $T_{r}$ blocks $L_{i},\ldots,L_{T_{r}}$ of size $B_{r}$ each.

3:  Initialize ${\mathbf{d}\mathbf{Q}} = {(0)}_{N \times d}$ in HBM and divide it into $T_{r}$ blocks ${\mathbf{d}\mathbf{Q}}_{1},\ldots,{\mathbf{d}\mathbf{Q}}_{T_{r}}$ of size $B_{r} \times d$ each. Divide ${{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}} \in {\mathbb{R}}^{N \times d}$ in to $T_{c}$ blocks ${\mathbf{d}\mathbf{K}}_{1},\ldots,{\mathbf{d}\mathbf{K}}_{T_{c}}$ and ${\mathbf{d}\mathbf{V}}_{1},\ldots,{\mathbf{d}\mathbf{V}}_{T_{c}}$, of size $B_{c} \times d$ each.

4:  Compute $D = {{rowsum}\hspace{0pt}{({{\mathbf{d}\mathbf{O}} \circ \mathbf{O}})}} \in {\mathbb{R}}^{d}$ (pointwise multiply), write $D$ to HBM and divide it into $T_{r}$ blocks $D_{1},\ldots,D_{T_{r}}$ of size $B_{r}$ each.

5:  for $1 \leq j \leq T_{c}$ do

6:     Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.

7:     Initialize ${{\mathbf{d}\mathbf{K}}_{j} = {(0)}_{B_{c} \times d}},{{\mathbf{d}\mathbf{V}}_{j} = {(0)}_{B_{c} \times d}}$ on SRAM.

8:     for $1 \leq i \leq T_{r}$ do

9:        Load $\mathbf{Q}_{i},\mathbf{O}_{i},{\mathbf{d}\mathbf{O}}_{i},{\mathbf{d}\mathbf{Q}}_{i},L_{i},D_{i}$ from HBM to on-chip SRAM.

10:        On chip, compute $\mathbf{S}_{i}^{(j)} = {\mathbf{Q}_{i}\hspace{0pt}\mathbf{K}_{j}^{T}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

11:        On chip, compute $\mathbf{P}_{i}^{(j)} = {\exp{({\mathbf{S}_{i\hspace{0pt}j} - L_{i}})}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

12:        On chip, compute ${\mathbf{d}\mathbf{V}}_{j}\leftarrow{{\mathbf{d}\mathbf{V}}_{j} + {{(\mathbf{P}_{i}^{(j)})}^{\top}\hspace{0pt}{\mathbf{d}\mathbf{O}}_{i}}} \in {\mathbb{R}}^{B_{c} \times d}$.

13:        On chip, compute ${\mathbf{d}\mathbf{P}}_{i}^{(j)} = {{\mathbf{d}\mathbf{O}}_{i}\hspace{0pt}\mathbf{V}_{j}^{\top}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

14:        On chip, compute ${\mathbf{d}\mathbf{S}}_{i}^{(j)} = {\mathbf{P}_{i}^{(j)} \circ {({{\mathbf{d}\mathbf{P}}_{i}^{(j)} - D_{i}})}} \in {\mathbb{R}}^{B_{r} \times B_{c}}$.

15:        Load ${\mathbf{d}\mathbf{Q}}_{i}$ from HBM to SRAM, then on chip, update ${\mathbf{d}\mathbf{Q}}_{i}\leftarrow{{\mathbf{d}\mathbf{Q}}_{i} + {{\mathbf{d}\mathbf{S}}_{i}^{(j)}\hspace{0pt}\mathbf{K}_{j}}} \in {\mathbb{R}}^{B_{r} \times d}$, and write back to HBM.

16:        On chip, compute 𝐝𝐊j←𝐝𝐊j+𝐝𝐒i(j)⊤​𝐐i∈ℝBc×d←subscript𝐝𝐊𝑗subscript𝐝𝐊𝑗superscriptsuperscriptsubscript𝐝𝐒𝑖𝑗topsubscript𝐐𝑖superscriptℝsubscript𝐵𝑐𝑑\\mathbf{dK}\_{j}\\leftarrow\\mathbf{dK}\_{j}+{\\mathbf{dS}\_{i}^{(j)}}^{\\top}\\mathbf{Q}\_{i}\\in\\mathbb{R}^{B\_{c}\\times d}.

17:     end for

18:     Write ${\mathbf{d}\mathbf{K}}_{j},{\mathbf{d}\mathbf{V}}_{j}$ to HBM.

19:  end for

20:  Return ${\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}$.

##### Multi-query attention and grouped-query attention.

Multi-query attention (MQA) [15] and grouped-query attention (GQA) [1] are variants of attention where multiple heads of query attend to the same head of key and value, in order to reduce the size of KV cache during inference. Instead of having to duplicate the key and value heads for the computation, we implicitly manipulate the indices into the head to perform the same computation. In the backward pass, we need to sum the gradients $\mathbf{d}\mathbf{K}$ and $\mathbf{d}\mathbf{V}$ across different heads that were implicitly duplicated.

### 3.2 Parallelism

The first version of FlashAttention parallelizes over batch size and number of heads. We use 1 thread block to process one attention head, and there are overall $\text{batch\ size} \cdot \text{number\ of\ heads}$ thread blocks. Each thread block is scheduled to run on a streaming multiprocessor (SM), and there are 108 of these SMs on an A100 GPU for example. This scheduling is efficient when this number is large (say $\geq 80$), since we can effectively use almost all of the compute resources on the GPU.

In the case of long sequences (which usually means small batch sizes or small number of heads), to make better use of the multiprocessors on the GPU, we now additionally parallelize over the sequence length dimension. This results in significant speedup for this regime.

##### Forward pass.

We see that the outer loop (over sequence length) is embarrassingly parallel, and we schedule them on different thread blocks that do not need to communicate with each other. We also parallelize over the batch dimension and number of heads dimension, as done in FlashAttention. The increased parallelism over sequence length helps improve occupancy (fraction of GPU resources being used) when the batch size and number of heads are small, leading to speedup in this case.

These ideas of swapping the order of the loop (outer loop over row blocks and inner loop over column blocks, instead of the other way round in the original FlashAttention paper), as well as parallelizing over the sequence length dimension were first suggested and implemented by Phil Tillet in the Triton [17] implementation.333https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py

##### Backward pass.

Notice that the only shared computation between different column blocks is in update $\mathbf{d}\mathbf{Q}$ in Algorithm 2, where we need to load ${\mathbf{d}\mathbf{Q}}_{i}$ from HBM to SRAM, then on chip, update ${\mathbf{d}\mathbf{Q}}_{i}\leftarrow{{\mathbf{d}\mathbf{Q}}_{i} + {{\mathbf{d}\mathbf{S}}_{i}^{(j)}\hspace{0pt}\mathbf{K}_{j}}}$, and write back to HBM. We thus parallelize over the sequence length dimension as well, and schedule 1 thread block for each column block of the backward pass. We use atomic adds to communicate between different thread blocks to update $\mathbf{d}\mathbf{Q}$.

We describe the parallelization scheme in Fig. 2.

Figure 2: In the forward pass (left), we parallelize the workers (thread blocks) where each worker takes care of a block of rows of the attention matrix. In the backward pass (right), each worker takes care of a block of columns of the attention matrix.

### 3.3 Work Partitioning Between Warps

As Section 3.2 describe how we schedule thread blocks, even within each thread block, we also have to decide how to partition the work between different warps. We typically use 4 or 8 warps per thread block, and the partitioning is described in Fig. 3.

##### Forward pass.

For each block, FlashAttention splits $\mathbf{K}$ and $\mathbf{V}$ across 4 warps while keeping $\mathbf{Q}$ accessible by all warps. Each warp multiplies to get a slice of ${\mathbf{Q}\mathbf{K}}^{\top}$, then they need to multiply with a slice of $\mathbf{V}$ and communicate to add up the result. This is referred to as the “split-K” scheme. However, this is inefficient since all warps need to write their intermediate results out to shared memory, synchronize, then add up the intermediate results. These shared memory reads/writes slow down the forward pass in FlashAttention.

In FlashAttention-2, we instead split $\mathbf{Q}$ across 4 warps while keeping $\mathbf{K}$ and $\mathbf{V}$ accessible by all warps. After each warp performs matrix multiply to get a slice of ${\mathbf{Q}\mathbf{K}}^{\top}$, they just need to multiply with their shared slice of $\mathbf{V}$ to get their corresponding slice of the output. There is no need for communication between warps. The reduction in shared memory reads/writes yields speedup (Section 4).

(a) FlashAttention

(b) FlashAttention-2

Figure 3: Work partitioning between different warps in the forward pass

##### Backward pass.

Similarly for the backward pass, we choose to partition the warps to avoid the “split-K” scheme. However, it still requires some synchronization due to the more complicated dependency between all the different inputs and gradients $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},{\mathbf{d}\mathbf{O}},{\mathbf{d}\mathbf{Q}},{\mathbf{d}\mathbf{K}},{\mathbf{d}\mathbf{V}}$. Nevertheless, avoiding “split-K” reduces shared memory reads/writes and again yields speedup (Section 4).

##### Tuning block sizes

Increasing block sizes generally reduces shared memory loads/stores, but increases the number of registers required and the total amount of shared memory. Past a certain block size, register spilling causes significant slowdown, or the amount of shared memory required is larger than what the GPU has available, and the kernel cannot run at all. Typically we choose blocks of size ${\{ 64,128\}} \times {\{ 64,128\}}$, depending on the head dimension $d$ and the device shared memory size.

We manually tune for each head dimensions since there are essentially only 4 choices for block sizes, but this could benefit from auto-tuning to avoid this manual labor. We leave this to future work.

4 Empirical Validation
-------------------------------------------------------------------

We evaluate the impact of using FlashAttention-2 to train Transformer models.

- •
    Benchmarking attention. We measure the runtime of FlashAttention-2 across different sequence lengths and compare it to a standard implementation in PyTorch, FlashAttention, and FlashAttention in Triton. We confirm that FlashAttention-2 is 1.7-3.0$\times$ faster than FlashAttention, 1.3-2.5$\times$ faster than FlashAttention in Triton, and 3-10$\times$ faster than a standard attention implementation. FlashAttention-2 reaches up to 230 TFLOPs/s, 73% of the theoretical maximum TFLOPs/s on A100 GPUs.
- •
    End-to-end training speed When used end-to-end to train GPT-style models of size 1.3B and 2.7B on sequence lengths either 2k or 8k, FlashAttention-2 yields up to 1.3$\times$ speedup compared to FlashAttention and 2.8$\times$ speedup compared to a baseline without FlashAttention. FlashAttention-2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization) per A100 GPU.

### 4.1 Benchmarking Attention

We measure the runtime of different attention methods on an A100 80GB SXM4 GPU for different settings (without / with causal mask, head dimension 64 or 128). We report the results in Fig. 4, Fig. 5 and Fig. 6, showing that FlashAttention-2 is around 2$\times$ faster than FlashAttention and FlashAttention in xformers (the “cutlass” implementation). FlashAttention-2 is around 1.3-1.5$\times$ faster than FlashAttention in Triton in the forward pass and around 2$\times$ faster in the backward pass. Compared to a standard attention implementation in PyTorch, FlashAttention-2 can be up to 10$\times$ faster.

Benchmark setting: we vary the sequence length from 512, 1k, …, 16k, and set batch size so that the total number of tokens is 16k. We set hidden dimension to 2048, and head dimension to be either 64 or 128 (i.e., 32 heads or 16 heads). To calculate the FLOPs of the forward pass, we use:

| | | |
|-----|----------------------------------------------------------------------------------------------|-----|
| | $${4 \cdot \text{seqlen}^{2} \cdot \text{head\ dimension} \cdot \text{number\ of\ heads}}.$$ | |

With causal mask, we divide this number by 2 to account for the fact that approximately only half of the entries are calculated. To get the FLOPs of the backward pass, we multiply the forward pass FLOPs by 2.5 (since there are 2 matmuls in the forward pass and 5 matmuls in the backward pass, due to recomputation).

(a) Without causal mask, head dimension 64

(b) Without causal mask, head dimension 128

(c) With causal mask, head dimension 64

(d) With causal mask, head dimension 128

Figure 4: Attention forward + backward speed on A100 GPU

(a) Without causal mask, head dimension 64

(b) Without causal mask, head dimension 128

(c) With causal mask, head dimension 64

(d) With causal mask, head dimension 128

Figure 5: Attention forward speed on A100 GPU

(a) Without causal mask, head dimension 64

(b) Without causal mask, head dimension 128

(c) With causal mask, head dimension 64

(d) With causal mask, head dimension 128

Figure 6: Attention backward speed on A100 GPU

Just running the same implementation on H100 GPUs (using no special instructions to make use of new features such as TMA and 4th-gen Tensor Cores), we obtain up to 335 TFLOPs/s (Fig. 7). We expect that by using new instructions, we can obtain another 1.5x-2x speedup on H100 GPUs. We leave that to future work.

(a) Without causal mask, head dimension 64

(b) Without causal mask, head dimension 128

(c) With causal mask, head dimension 64

(d) With causal mask, head dimension 128

Figure 7: Attention forward + backward speed on H100 GPU

### 4.2 End-to-end Performance

We measure the training throughput of GPT-style models with either 1.3B or 2.7B parameters, on 8$\times$A100 80GB SXM. As shown in Table 1, FlashAttention-2 yields 2.8$\times$ speedup compared to a baseline without FlashAttention and 1.3$\times$ speedup compared to FlashAttention-2, reaching up to 225 TFLOPs/s per A100 GPU.

Note that we calculate the FLOPs by the formula, following Megatron-LM [16] (and many other papers and libraries):

| | | |
|-----|-----------------------------------------------------------------------------------------|-----|
| | $${{6 \cdot \text{seqlen} \cdot \text{number\ of\ params}} + {12 \cdot \text{number\ of 
       layers} \cdot \text{hidden\ dim} \cdot \text{seqlen}^{2}}}.$$ | |

The first term accounts for the FLOPs due to weight–input multiplication, and the second term accounts for the FLOPs due to attention. However, one can argue that the second term should be halved, as with causal mask we only need to compute approximately half the number of elements in attention. We choose to follow the formula from the literature (without dividing the attention FLOPs by 2) for consistency.

Table 1: Training speed (TFLOPs/s/GPU) of GPT-style models on 8$\times$A100 GPUs. FlashAttention-2 reaches up to 225 TFLOPs/s (72% model FLOPs utilization). We compare against a baseline running without FlashAttention. 

| Model | Without FlashAttention | FlashAttention | FlashAttention-2 |
|----------------------|----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| GPT3-1.3B 2k context | 142 TFLOPs/s | 189 TFLOPs/s | 196 TFLOPs/s |
| GPT3-1.3B 8k context | 72 TFLOPS/s | 170 TFLOPs/s | 220 TFLOPs/s |
| GPT3-2.7B 2k context | 149 TFLOPs/s | 189 TFLOPs/s | 205 TFLOPs/s |
| GPT3-2.7B 8k context | 80 TFLOPs/s | 175 TFLOPs/s | 225 TFLOPs/s |

5 Discussion and Future Directions
-------------------------------------------------------------------------------

FlashAttention-2 is 2$\times$ faster than FlashAttention, which means that we can train models with 16k longer context for the same price as previously training a 8k context model. We are excited about how this can be used to understand long books and reports, high resolution images, audio and video. FlashAttention-2 will also speed up training, finetuning, and inference of existing models.

In the near future, we plan to collaborate with researchers and engineers to make FlashAttention widely applicable in different kinds of devices (e.g., H100 GPUs, AMD GPUs), as well as new data types such as FP8. As an immediate next step, we plan to optimize FlashAttention-2 for H100 GPUs to use new hardware features (TMA, 4th-gen Tensor Cores, fp8). Combining the low-level optimizations in FlashAttention-2 with high-level algorithmic changes (e.g., local, dilated, block-sparse attention) could allow us to train AI models with much longer context. We are also excited to work with compiler researchers to make these optimization techniques easily programmable.

#### Acknowledgments

We thank Phil Tillet and Daniel Haziza, who have implemented versions of FlashAttention in Triton [17] and the xformers library [10]. FlashAttention-2 was motivated by exchange of ideas between different ways that attention could be implemented. We are grateful to the Nvidia CUTLASS team (especially Vijay Thakkar, Cris Cecka, Haicheng Wu, and Andrew Kerr) for their CUTLASS library, in particular the CUTLASS 3.x release, which provides clean abstractions and powerful building blocks for the implementation of FlashAttention-2. We thank Driss Guessous for integrating FlashAttention to PyTorch. FlashAttention-2 has benefited from helpful discussions with Phil Wang, Markus Rabe, James Bradbury, Young-Jun Ko, Julien Launay, Daniel Hesslow, Michaël Benesty, Horace He, Ashish Vaswani, and Erich Elsen. Thanks for Stanford CRFM and Stanford NLP for the compute support. We thank Dan Fu and Christopher Ré for their collaboration, constructive feedback, and constant encouragement on this line of work of designing hardware-efficient algorithms. We thank Albert Gu and Beidi Chen for their helpful suggestions on early drafts of this technical report.
