Online normalizer calculation for softmax
=========================================

  Maxim Milakov  
NVIDIA  
mmilakov@nvidia.com &Natalia Gimelshein  
NVIDIA  
ngimelshein@nvidia.com 

(April 2018)

###### Abstract

The Softmax function is ubiquitous in machine learning, multiple previous works suggested faster alternatives for it. In this paper we propose a way to compute classical Softmax with fewer memory accesses and hypothesize that this reduction in memory accesses should improve Softmax performance on actual hardware. The benchmarks confirm this hypothesis: Softmax accelerates by up to $1.3$x and Softmax+TopK combined and fused by up to $5$x.

1 Introduction
-----------------------------------------------------------

Neural networks models are widely used for language modeling, for tasks such as machine translation [1] and speech recognition [2]. These models compute word probabilities taking into account the already generated part of the sequence. The probabilities are usually computed by a Projection layer, which "projects" hidden representation into the output vocabulary space, and a following Softmax function, which transforms raw logits into the the vector of probabilities. Softmax is utilized not only for neural networks, for example, it is employed in multinomial logistic regression [3].

A number of previous works suggested faster alternatives to compute word probabilities. Differentiated Softmax [4] and SVD-Softmax [5] replace the projection layer - which is usually just a matrix multiplication - with more computationally efficient alternatives. Multiple variants of Hierarchical Softmax [6, 7, 8] split a single Projection+Softmax pair into multiple much smaller versions of these two functions organized in tree-like structures. Sampled-based approximations, such as Importance Sampling [9], Noise Contrastive Estimation [10], and Blackout [11] accelerate training by running Softmax on select elements of the original vector. Finally, Self-Normalized Softmax [12] augments the objective function to make the softmax normalization term close to $1$ (and skip computing it during inference).

This is not an exhaustive list, but, hopefully, a representative one. Almost all of the approaches still need to run the original Softmax function, either on full vector or reduced one. There are two exceptions that don’t need to compute the softmax normalization term: training with Noise Contrastive Estimation and inference with Self-Normalized Softmax. All others will benefit from the original Softmax running faster.

To the best of our knowledge there has been no targeted efforts to improve the performance of the original Softmax function. We tried to address this shortcoming and figured out a way to compute Softmax with fewer memory accesses. We benchmarked it to see if those reductions in memory accesses translate into performance improvements on a real hardware.

2 Original softmax
---------------------------------------------------------------

Function $y = {S\hspace{0pt}o\hspace{0pt}f\hspace{0pt}t\hspace{0pt}m\hspace{0pt}a\hspace{0pt}x\hspace{0pt}{(x)}}$ is defined as:

| | | | |
|-----|----------------------------------------------------------------|-----|-------------------------------------------------------------------|
| | $$y_{i} = \frac{e^{x_{i}}}{\sum\limits_{j = 1}^{V}e^{x_{j}}}$$ | | (1) |

where ${x,y} \in {\mathbb{R}}^{V}$. The naive implementation (see algorithm 1) scans the input vector two times - one to calculate the normalization term $d_{V}$ and another to compute output values $y_{i}$ - effectively doing three memory accesses per vector element: two loads and one store.

Algorithm 1 Naive softmax

1:$d_{0}\leftarrow 0$

2:for $j\leftarrow{1,V}$ do

3:     $d_{j}\leftarrow{d_{j - 1} + e^{x_{j}}}$

4:end for

5:for $i\leftarrow{1,V}$ do

6:     $y_{i}\leftarrow\frac{e^{x_{i}}}{d_{V}}$

7:end for

Unfortunately, on real hardware, where the range of numbers represented is limited, the line 3 of the algorithm 1 can overflow or underflow due to the exponent. There is a safe form of (1), which is immune to this problem:

| | | | |
|-----|----------------------------------------------------------------------------------------------------------------------------------|-----|-------------------------------------------------------------------|
| | $$y_{i} = \frac{e^{x_{i} - {\max\limits_{k = 1}^{V}x_{k}}}}{\sum\limits_{j = 1}^{V}e^{x_{j} - {\max\limits_{k = 1}^{V}x_{k}}}}$$ | | (2) |

Algorithm 2 Safe softmax

1:$m_{0}\leftarrow{- \infty}$

2:for $k\leftarrow{1,V}$ do

3:     $m_{k}\leftarrow{\max{(m_{k - 1},x_{k})}}$

4:end for

5:$d_{0}\leftarrow 0$

6:for $j\leftarrow{1,V}$ do

7:     $d_{j}\leftarrow{d_{j - 1} + e^{x_{j} - m_{V}}}$

8:end for

9:for $i\leftarrow{1,V}$ do

10:     $y_{i}\leftarrow\frac{e^{x_{i} - m_{V}}}{d_{V}}$

11:end for

All major DL frameworks are using this safe version for the Softmax computation: TensorFlow [13] v1.7, PyTorch [14] (with Caffe2) v0.4.0, MXNET [15] v1.1.0, Microsoft Cognitive Toolkit [16] v2.5.1, and Chainer [17] v5.0.0a1. But Safe Softmax does three passes over input vector: The first one calculates the maximum value $m_{V}$, the second one - normalization term $d_{V}$, and the third one - final values $y_{i}$, see algorithm 2; This results in 4 memory access per vector element overall. We want to improve on that.

3 Online normalizer calculation
----------------------------------------------------------------------------

The algorithm 3 calculates both the maximum value $m$ and the normalization term $d$ in a single pass over input vector with negligible additional cost of two operations per vector element. It reduces memory accesses from 4 down to 3 per vector element for the Softmax function evaluation. Inspiration came from the numerically stable variance calculation online algorithm, see [18].

Algorithm 3 Safe softmax with online normalizer calculation

1:$m_{0}\leftarrow{- \infty}$

2:$d_{0}\leftarrow 0$

3:for $j\leftarrow{1,V}$ do

4:     $m_{j}\leftarrow{\max\left( m_{j - 1},x_{j} \right)}$

5:     $d_{j}\leftarrow{{d_{j - 1} \times e^{m_{j - 1} - m_{j}}} + e^{x_{j} - m_{j}}}$

6:end for

7:for $i\leftarrow{1,V}$ do

8:     $y_{i}\leftarrow\frac{e^{x_{i} - m_{V}}}{d_{V}}$

9:end for

Essentially, the algorithm keeps the maximum value $m$ and the normalization term $d$ as it iterates over elements of the input array. At each iteration it needs to adjust the normalizer $d$ to the new maximum $m_{j}$ and only then add new value to the normalizer.

###### Theorem 1.

The lines 1-6 of the algorithm 3 compute $m_{V} = {\max\limits_{k = 1}^{V}x_{k}}$ and $d_{V} = {\sum_{j = 1}^{V}e^{x_{j} - m_{V}}}$

###### Proof.

We will use a proof by induction.

- $◆$
    *Base case*: $V = 1$
- $◆$
    *Inductive step*: We assume the theorem statement holds for $V = {S - 1}$, that is the lines 1-6 of the algorithm 3 compute $m_{S - 1} = {\max\limits_{k = 1}^{S - 1}x_{k}}$ and $d_{S - 1} = {\sum_{j = 1}^{S - 1}e^{x_{j} - m_{S - 1}}}$. Let’s see what the algorithm computes for $V = S$  

The algorithm 3 is proved to compute the Softmax function as defined in (2). It is also safe:

- •
    $m_{j}$ is the running maximum, ${m_{j} \in \left\lbrack {\min\limits_{k = 1}^{V}m_{k}},{\max\limits_{k = 1}^{V}m_{k}} \right\rbrack},{{\forall j} \in {1,V}}$; $m_{j}$ cannot underflow or overflow.
- •
    $d_{j}$ is also bounded: ${1 \leq d_{j} \leq j},{{\forall j} \in {1,V}}$. It can be easily proven by induction. The 32-bit floating point storage for $d_{j}$ guarantees processing of up to $1.7 \ast 10^{37}$ elements in vector $x$ without overflow. It is a reasonably large amount, but if your vector is even larger you need to use the 64-bit floating point storage for $d_{j}$.

The algorithm 2 provides the same guarantees: ${1 \leq d_{j} \leq j},{{\forall j} \in {1,V}}$.

In the remainder of this paper we will call algorithm 3 "Online Softmax".

### 3.1 Parallel online normalizer calculation

The lines 1-6 of the algorithm 3 define a sequential way of calculating the normalization term in a single pass over input vector. Modern computing devices allow running multiple threads concurrently; We need to have a parallel version of the algorithm to fully utilize devices. We define a generalized version of the online normalizer calculation:

| | | | |
|-----|---------------------------------------------------------------------------|-----|-------------------------------------------------------------------|
| | $$\begin{bmatrix}                                                         
       m_{V} \\                                                                   
       d_{V} \\                                                                   
       \end{bmatrix} = {\begin{bmatrix}                                           
       x_{1} \\                                                                   
       1 \\                                                                       
       \end{bmatrix}\hspace{0pt}{\oplus{\begin{bmatrix}                           
       x_{2} \\                                                                   
       1 \\                                                                       
       \end{bmatrix}\hspace{0pt}{\oplus{\ldots\hspace{0pt}{\oplus\begin{bmatrix}  
       x_{V} \\                                                                   
       1 \\                                                                       
       \end{bmatrix}}}}}}}$$ | | (3) |

where ${x_{i},m_{V},d_{V}} \in {\mathbb{R}}$. The binary operation $\oplus:{{{\mathbb{R}}^{2} \times {\mathbb{R}}^{2}}\rightarrow{\mathbb{R}}^{2}}$ is defined as:

| | | | |
|-----|----------------------------------------------------------------------------------------------------------|-----|-------------------------------------------------------------------|
| | $${\begin{bmatrix}                                                                                       
       m_{i} \\                                                                                                  
       d_{i} \\                                                                                                  
       \end{bmatrix}\hspace{0pt}{\oplus\begin{bmatrix}                                                           
       m_{j} \\                                                                                                  
       d_{j} \\                                                                                                  
       \end{bmatrix}}} = \begin{bmatrix}                                                                         
       {\max\left( m_{i},m_{j} \right)} \\                                                                       
       {{d_{i} \times e^{m_{i} - {\max{(m_{i},m_{j})}}}} + {d_{j} \times e^{m_{j} - {\max{(m_{i},m_{j})}}}}} \\  
       \end{bmatrix}$$ | | (4) |

Applying (3) sequentially from left to right is equivalent to running lines 1-6 of the algorithm 3. The operation $\oplus$ is associative, which enables parallel evaluation of (3). It is also commutative, which provides the flexibility needed to make parallel implementations more efficient. We omit the proofs for these two statements for brevity.

4 Softmax and top-k fusion
-----------------------------------------------------------------------

Online Softmax (algorithm 3) does three memory accesses per vector element: one load for the normalizer calculation, one load and one store for computing Softmax function values $y_{i}$. Inference with the beam search for auto-regressive models has TopK following Softmax, and this TopK doesn’t need to compute all $y_{i}$ values. This enables even bigger improvements.

The TopK function is producing the vector of K integer indices referencing the largest values in the input vector, along with those values:

| | | | |
|-----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|-------------------------------------------------------------------|
| | $${{T\hspace{0pt}o\hspace{0pt}p\hspace{0pt}K\hspace{0pt}\left( y \right)} = {(v,z)}}:{{v_{i} = y_{z_{i}}},{{v_{i} \geq y_{j}},{{{\forall i} \in \left\lbrack 1,K \right\rbrack},{{\forall j} \notin z}}}}$$ | | (5) |

where ${y \in {\mathbb{R}}^{V}},{{z \in {\mathbb{Z}}^{K}},{v \in {\mathbb{R}}^{K}}}$.

Algorithm 4 Online softmax and top-k

1:$m_{0}\leftarrow{- \infty}$

2:$d_{0}\leftarrow 0$

3:${u\leftarrow\left\{ {- \infty},{- \infty},\ldots,{- \infty} \right\}^{T}},{u \in {\mathbb{R}}^{K + 1}}$ $\rhd$ The 1st $K$ elems will hold running TopK values 

4:${p\leftarrow\left\{ {- 1},{- 1},\ldots,{- 1} \right\}^{T}},{p \in {\mathbb{Z}}^{K + 1}}$ $\rhd$ … and their indices 

5:for $j\leftarrow{1,V}$ do

6:     $m_{j}\leftarrow{\max\left( m_{j - 1},x_{j} \right)}$

7:     $d_{j}\leftarrow{{d_{j - 1} \times e^{m_{j - 1} - m_{j}}} + e^{x_{j} - m_{j}}}$

8:     $u_{K + 1}\leftarrow x_{j}$ $\rhd$ Initialize $K + 1$ elem with new value from input vector 

9:     $p_{K + 1}\leftarrow j$ $\rhd$ … and its index 

10:     $k\leftarrow K$ $\rhd$ Sort $u$ in descending order, permuting $p$ accordingly. The first K elements are already sorted, so we need just a single loop, inserting the last element in the correct position. 

11:     while $k \geq {1\hspace{0pt}\text{~and~}\hspace{0pt}u_{k}} < u_{k + 1}$ do

12:         swap$\left( u_{k},u_{k + 1} \right)$

13:         swap$\left( p_{k},p_{k + 1} \right)$

14:         $k\leftarrow{k - 1}$

15:     end while

16:end for

17:for $i\leftarrow{1,K}$ do $\rhd$ The algorithm stores only K values and their indices 

18:     $v_{i}\leftarrow\frac{e^{u_{i} - m_{V}}}{d_{V}}$

19:     $z_{i}\leftarrow p_{i}$

20:end for

The TopK needs to load each element of the input vector at least once. Running Safe Softmax and the TopK separately requires 5 accesses per input element and 4 accesses if we use Online Softmax instead of Safe Softmax (but still run them separately, one after another). If we improve on the algorithm 3 and keep not only running values of $m$ and $d$ (when iterating over the input vector), but also the vectors of TopK input values $u$ and their indices $p$ - as in the algorithm 4 - we can run this Softmax+TopK fusion with just one memory access per element of the input vector.

5 Benchmarking
-----------------------------------------------------------

Online normalizer calculation reduces the number of memory accesses for the Softmax and Softmax+TopK functions. The softmax function has a very low flops per byte ratio; that means the memory bandwidth should be limiting the performance, even for Online Softmax with its additional few floating point operations per element. Fewer memory accesses should translate into performance improvements, and experiments confirm this.

We implemented a benchmark for GPUs using CUDA C. The benchmark utilizes CUB v1.8.0 for fast parallel reductions. All experiments were run on NVIDIA Tesla V100 PCIe 16 GB, ECC on, persistent mode on, CUDA Toolkit 9.1. Source code of the benchmark is available at github.com/NVIDIA/online-softmax.

### 5.1 Benchmarking softmax

We benchmarked all 3 Softmax algorithms - Naive, Safe, and Online - on different vector sizes for the batch sizes of 4,000 and 10. The large batch case corresponds to the training or batch inference with enough input vectors to saturate the device and and the small batch case corresponds to online inference with too few vectors to occupy the device fully.

Performance improvementVector size $V$Elements per second$0.8$$1$$1.2$$1.4$$1.6$$1.8$$2$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$0$$2 \cdot 10^{10}$$4 \cdot 10^{10}$$6 \cdot 10^{10}$Online/SafeNaiveOnlineSafe

Figure 1: Benchmarking softmax, Tesla V100, fp32, batch size 4000 vectors

For the large batch case (see figure 1) all three algorithms perform similarly up until $V = 1000$ vector size. The NVIDIA Visual Profiler shows that at that point L1 and L2 cache thrashing starts to make all three algorithms limited by the DRAM bandwidth. When this happens Online and Naive algorithms are getting faster than Safe one, quickly achieving $\sim 1.3$x at $V = 4000$ (look for bars in the chart, they are showing performance improvement of Online Softmax over Safe Softmax). This is quite close to $1.33$x reduction in memory accesses for those algorithms.

Performance improvementVector size $V$Elements per second$0.8$$1$$1.2$$1.4$$1.6$$1.8$$2$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$0$$2 \cdot 10^{10}$$4 \cdot 10^{10}$$6 \cdot 10^{10}$$0.8$$1$$1.2$$1.4$$1.6$$1.8$$2$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$10^{6}$$10^{7}$$0$$5 \cdot 10^{9}$$1 \cdot 10^{10}$$1.5 \cdot 10^{10}$$2 \cdot 10^{10}$Online/SafeNaiveOnlineSafeOnline/SafeNaiveOnlineSafe

Figure 2: Benchmarking softmax, Tesla V100, fp32, batch size 10 vectors

The absolute performance for small batch case is lower for all algorithms, see figure 2. The benchmark is running one threadblock per vector; thus small batch case - with 10 vectors - has just 10 threadblocks in the grid. This is not enough to saturate the GPU, both compute and the memory subsystem are underutilized, various latencies are exposed. As in the batch inference case, all three algorithms show similar performance up to $V = 1000$ vector size. After that Naive and Online algorithms outperform Safe one by $\sim 1.15$x.

### 5.2 Benchmarking softmax and top-k

We benchmarked Safe Softmax followed by the TopK (running one after another), Safe Softmax fused with the TopK into a single function, and Online Softmax fused with TopK, again, for 2 cases: 4,000 and 10 vectors. We picked up $K = 5$ in TopK for all runs.

Performance improvementVector size $V$Elements per second$0.8$$1$$1.2$$1.4$$1.6$$1.8$$2$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$0$$2 \cdot 10^{10}$$4 \cdot 10^{10}$$6 \cdot 10^{10}$$0.8$$1$$1.2$$1.4$$1.6$$1.8$$2$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$10^{6}$$10^{7}$$0$$5 \cdot 10^{9}$$1 \cdot 10^{10}$$1.5 \cdot 10^{10}$$2 \cdot 10^{10}$$0$$1$$2$$3$$4$$5$$6$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$0$$5 \cdot 10^{10}$$1 \cdot 10^{11}$$1.5 \cdot 10^{11}$$2 \cdot 10^{11}$Online/SafeNaiveOnlineSafeOnline/SafeNaiveOnlineSafeOnline fused/Safe unfusedOnline Softmax + TopK fusedSafe Softmax + TopK fusedSafe Softmax + TopK unfused

Figure 3: Benchmarking softmax and top-k, Tesla V100, fp32, batch size 4000 vectors

Online fused version is running considerably faster than Safe unfused one. For large batch case - see figure 3 - the performance improvement starts at $1.5$x and goes up as vector size $V$ increases approaching $5$x at $V = 25000$, which corresponds to $5$x reduction in memory accesses. This $5$x comes from $2.5$x due to function fusion and $2$x due to Online Softmax itself.

Performance improvementVector size $V$Elements per second$0.8$$1$$1.2$$1.4$$1.6$$1.8$$2$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$0$$2 \cdot 10^{10}$$4 \cdot 10^{10}$$6 \cdot 10^{10}$$0.8$$1$$1.2$$1.4$$1.6$$1.8$$2$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$10^{6}$$10^{7}$$0$$5 \cdot 10^{9}$$1 \cdot 10^{10}$$1.5 \cdot 10^{10}$$2 \cdot 10^{10}$$0$$1$$2$$3$$4$$5$$6$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$0$$5 \cdot 10^{10}$$1 \cdot 10^{11}$$1.5 \cdot 10^{11}$$2 \cdot 10^{11}$$0$$1$$2$$3$$4$$10^{2}$$10^{3}$$10^{4}$$10^{5}$$10^{6}$$10^{7}$$0$$5 \cdot 10^{9}$$1 \cdot 10^{10}$$1.5 \cdot 10^{10}$Online/SafeNaiveOnlineSafeOnline/SafeNaiveOnlineSafeOnline fused/Safe unfusedOnline Softmax + TopK fusedSafe Softmax + TopK fusedSafe Softmax + TopK unfusedOnline fused/Safe unfusedOnline Softmax +TopK fusedSafe Softmax +TopK fusedSafe Softmax +TopK unfused

Figure 4: Benchmarking softmax and top-k, Tesla V100, fp32, batch size 10 vectors

In the small batch case (see figure 4) Online fused version outperforms Safe unfused one by $1.5$x-$2.5$x. It cannot achieve $5$x because the GPU is underutilized and the performance is limited not by the memory bandwidth, but by various latencies. Yet the reduction in memory accesses helps even in this latency limited case. In small batch case fusion only already brings substantial performance improvements, switching to Online Softmax helps improve performance even further.

The benchmark shows these levels of performance improvement for relatively small $K$ only. The cost of keeping partial TopK results - as in the lines 10-15 of the algorithm 4 - increases quickly as $K$ gets bigger: the performance improvement drops to $3.5$x for $K = 10$, $2$x for $K = 15$, $1.4$x for $K = 30$, and degrades further for bigger $K$s. For these cases the TopK is dominating (in terms of runtime) over the Softmax. Getting rid of separate Softmax and fusing the normalization term calculation into the TopK is still beneficial, but the value goes down as TopK is taking more and more time.

6 Results
------------------------------------------------------

We introduced the way to calculate the normalizer for the Softmax function in a single pass over input data, which reduces memory accesses by $1.33$x for the Softmax function alone. Benchmarks on Tesla V100 show that this materializes in $1.15$x performance improvements for $V \geq 1000$ vector sizes, and for the large batch mode it goes up to $1.3$x when $V \geq 4000$.

If one is using Naive Softmax then switching to Online version improves numerical accuracy with no performance hit or a negligible one.

When the TopK follows the Softmax the new single-pass normalizer calculation enables efficient fusion of these 2 functions resulting in $5$x fewer memory accesses for Softmax+TopK combined. We observed $1.5$x-$5$x performance improvement on Tesla V100, with this $5$x improvement coming from $2.5$x with fusion and $2$x with Online Softmax itself.

These performance improvements could be applied not only to the classical Softmax function; They are orthogonal to many other Softmax optimization techniques including Hierarchical Softmax, Importance Sampling, and SVD-Softmax.

7 Discussion
---------------------------------------------------------

Online Softmax is running up to $1.3$x faster on the latest generation GPU than the one used by major DL frameworks. It also enables very efficient fusion of the Softmax with following TopK showing up to $5$x performance improvement over the traditional Safe Softmax and TopK running separately.

Could we see significantly different speed-ups or even slow-downs on different compute devices, for example CPUs? We didn’t do experiments for those, but if the original code is vectorized and one manages to keep it vectorized for the online normalizer (and partial TopK) calculation then similar speedups could probably be expected.

There could be a way to improve the performance further. The resulting Softmax and even Softmax+TopK fused are still limited by the memory bandwidth, so fusing them with the preceding layer will avoid memory round trip, thus improving performance. This change is more challenging though.

#### Acknowledgments

We would like to thank Christoph Angerer for his valuable comments and suggestions.
