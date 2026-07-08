# 【原理篇·论文精读】量化数学：从 scale/zero-point 到 GPTQ、AWQ、SmoothQuant

![全书路线图：你在这里](../diagrams/roadmap.png)

> 你在这里：第 VII 部分「量化 / 采样 / 投机 / 模型」的开篇，先打原理地基。
> 上一站：[第 30 章](../../ch30-fusedmoe-batch-invariant/narrative/chapter.md)拆完 FusedMoE，收官算子与编译篇。
> 这一章：补上量化框架将要消费的三篇论文的数学。
> 下一站：[第 32 章](../../ch32-ascend-quantization-framework/narrative/chapter.md)看这套数学怎么接进 vLLM 框架、加载执行。

下一章会带你读 `vllm_ascend/quantization` 这套框架的**工程骨架**：`QuantType`（量化类型枚举）怎么分族、`AscendLinearScheme`（所有 linear 量化方案的抽象基类）怎么按粒度注册参数、W8A8 和 W4A16 各自的 `apply` 怎么走。但那套框架**消费**的量化权重和 scale，到底是怎么算出来的？为什么权重能压到 4 bit 还不塌？为什么激活量化比权重量化难得多？这些数学，得先在本章打好——不然那套框架里的每一个参数张量都是天书。

这一章就把这些数学一块块推清楚。这里的 W8A8（权重 8 bit、激活 8 bit）、W4A16（权重 4 bit、激活 16 bit）不是昇腾发明的记号——它们背后站着三篇奠基论文：**SmoothQuant**（arXiv:2211.10438）、**GPTQ**（arXiv:2210.17323）、**AWQ**（arXiv:2306.00978）。昇腾的量化框架是 vLLM 基座量化栈的**昇腾特化顶替**（out-of-tree 后端把 GEMM 换成昇腾 INT8 算子），但它加载的权重格式、scale 语义，全都源自这三篇论文的离线校准产物。读懂这三篇，第 32 章那些参数张量的形状才不再是天书。

我们按四步走。先讲**动机**：W8A8 为何能省一半显存、又为何是量化里最险的一步。再推**数学**：均匀量化的 scale / zero-point / 粒度打底，然后 GPTQ 的二阶补偿、AWQ 的激活感知缩放、SmoothQuant 的迁移难度因子各推一遍，每个关键公式都给论文的 § 和 Eq 锚点。接着上**数值推演**：在小矩阵上手算量化-反量化误差，跑参考实现把四种方法摆到一起对比。最后回**落地**：这些论文产物怎么落到 `vllm_ascend/quantization` 的真实参数形状上，接回第 32 章的框架。

论文的算法（GPTQ 的 Hessian 补偿、AWQ 的缩放搜索、SmoothQuant 的迁移）全都是**离线校准**过程——它们不在 `vllm_ascend` 里跑。本仓只消费它们的**产物**：量化好的 int4 权重、迁移过的 scale。所以本章配了一套论文忠实的小型参考实现（纯 NumPy，能在 host 上单步跑），让你能亲手把论文公式验一遍——因为真正的校准代码不在这个仓库里。

![本章地图：GPTQ / AWQ / SmoothQuant 三篇论文分道推导、汇总落地的剖面图](../diagrams/chapter-map.png)

只想弄清一种量化方案怎么来：只关心 W4A16，顺着「四、GPTQ 二阶补偿」→「五、GPTQ 三大工程优化」→「六、AWQ 激活感知缩放」跳读；只关心 W8A8，直接跳到「七、SmoothQuant 迁移难度」一节。三篇论文都要吃透，就从「一、动机」按序通读到「九、落地」。

---

## 一、动机：省的是访存，险的是 outlier

### 直觉：把 16 bit 的尺子换成 8 bit 的卡尺

一个 FP16（16 位浮点）权重占 2 字节，一个 int8 占 1 字节。把权重和激活都压到 int8，存储直接减半，而且昇腾的 INT8 GEMM（整数矩阵乘）吞吐接近 FP16 的两倍。大模型推理的瓶颈常常不是算力而是**访存带宽**——权重从显存搬到计算单元的那段路。位宽减半，搬运量减半，这才是量化省钱的真正来源：省的是访存，不是算力。

那为什么不无脑压？因为量化是把连续的实数塞进有限的整数档位。8 bit 只有 256 个档位。档位怎么摆、数据怎么落进去，直接决定误差有多大。而激活里藏着一个系统性的坑。

### 机制：outlier 撑大 absmax，非 outlier 通道有效级数塌缩

SmoothQuant §3（arXiv:2211.10438）给了个精确的定量论据。per-tensor（整矩阵共用一个 scale）量化下，通道 $i$ 的**有效量化级数** $\ell_i$ 是：

$$
\ell_i = 2^{N}\cdot \dfrac{m_i}{m}
$$

这里 $m_i$ 是通道 $i$ 的最大幅值、$m$ 是整个矩阵的最大幅值、$N$ 是位宽。人话翻译：整个矩阵共用一把尺子，尺子的量程被最大值 $m$ 撑到头；某个通道自己的幅值只有 $m_i$，它实际用到的刻度数就只有满量程的 $m_i/m$ 那么多。

坑就在这。大模型的激活里，有极少数通道（往往固定就那几个）的幅值系统性地比别人大上百倍——这就是 **outlier**（离群值）。一个 outlier 把 $m$ 撑大 ~100 倍，其余正常通道的有效级数就被压到个位数。SmoothQuant 论文原话：非 outlier 通道的有效级数掉到「2-3 级」，8 bit 的 256 个档位名存实亡。

### 数值：一个 outlier 让最差通道只剩 1.5 级

拿本章参考实现里的小例子说话。一个 3 通道的激活，通道幅值分别是 0.15、0.2、10.0——通道 2 是个约 50 倍的 outlier。整矩阵 absmax（absolute max，绝对值最大值）$m = 10.0$。代进公式：

![per-tensor 量化下 outlier 撑大 absmax，非 outlier 通道有效级数塌缩](../diagrams/fig35-1-outlier-collapse.png)

通道 0 的有效级数是 256 × 0.15 / 10.0 = 3.84，通道 1 是 5.12，而 outlier 通道 2 独占满量程 256 级。换句话说：正常通道本来能用 256 个档位，现在只剩三四个能用。图里再往极端推一步——如果 outlier 是 125 倍，最差的非 outlier 通道有效级数只剩 **1.536**，不足 2 档。8 bit 在这些通道上，退化成了不到 1 bit。

这就是「W8A8 为何险」的全部：不是位宽不够，是被 outlier 挤掉了刻度。权重那边相对温顺（分布集中），激活这边 outlier 猖獗——所以量化的核心矛盾，是怎么对付激活的 outlier。这也是后面三篇论文分道扬镳的起点。

### 源码：W8A8 在框架里的落点

这个「险」在昇腾代码里对应的就是 W8A8 这个量化族。`QuantType` 枚举把落地支持的量化类型一字排开：

```python
# vllm_ascend/quantization/quant_type.py:L26-L36
class QuantType(Enum):
    """Quantization type enum for MoE schemes."""

    NONE = 0
    W8A8 = 1
    W4A8 = 2
    MXFP8 = 3
    W4A16 = 4
    MXFP4 = 5
    W4A8MXFP = 6
```

`W8A8` 是权重和激活都压 8 bit ——正是 SmoothQuant 的领地，也是最险的那一步（激活也要量化）。`W4A16`（权重 4 bit、激活留 16 bit）避开了激活量化，是 GPTQ / AWQ 的领地。`W4A8` 是两者的混合。这张枚举表，就是「论文方法 → 落地命名」的对照锚：每个名字背后，对应本章推导的一支。下面就从最基础的均匀量化开始，一支一支拆。

---

## 二、均匀量化基础：scale、zero-point、粒度

这是本章第一个核心机制，我们仍按「直觉 → 机制 → 源码」三层讲透。

### 直觉：scale 定档宽，zero-point 定零点落在哪

量化像把连续的刻度尺换成有限档位的卡尺。**scale**（记作 $\Delta$，量化步长）决定每一档多宽；**zero-point**（零点）决定「实数 0」落在哪一档。对称量化把零点钉死在正中间。这对权重挺合适——权重大致以 0 为中心、正负都有。但激活常常一批全是正数（比如 ReLU 之后），钉死在中间就白白浪费掉一半负档位。这时加一个 zero-point，把档位整体挪到数据真正的区间上，误差立刻小一截。

### 机制：对称 absmax 量化，误差被步长界住

SmoothQuant §2 Eq.1（arXiv:2211.10438）给了最基础的对称均匀量化定义：

$$
\overline{\mathbf{X}}^{\mathrm{INT8}} = \left\lceil \dfrac{\mathbf{X}^{\mathrm{FP16}}}{\Delta} \right\rfloor, \quad \Delta = \dfrac{\max(|\mathbf{X}|)}{2^{N-1}-1}
$$

$\lceil\cdot\rfloor$ 是四舍五入取整。量化 = 除以步长再取整；反量化 = 乘回步长。步长 $\Delta$ 由 absmax 定：把最大幅值映到最大码位 $2^{N-1}-1$（N=8 时是 127）。

这里有个小陷阱要提前说破。AWQ §3.2 Eq.1（arXiv:2306.00978）的量化函数长得几乎一样，但分母不同：

$$
Q(\mathbf{w}) = \Delta \cdot \mathrm{Round}\!\left(\dfrac{\mathbf{w}}{\Delta}\right), \quad \Delta = \dfrac{\max(|\mathbf{w}|)}{2^{N-1}}
$$

AWQ 用 $2^{N-1}$（满量程），SmoothQuant 用 $2^{N-1}-1$（留一个码位）。两种约定文献里都常见，N=8 时二者之比是 $128/127\approx1.0079$，只差一个码位，**不是矛盾**。本章参考实现刻意把两个函数分开写，不硬凑成一个，就是为了不让你以为哪篇写错了。

误差有个漂亮的界。取整误差最多半个码位，乘回 $\Delta$，反量化误差就不超过半个步长：$|\hat w - w| \le \Delta/2$。位宽每降 1 bit，$2^{N-1}-1$ 近似减半，$\Delta$ 翻倍，误差上界也翻倍——这就是「低位宽更糙」的定量说法。

### 数值：4 个权重的量化-反量化，误差贴着界

拿一个 4 元权重向量手算一遍。$w = [1.27, -0.633, 0.307, -0.951]$，8 bit，SmoothQuant 约定。absmax = 1.27，步长 $\Delta = 1.27/127 = 0.01$。逐个量化再反量化：

<!-- trace: M2 -->

| 权重 $w_i$ | $w_i/\Delta$（$\Delta=0.01$） | 量化码 $\mathrm{round}(w/\Delta)$ | 反量化 $\hat w_i = \mathrm{code}\cdot\Delta$ | $\lvert$误差$\rvert$ |
|---|---|---|---|---|
| 1.27 | 127.0 | 127.0 | 1.27 | 0.0 |
| -0.633 | -63.3 | -63.0 | -0.63 | 0.003 |
| 0.307 | 30.7 | 31.0 | 0.31 | 0.003 |
| -0.951 | -95.1 | -95.0 | -0.95 | 0.001 |

absmax 那个权重（1.27）正好落在码位 127 上，零误差。其余的取整误差最大 0.003——严格小于误差上界 $\Delta/2 = 0.005$，界成立。

zero-point 的威力，换一批全正数据看。激活 $a = [0.1, 0.55, 0.9, 0.32]$，3 bit（8 个档位）。对称量化把零点钉在 0，负半轴那 4 个档位全程用不上，最大误差 0.1。改成非对称量化，用 (scale=0.1143, zero_point=-1) 把 8 个档位全铺在 [0.1, 0.9] 上，最大误差降到 0.0229——只差不到四分之一。

![非对称量化把零点摆对，全正数据的最大误差从 0.1 降到 0.0229](../diagrams/fig35-2-zero-point.png)

同样 8 个档位，零点摆对地方，误差就小四倍。这不是纸上谈兵——它正是落地代码里 `input_offset`（激活的 zero-point）存在的理由。

### 源码：粒度直接编码成参数张量的形状

回到 `vllm_ascend`。论文里抽象的「scale / zero-point / 粒度」，落地代码把它们**直接编码成参数张量的形状**——加载时就定型，`apply` 里不用再判断粒度。看 W8A8 静态方案的参数注册：

```python
# vllm_ascend/quantization/methods/w8a8_static.py:L33-L72
@register_scheme("W8A8", "linear")
class AscendW8A8LinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8 static quantization.

    This scheme uses static per-tensor quantization for activations
    and per-channel quantization for weights.
    """
    # … 省略：get_weight（声明 int8 权重张量）…

    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        params_dict = {}
        params_dict["input_scale"] = torch.empty(1, dtype=params_dtype)      # 激活 per-tensor：单标量 scale
        params_dict["input_offset"] = torch.empty(1, dtype=torch.int8)       # 非对称 zero-point
        return params_dict

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {}
        params_dict["quant_bias"] = torch.empty(output_size, dtype=torch.int32)
        # … 省略：deq_scale 按 dtype 分支声明 …
        params_dict["weight_scale"] = torch.empty(output_size, 1, dtype=params_dtype)   # 权重 per-channel：每输出通道一个 scale
        params_dict["weight_offset"] = torch.empty(output_size, 1, dtype=params_dtype)
        return params_dict
```

`register_scheme("W8A8", "linear")` 是把这个方案注册进第 32 章那套工厂的装饰器。看两个 scale 的形状就懂了粒度：

- `input_scale = torch.empty(1)` ——激活是**一个标量** scale，这就是 **per-tensor**（整矩阵一个 $\Delta$）。
- `weight_scale = torch.empty(output_size, 1)` ——权重是**每输出通道一个** scale，这就是 **per-channel**。
- `input_offset` 的存在，说明激活走的是**非对称**量化，带 zero-point。

docstring 写得明明白白：激活 per-tensor 静态、权重 per-channel。激活这一侧（per-tensor 静态）正对应 SmoothQuant Table 2 里最激进的 **O3** 设置。要说清一处易被误读的细节：Table 2 的 O1-O3 分级只定义**激活**量化的粒度与时机（O3=per-tensor 静态、O2=per-tensor 动态、O1=per-token 动态），三档的**权重都是 per-tensor**；这里的权重 per-channel 不在 Table 2 的 O1-O3 定义里，而是论文后续实验（Table 7 前的原文：per-token 激活量化 + per-channel 权重量化，用于 Llama-2 / Falcon / Mistral / Mixtral）采用的另一项精化。落地代码把 O3 的激活粒度和这项权重 per-channel 精化**组合**到了一起，而不是照搬 Table 2 某一行。形状即粒度——这是本章第一处「论文概念 → 真实参数」的硬对齐。但为什么激活只能 per-tensor 或 per-token，偏偏不能像权重那样按最优的方式切？这就要说到下一节的硬件约束。

---

## 三、粒度与 INT8 GEMM 的硬件约束

### 直觉：GEMM 只能在外维还原 scale

上一节看到激活是 per-tensor、权重是 per-channel。你可能会问：既然 outlier 集中在少数**输入通道**上，那给激活按输入通道各配一个 scale（per-channel），不就把 outlier 单独隔离了吗？数学上这确实最优。可惜硬件做不到。原因在 INT8 GEMM 怎么算。

### 机制：只能沿外维缩放

线性层是 $\mathbf{Y} = \mathbf{X}\mathbf{W}$，SmoothQuant §2（arXiv:2211.10438）先把维度摆清楚（这是论文 Preliminaries 的一句叙述，未编号）：

$$
\mathbf{Y} = \mathbf{X}\,\mathbf{W}, \quad \mathbf{X}\in\mathbb{R}^{T\times C_i},\ \mathbf{W}\in\mathbb{R}^{C_i\times C_o}
$$

$T$ 是 token 数、$C_i$ 是输入通道、$C_o$ 是输出通道。INT8 GEMM 在 int8 域把 $\mathbf{X}\mathbf{W}$ 算成 int32，最后**乘 scale 一次性反量化**回浮点。SmoothQuant §3 Eq.2（arXiv:2211.10438）把这一步写死了——反量化只能在矩阵乘**之后**、从两个外维还原：

$$
\mathbf{Y} = \mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{X}})\cdot\big(\overline{\mathbf{X}}^{\mathrm{INT8}}\,\overline{\mathbf{W}}^{\mathrm{INT8}}\big)\cdot\mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{W}})
$$

$\mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{X}})$ 是激活的 scale（沿外维 token $T$）、$\mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{W}})$ 是权重的 scale（沿外维输出通道 $C_o$）——两者都乘在结果 $\mathbf{Y}$ 的外维上，因为那两维在输出里还在。而输入通道 $C_i$ 是 GEMM 的**收缩维**（累加维）：每个输出元素都要沿所有 $C_i$ 求和，一旦求和，通道的身份就没了——你再没法回头说「这部分和来自通道 0，给它乘 $s_0$」，整个和已经坍成一个数。所以每个 $C_i$ 的 scale 拆不回来。

所以激活的合法粒度只有两种：per-tensor（一个标量）或 per-token（每行 $T$ 一个）。权重则是 per-channel（每输出通道 $C_o$ 一个）。数学上最优的「激活 per-channel（沿 $C_i$）」，硬件上不可行。

### 数值：三种粒度的 scale 长什么样

一个 2 token × 3 通道的激活 X，通道 2 是 outlier：

![三种量化粒度：per-tensor / per-token 硬件可行，激活 per-channel 数学最优却做不到](../diagrams/fig35-3-granularity.png)

per-tensor 全矩阵一个 scale 0.0787，被 outlier 主导；per-token 每行一个（0.0787 和 0.063）；而沿输入通道切的话，scale 是 [0.0012, 0.0016, 0.0787]——差异悬殊，确实能把 outlier 通道单独隔离。但这一维是收缩维，GEMM 还原不了。这就是为什么落地代码只提供 per-tensor 和 per-channel（输出维）两种参数形状。

### 源码：三种粒度接口，独缺内维缩放

框架的抽象基类把这个约束固化成了三个粒度接口：

```python
# vllm_ascend/quantization/methods/base.py:L65-L103
    def get_pertensor_param(self, params_dtype: torch.dtype, **kwargs: Any) -> dict[str, Any]:
        """Return per-tensor parameter specifications (e.g., input_scale)."""
        return {}

    def get_perchannel_param(self, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        """Return per-channel parameter specifications (e.g., weight_scale)."""
        return {}

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        """Return per-group parameter specifications."""
        return {}
```

三个接口——per-tensor、per-channel、per-group（后面 W4A16 会用到），**唯独没有沿内维（输入通道）缩放激活的接口**。框架用「哪个接口非空」来编码方案的粒度，而不是运行时开关。这不是漏写，是「兼容 INT8 GEMM 只能外维」这条硬约束的工程体现。

激活的 outlier 甩不掉、per-channel 又不让用——那怎么办？三篇论文给了三条不同的路。GPTQ 说：干脆别碰激活，只量化权重，但要用二阶信息把误差补回来。AWQ 说：只量化权重，但按激活的重要性来保护关键权重。SmoothQuant 说：把激活的难度「挪账」到权重上去。先看 GPTQ。

---

## 四、GPTQ 二阶补偿：让整层输出尽量不变

### 直觉：量化不是各扫门前雪，是把误差摊给邻居

朴素量化 RTN（round-to-nearest，直接四舍五入）逐个权重独立取整，每个权重「各扫门前雪」。GPTQ / OBQ（Optimal Brain Quantization，最优脑量化，GPTQ 的前身算法）的洞见是：量化不该逐权重最小化 $|w-\hat w|$，而该最小化整**层输出**的误差。量化一个权重会留下取整误差；OBQ 不是放着不管，而是立刻按 Hessian（二阶导矩阵）记录的「谁和谁相关」，把这点误差摊派给还没量化的邻居去抵消——像结账时把多收的零头当场补给下一位顾客。

### 机制：逐层重构目标 + OBQ 贪心量化与补偿

GPTQ §3 Eq.1（arXiv:2210.17323）先定目标——逐层重构：

$$
\underset{\widehat{\mathbf{W}}}{\arg\min}\ \big\| \mathbf{W}\mathbf{X} - \widehat{\mathbf{W}}\mathbf{X} \big\|_2^2
$$

不是让 $\hat{\mathbf{W}}$ 逼近 $\mathbf{W}$，而是让量化后的层输出 $\hat{\mathbf{W}}\mathbf{X}$ 逼近原输出 $\mathbf{W}\mathbf{X}$。这是一切二阶补偿的出发点。

GPTQ 建在 OBQ 之上——OBQ 是更早的一套后训练量化方法，它引入了「用二阶信息（Hessian）在权重间再分配量化误差」这个核心思想；GPTQ 保留 OBQ 的哲学，只把它优化到能上真实规模（下一节的三大工程优化就是干这个的）。OBQ §3 Eq.2（arXiv:2210.17323）给了核心两式——贪心选权重、补偿邻居：

$$
w_q = \underset{w_q}{\arg\min}\ \dfrac{\big(\mathrm{quant}(w_q)-w_q\big)^2}{[\mathbf{H}_F^{-1}]_{qq}}, \qquad \boldsymbol{\delta}_F = -\dfrac{w_q-\mathrm{quant}(w_q)}{[\mathbf{H}_F^{-1}]_{qq}}\cdot(\mathbf{H}_F^{-1})_{:,q}
$$

其中 Hessian $\mathbf{H}_F = 2\,\mathbf{X}_F\mathbf{X}_F^{\top}$（层输出误差对权重的二阶导矩阵）：$\mathbf{X}$ 是喂进这一层的**校准激活**（一小批真实样本的输入），下标 $F$ 表示只取还没量化的权重集合对应的那些激活列，$F$ 本身就是这批尚未量化的全精度权重的下标集合。它是对称矩阵，非对角元 $H_{ij}$ 衡量「量化权重 $i$ 会给权重 $j$ 贡献的输出误差带来多大扰动」——相关性高，就该协同量化、而非各自独立取整。左式：在所有剩余权重里，挑那个「量化误差除以 Hessian 逆对角」最小的先量化。右式：量化掉 $w_q$ 后，把它的误差摊给所有剩余的全精度权重，摊派方向 $\boldsymbol{\delta}_F$ 正比于 Hessian 逆的第 $q$ 列 $(\mathbf{H}_F^{-1})_{:,q}$——这一列刻画了在 $q$ 处注入单位误差、每个剩余权重该分摊多少来抵消。

量化掉一个权重后，Hessian 逆要更新。OBQ §3 Eq.3（arXiv:2210.17323）用一次秩-1 高斯消元，从 $\mathbf{H}^{-1}$ 里剔除与第 $q$ 个权重相关的那一行一列，直接得到「去掉该权重后」的 Hessian 逆 $\mathbf{H}_{-q}^{-1}$——注意这不是把矩阵某行某列简单删掉，而是一次秩-1 更新（下式括号里的减项就是这个更新，外层下标 $-p$ 才是删去对应行列），免得每步都从头重算矩阵求逆：

$$
\mathbf{H}_{-q}^{-1} = \left( \mathbf{H}^{-1} - \dfrac{1}{[\mathbf{H}^{-1}]_{qq}}\,\mathbf{H}^{-1}_{:,q}\,\mathbf{H}^{-1}_{q,:} \right)_{-p}
$$

这套贪心逐权重量化+补偿+收缩 Hessian，有个漂亮的**不变量**保证它一定收敛：每轮严格量化并移除一个权重，剩余全精度集合大小单调减 1；$d$ 个权重必在 $d$ 轮内全部量化并终止。

### 数值：3 个权重、3 轮，补偿把 w₂ 从 0.3347 翻到 0.0

拿参考实现在一个 3 权重、2 bit 的行上跑一遍。权重 $w = [-0.6, -0.812, 0.192]$，配一批强相关的校准激活（让 Hessian 有大非对角，补偿才真会翻转决定）。逐轮追踪：

<!-- trace: M4 -->

| 轮次 | 各候选代价 Δq²/[H⁻¹]_qq | 选中（原索引） | 量化值 q | 量化误差 w−q | 补偿 δ_F（给其余全精度权重） | 补偿后剩余权重 |
|---|---|---|---|---|---|---|
| 1 | [0.001411, 0.001402, 0.002587] | 1 | -0.6693 | -0.1427 | [-0.0297, 0.0, -0.0955] | [-0.6297, -0.6693, 0.0965] |
| 2 | [0.000566, 0.006811] | 0 | -0.6693 | 0.0397 | [0.0, 0.026] | [-0.6693, 0.1225] |
| 3 | [0.086854] | 2 | 0.0 | 0.1225 | [0.0] | [0.0] |

第 1 轮选代价最小的权重 1，量化成 -0.6693，误差 -0.1427 摊给邻居——权重 0 被推到 -0.6297、权重 2 被推到 0.0965。第 2 轮量化权重 0。到第 3 轮，只剩权重 2，此时它已经被前两轮的补偿从原值 0.192 推到了 0.1225，量化到网格上恰好落到 **0.0**。

对比一下：RTN 会直接把原始的 0.192 四舍五入到网格点 0.3347。而 OBQ 因为补偿，把权重 2 翻到了 0.0：

![二阶补偿不是取整微调：w₂ 被翻转，层输出误差降到不足 RTN 的三分之一](../diagrams/fig35-4-obq-compensation.png)

两条码 $[-0.6693, -0.6693, 0.0]$ 与 RTN 的 $[-0.6693, -0.6693, 0.3347]$ 只差最后一个权重。但就这一个翻转，让层输出误差 $\|\mathbf{W}\mathbf{X}-\hat{\mathbf{W}}\mathbf{X}\|^2$ 从 RTN 的 0.13116 降到 0.04441——降了 **2.95 倍**。这就是「量化不是孤立取整」的一图证据。

### 源码：GPTQ 产出的 int4 权重是什么格式

GPTQ 的二阶补偿算法在 `vllm_ascend` 里没有对应代码——它是离线跑的（msmodelslim / llm-compressor 之类的工具链）。本仓消费的是它的**产物**：group 量化的 int4 权重。先看 W4A16 方法怎么声明这套格式：

```python
# vllm_ascend/quantization/methods/w4a16.py:L180-L186
    def __init__(self) -> None:
        self.num_bits = 4  # dtype = torch.int4
        self.pack_factor = 8  # pack 8 of torch.int4 tensors to torch.int32

        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        # … 省略：dynamic_eplb 等无关字段 …
```

```python
# vllm_ascend/quantization/methods/w4a16.py:L79-L82
    offset = pow(2, num_bits) // 2                 # 对称零点：4bit 无符号码位 [0,16) 平移到对称区间 [-8,8)，offset=8 恰为中点
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)

    return unpacked_weight
```

`num_bits = 4`、`pack_factor = 8` 把 8 个 int4 塞进一个 int32——4 bit 权重的带宽降到 FP16 的四分之一。`offset = 2^(N-1) = 8` 把无符号码位平移成对称区间，反量化时减掉。`group_size`（默认 32）就是 GPTQ 论文里的**分组量化**（group-wise quantization）——每组独立 scale，抵抗组内幅值差异（论文的 g128 / g64 / g32 就是不同 group_size）。

OBQ 逐权重贪心的复杂度是 $O(d_{\mathrm{row}}\cdot d_{\mathrm{col}}^3)$，在真实层上贵得离谱。GPTQ 的三大工程优化就是来治这个的。

---

## 五、GPTQ 三大工程优化：同结果、更省访存

### 直觉：把逐笔结账改成整箱结账

GPTQ 在 OBQ 之上加了三处优化，全是「同结果、更省访存 / 更稳数值」的重排，不改算法本身。核心那个叫**懒惰批更新**：把逐笔结账改成整箱结账——只改变「什么时候把账合并写回」，不改每一笔账，所以最终结果一模一样，但少跑很多趟仓库（访存）。

### 机制：全行同序 + 懒惰批 + Cholesky/阻尼

三处优化，逐个看（GPTQ §4，arXiv:2210.17323）：

**其一，全行同序。** OBQ 里每行可以按不同顺序量化。GPTQ 发现：让所有行都按同一列序量化，效果几乎不变，却能让所有行共享同一个 Hessian 逆。这把整体复杂度降下来（GPTQ §4，arXiv:2210.17323）：

$$
O(d_{\mathrm{row}}\cdot d_{\mathrm{col}}^3)\ \longrightarrow\ O\!\big(\max\{d_{\mathrm{row}}\cdot d_{\mathrm{col}}^2,\ d_{\mathrm{col}}^3\}\big)
$$

快约 $\min\{d_{\mathrm{row}}, d_{\mathrm{col}}\}$ 倍。

**其二，懒惰批。** GPTQ §4 Eq.4-5（arXiv:2210.17323）把逐列补偿攒成一批（B=128 列）：

$$
\boldsymbol{\delta}_F = -\big(\mathbf{w}_Q-\mathrm{quant}(\mathbf{w}_Q)\big)\big([\mathbf{H}_F^{-1}]_{QQ}\big)^{-1}(\mathbf{H}_F^{-1})_{:,Q}
$$

块内逐列即时更新、块末才对块外做一次全局补偿。这不是近似——它是逐列 OBQ 的**代数重排**。好处是把访存密集的操作攒成算力密集的批操作，实测再快一个数量级。

**其三，Cholesky + 阻尼。** 大模型上 Hessian 逆可能变不定（indefinite），补偿方向就错了。对策：给 Hessian 对角加 1% 均值的阻尼（dampening）——把对角整体抬高一点，保证矩阵**正定**（所有特征值为正，不再出现导致分解/求逆失败的负特征值），再对这个阻尼后的 Hessian 做一次 Cholesky 分解 $\mathbf{H}=\mathbf{L}\mathbf{L}^{\top}$，用数值稳定的三角求解一次预算好所有需要的行，替代反复的矩阵求逆。

完整流程就是论文的 Algorithm 1：

```text
# GPTQ Algorithm 1 (arXiv:2210.17323 §4)
Q ← 0_{d_row × d_col}                       // 量化输出
E ← 0_{d_row × B}                           // 块量化误差
H^{-1} ← Cholesky(H^{-1})^T                  // 预计算：对(阻尼后)Hessian H=(2XX^T+λI)做Cholesky，λ=1%均值对角；此后用三角求解替代逐步矩阵求逆
for i = 0, B, 2B, ... do
    for j = i, ..., i+B-1 do
        Q[:,j] ← quant(W[:,j])                                  // 量化当前列
        E[:,j-i] ← (W[:,j] - Q[:,j]) / [H^{-1}]_{jj}            // 量化误差
        W[:, j:i+B] ← W[:, j:i+B] - E[:,j-i] · H^{-1}_{j, j:i+B} // 块内补偿
    end for
    W[:, i+B:] ← W[:, i+B:] - E · H^{-1}_{i:i+B, i+B:}          // 块末全局补偿
end for
```

关键**不变量**：`gptq_quantize` 的输出与 blocksize（批大小）无关。因为懒惰批是逐列 OBQ 的代数恒等重排，blocksize=1 到 $d_{\mathrm{col}}$ 都得到同一个 Q。

### 数值：四个 blocksize，输出误差逐位相同

参考实现在一个 3×6、3 bit 的权重上，把 blocksize 扫过 {1, 2, 3, 6}：

<!-- trace: M5 -->

| blocksize | GPTQ 层输出误差 $\|\mathbf{W}\mathbf{X}-\hat{\mathbf{W}}\mathbf{X}\|^2$ | 相对 RTN（1.15236）倍数 |
|---|---|---|
| 1 | 0.99387 | 1.159 |
| 2 | 0.99387 | 1.159 |
| 3 | 0.99387 | 1.159 |
| 6 | 0.99387 | 1.159 |

四个 blocksize，输出误差逐位相同（0.99387），跨块最大差 0.0——懒惰批是效率重排，不是新算法，这一栏数字是该不变量的数值印证。

![懒惰批只改访存效率、不改结果：四个 blocksize 的层输出误差逐位相同](../diagrams/fig35-5-lazybatch-invariance.png)

同时四个都优于 RTN 基线的 1.15236：补偿确实降了误差，且与批大小无关。（这个小玩具上 GPTQ 相对 RTN 收益温和，规模化到真实层才显著——真实层的 Hessian 相关性远比 3×6 丰富。）这一切的省，都是访存 / 带宽层面的，结果一分不改。

### 源码：懒惰批产出的打包权重怎么被解开

懒惰批是离线量化时的优化，`vllm_ascend` 里同样没有它的代码。它跑完，产出的仍是上一节那套紧凑的 int4 打包格式——8 个 int4 压进一个 int32。本仓加载时得先把它解开。`unpack_from_int32`（int32 解包函数）逐个移位、取低位，把 8 个权重从一个 int32 里抠出来：

```python
# vllm_ascend/quantization/methods/w4a16.py:L55-L82
    pack_factor = 32 // num_bits                   # 一个 int32 里塞了几个权重：4bit → 8
    mask = (1 << num_bits) - 1                      # 取低 num_bits 位的掩码
    # … 省略：按 packed_dim 分配输出张量 …
    for i in range(pack_factor):
        unpacked_weight[:, i::pack_factor] = (weight >> (num_bits * i)) & mask   # 移位+掩码拆出第 i 个
    # … 省略：裁掉打包对齐的尾部 …
    offset = pow(2, num_bits) // 2                  # 对称零点：4bit → offset=8
    unpacked_weight = (unpacked_weight - offset).to(torch.int8)                  # 平移成对称区间
    return unpacked_weight
```

`>> (num_bits*i)` 把第 $i$ 个权重移到低位、`& mask`（按位与掩码）取出，最后减 `offset` 平移成对称区间。解包只是第一步——权重还得重排到昇腾 kernel 期望的内存布局。`process_weights_after_loading`（加载后权重后处理）把「解包 → transpose 重排 → 再打包」连成一趟：

```python
# vllm_ascend/quantization/methods/w4a16.py:L335-L343
        unpacked_w13_weight = (
            unpack_from_int32(layer.w13_weight_packed.data.flatten(0, 1), ..., self.num_bits)
            .view(w13_shape[0], w13_shape[1], -1)
            .transpose(1, 2)                        # 重排到昇腾 kernel 期望的布局
            .contiguous()
            .int()
        )
        layer.w13_weight_packed.data = pack_to_int32(unpacked_w13_weight)        # 重排后再打回 int32
```

「解包 → transpose 重排 → 再打包」这一趟，正是 GPTQ「省的是访存」在消费端的对应体现：离线懒惰批把权重压成紧凑的 int4 格式省了带宽，本仓加载时把它摊开、按昇腾算子布局重排、再压回去——算法留在离线，本仓只做格式适配。GPTQ 只管让权重误差最小；下一节的 AWQ 换了个角度：不是所有权重都一样重要。

---

## 六、AWQ 激活感知缩放：重要的不是权重大，是激活大

### 直觉：给重要的字用更粗的笔

AWQ 的反直觉洞见：一个权重重不重要，不看它自己多大，看它乘的**激活**多大。为什么是激活？因为一个权重 $w$ 量化成 $\hat w$，在输出里留下的误差是 $(w-\hat w)\cdot x$——激活 $x$ 越大，同样的权重取整误差就被放得越大。所以「和大激活相乘的权重」比「自己数值大、却只乘小激活的权重」更该保护。AWQ §3.1（arXiv:2306.00978）发现，只有 0.1%-1% 的权重是「显著」的，而且正是**按激活幅值**判定的。保护这批显著权重的办法：把它放大 $s$ 倍、对应激活缩小 $s$ 倍，乘积不变，但放大后的权重在量化格子里占的相对误差缩到约 $1/s$——像给重要的字用更粗的笔写，手抖的相对影响更小。

### 机制：缩放降相对误差 ~1/s，再搜最优 α

AWQ §3.2 Eq.2 及其后的误差推论（arXiv:2306.00978）给了核心推导。显著权重 $w$ 乘 $s>1$、激活 $x$ 除 $s$，数学等价（下式左半即论文 Eq.2）；论文紧接着写出未缩放的基础误差式（编号 Eq.3）与缩放后的误差式，再由二者相除得到缩放前后的误差比（下式右半的 $\mathrm{Err}_{s}/\mathrm{Err}_{1}$ 是本章为这个比值取的记号，论文以文字给出、并未单独编号）：

$$
Q(w\cdot s)\cdot\dfrac{x}{s} = \Delta'\cdot\mathrm{Round}\!\left(\dfrac{ws}{\Delta'}\right)\cdot x\cdot\dfrac{1}{s}, \qquad \dfrac{\mathrm{Err}_{s}}{\mathrm{Err}_{1}} = \dfrac{\Delta'}{\Delta}\cdot\dfrac{1}{s}
$$

取整误差 $\mathrm{RoundErr}(\cdot)$ 期望约 0.25、与 $s$ 无关；而放大**单个**权重通常不改组内 absmax，所以新步长 $\Delta'\approx\Delta$。于是相对误差 $\approx 1/s < 1$。这就是「缩放为何能保护显著权重」的数学证明。

但 $s$ 不能无脑放大——前提是「$\Delta'\approx\Delta$」。一旦把 $w$ 放大到它自己成了组内新的最大值，步长 $\Delta'$ 就随之涨大（$\Delta$ 由组 absmax 定），这一涨会拖累组内**所有**非显著权重的量化，$1/s$ 的好处也被 $\Delta'/\Delta>1$ 抵消。所以要搜一个既能保护显著权重、又不至于改写组 absmax 的 $s$。AWQ §3.2 Eq.4-5（arXiv:2306.00978）用一个超参 $\alpha$ 搜这个最优缩放：

$$
\mathbf{s}^{*} = \underset{\mathbf{s}}{\arg\min}\ \big\| Q(\mathbf{W}\cdot\mathrm{diag}(\mathbf{s}))\,(\mathrm{diag}(\mathbf{s})^{-1}\mathbf{X}) - \mathbf{W}\mathbf{X} \big\|, \qquad \mathbf{s} = \mathbf{s_X}^{\alpha}
$$

$\mathbf{s_X}$ 是逐输入通道的激活平均幅值，$\alpha\in[0,1]$ 网格搜：$\alpha=0$ 不缩放、$\alpha=1$ 最激进。**不变量**：只要放大不改组 absmax（$\Delta'=\Delta$），相对误差期望就精确等于 $1/s$。

### 数值：s=2/4 实测误差比贴着 1/s

参考实现在 4 bit 上，对多次随机显著权重取误差均值（因为 $1/s$ 是期望意义的规律），扫 $s$：

<!-- trace: M6 -->

| 缩放 s | 朴素预测 1/s | 实测平均误差比（均值误差之比） | 组 absmax 被位移比例 |
|---|---|---|---|
| 1.0 | 1.0 | 1.0 | 0.0 |
| 2.0 | 0.5 | 0.5694 | 0.0 |
| 4.0 | 0.25 | 0.2658 | 0.0 |

$s=2$ 实测误差比 0.5694（朴素 0.5）、$s=4$ 实测 0.2658（朴素 0.25），都贴着 $1/s$——因为组 absmax 被位移比例恒为 0.0，$\Delta$ 确实没变。

![AWQ 放大显著权重把相对误差压到约 1/s，但 s 过大会撑大 Δ′ 反噬非显著通道](../diagrams/fig35-6-awq-scaling.png)

图的下半截给了反面：$s$ 一旦大到让显著权重反成组内 max（$s=8$ 时 $\Delta'$ 涨 8 倍），整组刻度变粗，非显著通道反受害——所以要网格搜一个最优 $\alpha$。参考实现在一个激活-显著通道的玩具层上搜出最优 $\alpha=0.25$，把重构损失从不缩放（$\alpha=0$）的 0.7094 降到 0.2484，即 2.86 倍。AWQ 和 GPTQ 都走权重-only（激活留 16 bit），落地都是 W4A16。可激活量化那道坎（W8A8）终究得有人迈——那就是 SmoothQuant。

### 源码：AWQ 权重同样落到 W4A16

AWQ 的缩放搜索也是离线跑的，缩放融进权重后，产物同样是 W4A16 的 int4 打包权重——和 GPTQ 共用 `w4a16.py` 里那套 `num_bits=4`、`group_size` 分组的格式声明（第四节）和 `unpack_from_int32` 解包路径（第五节）。落地看不出是 GPTQ 还是 AWQ 产的权重，只看到统一的 4 bit 打包格式。差异全在离线那一端：GPTQ 用 Hessian 补偿，AWQ 用激活感知缩放。

---

## 七、SmoothQuant 迁移难度：把激活的坑挪给权重

### 直觉：激活难量化，就把难度挪账给权重

前面 GPTQ、AWQ 都绕开了激活量化（只碰权重）。SmoothQuant 正面迎上 W8A8：激活难量化、权重好量化，那就把一部分难度「挪账」过去。激活除以 $s$、权重乘以 $s$，乘积完全不变（代数恒等）。$\alpha$ 控制挪多少——挪得恰到好处，激活的 outlier 被摊平，权重也没被压垮。而且这个 $s$ 能离线融进上一层（LayerNorm / Linear）的参数，运行时零额外 kernel。

### 机制：等价迁移 + α 均分难度

SmoothQuant §4 Eq.3（arXiv:2211.10438）是迁移变换：

$$
\mathbf{Y} = \big(\mathbf{X}\,\mathrm{diag}(\mathbf{s})^{-1}\big)\cdot\big(\mathrm{diag}(\mathbf{s})\,\mathbf{W}\big) = \widehat{\mathbf{X}}\,\widehat{\mathbf{W}}
$$

激活逐通道除 $s$、权重逐通道乘 $s$，$\mathrm{diag}(\mathbf{s})^{-1}\mathrm{diag}(\mathbf{s})$ 抵消，乘积恒等于原 $\mathbf{X}\mathbf{W}$。这是**代数恒等**，对任意正 $s$ 成立，不是近似。

难度挪多少，由 SmoothQuant §4 Eq.4（arXiv:2211.10438）的难度因子定：

$$
\mathbf{s}_j = \dfrac{\max(|\mathbf{X}_j|)^{\alpha}}{\max(|\mathbf{W}_j|)^{1-\alpha}}
$$

$\alpha$ 是迁移强度（migration strength）。为什么甜点常在 0.5？$\alpha<0.5$ 时难度仍偏在激活一侧（outlier 没摊平够），$\alpha>0.5$ 时又过度倒向权重（权重的 max 被抬起来）；恰好 $\alpha=0.5$ 时，迁移后激活与权重在通道 $j$ 的 max 收敛到二者的**几何平均** $\sqrt{\max(|\mathbf{X}_j|)\max(|\mathbf{W}_j|)}$，两侧最大值齐平、量化难度均分——这是 OPT / BLOOM 的甜点，下面数值表里 $\alpha$ 扫描呈 U 形、谷底正落在 0.5，就是它的经验印证。$\alpha\to1$ 全推给权重（权重误差爆）、$\alpha\to0$ 全留激活，两极端都恶化。**不变量**：迁移对任意正 $s$ 乘积无损；$\alpha$ 只改量化友好度，不改这个乘积。

### 数值：α 呈 U 形，甜点在 0.5

参考实现在一个 T=16、$C_i$=4 的玩具层上（输入通道 0 是约 44 倍的激活 outlier），扫 $\alpha$：

<!-- trace: M7 -->

| α（迁移强度） | 通道0 迁移因子 s | 原始 per-tensor 误差（raw） | 迁移后误差 |
|---|---|---|---|
| 0.0 | 6.6752 | 0.082 | 0.0325 |
| 0.25 | 7.1736 | 0.082 | 0.0273 |
| 0.5 | 7.7092 | 0.082 | 0.0233 |
| 0.75 | 8.2849 | 0.082 | 0.0261 |
| 1.0 | 8.9035 | 0.082 | 0.0285 |

原始 per-tensor 误差恒 0.082（不迁移的基线）。迁移后误差随 $\alpha$ 呈 U 形：两端（0.0 → 0.0325、1.0 → 0.0285）都更差，$\alpha=0.5$ 降到最低 0.0233，即 3.52 倍。这印证了「难度均分」的甜点在中间。

![α=0.5 迁移把激活 outlier 从 8.9 压到与权重相等的几何均值 1.1549，乘积无损](../diagrams/fig35-7-migration.png)

图里看得最直观：迁移前激活通道 0 是 8.9 的 outlier、其余不到 0.4，per-tensor 量化必被它主导；$\alpha=0.5$ 迁移后，激活和权重每通道的 max 相等（几何均值，通道 0 两侧都是 1.1549），outlier 被摊平，激活通道最大值之差从 44.3 倍压到 4.2 倍，而 Eq.3 迁移残差是 0.0（乘积一分不差）。

### 源码：W8A8 消费迁移后的 scale

SmoothQuant 的迁移 $s$ 在离线校准时算好、融进上一层参数。本仓的 W8A8 消费其产物。反量化那一步最能看出论文和代码的对齐：

```python
# vllm_ascend/quantization/methods/w8a8_static.py:L158-L161
        if ascend_quant_method == COMPRESSED_TENSORS_METHOD:
            deq_scale = layer.input_scale.data * layer.weight_scale.data
            layer.deq_scale = torch.nn.Parameter(deq_scale, requires_grad=False)
```

反量化 scale = 激活 scale × 权重 scale。INT8 GEMM 在 int8 域算完，两侧 scale 相乘一次性还原浮点——对应 SmoothQuant §3 Eq.2：

$$
\mathbf{Y} = \mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{X}})\cdot\big(\overline{\mathbf{X}}^{\mathrm{INT8}}\,\overline{\mathbf{W}}^{\mathrm{INT8}}\big)\cdot\mathrm{diag}(\boldsymbol{\Delta}_{\mathbf{W}})
$$

这正是第三节推过的那条 Eq.2。代码里的 `input_scale` 就是 $\boldsymbol{\Delta}_{\mathbf{X}}$、`weight_scale` 就是 $\boldsymbol{\Delta}_{\mathbf{W}}$——两者都在离线校准时算好（SmoothQuant 的迁移因子 $s$ 已经融进 `input_scale`，前向时不必再单独乘一次 $s$）；前向时把它们相乘，一次反量化还原。这是把论文公式和落地实现对齐的最直接一行。

激活量化 W8A8 还分静态和动态两套。静态（上面那套）预存 `input_scale`，省一次运行时求 scale，更快，对应 SmoothQuant 的 O3。动态则每 token 现算 scale：

```python
# vllm_ascend/quantization/methods/w8a8_dynamic.py:L48-L80
@register_scheme("W8A8_DYNAMIC", "linear")
class AscendW8A8DynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W8A8_DYNAMIC.

    This scheme uses dynamic per-token quantization for activations
    and per-channel quantization for weights.
    """
    # … 省略：get_weight / get_perchannel_param …

    def apply(self, layer, x, bias=None, tp_rank=0):
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x)   # 运行时逐 token 求 scale
        # … 省略：chunk 分块 matmul 分支 …
```

`npu_dynamic_quant`（昇腾的动态量化算子）在前向时按 token 现算 scale，没有预存的 `input_scale`。这对应 SmoothQuant 的 O1 **激活**设置——per-token 动态（O1-O3 是论文按激活量化粒度/时机分的三档效率级别，见[第二节](#二均匀量化基础scalezero-point粒度)；O1 在 Table 2 里的权重本是 per-tensor，昇腾则配上前面那项 per-channel 权重精化）。静态 vs 动态，就是论文 O1-O3 那条效率 / 精度权衡谱：静态省运行时开销（更快），动态更贴每 token 真实分布（更准）。

---

## 八、数值推演：四法同台，各在自己的赛道降误差

三篇论文讲完，把四种方法（含朴素 RTN）摆到一张台上对比。同一道量化题，三种解法给出三张答卷：GPTQ 靠二阶补偿把误差摊给邻居，AWQ 靠激活感知缩放保护关键权重，SmoothQuant 靠等价迁移把难度挪给权重——现在把它们摆到一张台上，看各自压低了多少误差。但有个前提得先说清：**它们不在同一位宽赛道上**。GPTQ / AWQ 走 W4 权重-only，SmoothQuant 走 W8A8。所以只能「各自相对自己的朴素基线」比降幅，不能横向比谁的绝对误差小。

参考实现让每种方法对照「自己的」朴素基线、在「自己的」regime 里跑：

<!-- trace: M8 -->

| 方法（regime） | 朴素基线误差 | 方法误差 | 降幅 × |
|---|---|---|---|
| GPTQ (W4 权重-only) | 1.0735 | 0.9969 | 1.0768 |
| AWQ (W4 权重-only) | 0.7094 | 0.2484 | 2.8558 |
| SmoothQuant (W8A8) | 0.082 | 0.0233 | 3.5177 |

**不变量**：每种方法的「方法误差」都不超过它的朴素基线——补偿 / 缩放 / 迁移只做量化误差的再分配或难度迁移，只减不增。三法降幅分别 1.0768、2.8558、3.5177 倍，都 ≥1。这三个 regime 各自落回框架的哪一端也一目了然：GPTQ / AWQ 的降幅对应 W4A16 消费端的 `unpack_from_int32` 反量化（第四、五节的 `offset`），SmoothQuant 的降幅对应 W8A8 消费端的 `deq_scale`（下一节展开），量化-反量化在真实框架里的对应参数形状，就在紧接着的落地一节。

![三法各自相对朴素基线降低量化误差，各在其位宽 regime 内部对照](../diagrams/fig35-8-showdown.png)

三根降幅箭头都朝下，但分属不同赛道：GPTQ 在小玩具上补偿收益温和（规模化后才显著），AWQ 和 SmoothQuant 收益明显。图的要点是「各自相对基线的改进」，不是谁的绝对误差小。这也回答了本章开篇的问题——为什么权重能压到 4 bit 还不塌：因为 GPTQ 用二阶信息补、AWQ 按激活重要性护，都不是无脑取整。

---

## 九、落地：论文数学接回 vllm_ascend 框架

把这一路数学收束回 `vllm_ascend/quantization`。这套框架（第 32 章的主角）是 vLLM 基座量化栈的昇腾特化顶替——同样的「按方法注册 scheme、按粒度注册参数」范式，把 GEMM 换成昇腾 INT8 算子。本章三篇论文，正好一一对上框架里的量化族：

- **粒度即参数形状**。`input_scale=torch.empty(1)` 是 per-tensor、`weight_scale=torch.empty(output_size, 1)` 是 per-channel。加载即定型，`apply` 里无需判断粒度——这是 SmoothQuant Fig.3 粒度分类的代码固化。

- **W8A8 = SmoothQuant 的 O3 / O1**。静态（`AscendW8A8LinearMethod`，激活 per-tensor 静态）对应 O3、动态（`AscendW8A8DynamicLinearMethod`，`npu_dynamic_quant` 逐 token）对应 O1。`deq_scale = input_scale * weight_scale` 是两侧 scale 一次反量化，对应 SmoothQuant Eq.2。

- **W4A16 = GPTQ / AWQ 的权重落地形态**。`num_bits=4`、`pack_factor=8`（8 个 int4 打包进 int32）、`group_size`（默认 32，对应论文 g128 / g64 / g32）、`offset=2^(N-1)` 对称零点。GPTQ 和 AWQ 离线产出的 4 bit 权重，走的是同一条解包消费路径——差异全在离线校准端。

- **`QuantType` 枚举是论文与落地的对照锚**。W8A8 是 SmoothQuant 领地、W4A16 是 GPTQ / AWQ 领地、W4A8 是混合。

一句话收尾：论文的算法（Hessian 补偿、激活感知缩放、迁移变换）全在**离线校准**跑；`vllm_ascend` 只消费它们产出的权重和 scale。所以你在框架里看到的 `input_scale`、`weight_scale`、`group_size`、`deq_scale`，每一个背后都站着本章推过的一支数学。这些参数怎么被工厂注册、怎么在前向里被 `apply` 消费，正是[下一章](../../ch32-ascend-quantization-framework/narrative/chapter.md)的主线——现在你手里先有了它们的数学出处，读那套框架时就不会再把它当黑盒。
