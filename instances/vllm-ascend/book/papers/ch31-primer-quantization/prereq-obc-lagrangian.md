# 先修补充：OBS→OBQ 的拉格朗日闭式解 与 Hessian 逆秩-1 更新

> 本文件是论文包的**先修子论文摘录**，服务于正文「四、GPTQ 二阶补偿」一节引用的两个核心公式
> （本章记法：OBQ §3 Eq.2 贪心选权重+补偿方向、OBQ §3 Eq.3 Hessian 逆秩-1 更新）。GPTQ 论文自己
> 在这一段也只是 brief summarize，不展开推导；完整推导来自 GPTQ 作者同一批人的前作：
>
> **《Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning》
> Frantar & Alistarh, 2022, arXiv:2208.11580**（下文简称 **OBC**）。
>
> 只摘录与这两个公式直接相关的几段核心推导，不收录整篇论文。所有 `# PAPER: OBC §x Eq.y` 锚
> 指向 OBC 论文**自己的**章节/公式编号（与正文引用的 GPTQ 编号是两套体系，见第 5 节的对应表）。

## 0. 命名落差：GPTQ 论文说的「OBQ」，就是这篇《Optimal Brain Compression》

GPTQ 论文正文把它称作 **Optimal Brain Quantization（OBQ）**，读起来像是在引用一个专门叫这个
名字的方法。但如果去搜「Optimal Brain Quantization」这个论文题目，搜不到——因为原论文的题目
其实是 *Optimal Brain **Compression***（OBC），OBQ 只是 OBC 论文内部第 5 节提出的一个**子方法**
的名字（OBC 论文自己也把「量化」这个应用叫作 Optimal Brain Quantizer，见下文 §5 标题）。也就是说：

- 论文题目：**Optimal Brain Compression**（arXiv:2208.11580）——这篇论文同时讲剪枝（第 4 节
  ExactOBS）和量化（第 5 节 OBQ）两件事。
- GPTQ 论文引用/建在其上的，只是这篇论文里量化那一半——OBQ。
- GPTQ 论文的行文里始终只提「OBQ」，从未提「Optimal Brain Compression」这个真实标题——如果照
  着 GPTQ 论文里的名字去搜引用文献，会搜错论文名，这是本节要先点破的第一个坑。

## 1. 出发点：逐层重构目标（回顾）

OBC §3「The Layerwise Compression Problem」把要解的问题写成（对应正文 GPTQ §3 Eq.1 的来源）：

```text
# PAPER: OBC §3 Eq.2
argmin_{Ŵ}  ‖ W X - Ŵ X ‖²_2   s.t.  C(Ŵ) > C
```

其中 `C(Ŵ) > C` 是一个通用的「压缩约束」占位符——剪枝时它是「非零权重数 ≤ 预算」，量化时它是
「每个权重必须落在量化网格上」。这一步就是正文 GPTQ §3 Eq.1（逐层重构目标）的原始出处；GPTQ
论文把约束具体化成了「量化」，本节接下来要展开的，就是这个具体化怎么一步步推出闭式解。

## 2. OBS 剪枝闭式解——为什么下标是 p 而不是 q

OBC §3「The Optimal Brain Surgeon (OBS) Framework」先给出**剪枝**（把某权重设为 0）版本的经典
OBS 公式，这是后面一切推导的母版：

```text
# PAPER: OBC §3 Eq.3 (OBS Framework)
w_p = argmin_{w_p}  w_p² / [H⁻¹]_pp,     δ_p = − w_p / [H⁻¹]_pp · H⁻¹_{:,p}
```

注意这里的下标从一开始就是 **p**（不是 q）——p 来自英文 *prune*（剪枝）。OBC 论文第 4 节
「An Optimal Greedy Solver for Sparsity」把这套公式实例化成剪枝算法时，全程也用 p。这是本节
第 5 部分要点破的关键背景：**p 是这篇母论文里剪枝语境下的原生下标**，不是量化语境专属的记号。

## 3. 从「剪枝到 0」到「量化到网格点」：拉格朗日推导

OBC §5「The Optimal Brain Quantizer (OBQ)」把 §3 Eq.3 的剪枝公式重新推导了一遍，只是把约束从
「权重变成 0」换成「权重变成量化网格点」。完整推导分两步。

### 3.1 拉格朗日函数

设 `δ_p` 是对**所有剩余权重**的补偿更新（向量），约束是「第 p 个分量补偿到目标值 t」：剪枝时
`t = −w_p`（补偿到 0），量化时 `t = quant(w_p) − w_p`（补偿到最近网格点）。OBC 论文给出的拉格
朗日函数（原始剪枝形式，随后代入量化目标）：

```text
# PAPER: OBC §5 Eq.6
L(δ_p, λ) = δ_p^T H δ_p + λ·(e_p^T δ_p − (−w_p))
```

`e_p` 是第 p 个标准基向量（只在第 p 位是 1，其余是 0）。论文原话：「The optimal solution is
then derived by first finding the optimal solution to δ_p via setting the derivative ∂L/∂δ_p
to zero and then substituting this solution back into L and solving for λ」——论文到这里就不
再展开代数，直接给出结果（下面 3.2 是本节按标准拉格朗日乘子法把这几步代数补全，非论文原文逐
字展开，但每一步都对应论文这句话描述的两次求导+回代）：

### 3.2 展开代数（补全，非论文逐字展示）

记通用目标 `t`（剪枝时 t=−w_p，量化时 t=quant(w_p)−w_p，用一般记号先推，最后代入）：

**第一步：对 δ_p 求导置零。**

$$
\frac{\partial L}{\partial \boldsymbol{\delta}_p} = 2\mathbf{H}\boldsymbol{\delta}_p + \lambda\,\mathbf{e}_p = \mathbf{0}
\quad\Longrightarrow\quad
\boldsymbol{\delta}_p = -\frac{\lambda}{2}\,\mathbf{H}^{-1}\mathbf{e}_p = -\frac{\lambda}{2}\,\mathbf{H}^{-1}_{:,p}
$$

**第二步：代回约束 `e_p^T δ_p = t` 解出 λ。**

$$
\mathbf{e}_p^{\top}\boldsymbol{\delta}_p = -\frac{\lambda}{2}\,\big[\mathbf{H}^{-1}\big]_{pp} = t
\quad\Longrightarrow\quad
\lambda = -\frac{2t}{[\mathbf{H}^{-1}]_{pp}}
$$

**第三步：把 λ 代回 δ_p，得到补偿方向。**

$$
\boldsymbol{\delta}_p = -\frac{\lambda}{2}\,\mathbf{H}^{-1}_{:,p}
= \frac{t}{[\mathbf{H}^{-1}]_{pp}}\,\mathbf{H}^{-1}_{:,p}
$$

代入量化目标 `t = quant(w_p) − w_p`（等价地写成 `−(w_p − quant(w_p))`），恰好就是论文给出的：

```text
# PAPER: OBC §5 Eq.7 (第二式：补偿方向)
δ_p = − (w_p − quant(w_p)) / [H⁻¹]_pp · H⁻¹_{:,p}
```

**第四步：把 δ_p 代回目标函数，求这一步造成的最小损失增量（决定挑哪个权重先量化）。**

$$
\boldsymbol{\delta}_p^{\top}\mathbf{H}\boldsymbol{\delta}_p
= \left(\frac{t}{[\mathbf{H}^{-1}]_{pp}}\right)^{2}\mathbf{H}^{-1}_{:,p}{}^{\top}\mathbf{H}\,\mathbf{H}^{-1}_{:,p}
$$

利用 `H⁻¹_{:,p} = H⁻¹ e_p` 和 `H⁻¹H=I`：

$$
\mathbf{H}^{-1}_{:,p}{}^{\top}\mathbf{H}\,\mathbf{H}^{-1}_{:,p}
= \mathbf{e}_p^{\top}\mathbf{H}^{-1}\mathbf{H}\mathbf{H}^{-1}\mathbf{e}_p
= \mathbf{e}_p^{\top}\mathbf{H}^{-1}\mathbf{e}_p = [\mathbf{H}^{-1}]_{pp}
$$

两式合并，损失增量精确化简为：

$$
\boldsymbol{\delta}_p^{\top}\mathbf{H}\boldsymbol{\delta}_p = \frac{t^2}{[\mathbf{H}^{-1}]_{pp}}
$$

代入 `t = quant(w_p) − w_p`，正是论文给出的挑选准则：

```text
# PAPER: OBC §5 Eq.7 (第一式：贪心选权重)
w_p = argmin_{w_p}  (quant(w_p) − w_p)² / [H⁻¹]_pp
```

这就是正文 GPTQ §3 Eq.2（本章记法）的完整来源：**在「剩余全精度权重」这个约束二次型上，用拉
格朗日乘子法求出的闭式解**——分母 `[H⁻¹]_pp` 衡量「这个权重方向有多容易被 Hessian 修正」，
分子是量化误差的平方，两者相除就是「量化这个权重、其余权重最优补偿之后，层输出损失会增加多少」。

## 4. Hessian 逆秩-1 更新（Lemma 1）

量化/剪枝掉一个权重后，剩下权重的 Hessian 逆要更新——OBC §4「An Optimal Greedy Solver for
Sparsity」给出的核心引理：

```text
# PAPER: OBC §4 Eq.4 (Lemma 1, Row & Column Removal)
H⁻¹_{-p} = ( H⁻¹ − 1/[H⁻¹]_pp · H⁻¹_{:,p} H⁻¹_{p,:} )_{-p}
```

论文原文：「which corresponds to performing Gaussian elimination of row and column p in H⁻¹
followed by dropping them completely. This has Θ(d_col²) time complexity.」——这正是正文
GPTQ §3 Eq.3（本章记法）的直接出处：GPTQ 论文只是原样搬了这个引理，把外层的 H、H⁻¹ 换成量化
语境下已经用 q 记的量化权重集合的 Hessian，公式结构完全没变。

### 4.1 证明（论文 Appendix A.1，简化到「p 是最后一个下标」的等价情形以便读者跟手推）

论文附录给出的是「p 处在矩阵任意位置」的一般证明（需要引入置换分块记号）。这里用**等价的简化
版本**——不失一般性地设 p 是最后一个下标（对称矩阵可以同时置换行列把任意 p 换到最后一位，结论
不变）——把论文「对 `H⁻¹H=I` 两边做初等行列变换」这个核心思路完整走一遍：

把 `H` 和 `H⁻¹` 按「前 d−1 个下标 / 第 p 个下标」分块：

$$
\mathbf{H} = \begin{bmatrix} \mathbf{H}_{11} & \mathbf{h}_{12} \\ \mathbf{h}_{12}^{\top} & h_{22} \end{bmatrix},
\qquad
\mathbf{H}^{-1} = \begin{bmatrix} \mathbf{G} & \mathbf{g} \\ \mathbf{g}^{\top} & \gamma \end{bmatrix}
$$

这里 `H_11` 就是去掉第 p 行第 p 列后的原 Hessian，即 `H_{-p}`；`γ = [H⁻¹]_pp`、`g = H⁻¹_{:,p}`
去掉最后一个分量后的部分。展开 `H⁻¹H = I` 的分块乘法，取左上角这一块：

$$
\mathbf{G}\mathbf{H}_{11} + \mathbf{g}\,\mathbf{h}_{12}^{\top} = \mathbf{I}
$$

——记为式 (a)。再取左下角这一块：

$$
\mathbf{g}^{\top}\mathbf{H}_{11} + \gamma\,\mathbf{h}_{12}^{\top} = \mathbf{0}
\quad\Longrightarrow\quad
\mathbf{g}^{\top}\mathbf{H}_{11} = -\gamma\,\mathbf{h}_{12}^{\top}
$$

——记为式 (b)。现在直接验证「`G − (1/γ)·g g^T` 就是 `H_11` 的逆」——用它去乘 `H_11`，代入式 (b)：

$$
\left(\mathbf{G} - \frac{1}{\gamma}\mathbf{g}\mathbf{g}^{\top}\right)\mathbf{H}_{11}
= \mathbf{G}\mathbf{H}_{11} - \frac{1}{\gamma}\mathbf{g}\,(\mathbf{g}^{\top}\mathbf{H}_{11})
= \mathbf{G}\mathbf{H}_{11} - \frac{1}{\gamma}\mathbf{g}\,(-\gamma\,\mathbf{h}_{12}^{\top})
= \mathbf{G}\mathbf{H}_{11} + \mathbf{g}\,\mathbf{h}_{12}^{\top} = \mathbf{I}
$$

——最后一步用回式 (a)。

所以 `H_11⁻¹ = G − (1/γ)·g g^T`，也就是：

$$
\mathbf{H}^{-1}_{-p} = \mathbf{G} - \frac{1}{\gamma}\mathbf{g}\mathbf{g}^{\top}
= \left(\mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_{pp}}\mathbf{H}^{-1}_{:,p}\mathbf{H}^{-1}_{p,:}\right)_{-p}
$$

——恰好就是 Lemma 1（Eq.4）。这个证明思路和论文 Appendix A.1 完全一致（都是利用 `H⁻¹H=I` 这
个恒等式做分块/消元），只是论文为了覆盖「p 在任意位置」写了更一般的三块置换记号；上面简化到
「p 是最后一个下标」不影响结论——对称矩阵总能通过同时置换行和列，把任意 p 挪到最后一位。想看
论文一般位置的完整证明，直接查 `# PAPER: OBC Appendix A.1 Eq.8`。

这个恒等式属于「利用矩阵求逆的分块结构做秩-1 修正」这一类（和 Sherman–Morrison / Woodbury
公式同源：都是「已知大矩阵逆，去掉/更新一行一列后不用重新求逆，用一次秩-1 修正即可」的思想），
但论文自己全程没有点名 Sherman–Morrison，而是叫它「Gaussian elimination of row and column p」
——这个别名上的差异值得注意，避免误以为论文直接照抄了教科书上的 Sherman–Morrison 公式。

## 5. 命名继承的具体证据：GPTQ Eq.3 里那个「漏改」的 -p

把 OBC 论文的 Lemma 1（Eq.4）和正文引用的 GPTQ §3 Eq.3（本章记法）并排放在一起看：

```text
OBC  §4 Eq.4:  H⁻¹_{-p} = ( H⁻¹ − 1/[H⁻¹]_pp · H⁻¹_{:,p} H⁻¹_{p,:} )_{-p}
GPTQ §3 Eq.3:  H⁻¹_{-q} = ( H⁻¹ − 1/[H⁻¹]_qq · H⁻¹_{:,q} H⁻¹_{q,:} )_{-p}
```

两条公式逐项对比：GPTQ 把 OBC 原公式里每一处 `p` 都换成了 `q`（因为 GPTQ 论文全篇用 q 表示
「当前正在量化的权重下标」），**唯独最外层的下标 `-p` 没有跟着换**——这不是本章的笔误，而是
GPTQ 论文自己从 OBC 论文抄公式、改记号时留下的一处不一致（把内部的 q 都改了，外层框住整个矩
阵的下标忘了改）。核对 OBC 原文，`-p` 在 OBC 里本来就是「删去第 p 行第 p 列」的意思，语义上和
GPTQ 想表达的「删去第 q 行第 q 列」完全一致，只是符号没对齐——这正是正文提示读者「p 就是 q」
的依据来源。

## 6. 记号对应关系小结

| GPTQ 论文记号（本章引用） | OBC 论文记号（本文件出处） | 含义 |
|---|---|---|
| `q`（正文全程使用） | `p`（OBC 全程使用，来自 *prune*） | 当前正在处理（剪枝/量化）的权重下标 |
| `H_{-q}^{-1}` 外层下标 `-p` | `H_{-p}^{-1}` | 两者是同一个操作「删去第 q(=p) 行列」，GPTQ 只是没把这一处的记号换过来 |
| `F`（本章记法，未量化权重集合） | 未显式命名，隐含于 `w_p`/`δ_p` 定义域 | 剩余待量化/待剪枝的权重下标集合 |
| GPTQ §3 Eq.1（逐层重构目标） | OBC §3 Eq.2 | 完全相同的目标函数，只是记号简化 |
| GPTQ §3 Eq.2（本章记法：贪心选权重+补偿方向） | OBC §5 Eq.7 | 量化版本的 OBS 闭式解，本文件 §3 完整推导 |
| GPTQ §3 Eq.3（本章记法：Hessian 逆秩-1 更新） | OBC §4 Eq.4（Lemma 1） | 完全相同的秩-1 更新公式，本文件 §4 完整证明 |
