# ch09 implementation notes —— MLIR/Linalg 结构化算子代数：论文忠实的小型参考实现

本章是 **primer 原理章**（豁免"只做减法"）：`implementation/` 不是任何真实代码仓
的精简版，而是把两篇论文——[MLIR] arXiv:2002.11054 §2-§4 与 [Linalg]
arXiv:2202.03293 §2.3、§3——里关于 structured op 的数学（索引表达式、隐式迭代域、
求像即子集、tiling 的不动点性质、padding 的幺元条件、向量化分情形、
destination-passing style、bufferization）逐条变成可跑、可打断点的 NumPy 代码。

**不建模的东西**（forbidden，见 dossier `reference_impl_plan.forbidden`）：
MLIR 的 Op/Region/Block 类层次（那是叙事 + 配图的活）、任何 HFusion/HIVM 算子名
或 pass 行为、任何昇腾运行时或伪造的真机计时——宿主无 NPU/CANN，
`trace_source: cpu-numpy-reference`。

## 文件划分

| 文件 | 覆盖论文小节 | 内容 |
|---|---|---|
| `structured_op.py` | [Linalg §2.3.6][§3] | `AffineExpr`/`IndexingMap`/`StructuredOp`：索引表达式、`derive_iteration_domain`（t2）、`StructuredOp.apply`（标量参考求值） |
| `tiling.py` | [Linalg §3.1] | `image_of_domain`（t3 求像）、`extract_slice`/`insert_slice`、`tile`/`tile_and_run`（m16 不动点性质） |
| `padding.py` | [Linalg §3.2] | `neutral_element`、`pad_to_static`（t5 幺元条件） |
| `vectorization.py` | [Linalg §3.3] | `build_einsum_subscripts`、`vectorize`（情形 1 逐点 + 情形 4 归约，情形 2/3/5 显式拒绝） |
| `named_ops.py` | [Linalg §3.3 脚注6][§5.1] | `named_op_registry`、`to_generic`、`make_conv_1d_nwc_wcf`/`make_matmul`/`make_pointwise_add`（m19） |
| `bufferization.py` | [Linalg §3.4] | `bufferize_naive`/`bufferize_dps`/`BufferizeReport`（m21/t7，分配计数对照） |
| `verify.py` | [Linalg §3.6] | `assert_same_result`（"legal by design" 的可执行校验） |

## Paper Map（论文机制 ↔ 参考实现符号）

| 论文出处 | 参考实现符号 | 覆盖的 dossier 机制/theory |
|---|---|---|
| [Linalg §3] 索引记法 O[n,w,f]=I[n,w+kw,c]·K[kw,c,f]（paper.md:L248-L264） | `named_ops.make_conv_1d_nwc_wcf` + `structured_op.AffineExpr` | m13, t1 |
| [Linalg §3] "iterators span the entire data of the operands"（paper.md:L266-L278） | `structured_op.derive_iteration_domain` / `StructuredOp.iteration_domain` | m14, t2 |
| [Linalg §3.1] "computing the image of the iteration domain by the indexing function"（paper.md:L280-L284） | `structured_op.AffineExpr.image` / `tiling.image_of_domain` | m15, t3 |
| [Linalg §3.1] "tiled form of the operation is itself a linalg.conv_1d_nwc_wcf"（paper.md:L333-L339） | `tiling.tile` / `tiling.tile_and_run` | m16, t4 |
| [Linalg §3.2] padding 的幺元条件（paper.md:L341-L346） | `padding.neutral_element` / `padding.pad_to_static` | m17, t5 |
| [Linalg §3.3] 向量化情形 (1)/(4)（paper.md:L353-L367） | `vectorization.vectorize` / `vectorization.build_einsum_subscripts` | m18 |
| [Linalg §3.3 脚注6][§5.1] named vs generic（paper.md:L296-L313） | `named_ops.to_generic` / `named_ops.named_op_registry` | m19 |
| [Linalg §3.4] destination-passing style（paper.md:L315-L325） | `structured_op.StructuredOp.apply` 的 `out_shape` 参数（`outs` 一等操作数） | m20, t7 |
| [Linalg §3.4] bufferization 少分配少拷贝（paper.md:L369-L373） | `bufferization.bufferize_naive` / `bufferization.bufferize_dps` | m21, t7 |
| [Linalg §3.6] "legal by design"（paper.md:L385） | `verify.assert_same_result` | t6（全章交叉校验） |

## 关键设计取舍

1. **迭代域推导是受限的，不是通用仿射求解器**：`derive_iteration_domain` 只认「某
   操作数某轴是该迭代维的纯恒等映射」这一种边界来源，精确复现论文对
   conv_1d_nwc_wcf 的推导，但论文自己说通用稠密情形要靠"连续施加 Fourier-Motzkin
   消元"（paper.md:L278）——本参考实现不实现这一步，找不到纯恒等来源时显式
   `ValueError`，不去猜。
2. **`outs`/`out_shape` 是 `StructuredOp.apply` 的显式参数，不是内部算出来的**：
   这不只是工程方便——论文自己的例子里 `w` 的边界只能从 `O`（输出）的形状读出，
   `I`/`K` 都给不出来；这一设计选择直接把 §7.4 destination-passing style 的
   "`outs` 是一等操作数"落到了代码接口上。
3. **`tiling.py` 只允许对 parallel 维分块**：归约维（论文的 tile size 例子
   `1x8x32x1x8` 覆盖全部 5 维）每个 tile 都取满量程。对归约维分块需要跨 tile
   累加（partial reduction），是另一层复杂度；dossier 的 worked example 不需要，
   `test_padding.py` 里用手工搭的两段式归约分块场景单独演示了这个模式，
   不依赖 `tile()`/`tile_and_run` 的通用支持。
4. **`vectorization.vectorize` 只信 `StructuredOp.vectorizable_reduce` 标记，
   不反射算子体是不是乘加**：对应论文原话"视对算子体的进一步分析"
   （paper.md:L364）——本参考实现故意不去做这个通用分析，只覆盖乘加这一种
   归约（`sum_of_products`），其余（情形 2/3/5）一律 `NotImplementedError`。
5. **`bufferize_naive`/`bufferize_dps` 数的是 CPU 上的分配次数，不是任何性能
   数字**——宿主无昇腾 NPU/CANN，`trace_source: cpu-numpy-reference`，不能也不
   去伪造真机耗时；分配计数是一个纯结构性的、可在 CPU 上诚实数出来的量。
6. **`pointwise_add` 不是论文点名的算子**：两篇论文对向量化情形 (1) 只有泛泛
   描述，没有点名一个具体的 linalg 逐点算子——`named_ops.make_pointwise_add`
   是本参考实现自己挑的最简单实例，用来让"情形 1"可以被跑一遍；因此它不进
   `named_op_registry`（该表只登记论文真正点名过的 `conv_1d_nwc_wcf`/`matmul`）。

## 测试

`tests/`（24 例，host `python3 -m pytest`）覆盖：
- t1/t2 worked example：论文原始形状（O:1x988x64/I:1x990x32/K:3x32x64）下的域推导
  + 小参数（N=1,W=16,C=2,F=3,KW=3）下 `apply` 与手写三重循环逐元素一致
  （`test_structured_op.py`）；
- t3 worked example：卷积滑窗的"像比 tile 宽"（halo）——tile_w=8、核宽 3 时输入
  侧的像宽度精确等于 8+3-1=10（`test_image_of_domain.py`）；
- m16 worked example：tiling 结果（含整除与不整除两种边界 tile）与不切时数值一致，
  归约维分块被显式拒绝（`test_tiling.py`）；
- t5 worked example：手工搭建归约维两段分块场景，正确幺元(0)的部分和相加等于
  全量参考值，错误幺元(1)则偏离——把"补错幺元→结果错"变成一个可运行的反例
  （`test_padding.py`）；
- m19 worked example：named/generic 展开前后数值相同、索引映射/迭代器类型/算子体
  是同一份对象而非"恰好配置成相等"（`test_named_vs_generic.py`）；
- 向量化情形 (1)/(4) 与 `apply` 一致，情形 (5)（滑窗）与缺失 `vectorizable_reduce`
  标记均显式拒绝（`test_vectorization.py`）；
- m21/t7 worked example：naive 与 DPS 两条 bufferization 路径数值完全一致，分配
  次数分别为 `len(tiles)+1` 与 `1`（`test_bufferization.py`）。
全部通过。
