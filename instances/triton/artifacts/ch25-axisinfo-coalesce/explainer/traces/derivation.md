# ch25 手工推演底稿（trace_source=manual）

本章是 **编译器 pass 源码解读章**，无 subtract-only Python 精简版（skip_impl）。AxisInfo 三元组由 MLIR
`SparseForwardDataFlowAnalysis` 在**编译期**算出，不是运行时 numeric trace。下列每个数字都由源码公式**逐步手工推演**，
锚点行号来自 pin `9641643da` @ triton v3.2.0。此文件是 explainer.json 每张表数字的可溯源底稿。

## 源码常量 / 公式清单（file:Lxxx）

- `highestPowOf2Divisor(n) = n & ~(n-1)`（n==0 或 min 时返回 `1<<(sizeof(T)*8-2)`）
  —— `include/triton/Dialect/Triton/IR/Utility.h:L33-L42`。i32（sizeof=4）：`highestPowOf2Divisor(0)=1<<30=1073741824`。
- MakeRange：`contiguity=end-start`，`divisibility=highestPowOf2Divisor(start)`，`constancy=1`
  —— `lib/Analysis/AxisInfo.cpp:L229-L232`。
- Splat：`contiguity=1`，`divisibility=opInfo.getDivisibility(0)`，`constancy=retShape[d]`
  —— `lib/Analysis/AxisInfo.cpp:L540-L542`。
- Broadcast（沿 size==1 维）：`contiguity= opShape[d]==1 ? 1 : contig(d)`，`divisibility=div(d)`，
  `constancy= opShape[d]==1 ? retShape[d] : constancy(d)` —— `lib/Analysis/AxisInfo.cpp:L633-L636`。
- AddSub/AddPtr contiguity：`max(gcd(lhs.constancy, rhs.contiguity), gcd(lhs.contiguity, rhs.constancy))`
  （SubI 特例走 `gcd(lhs.contig, rhs.constancy)`）—— `lib/Analysis/AxisInfo.cpp:L283-L288`。
- AddPtr divisibility：`gcd(lhs.div, rhs.div × elemSize)`，`elemSize = max(1, pointeeBitWidth/8)`
  —— `lib/Analysis/AxisInfo.cpp:L295-L314`。
- Load（结果 = 被读的数据，非指针）：`contiguity=1, divisibility=1, constancy=gcd(ptr.constancy, mask.constancy)`
  —— `lib/Analysis/AxisInfo.cpp:L567-L575`。
- join：逐轴 `gcd(contiguity/divisibility/constancy)`，constantValue 仅两侧相等才留
  —— `lib/Analysis/AxisInfo.cpp:L1195-L1205`。
- 悲观 seed：函数入口参数逐个读 `tt.divisibility/contiguity/constancy` arg attr 填三元组，无 attr 默认 `(1,1,1)`
  —— `lib/Analysis/AxisInfo.cpp:L1102-L1122`（`initPessimisticStateFromFunc`）、`L1132-L1157`。
- `getNumElementsPerThread`：`maxMultiple=max(divisibility(order0)/elemBytes,1)`；
  `maxContig=min(contiguity(order0), shapePerCTA[order0])`；`alignment=min(maxMultiple,maxContig)`；
  `perThread=min(alignment, 128/elemBits)` —— `lib/Dialect/TritonGPU/Transforms/Utility.cpp:L117-L124`。
- `argSort`：`stable_sort` 按值降序 → `order[0]=argmax` —— `lib/Dialect/TritonGPU/Transforms/Utility.cpp:L81-L87`。
- setCoalescedEncoding：`order=argSort(contiguity)`；`perThread=min(perThread, max(numElems/numThreads,1))`；
  非 load 额外夹 `getNumElementsPerThread`（≤128-bit）；`sizePerThread[order[0]]=perThread`
  —— `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:L33-L104`。
- coalesceOp：操作数插 `ConvertLayoutOp`→造同名新 op→结果 `ConvertLayoutOp` 回原布局→`replaceAllUsesWith`→`erase`
  —— `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:L113-L154`。
- getPtrAlignment：`maxMultiple=max(div(order0)/elemBytes,1)`，`alignment=min(maxMultiple, contig(order0))`
  —— `lib/Analysis/AxisInfo.cpp:L1239-L1244`。

## 主推演：canonical 1D load 链（贯穿多机制的 running example）

Kernel 片段（f32，`BLOCK=1024` 是 constexpr）：`offs = tl.arange(0,1024); ptr = X + offs; x = tl.load(ptr)`
函数签名把 `X` 标了 `tt.divisibility = 16`（前端对齐提示，见 ch16 回指）。

| op | 规则(锚点) | contiguity | divisibility(字节) | constancy |
|----|-----------|-----------|-----|-----------|
| seed X (!ptr<f32>, tt.divisibility=16) | L1102-1122 | 1 | 16 | 1 |
| arange(0,1024) | L229-232 | 1024 | 2^30 (start=0) | 1 |
| splat(X) → tensor<1024x!ptr> | L540-542 | 1 | 16 | 1024 |
| addptr(splat, arange) | L283-314 | max(gcd(1024,1024),gcd(1,1))=**1024** | gcd(16, 2^30·4)=**16** | gcd(1024,1)=1 |
| load(ptr) 的结果数据 | L567-575 | 1 | 1 | 1 |

关键：Coalesce 消费的是 **addptr 那一行（指针）** 的三元组 `contiguity=1024, divisibility=16`，不是 load 结果那行。

perThread（f32, order0=0, shapePerCTA=1024, div=16, contig=1024）：
`maxMultiple=16/4=4；maxContig=min(1024,1024)=1024；alignment=min(4,1024)=4；perThread=min(4,128/32)=min(4,4)=4`。
→ 每线程 4 个 f32 = 128-bit 向量 load（`ld.global.v4.f32`）。

配置假设（worked-example config，非源码常量）：`num_warps=4, threadsPerWarp=32` → `numThreads=128`，
`numElems=1024`，`perThread=min(4, max(1024/128,1))=min(4,8)=4`。BlockedEncoding：
`sizePerThread=[4], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]`；tile=4·32·4=512 elems，1024/512=2 tiles。

失败面（f9）：若 `X` 没 `tt.divisibility`（seed=1）→ addptr div=gcd(1,·)=1 → maxMultiple=1 → perThread=min(1,4)=1 → 标量 load。

## highestPowOf2Divisor 抽查（f3）

`n & ~(n-1)`：1024→1024；256→256；96=32·3→32；12=4·3→4；0→2^30（cap）；运行时未知→退回 seed 默认 1。
