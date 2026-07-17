# ch25 手工推演工作底稿（trace_source=manual）

本章解读的是编译器 C++ 内部的**静态分析 + 改写 pass**，没有「精简版可运行程序」这一交叉验证载体
（analysis 跑在 MLIR IR 上，产物是格值/布局而非数值张量）。宿主机 triton 为 3.6.0，与本书钉版
v3.2.0 @ 9641643 不一致，headless 跑出的 AxisInfo dump 行号/数值可能与所引源码不符——故全部数字
**手工按 v3.2.0 已核验的 transfer rule 逐步推演**，每个常量标 `file:Lxxx`。下面是各 worked example
的完整推导，explainer.json 的表格数字均来自此处。

## 已核验的 transfer rule（源码出处）

- `make_range(start,end)`: contiguity=`end-start`, divisibility=`highestPowOf2Divisor(start)`,
  constancy=`1`。  —— lib/Analysis/AxisInfo.cpp:L229-L232
- `highestPowOf2Divisor(n) = n & ~(n-1)`；`n==0` 返回 `1<<(sizeof(T)*8-2)`。
  i32 的 `highestPowOf2Divisor(0)=1<<30=1073741824`；`highestPowOf2Divisor(1024)=1024`。
  —— include/triton/Dialect/Triton/IR/Utility.h:L33-L42
- `arith.constant(v)`(int): contiguity=1, divisibility=`highestPowOf2Divisor(v)`, constancy=1,
  constantValue=v。  —— lib/Analysis/AxisInfo.cpp:L252-L255
- `splat(scalar)→tensor`: 每轴 contiguity=1, divisibility=`opInfo.divisibility(0)`,
  constancy=`retShape[d]`。  —— lib/Analysis/AxisInfo.cpp:L539-L543
- `broadcast`(沿 size==1 维): 该维 contiguity=1、constancy=retShape；其余维原样。
  —— lib/Analysis/AxisInfo.cpp:L632-L637
- `addptr`(= AddSubOpAxisInfoVisitor<AddPtrOp>):
  - contiguity(非 sub) = `max(gcd(lhs.constancy, rhs.contiguity), gcd(lhs.contiguity, rhs.constancy))`
    —— L286-L287
  - divisibility = `gcd(lhs.div, multiplyDivisor(rhs.div, elemSize))`，`elemSize=pointeeBitWidth/8`
    —— L296-L313
  - constancy = `gcd(lhs.constancy, rhs.constancy)` —— L318-L319
- `mul`: divisibility = `multiplyDivisor(lhs.div, rhs.div)` —— L384
- `multiplyDivisor(a,b)`: 溢出则夹到 `2^62`，否则 `a*b` —— L47-L52
- `join(lhs,rhs)`: 逐轴 `gcd(contiguity)`,`gcd(divisibility)`,`gcd(constancy)`；constantValue 仅两侧
  相等才保留。  —— lib/Analysis/AxisInfo.cpp:L1186-L1206
- 悲观初值 seed：入口块函数参数 → `initPessimisticStateFromFunc` 逐个读 `tt.contiguity/
  tt.divisibility/tt.constancy` arg attr 填三元组，缺省全 1。  —— L1124-L1157, L1102-L1122
- `getNumElementsPerThread`: `elemBytes=elemBits/8`; `maxMultiple=max(div(order0)/elemBytes,1)`;
  `maxContig=min(contiguity(order0), shapePerCTA[order0])`; `alignment=min(maxMultiple,maxContig)`;
  `perThread=min(alignment, 128/elemBits)`。  —— lib/Dialect/TritonGPU/Transforms/Utility.cpp:L111-L128
- `argSort(arr)`: 稳定排序，`arr[x]>arr[y]` 降序 → 最连续轴排 order[0]。  —— Utility.cpp:L81-L87
- `setCoalescedEncoding`: `perThread=min(perThread, max(numElems/numThreads,1))`；非 load 额外夹
  `perThread=min(perThread, getNumElementsPerThread(op))`；`sizePerThread[order[0]]=perThread`。
  —— lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:L85-L99
- `getPtrAlignment`: `min(max(div(order0)/elemBytes,1), contiguity(order0))`。 —— AxisInfo.cpp:L1244-L1250
- `getPtrContiguity`: `min(getPtrAlignment, uniqueContigPerThread[order0])`。 —— AxisInfo.cpp:L1208-L1226

## 主线例子：`x_ptr + tl.arange(0, 1024)`，i32（elemBits=32, elemBytes=4）

前端 JIT 特化（见第 16 章）默认给 16 字节对齐的数据指针打 `tt.divisibility=16`。
numWarps=4, threadsPerWarp=32 → numThreads=128；BLOCK=1024 constexpr。

前向传播（rank 1）:

| Value | op | contiguity | divisibility(字节语义) | constancy | 推导 |
|---|---|---|---|---|---|
| `x_ptr`(arg) | seed | 1 | 16 | 1 | tt.divisibility=16 arg attr（其余缺省 1）|
| `%r=make_range 0..1024` | make_range | 1024 | highestPowOf2Divisor(0)=1073741824 | 1 | contiguity=1024-0 |
| `%b=splat x_ptr` | splat | 1 | 16 | 1024 | div=opInfo.div=16, constancy=shape=1024 |
| `%p=addptr %b,%r` | addptr | 1024 | 16 | 1 | 见下 |

`%p` 三步：
- contiguity = max(gcd(lhs.constancy=1024, rhs.contiguity=1024), gcd(lhs.contiguity=1, rhs.constancy=1))
  = max(gcd(1024,1024)=1024, gcd(1,1)=1) = **1024**
- divisibility = gcd(lhs.div=16, multiplyDivisor(rhs.div=1073741824, elemSize=4))
  = gcd(16, 4294967296) = **16**（elemSize=32/8=4）
- constancy = gcd(1024, 1) = **1**

Coalesce 消费 `%p`：
- argSort([1024]) → order=[0]
- getNumElementsPerThread: elemBytes=4; maxMultiple=max(16/4,1)=4; maxContig=min(1024,1024)=1024;
  alignment=min(4,1024)=4; perThread=min(4, 128/32=4)= **4**
- setCoalescedEncoding: numElems=1024, numThreads=128, numElems/numThreads=8;
  perThread=min(4, max(8,1))= **4**；sizePerThread=[4]；BlockedEncodingAttr(order=[0])
- 结论：每线程 4 个连续 i32 = 128 bit = 一条 ld.global.v4.b32；1024 元素 → 1024/4=**256** 笔向量事务
  （标量则 1024 笔），4× 减少。

## f9 失败面：seed divisibility=1（无 tt.divisibility）

`%b` div=1 → `%p` div=gcd(1, 4294967296)=1 → maxMultiple=max(1/4,1)=1 → alignment=min(1,1024)=1 →
perThread=min(1,4)= **1** → 每线程 1 个 i32（32 bit 标量），1024 笔事务，合并不动。

## join 失败面：两分支在 scf 汇合处 join

分支 A（对齐路径）`%pA`: contiguity=1024, div=16, constancy=1；
分支 B（错位路径）`%pB`: contiguity=1024, div=4, constancy=1。
join：contiguity=gcd(1024,1024)=1024；divisibility=gcd(16,4)=4；constancy=gcd(1,1)=1。
merged div=4 → maxMultiple=max(4/4,1)=1 → alignment=min(1,1024)=1 → perThread=1。
→ A 单独本可 perThread=4，合并后塌成 1。若某分支 contiguity=1（如 gather 指针），
join contiguity=gcd(1024,1)=1，同样 perThread=1。

## addptr 元素连续→字节对齐（源码注释例）

`addptr [16]:!ptr<i32>, [0,1,2,3]:i32`（AxisInfo.cpp:L303-L311 注释）：

| lane | range 值 | 字节地址 = 16 + 4×range | 元素位置 |
|---|---|---|---|
| 0 | 0 | 16 | 4 |
| 1 | 1 | 20 | 5 |
| 2 | 2 | 24 | 6 |
| 3 | 3 | 28 | 7 |

元素位置 [4,5,6,7] 连续 → contiguity=4；字节地址步长 4，起点 16 →
divisibility=gcd(16, multiplyDivisor(highestPowOf2Divisor(0),4))=gcd(16, 4294967296)=16 字节。
「strided contiguous, divisibility 16 bytes」。

## f3：constexpr→精确 divisibility（`pid * BLOCK`）

BLOCK=1024 constexpr → `arith.constant 1024`: divisibility=highestPowOf2Divisor(1024)=1024（1024=2^10）。
`pid=program_id`: seed div=1。`muli(pid,1024)`: div=multiplyDivisor(1,1024)=1024。
→ offset `pid*1024` 静态可知「必是 1024 的倍数」。
若 BLOCK 是运行时值：constant 路径不触发，stride div=1 → muli div=multiplyDivisor(1,1)=1。
（且运行时 BLOCK 连 arange(0,BLOCK) 都无法给出静态 contiguity——contiguity 退回未知。）

## coalesce-order：2D 例子

tensor<32x64xi32>，行主序 → contiguity=[1,64]（axis1 内层连续 64，axis0 不连续）。
argSort([1,64])：降序 → [64@idx1, 1@idx0] → order=[1,0]，最连续的 axis1 排 order[0]（最内层）。
sizePerThread[order[0]=1]=perThread。若误用 order=[0,1] 把非连续 axis0 放最内 → 相邻 lane 跨 64
元素跳，完全不合并。

## perthread-vec-width：divisibility 是绑定约束（i32, contiguity=1024, shape=1024）

| divisibility(字节) | maxMultiple=div/4 | maxContig | alignment | 128/32 cap | perThread | 向量宽 |
|---|---|---|---|---|---|---|
| 16 | 4 | 1024 | 4 | 4 | 4 | 128-bit (v4) |
| 8 | 2 | 1024 | 2 | 4 | 2 | 64-bit (v2) |
| 4 | 1 | 1024 | 1 | 4 | 1 | 32-bit 标量 |
| 1 | 1（max(0,1)）| 1024 | 1 | 4 | 1 | 32-bit 标量 |

i32 下 divisibility≥16 字节即饱和 128-bit 向量；divisibility 掉到 4 就退回标量。

## ptr-alignment-query（%p: div=16, contiguity=1024, i32）

- getPtrAlignment = min(max(16/4,1)=4, contiguity(order0)=1024) = **4**（AxisInfo.cpp:L1244-L1250）
- getPtrContiguity = min(getPtrAlignment=4, uniqueContigPerThread=4) = **4**（合并后 Blocked
  sizePerThread=4 → uniqueContig=4；AxisInfo.cpp:L1208-L1226）

## coalesce-rewrite：coalesceOp 的 IR 缝合（Coalesce.cpp:L113-L154）

| 步骤 | IR | 布局 |
|---|---|---|
| 改写前 | `%v = tt.load %p` | %p,%v : L_old |
| 插操作数 convert | `%p2 = convert_layout %p` | L_old → L_new |
| 造同名新 op | `%v2 = tt.load %p2` | L_new |
| 结果 convert 回 | `%v3 = convert_layout %v2` | L_new → L_old |
| 替换&删除 | `replaceAllUsesWith(%v → %v3); %oldOp.erase()` | 下游仍见 L_old |

1 个 load → 1 load + 2 convert_layout；冗余 convert 由后续 layout 传播/RemoveLayoutConversions 化简。
