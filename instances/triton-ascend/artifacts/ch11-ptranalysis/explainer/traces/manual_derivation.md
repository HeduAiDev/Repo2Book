# ch11 PtrAnalysis — 手工推导取证记录（trace_source=manual）

## 为什么是 manual，不是 run

本章 `skip_impl=true`：`TritonToStructured::PtrAnalysis` 是纯 C++ MLIR pass，无精简 `.py`
可跑。宿主机无 CANN/NPU 工具链，无法跑 `triton-opt --triton-to-structured` 产出真实
`PtrState` dump（且 pin 里的 dump 走 `LLVM_DEBUG`，需 debug build）。因此每个 worked example
的 `PtrState` 逐节点值，由**依 `PtrAnalysis.cpp` 源码算法手工代入 lit 夹具的真实 IR** 推导，
数字全部标注 `PtrAnalysis.cpp:Lxxx` 或夹具行号出处。交叉验证走两条：
1. **pin 精确源码**：算法逐行（`sed -n` 对眼，见下方每机制的源码锚点）。
2. **lit 夹具**（RUN + CHECK 前后对照）：IR 输入真实存在于仓库，非杜撰。

## 夹具清单（真实存在，已 `grep`/`sed` 核对）

- 2D 主链：`third_party/ascend/unittest/Conversion/General/TritonAscendAllPass/simplify_for_loop.mlir`
  `@matmul_kernel`，b_ptrs 子链 `%14 → %27`（L52–L65）。
  - `%cst_7 = arith.constant dense<1024> : tensor<256xi32>`（L33）——即 rem 的除数 1024。
- rem 安全路径：`third_party/ascend/unittest/Conversion/General/TritonToStructured/parseRem.mlir`
  `@kernel_with_rem_safe`（L3–L25），`%2 = arith.remsi %1, %cst_2`，`%cst_2 = dense<128>`，
  `%1 = make_range(0,256)`。RUN 行带 `optimize-dynamic-offset=true`。

## 核过的算法常量 / 分支（sed 对眼）

| 机制 | 源码锚点 | 核到的事实 |
|------|----------|------------|
| m2 dispatch | PtrAnalysis.cpp:L1280–L1355 | 14 个 `getDefiningOp<>` 分支：12 产状态，2（LoadOp/FPToSIOp）保守失败 |
| m2 scalar 判定 | PtrAnalysis.cpp:L323–L335 | `operandIsScalar` 只认 Integer/Index；`!tt.ptr` 非标量 |
| m2 pointer 入口 | PtrAnalysis.cpp:L370–L401 | 裸指针 block-arg：`!getDefiningOp()` → `newSource=operand`, offset=0 → **成功** |
| m6 make_range | PtrAnalysis.cpp:L778 | `stride=(end-start+shape[0]-1)/shape[0]`；`!=1` → failure |
| m4 addState | PtrAnalysis.cpp:L520–L583 | dimIndex 双指针归并 + isMultiple 兼容 + min/div 拆维 + source=lhs?lhs:rhs |
| m5 mulState | PtrAnalysis.cpp:L440–L451 | `if(!rhs->isScalar() && lhs->isScalar()) swap`；lhs 每维 stride×rhs.offset |
| m5 isScalar | PtrAnalysis.cpp:L165–L174 | 所有 stride 静态为 0 且 (offset‖source) |
| m10 normalize | PtrAnalysis.cpp:L209–L228 | 连续同 dimIndex 零 stride 维乘并；孤立单元维（sizes 也=1）保留 |
| m11 rem | PtrAnalysis.cpp:L1083–L1122 | `isMultiple(a,b)=a%b==0`；divisor 是 stride 倍数→拆 (0,nonContig)+(stride,contig) |
| m11 isMultiple | PtrAnalysis.cpp:L(isMultiple) | `a % b == 0` |

## m8 主链逐节点手工推导（代入 simplify_for_loop.mlir 真实 IR）

起点：base 指针 `%arg1 : !tt.ptr<i8>`，行 stride 入参 `%arg4 : i32`（运行期）。

```
%14 make_range(0,64)   -> stateInfo=[(1,64,d0)]                     sizes=[64]    off=0     src=∅
%19 expand_dims(ax=1)  -> stateInfo=[(1,64,d0),(0,1,d1)]           sizes=[64,1]  off=0     src=∅
%20 splat(%arg4:i32)   -> stateInfo=[(0,64,d0),(0,1,d1)]           sizes=[64,1]  off=%arg4 src=∅   (isScalar=true)
%21 muli(%19,%20)      -> stateInfo=[(%arg4,64,d0),(0,1,d1)]       sizes=[64,1]  off=0     src=∅   (每维 stride×%arg4)
%23 broadcast(%21)     -> stateInfo=[(%arg4,64,d0),(0,256,d1)]     sizes=[64,256] off=0    src=∅   (d1: 1→256, stride 不动)
%10 make_range(0,256)  -> stateInfo=[(1,256,d0)]                    sizes=[256]   off=0     src=∅
%12 addi(splat %9,%10) -> stateInfo=[(1,256,d0)]                    sizes=[256]   off=%9    src=∅
%13 remsi(%12,1024)    -> stateInfo=[(1,256,d0)]                    sizes=[256]   off=rem(%9,1024) shouldLinearize=true
                          （divisor=1024，stride=1：isMultiple(1024,1)→contig=min(1024,256)=256, nonContig=1 不加）
%22 expand_dims(ax=0)  -> stateInfo=[(0,1,d0),(1,256,d1)]           sizes=[1,256] off=rem(%9,1024)
%24 broadcast(%22)     -> stateInfo=[(0,64,d0),(1,256,d1)]          sizes=[64,256] off=rem(%9,1024)
%25 addi(%23,%24)      -> stateInfo=[(%arg4,64,d0),(1,256,d1)]      sizes=[64,256] off=rem(%9,1024)
                          （d0:%arg4+0=%arg4  d1:0+1=1）
%26 splat(%arg1:!ptr)  -> stateInfo=[(0,64,d0),(0,256,d1)]         sizes=[64,256] off=0 src=%arg1
                          （initStateByPointer: 裸指针→source=%arg1；splat 铺零 stride）
%27 addptr(%26,%25)    -> ptr 子状态=%26(source=%arg1), offset 子状态=%25 → addState
                          -> stateInfo=[(%arg4,64,d0),(1,256,d1)]  sizes=[64,256]
                             off=rem(%9,1024)  src=%arg1
```

结论：base=`%arg1`、行 stride=`%arg4`（=stride_bk 运行期）、列 stride=1（contiguous）、
块内 offset=rem(%9,1024)。正是 `reinterpret_cast(offset=rem(%9,1024), sizes=[64,256],
strides=[%arg4,1])` 所需——ch12 落 memref。

## m11 rem 安全路径逐步（parseRem.mlir @kernel_with_rem_safe）

```
%1 make_range(0,256)  -> stateInfo=[(1,256,d0)]  off=0
%2 remsi(%1, 128)     -> divisor=128, staticOffset=0 (isMultiple(0,128)=true, 放行)
                         info=(stride=1,shape=256):
                           isMultiple(1,128)? 1%128≠0 → 否
                           isMultiple(128,1)? 128%1=0 → 是
                             contiguousSize = 128/1 = 128; min(128,256)=128
                             nonContiguousSize = 256/128 = 2 (>1 → 保留)
                             emplace (0,2,d0) 再 emplace (1,128,d0)
                      -> stateInfo=[(0,2,d0),(1,128,d0)]  off=rem(0,128)=0  shouldLinearize=true
```

含义：`i mod 128`（i∈[0,256)）= [0..127] 重复 2 遍 → 外层 size=2 stride=0（取模抹平的重复），
内层 size=128 stride=1（连续段）。
