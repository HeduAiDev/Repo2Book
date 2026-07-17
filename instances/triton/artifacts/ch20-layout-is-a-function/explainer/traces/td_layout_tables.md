# ch20 素材真相源 — 源码 verbatim 表格（trace_source=manual 的原始出处）

本章为 primer 概念地基，**无精简版可跑**：主真相源是 `TritonGPUAttrDefs.td` / `TritonGPUDialect.td` 的
`description` 块 —— 这是 MLIR 的 `.td` 声明文件（TableGen），**非可执行代码**，没有 runtime 可跑出轨迹。
因此 explainer 的每张数值表逐格取自下方 `.td` 的 **verbatim 示例**，每个数字标 `file:Lxxx`。

以下摘录均从 `instances/triton/source/` 下的源码逐字复制（引用时规范路径去掉 `instances/triton/source/` 前缀）。

---

## A. m02 顿悟例 —— `TritonGPUAttrDefs.td:L40-L50`（file 实际 L41-L44 为 L(..) 四行）

```
\mathcal{L}(0, 0) = {0, 4}
\mathcal{L}(0, 1) = {1, 5}
\mathcal{L}(1, 0) = {2, 6}
\mathcal{L}(1, 1) = {3, 7}
```
含义（.td L46-49 逐字）：
- T[0,0] is owned by both cuda thread 0 and 4
- T[0,1] is owned by both cuda thread 1 and 5
- T[1,0] is owned by both cuda thread 2 and 6
- T[1,1] is owned by both cuda thread 3 and 7

并集 = {0,1,2,3,4,5,6,7}（8 线程），每格集合大小 = 2（值域是集合而非单点）。

---

## B. m05 Blocked 逐格线程号表 —— `TritonGPUAttrDefs.td:L601-L619`

16×16 张量、2 warp（64 线程），`sizePerThread={2,2}`、`threadsPerWarp={8,4}`、`warpsPerCTA={1,2}`：

```
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]   <- rows 0,1
[ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]   <- rows 2,3
[ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
...                                                     <- rows 4..13 elided in .td
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]   <- rows 14,15
[ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
```

只用 .td 显式给出的 cell 做核对（不碰被 `...` 省略的 rows 4-13）：
- (row 0, col 0)  -> thread 0     (左半，warp 0)
- (row 0, col 2)  -> thread 1
- (row 2, col 0)  -> thread 4
- (row 0, col 8)  -> thread 32    (右半，warp 1 起点，";" 分隔)
- (row 14, col 14)-> thread 63    (右下角)

每个线程占一个连续 2×2 小块（`sizePerThread={2,2}`），故每个号在表里出现 2×2=4 次。
256 元素 / 64 线程 = 4 元素/线程，与 ∏sizePerThread = 2×2 = 4 一致（严格 partition）。
block 线程总数 = warpsPerCTA(1×2=2) × threadsPerWarp(8×4=32) = 64。

---

## C. m06 broadcast / wrap-around 逐格 L(T) —— `TritonGPUAttrDefs.td:L559-L569`

张量 T = 2×8，布局 L = 4×4：

```
T = [x  x  x  x  x  x  x  x]
    [x  x  x  x  x  x  x  x]
L = [0  1  2  3 ]
    [4  5  6  7 ]
    [8  9  10 11]
    [12 13 14 15]

L(T) = [ {0,8} , {1,9} , {2,10}, {3,11}, {0,8} , {1, 9} , {2, 10}, {3, 11},   <- T row 0, cols 0-7
         {4,12}, {5,13}, {6,14}, {7,15}, {4,12}, {5, 13}, {6, 14}, {7, 15} ]  <- T row 1, cols 0-7
```

- 行方向：T 高 2 < L 高 4 → **broadcast**，每格是 2 个线程集合 {a, a+8}（如 (0,0)={0,8}）。
- 列方向：T 宽 8 > L 宽 4 → **wrap-around**，col 4-7 复用 col 0-3 的线程号（(0,4)={0,8} 与 (0,0) 同）。

逐格核对样点（均在上方 L(T) 数组里逐字可见）：
- (0,0) = {0,8}
- (0,3) = {3,11}
- (0,4) = {0,8}    （wrap：col 4 == col 0）
- (1,0) = {4,12}
- (1,7) = {7,15}   （wrap：col 7 == col 3）

计数：16 张量元素 × broadcast 因子 2 = 32 thread-slots；16 线程 × wrap 因子 2 = 32。每线程恰出现 2 次。

---

## D. m03/m04/m07 结构性常量（图/正文引用）

- shared 布局：对所有 i，L(i) = {0,1,...,32*num_warps-1}（`TritonGPUAttrDefs.td:L158-L161`）。
- 四级层次 CTA→Warp→Thread→Value（`TritonGPUAttrDefs.td:L470-L471`）。
- 上两级 linear-id contiguous，shape=[4,4]/order=[0,1] 列优先填号（`TritonGPUAttrDefs.td:L473-L481`）：
  ```
  layout = [0  4  8  12]
           [1  5  9  13]
           [2  6  10 14]
           [3  7  11 15]
  ```
- 模块契约（`TritonGPUDialect.td:L24-L46`）：num-warps 强制（缺失 report_fatal_error）；
  num-ctas 缺省 1；threads-per-warp 缺省 32。block 线程总数 = num_warps × threads_per_warp。
