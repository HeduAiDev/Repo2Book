# 手工推演：fuseBlock 在 @testA（ShallowCV，静态 7x7）上的融合

源：`third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HFusion/OpFusion/test_static_shallow_cv.mlir` @testA
RUN 行：`bishengir-opt --test-assign-fusion-kind --fusion-kind="SHALLOW_CV" -hfusion-fuse-ops`
FileCheck 断言（pin 亲自给出的期望）：@testA_0 恰含
`linalg.elemwise_unary / linalg.elemwise_binary / linalg.elemwise_unary / linalg.matmul / linalg.elemwise_unary`（5 op），且 @testA 里出现 `call @testA_0`。

## 重要算子（linalg op）与依赖边
（tensor.empty / tensor.dim / arith.constant 非重要算子，不进融合图）

| 节点 | op | OpPattern | 生产者→消费者边 |
|---|---|---|---|
| n3  | %3 = elemwise_unary ceil(arg2) | kElementWise | %3→%5 |
| n5  | %5 = elemwise_binary add(arg2,%3) | kElementWise | %5→%7 |
| n7  | %7 = elemwise_unary log(%5) | kElementWise | %7→%9, %7→%11 |
| n9  | %9 = matmul(arg2,%7) | kMatmul | （消费者是 return） |
| n11 | %11 = elemwise_unary ceil(%7) | kElementWise | （return） |
| n13 | %13 = broadcast(arg2) dims=[0] → 7x7x7 | broadcast | %13→%17 |
| n15 | %15 = elemwise_unary abs(arg0) | kElementWise | %15→%19 |
| n17 | %17 = broadcast(%13) dims=[3] → 7x7x7x7 | broadcast | （return） |
| n19 | %19 = transpose(%15) | kTranspose | （return） |

三个连通分量：A={3,5,7,9,11}（matmul 链），B={13,17}（纯 broadcast），C={15,19}（abs→transpose）。

## 逐候选边推演（fuseBlock，纵向）
候选边按被消费者拓扑秩升序尝试（`llvm::sort`，FusibleBlockAnalyzer.cpp:L198-L203）。
ShallowCV 下 isFusible 对 matmul↔vector、vector↔vector 全允许（isShallowCVFusible 兼容表 L673-L712）；
关卡里 reduceRank/reduceDim 仅 LastAxisPBR/AnyPBR 生效、nodeType 仅 MixCV 生效——ShallowCV 全关，
故只剩 dependency 关卡（简单树 indegree=1，不拦）。每条边都 join：

1. %3→%5 : isFusible=T, 关卡全过 → join → {3,5}
2. %5→%7 : isFusible=T → join → {3,5,7}
3. %7→%9 : isFusible=T（vector↔matmul，ShallowCV 允许）→ join → {3,5,7,9}
4. %7→%11: isFusible=T → join → {3,5,7,9,11}
5. %13→%17: isFusible=T → join → {13,17}
6. %15→%19: isFusible=T → join → {15,19}

横向融合段（L246-L266）：受 maxHorizontalFusion 上限约束，本例三组之间无“无依赖的重要 op”可横合，跳过。

## 出组过滤（checkGroupRequirements，L149-L173）
- {3,5,7,9,11}: matmulCount=1(%9), importantCount=5 → matmul≥1 ✓, important>1 ✓ → **保留**
- {13,17}: matmulCount=0 → ShallowCV 要求含 matmul → **拒绝**
- {15,19}: matmulCount=0 → **拒绝**

最终 fusedGroups = [ {3,5,7,9,11} ] → 外提成 @testA_0（5 op）。
**与 FileCheck 断言逐 op 吻合**：elemwise_unary(ceil %3)/elemwise_binary(add %5)/elemwise_unary(log %7)/matmul(%9)/elemwise_unary(ceil %11)。
