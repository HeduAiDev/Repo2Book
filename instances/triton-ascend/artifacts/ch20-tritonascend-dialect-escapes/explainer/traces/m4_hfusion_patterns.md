# m4 — TritonToHFusion 三 pattern 贪婪舱（手工推演，trace_source=manual）

**为何 manual**：skip_impl，无 .py 精简版；不伪造 IR dump。下表逐 pattern 沿
`lib/TritonToHFusion/TritonToHFusion.cpp` 的 matchAndRewrite 控制流手工走一遍，行号标注。

## pattern set 挂载（贪婪驱动）
`runOnOperation` L145-L160：三 pattern 全 add 进 set（L151-153），
`applyPatternsAndFoldGreedily`（L157）驱动——注释 L148-149、L155-156 明写
「pattern 自行决定是否转换（返 success/failure）」「failure() 不致 pass 失败」。

## 逐输入走控制流
选一组具体输入，逐 op 追分支：

1. **ascend.mod**（lhs/rhs 均 `tensor<128xi32>`）→ TritonModToHFusionConversion（L31-55）
   - L37-38 dyn_cast<RankedTensorType> lhs/rhs → 均非空 → 不进 L39 failure 分支
   - L43 建 tensor.empty(shape=128, i32)；L45-50 hfusion::createBinaryOp(BinaryFn::mod)
   - L52 replaceOp → **hfusion.elemwise_binary**（BinaryFn::mod）；L53 return success()

2. **tt.histogram**（result `tensor<64xi32>`，静态形状）→ TritonHistogramToHFusionConversion（L57-80）
   - L67 numBins = 256（缺省）；L68-70 result 是 RankedTensorType 且 static 且 numElements=64>0
     → numBins = 64（**走静态分支，非 256 回退**）
   - L74-75 建 hfusion.histogram(numBins=64)；L77 return success()

3. **tt.fp_to_fp**（rounding = RTZ）→ TritonFpToFpToHFusionConversion（L82-134）
   - L102 roundingMode = RTZ；L103 RTZ ≠ RTNE 且 has_value → 不进 L104-106 failure 放行
   - L110-112 switch RTZ → hfusion::RoundMode::TRUNC
   - L129-130 replaceOpWithNewOp<hfusion::CastOp>(mode=TRUNC)；return success()

4. **tt.fp_to_fp**（rounding = RTNE，默认）→ 同 pattern
   - L103 `roundingMode.value() == RTNE` **成立** → L105 `return failure()`
   - **op 原样留存**，交主链 TritonToLinalg 用 arith.truncf/extf 处理（L92-93 注释）
   - 贪婪驱动下 failure() 不使 pass 失败（L155-156）

## 计数
- 3 pattern（L151-153），治 3 类源 op：ascend.mod / tt.histogram / tt.fp_to_fp(非RTNE)
- 其中仅 ascend.mod 是 ascend 方言 op；histogram/fp_to_fp 是核心 tt.* op（因需专用硬件下降也走此舱）
- HFusion 舱宽度 = **3**（不是 brief 的「各 2-3」）
