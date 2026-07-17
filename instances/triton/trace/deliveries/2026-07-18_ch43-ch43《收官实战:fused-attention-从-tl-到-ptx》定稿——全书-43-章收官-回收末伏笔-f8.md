# ch43《收官实战:fused-attention 从 tl.* 到 PTX》定稿——全书 43 章收官,回收末伏笔 f8(ch07→ch43)

- **Type**: delivery
- **Chapter**: ch43
- **Date**: 2026-07-18
- **Timestamp**: 2026-07-18T22:10:00+08:00
- **Agents involved**: archivist, ch43writer(+dossier/explainer/illustrator/reviewer/map,均已 idle)
- **User present**: False
- **Tags**: triton, part-9, deep, skip_impl, capstone, 全书收官, f8-payoff, TRITON_KERNEL_DUMP, add_rewrite_tensor_pointer, mma.sync-m16n8k16, ex2.approx, cp.async, ptx_kernel, tcgen05, 验尸清单, tutorial-06-fused-attention

## What happened

全书收官 capstone(kind=deep/skip_impl),verdict=APPROVED。主题:拿官方 tutorial 06 的 Flash-Attention v2 前向核 `_attn_fwd`(`python/tutorials/06-fused-attention.py`)当活体标本,对它设 `TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1` 抓 ttir/ttgir/llir/ptx 四层真机 dump,把前 42 章每一层在**同一个真核**上串一遍、逐层指认哪个 pass 改了什么、回链对应章。全章脊柱=`compile()` 里 `add_stages` 那圈逐阶段降级循环(六步):

1. **`tl.*` 表面**——在线 softmax 主循环(两个 `tl.dot`、两个 `tl.math.exp2`、一个常驻 `q`、两条 `advance`);配一趟两块 K 的数值走查(首块 alpha=0 抹哨兵、次块 alpha=0.2231<1 重标定)。**这一步回收 f8**。
2. **JIT 特化与缓存键**——内存级 `sig_and_spec+constexpr_vals`(jit.py:L580)/磁盘级 `sha256(triton_key+src.hash+backend.hash+options.hash+env)`(compiler.py);`src.hash` 含 `fn.cache_key`(DependenciesFinder AST + starting_line_number),故改核体即失效——这正是复现要 `TRITON_ALWAYS_COMPILE=1` 的原因。
3. **AST→TTIR**——`make_ttir` 里 `add_rewrite_tensor_pointer` 把 `make_block_ptr` 抹平成 `tt.splat`+`tt.addptr`+`tt.load`;`tt.dot/tt.reduce/math.exp2` 成形但结果还是裸 `tensor<128x64xf32>`、没有布局。
4. **TTIR→TTGIR**——`convert_to_ttgpuir` 打上 num-warps/threads-per-warp 并派三种布局 `#blocked`(访存)/`#mma`(versionMajor=2 instrShape[16,8])/`#shared`(swizzle);`tt.dot` 操作数戴 `dot_op<parent=#mma>`,P 经 `ttg.convert_layout` 转布局喂第二个 dot(非免费搬运)。
5. **Coalesce/AccelerateMatmul/Pipeliner 各改一处**——Coalesce 选 `#blocked sizePerThread=[8,1]`(带宽)、AccelerateMatmul 把 `tt.dot` 改成结果 `#mma`(算力)、Pipeliner 把 K/V load 做成 `memdesc<2x64x64>` 双缓冲+`async_copy`(延迟隐藏)。
6. **TTGIR→LLVM→PTX**——`make_llir` 的 `allocate_shared_memory` 给三块 memdesc 分配 **49152 字节**、`to_llvmir` 降成 NVVM/inline-asm(kernel 变 `ptx_kernel`);`make_ptx` 落到 **256 条** `mma.sync.aligned.m16n8k16`、**136 条** `ex2.approx.ftz.f32`(答第一步悬念:qk_scale 预乘 1/ln2 换 exp2 就为吃这条硬件指令)、**48 条** `cp.async.cg.shared.global`。反直觉观察:即便 target 是 sm_120a(Blackwell),fp16 dot 仍走 Ampere 系 `mma.sync` 而非 tcgen05——这也是本章 IR 地标跨 3.2.0↔3.6.0/三代架构稳定的原因。

结尾一张「验尸清单」表把六层地标倒着串成读者能立刻复用的工具:给自己的核设 `TRITON_KERNEL_DUMP=1`,照表逐层查布局对不对、dot 上没上 Tensor Core、循环有没有流水。

**f8 回收(全书最后一个伏笔,ch07→ch43,跨 36 章)**:f8 埋于 ch07(block pointer 的 `advance(offsets)` 沿滑窗维守恒——`block_shape`/`order`/`strides` 不变、只挪起点 `offset`)。ch43 是它的真实 kernel 落点:主循环 `K_block_ptr = advance(0, BLOCK_N)`、`V_block_ptr = advance(BLOCK_N, 0)` 沿序列维滑窗,narrative L212 显式兑现并回链第 7 章(『block_shape、order、strides 全程不变,只挪起点 offset』)。已 `bible.py payoff --resolve f8 --in ch43`→arc-map f8 status=resolved/resolved_in=ch43,`bible.py due ch43` 回收栏已空。注:发车 focus 误写『无伏笔』,但 dossier/writer 正确交付 f8,仅归档记账需补——已补齐。

诚实边界:四层 dump 抓自真机(triton 3.6.0 / sm_120a Blackwell,num_warps=4/num_stages=3/BLOCK_M=128/BLOCK_N=64/HEAD_DIM=64/fp16/causal),正文有醒目版本声明说明与 pin 3.2.0 的细节数字(寄存器名/loc/shared 字节)会略有出入、引用的都是跨版本稳定的**地标**,SSA 值名已重命名为语义短名、loc 一律省去、类型/属性/指令逐字未改。~15 处源码引用逐行核对 pin v3.2.0 精确(唯 1 anchor off-by-one `jit.py:L717-L724→L717-L725` 已派 writer 修);关键数字锚定真机 dump(49152 字节 / 256 mma / 136 ex2 / 48 cp.async 均 grep 三方一致);数值走查用 HEAD_DIM=4/BLOCK=2/N_CTX=4 最小非退化例,`N_CTX=1024/8192` 的显存量级对比已标『示意,与实抓 dump 配置无关』。

质量事件:3 轮 write-review + blind round 1(PASS,无 failure)+ map round 1(PASS),impl/test 按 skip_impl 跳过。review APPROVED、8 项 issue 全 non-blocking。**Lead 修 explainer.json m1.quantified 算术漂移**(素材真相源错、正文对):32768 float→8192 float、128 KB→32 KB、4×→16×、32×→128×。**Lead 派 writer 补 6 处**:jit.py anchor off-by-one、`fp8_v` 标识符解释、make_llir 六 pass 省略括注、`N_CTX` 举例标示意、`cdiv` gloss、(可选)m5 三 pass 直觉总起句+指回 TTGIR。过程教训(experience 候选):issue 2 揭示自动化盲区——`lint_trace_consistency` 只比对 `<!-- trace -->` 标记后的表格,不查 explainer.json 自由文本 `quantified` 字段,素材真相源里的算术错可绕过全部门禁、仅靠正文暗自算对兜住(本次侥幸未误导读者);若『quantified 字段漂移』复发≥2,考虑 `lint_explainer` 扩展到校验 quantified 自洽性(至少 float 数×字节=KB 的量纲自检)。skip_impl 无精简版接口(按契约跳过 interfaces 登记)。

bible 回写:glossary +7(509→516,NEW:add_rewrite_tensor_pointer / ex2.approx / ptx_kernel / TRITON_ALWAYS_COMPILE / tcgen05 / STAGE 位掩码分趟 / fp8_v 分支——mma.sync/m16n8k16/cp.async/instrShape/memdesc/make_ttir·ttgir·ptx/三种布局等前 42 章已有的不重复登记)、concepts +14(345→359,全标 ch43,尤其『同一个真核上看每层降级的痕迹』收束视角 + 验尸清单)、figures +6(89→95,含 chapter-map)。

## Why it matters

这是全书 43 章的收官:前 42 章每章各磨一片透镜(一章讲 `tl.dot`、一章讲布局、一章讲软件流水),读者始终缺一根把它们叠成望远镜、在**一个真核**上看『这些机制到底是不是接力发生』的主线。ch43 用 tutorial 06 当活体标本,把 `tl.*`→JIT 特化→TTIR→布局指派→Coalesce/AccelerateMatmul/Pipeliner→LLVM→PTX 在同一核上串通,并交给读者一份可立即复用的验尸清单(设 `TRITON_KERNEL_DUMP=1` 照表逐层查)——把全书从『读懂 Triton 各层怎么工作』升级成『能对自己的核做同样的层层归因』。f8 的回收(ch07 埋下的 advance 守恒写法,在真实 kernel 主循环里兑现)是全书最后一个伏笔、跨 36 章闭合,象征前后呼应的连贯性设计走完全程。过程上,explainer quantified 字段能绕过门禁的盲区值得记入 experience ledger 观察——素材真相源(explainer.json)的自由文本数字目前无确定性校验,是 v3『素材先行』管线的一处未覆盖缝隙。

## What to remember

全书收官 capstone(deep/skip_impl,APPROVED,capstone=true)。标本=tutorial 06 `_attn_fwd`,`TRITON_KERNEL_DUMP=1 TRITON_ALWAYS_COMPILE=1` 抓四层 dump,沿 `add_stages` 六步串前 42 章:tl.* 表面(两 dot/两 exp2/常驻 q/两 advance)→JIT 缓存两级键(改核体即失效,故要 ALWAYS_COMPILE)→AST→TTIR(add_rewrite_tensor_pointer 抹平块指针、还没布局)→TTGIR(convert_to_ttgpuir 派 #blocked/#mma[v2 instrShape16,8]/#shared + convert_layout)→三 pass 各一处(Coalesce 带宽/AccelerateMatmul 算力/Pipeliner memdesc<2x> 双缓冲+async_copy)→LLVM/PTX(allocate_shared_memory 49152 字节、ptx_kernel、mma.sync.m16n8k16×256 / ex2.approx×136 / cp.async×48;fp16 dot 即便 Blackwell 仍走 mma.sync 非 tcgen05,地标跨三代稳)。收尾验尸清单表倒着串六层地标。**回收 f8=全书最后一个伏笔(ch07 advance 守恒→ch43 真核落点,跨 36 章),已 resolve,due ch43 空**。dump 版本 3.6.0/sm_120a 与 pin 3.2.0 有版本声明、引用地标跨版本稳、1 anchor 已修、数字锚真机 grep 三方一致、N_CTX 举例标示意。Lead 修 explainer m1.quantified 算术漂移(8192 float/32KB/16×/128×)+ 派 writer 补 6 处(non-blocking)。教训:lint_trace_consistency 不查 explainer.json quantified 自由文本,算术错可绕门禁(复发≥2 则扩 lint_explainer)。glossary +7 / concepts +14 / figures +6。**全书 43 章至此全部定稿——triton 主书完结。**
