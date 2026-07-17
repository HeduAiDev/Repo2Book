# ch38《对照落地:AMD HIP 后端——同一抽象的第二种实现》定稿——Part VIII 收官

- **Type**: delivery
- **Chapter**: ch38
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T20:15:23Z
- **Agents involved**: analyst, explainer, illustrator, writer, reviewer, archivist
- **User present**: False
- **Tags**: triton, part-8, amd-hip, backend, 配对脊柱, skip_impl

## What happened

Part VIII(硬件后端)收官章,也是全书配对脊柱主线的收官示范。以 AMD HIP 后端(third_party/amd/backend/compiler.py 的 HIPBackend)当活样板,与 ch36 CUDABackend/ch37 NVIDIA 工具链逐面对照,坐实『同一套 BaseBackend 抽象、两种落地』:六个面每面同一契约位置两种填法。(1)编译选项:HIPOptions 与 CUDAOptions 绝大多数字段逐字段对齐(num_warps 默认4),num_stages 默认 2(NVIDIA 3);AMD 专属 waves_per_eu/matrix_instr_nonkdim/kpack 三个 matrix core 旋钮 + warp_size 在 __post_init__ 按 gfx 档现算(RDNA gfx10/11/12=32、CDNA gfx9 等=64,object.__setattr__ 绕 frozen);NVIDIA warp 恒 32 无此字段直写字面量——最尖锐差异点。parse_options gfx940/941/942 才补 fp8e4b8/fp8e5b16。(2)五段骨架 add_stages:ttir/ttgir/llir 前三段两后端键名一字不差,末两段 amdgcn/hsaco 对 ptx/cubin(binary_ext='hsaco'),末段返 bytes 对齐契约。(3)make_ttgir 站位顺序不变、只换后端专属 pass 与门控:加速矩阵乘站 amd.add_accelerate_matmul 带 arch/nonkdim/kpack 三参 vs NVIDIA 零参(机制回指 ch28);软件流水站门控 has_matrix_core_feature 探测 vs capability//10>=8 分档、开 add_stream_pipelinev2(原理回指 ch29/ch30)。(4)工具链末端两后端共用汇编入口 llvm.translate_to_asm 后分叉:AMD assemble_amdgcn+ld.lld -shared 出 hsaco 两步 vs NVIDIA ptxas 一步出 cubin(回指 ch37)。(5)双语接缝 load_dialects/get_module_map:amd.* vs nvidia.* 命名空间。(6)专属特化:HIPAttrsDescriptor 覆写基类空钩子 _add_backend_properties 注入 tt.pointer_range=32(断言指针 ≤2GiB is_within2gb),解锁 AMD buffer load/store(硬件级越界+降寄存器压力),经 AMDGCN_USE_BUFFER_OPS 下 add_convert_to_buffer_ops 消费;NVIDIA 不覆写用基类空 pass——配对脊柱第四面『专属能力从基类预留空位长出』。另 make_llir 设 AMDGPU control constants(__oclc_wavefrontsize64=(warp_size==64)/amdgpu-flat-work-group-size=num_warps*warp_size/amdgpu-waves-per-eu),三个 HIPOptions 字段落成 LLVM 内核属性。逐机制 10/10 覆盖,全 linter green,reviewer APPROVED。归档:Lead 派 writer 补 4 处(3 处不变量加粗 + matrix core/wave-wavefront 用词统一/『1,』前缀),归档时 narrative 仍在小修,archivist 只写 bible/trace 无冲突。无精简版接口(skip_impl)。本章无伏笔埋/回收(bible.py due ch38 为空)。

## Why it matters

全书配对脊柱(hardware-backend 抽象『6 面两种落地』)主线的收官示范:NVIDIA 一份 CUDABackend(ch36/ch37)+AMD 一份 HIPBackend(本章)→姊妹篇《Triton-Ascend 源码解读》AscendBackend 第三份。本章交付=Part VIII(ch36-38)全部完成、triton 主书 ch01-38 连续。回答『一块新卡怎么接进 Triton』:填 BaseBackend 六方法+add_stages 钉五段(前三照抄末两换目标格式)+make_ttgir 既有站位换 pass+按需覆写钩子——开放-封闭原则教科书落地,加后端=新填一份而非改一遍编译器。

## What to remember

ch38=Part VIII 收官+配对脊柱主线收官示范。HIPBackend 是 BaseBackend 第二份落地,与 CUDABackend 同契约面六种『同一位置两种填法』:选项(HIPOptions warp_size 按 gfx 32/64、waves_per_eu/matrix_instr_nonkdim/kpack 专属)、五段骨架(前三同名末两 amdgcn/hsaco)、make_ttgir 换 pass 换门控(has_matrix_core_feature vs capability)、末端 ld.lld 出 hsaco vs ptxas 出 cubin、双语 amd.* vs nvidia.*、专属特化 tt.pointer_range=32 覆写钩子启 buffer ops。ascend 是等填的第三份。skip_impl,无伏笔。glossary +21、concepts +9。
