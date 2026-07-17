# ch38 素材取证账本（trace_source=manual）

本章 kind=skip_impl，无精简版，且 AMD 编译末端（make_amdgcn/make_hsaco）需 ROCm 工具链
（ld.lld / assemble_amdgcn），host 无 AMD GPU 与 ROCm 环境，无法运行取真实字节轨迹。
故全部 figure 数字均为**源码常量/结构事实静态取证**（pin v3.2.0），逐个标 `file:Lxxx`。
本文件是 explainer.json 中 figure numbers 的 provenance 落地清单，供盲审逐条核对。

## m1 — BaseBackend 契约面（python/triton/backends/compiler.py, pin v3.2.0）
- 6 个 @abstractmethod：supports_target / hash / parse_options / add_stages /
  load_dialects / get_module_map —— L226-L290。
- 2 个带默认实现的可覆写钩子：get_attrs_descriptor / compute_spec_key —— L292-L304。
- AttrsDescriptor._add_backend_properties 空钩子（pass）—— L100-L102。
- 落地份数：CUDABackend（nvidia/backend/compiler.py）+ HIPBackend（amd/backend/compiler.py）= 2 份。

## m2 — HIPOptions vs CUDAOptions 字段
AMD（third_party/amd/backend/compiler.py）：
- num_warps 默认 4 —— L30
- waves_per_eu 默认 1 —— L31（AMD 专属）
- num_stages 默认 2 —— L32
- matrix_instr_nonkdim 默认 0 —— L48（AMD 专属）
- kpack 默认 1 —— L49（AMD 专属）
- warp_size：__post_init__ 按 gfx10/11/12→32 否则 64 —— L64-L66（AMD 专属计算项）
- backend_name 'hip' —— L52

NVIDIA（third_party/nvidia/backend/compiler.py）：
- num_warps 默认 4 —— L93
- num_stages 默认 3 —— L95
- maxnreg 默认 None —— L102（NVIDIA 专属）
- ptx_version 默认 None —— L104（NVIDIA 专属）
- 无 warp_size 字段；warp 恒 32，在 add_convert_to_ttgpuir 处硬编码字面量 32 —— L218

## m4 — add_stages 五段骨架
AMD add_stages —— third_party/amd/backend/compiler.py:L358-L363
- ttir / ttgir / llir / amdgcn / hsaco（5 段）；前三段 ttir/ttgir/llir 与 NVIDIA 同名。
NVIDIA add_stages —— third_party/nvidia/backend/compiler.py:L384-L389
- ttir / ttgir / llir / ptx / cubin（5 段）；末两段 ptx/cubin。
- 共有前缀 3 段（ttir/ttgir/llir），差异末 2 段。

## m5 — 工具链末端 amdgcn/hsaco vs ptx/cubin
AMD（third_party/amd/backend/compiler.py）：
- make_amdgcn：llvm.translate_to_asm(src, amd.TARGET_TRIPLE, arch, ...) 出 amdgcn —— L330-L338
- make_hsaco：amd.assemble_amdgcn(src, arch, '') —— L346
- path_to_rocm_lld() 找 ld.lld —— L171-L188；subprocess.check_call([lld, '-flavor','gnu','-shared',...]) —— L353
NVIDIA（third_party/nvidia/backend/compiler.py）：
- make_ptx：llvm.translate_to_asm(...) 出 PTX —— L318-L324
- make_cubin：_path_to_binary("ptxas") + subprocess 调 ptxas —— L340-L353
- translate_to_asm 是两后端共用入口，只是 target triple/arch 不同（AMD L338 vs NVIDIA L324）。
- binary_ext：AMD 'hsaco'（L126）对照 NVIDIA 'cubin'（L142）。

## m6 — make_ttgir AMD pass 序列（同位置换 amd 专属 pass）
AMD（third_party/amd/backend/compiler.py）：
- amd.passes.ttgpuir.add_accelerate_matmul(pm, arch, matrix_instr_nonkdim, kpack) —— L218（3 个后端参数）
- 门控 amd.has_matrix_core_feature(arch) —— L222、L234
- amd.passes.ttgpuir.add_stream_pipelinev2(pm, num_stages) —— L228
NVIDIA（third_party/nvidia/backend/compiler.py）：
- passes.ttgpuir.add_accelerate_matmul(pm) —— L227（0 参）
- 门控 capability // 10 >= 8 —— L231
- passes.ttgpuir.add_pipeline(pm, num_stages) —— L239
