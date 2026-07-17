# ch37《从 PTX 到 cubin 到发射:ptxas、二进制装载与 launcher》定稿——Part VIII 落地段

- **Type**: delivery
- **Chapter**: ch37
- **Date**: 2026-07-17
- **Timestamp**: 2026-07-17T19:10:41Z
- **Agents involved**: writer, illustrator, reviewer, archivist
- **User present**: False
- **Tags**: delivery, ch37, part-8, skip_impl, ptxas, cubin, launcher, loadBinary, occupancy, 48KB-optin, sm_120a-honest-note, APPROVED

## What happened

Part VIII 硬件后端落地段章,skip_impl,APPROVED。承 ch36 五段注册,详解编译链最后两段真正产机器码 + 装载 + 发射全程。make_ptx=llvm.translate_to_asm 出 PTX 文本 + 三步后处理(正则抓唯一核名 assert len==1 写 metadata['name']/规约 .version 成 ptxas 认得的 major.minor/去 debug 标志);make_cubin 起 ptxas 子进程=编译链唯一外部工具进程(写死 -v 打 n_regs/n_spills 回执、--gpu-name 定架构,返回码非零按 255/128+SIGSEGV/其它分类拼 stderr+可复现命令报错,ptxas 是 CPU 程序无卡也跑到 cubin);loadBinary(C 侧)cuModuleLoadData 搬进显存→cuModuleGetFunction 取句柄→cuFuncGetAttribute 读 n_regs/n_spills(LOCAL_SIZE_BYTES÷4)+48KB opt-in AND 门(shared>49152 && shared_optin>49152 才 cuFuncSetAttribute 抬动态额度+cuFuncSetCacheConfig 偏 shared);occupancy=寄存器/线程/共享三闸取最小、n_regs 单增单调不升;make_launcher 按核签名现焊 C 发射器(ty_to_cpp/PyArg 格式/取址逐参映射,sha256(C源)编 .so 缓存=launch overhead 来源);_launch 按 num_ctas==1 二分(cuLaunchKernel vs cuLaunchKernelEx,dlsym 探测新符号);CudaDriver 配对脊柱落地端(is_active 判 hip is None 排 ROCm,AMD 侧镜像链)。skip_impl 无精简版接口。**诚实注**:worked example 的 sm_120a 来自 Blackwell(cap 120)真机+支持 Blackwell 的更新版 triton;pin v3.2.0 源码 'a' 后缀只对 capability==90 加,按 pin cap 120 本应得 sm_120——两版本已在正文注明。归档:Lead 派 writer 挑明 sm_120a 与 pin(==90)架构差异(exp-0718-1:本机 Blackwell+更新版 triton,数字真机取证保留)+4 处可读性;illustrator 修 2 图注(host→真机/示例框无因果)。bible.py due ch37 确为空(本章无伏笔埋/回收)。glossary +18(414→432:make_ptx/llvm.translate_to_asm/make_cubin/SASS/cubin/--gpu-name=sm_XX/n_regs·n_spills/loadBinary/cuModuleLoadData/cuFuncGetAttribute/cuFuncSetAttribute/cuFuncSetCacheConfig/make_launcher/CudaLauncher/ty_to_cpp/compile_module_from_src/cuLaunchKernel·Ex/cuOccupancyMaxActiveClusters);concepts +10(272→282)。本章交付后 Part VIII 仅剩 ch38,ch01-37 连续。

## Why it matters

编译链最后两段(make_ptx/make_cubin)+装载(loadBinary)+发射(make_launcher/_launch)是全书从字符串 IR 到 GPU 真跑起来的落地收束;把 n_regs/n_spills、48KB opt-in、launch overhead 三个性能抓手从机制落成可读可改的决策,承 ch02 occupancy·spill/ch26 48KB/ch32 五级阶梯/ch36 五段。诚实处理 sm_120a 与 pin(==90)的架构/版本差异是保真度关键(避免读者把真机数字误当 pin v3.2.0 对 cap 120 的输出)。

## What to remember

Part VIII 硬件后端落地段章,skip_impl,APPROVED。承 ch36 五段注册,详解编译链最后两段真正产机器码 + 装载 + 发射全程。make_ptx=llvm.translate_to_asm 出 PTX 文本 + 三步后处理(正则抓唯一核名 assert len==1 写 metadata['name']/规约 .version 成 ptxas 认得的 major.min...
