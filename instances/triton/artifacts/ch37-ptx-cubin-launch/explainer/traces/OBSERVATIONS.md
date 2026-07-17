# ch37 真实取证观测 (host 实跑)

> **环境**: NVIDIA RTX PRO 6000 Blackwell (sm_120a) / CUDA 12.8 (ptxas V12.8.93) /
> triton **3.6.0** (host 安装版) — 本书 **pin 为 v3.2.0**。故 explainer 用
> `trace_source="manual"`(见各机制 manual_reason):这些数字是**真实实跑取证**,用于
> 锚定量级与验证公式,不是凭空捏造;而载荷源码常量(49152 / `/4` / `32*num_warps` /
> format 字符表)一律以 **pin v3.2.0 源码 file:Lxxx** 为准。ptxas -v 的寄存器/spill 报告
> 与 loadBinary 的 `n_spills = LOCAL_SIZE_BYTES/4` 公式跨 triton 版本稳定,故这里的实测
> 与书中论断一致。

## 设备常量 (torch.cuda.get_device_properties, host 实测)
| 属性 | 值 |
|---|---|
| regs_per_multiprocessor | 65536 |
| max_threads_per_multi_processor | 1536 |
| shared_memory_per_block (静态硬上限) | **49152** (= 48KB, 恰对 driver.c L132 阈值) |
| shared_memory_per_block_optin | 101376 |
| shared_memory_per_multiprocessor | 102400 |
| warp_size | 32 |
| multi_processor_count | 188 |

## m1 make_ptx — add_kernel PTX 头 (add_kernel_ptx_head.txt / gen_ptx.py)
```
.version 8.8            <- LLVM NVPTX 后端原始输出
.target sm_120a
.address_size 64
.visible .entry add_kernel(
.reqntid 128           <- blockDim=128 (num_warps=4)
```
- make_ptx 正则抓出唯一 `.visible .entry add_kernel` → 写入 metadata['name']='add_kernel'。
- 版本后处理: 本机 bundled ptxas 只支持 PTX 8.7,LLVM 出的是 8.8 → make_ptx L331
  `re.sub(r'\.version \d+\.\d+', ...)` 把 `.version 8.8`→目标版本。**这正是为何要改版本**:
  实测直接喂 8.8 给 ptxas 报 `fatal: Unsupported .version 8.8; current version is '8.7'`。

## m2 make_cubin — ptxas -v 真实 stderr (ptxas_add_sm120.log)
命令(bundled ptxas,与 make_cubin L352 同形):
`ptxas -lineinfo -v --gpu-name=sm_120a add_kernel.ptx -o add_kernel.o`
```
ptxas info : Compiling entry function 'add_kernel' for 'sm_120a'
ptxas info : Function properties for add_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 28 registers, used 0 barriers
```
- cubin 产物 add_kernel.o = **10544 字节**。
- `-v` 打印 "Used 28 registers" + "0 bytes spill" → 这正是读者判寄存器压力的直接来源。

## m2/m4 spill 案例 — heavy_kernel (ptxas_heavy.log / gen_heavy.py)
triton 装载读回: **n_regs=26, n_spills=8**。
ptxas -v: `Used 26 registers` + `32 bytes stack frame`。
- **验证 loadBinary 公式**: n_spills = LOCAL_SIZE_BYTES/4 = 32/4 = **8** ✓
  (driver.c L124-126 `n_spills /= 4`)。32 字节 local memory 即 spill 的代理指标。

## m3/m4 大共享内存+高寄存器 — mm_kernel (ptxas_mm.log / gen_matmul.py)
签名 BM=BN=128,BK=32,num_warps=4。triton 装载读回:
**n_regs=212** (ptxas -v 静态计数 "Used 208 registers" — 驱动 NUM_REGS 含 ABI 预留,略高),
**shared=65536 字节 (64KB)**,n_spills=0,blockDim=128。
- shared=65536 **> 49152** → 触发 loadBinary L132 的 >48KB opt-in 动态共享内存路径
  (65536 ≤ optin 上限 101376,装载成功)。

## m4 occupancy 计算 (设备常量 host 实测 + 上面真实 n_regs/shared)
每 SM 可驻留 block 数 = min(寄存器限, 线程限, 共享内存限):
- **add_kernel** (n_regs=28, shared=0, blockDim=128):
  寄存器限 floor(65536/(28×128))=floor(18.28)=**18**;线程限 floor(1536/128)=**12**;
  共享限 ∞。min=12 block → 12×128=1536 线程 → occupancy=1536/1536=**100%** (线程受限)。
- **mm_kernel** (n_regs=212, shared=65536, blockDim=128):
  寄存器限 floor(65536/(212×128))=floor(2.41)=**2**;线程限 **12**;
  共享限 floor(102400/65536)=**1**。min=1 block → 128 线程 → occupancy=128/1536=**8.33%** (共享内存受限)。
对照鲜明: add_kernel 100% vs mm_kernel 8.3%。

## m5 make_launcher — 真实生成的 C 扩展 (launcher_add.c, 由 pin v3.2.0 源码 make_launcher 实跑)
签名 `{0:*fp32, 1:*fp32, 2:*fp32, 3:i32}` (constexpr BLOCK_SIZE 不入签名):
- arg_decls: `CUdeviceptr arg0, CUdeviceptr arg1, CUdeviceptr arg2, int32_t arg3`
- format 字符串 = `"iiiKKOOOO"` + args_format(`"OOOi"`) = **`"iiiKKOOOOOOOi"`**
  ("iii"=grid, "KK"=stream+function, "OOOO"=4 个固定对象 metadata/launch_metadata/两 hook,
   "OOOi"=x_ptr(O) y_ptr(O) out_ptr(O) n_elements(i))
- getPointer 施于 3 个指针实参: `ptr_info0/1/2 = getPointer(_arg0/1/2, ...)`,标量 _arg3 直传。
- `void *params[] = { &arg0, &arg1, &arg2, &arg3 };`
- `cuLaunchKernel(function, gridX, gridY, gridZ, 32*num_warps, 1, 1, shared_memory, stream, params, 0)`
- 生成 C 源 9027 字节 → compile_module_from_src 按 sha256 编成缓存 .so。
