# ch30 发射器素材 — 手工推演底稿(trace_source="manual")

本章无精简版(skip_impl)。发射器 wrapper 依赖 CANN 闭源运行时(rtKernelLaunch / msprof)
与昇腾 NPU 设备,host 无 CANN/无 NPU,无法真跑。以下所有数字是对
`third_party/ascend/backend/driver.py` 里 `generate_npu_wrapper_src` 的 Python f-string
拼装逻辑,针对一组选定的 kernel 签名 + metadata 做的手工推演;凡引用源码常量均标 file:Lxxx。

## 选定场景(读者可心算)

一个经典向量加 kernel:

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr): ...

- signature = {0:'*fp32', 1:'*fp32', 2:'*fp32', 3:'i32'}   (BLOCK_SIZE 为编译期 constexpr,
  已在编译期折叠,不进 launcher 签名 → constants 对 launcher 为空)
- metadata(ch28/ch29 落定): workspace_size=256(单 block 字节), lock_num=2,
  lock_init_value=0, force_simt_only=False, compile_on_910_95=False,
  target_support_ffts=False, enable_device_print=False
- enable_taskqueue = True (默认, driver.py:L512-513 环境变量 TRITON_ENABLE_TASKQUEUE 缺省 'true')
- 后端 = torch_npu
- 发射 A: grid=(4,1,1);  发射 B(同一编译产物第二次调用): grid=(8,1,1)

## m1 条件注入命中集

generate_npu_wrapper_src 里每个注入点由一个 metadata 布尔/数值开关控制:

| 注入点                         | 控制条件                        | 本例取值        | 命中? | 源 |
|--------------------------------|---------------------------------|-----------------|-------|----|
| workspace 分配段               | workspace_size > 0              | 256 > 0         | 是    | driver.py:L773-776 |
| syncBlockLock 段               | lock_num > 0                    | 2 > 0           | 是    | driver.py:L800-815 |
| 发射逻辑包成 lambda(异步)      | enable_taskqueue                | True            | 是    | driver.py:L777,841 |
| ffts_addr 字段/取址            | target_support_ffts             | False           | 否    | driver.py:L795,818 |
| device_print(DTData/DebugTunnel)| enable_device_print            | False           | 否    | driver.py:L752,793,823 |
| rtKernelLaunchWithFlagV2 变体  | compile_on_910_95 and enable_simt| False          | 否    | driver.py:L736-744 |

命中 3(workspace/lock/lambda),跳过 3(ffts/print/V2)。模板主体约 560 行 C++(driver.py:L750-965)。

## m2 format 串 + arg_decls

- 定长前缀恒 = "iiiKKOOOO"  (driver.py:L503) —— 9 个格式符对应 9 个定长参数:
  gridX/Y/Z(iii)、stream(K)、function(K)、packedMetadata/launch_metadata/enter_hook/exit_hook(OOOO)
- 逐 signature slot 追加(driver.py:L503 `_format_of(_extracted_ty(ty))`):

  | slot | 源类型 | _extracted_ty(L438) | _format_of(L459) 格式符 |
  |------|--------|---------------------|-------------------------|
  | 0    | *fp32  | PyObject*           | O |
  | 1    | *fp32  | PyObject*           | O |
  | 2    | *fp32  | PyObject*           | O |
  | 3    | i32    | int32_t             | i |

- 追加段 = "OOOi";  完整 format = "iiiKKOOOO" + "OOOi" = "iiiKKOOOOOOOi" (13 格式符)
- arg_decls(driver.py:L493, `_ty_to_cpp`): "void* arg0, void* arg1, void* arg2, int32_t arg3"
  (*fp32→void* L418; i32→int32_t L424)
- launch() 里对应追加 13-9=4 个 &_argi 取址(driver.py:L893),格式符数 == 取址数。

## m4 workspace 每次现分配

totalWorkSpaceSize = workspace_size × blockNum4Workspace,  blockNum4Workspace = gridX·gridY·gridZ
(driver.py:L771 定义, L774 相乘)

| 发射 | grid    | blockNum4Workspace | totalWorkSpaceSize(字节) |
|------|---------|--------------------|--------------------------|
| A    | (4,1,1) | 4                  | 256 × 4 = 1024           |
| B    | (8,1,1) | 8                  | 256 × 8 = 2048           |

每进 _launch 先 `void *workspace_addr_ptr = NULL;`(L770)再按本次 grid 重算,不跨调用复用。
分配走 torch_npu 的 at::empty(kPrivateUse1)(backend_register.py:L300)。

## m5 syncBlockLock 现分配 + 初值

syncBlockLockSize = lock_num × sizeof(int64_t) = 2 × 8 = 16 字节 (driver.py:L801)

| 步 | 动作                                             | 关键标量          | 源 |
|----|--------------------------------------------------|-------------------|----|
| 1  | 判是否注入(lock_num>0)                            | 2 > 0 → 注入      | driver.py:L815 |
| 2  | 算大小 syncBlockLockSize                          | 16 字节           | driver.py:L801 |
| 3  | allocate_sync_block_lock 现开(at_npu allocate_workspace) | ptr!=NULL   | backend_register.py:L311 |
| 4  | 建初值向量 lockInitData(lock_num, lock_init_value)| {0, 0}            | driver.py:L806 |
| 5  | rtMemcpy 16 字节 host→device                      | 16 字节, H2D      | driver.py:L807-811 |

## m6 packed args struct 布局(本例)

force_simt_only=False → 含 syncBlockLock+workspace 两个头字段;无 ffts/DTData。
字段声明顺序即初始化顺序(driver.py:L817-833),对齐规则 L821:
`4 if ty[0] != '*' and ty[-2:] != '64' else 8`(指针/64 位 → 8,余 → 4)。

| 顺 | 字段          | C++ 类型 | 对齐 | 宽度 | 累计偏移 | 初值来源 |
|----|---------------|----------|------|------|----------|----------|
| 0  | syncBlockLock | void*    | 8    | 8    | 0        | syncBlockLock_ptr (lock_num>0) |
| 1  | workspace_addr| void*    | 8    | 8    | 8        | workspace_addr_ptr (workspace_size>0) |
| 2  | arg0          | void*    | 8    | 8    | 16       | x_ptr 设备指针 |
| 3  | arg1          | void*    | 8    | 8    | 24       | y_ptr 设备指针 |
| 4  | arg2          | void*    | 8    | 8    | 32       | out_ptr 设备指针 |
| 5  | arg3          | int32_t  | 4    | 4    | 40       | n_elements |
| 6  | gridX         | int32_t  | 4    | 4    | 44       | 4 |
| 7  | gridY         | int32_t  | 4    | 4    | 48       | 1 |
| 8  | gridZ         | int32_t  | 4    | 4    | 52       | 1 |

sizeof(args) = 8+8+8+8+8+4+4+4+4 = 56 字节(packed 无补齐)。
rtKernelLaunch(func, blockNum=4, &args, sizeof(args)=56, NULL, stream) (driver.py:L734)。
constants 不进 struct(`if i not in constants`, L821/L829);argsSize 恒 = sizeof(args),随字段增减自动。
