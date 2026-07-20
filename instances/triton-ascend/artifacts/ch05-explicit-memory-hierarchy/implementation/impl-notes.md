# ch05 精简版实现说明——显式内存层级:UB/GM/L1/L0C、buffer 语言与 copy/fixpipe

## 范围

本章精简版只保留 dossier `mechanisms`(M1-M7)真正需要的四个文件的对应行区间：

- `third_party/ascend/language/cann/extension/core.py`——地址空间可见(`ascend_address_space`)
  + `copy`/`copy_from_ub_to_l1` 前端 + Fixpipe 四组模式枚举 + `fixpipe` 前端(对齐/dtype/芯片校验)。
- `third_party/ascend/language/cann/extension/semantic.py`——`copy`/`copy_from_ub_to_l1`/`fixpipe` 语义实现。
- `python/triton/extension/buffer/language/core.py`——`address_space`/`buffer_type`/`buffer` +
  `alloc`/`to_buffer`/`to_tensor`/`subview` 前端。
- `python/triton/extension/buffer/language/semantic.py`——上述四个 builtin 的语义实现。

按 dossier `subtraction_plan.delete` 批准，`third_party/.../extension/core.py` 的 CORE/PIPE/MODE/
IteratorType 枚举、sync_block_set/wait/all、create_sync_block、sub_vec_id、sub_vec_num、
debug_barrier、SYNC_IN_VF、int64、以及该文件自己的 `@builtin`/`is_builtin` 双标记装饰器机制
全部不纳入本章精简版——它们属于 ch08(scope/同步/流水线)与 ch04(双 builder 的 `@builtin` 契约，
已在该章讲清"打双标记→CodeGenerator.visit_Call 按标记路由到 ascend_builder"这件事)，与本章
"内存层级对程序员可见 + 显式 copy/fixpipe"这条主线正交。`mem_ops.py`(index_put/gather_out_to_ub/
scatter_ub_to_out/index_select_simd)按已审批大纲整体归 ch06，本章不引入。

`triton.language.core`(tl)与 `triton.language._utils` 不是本章机制主角——它们是 buffer 语言/
al.copy/al.fixpipe 依赖的基座 Triton 值系统，按 code_spine 之外的最小子集裁剪(同 ch04 对同一文件
的处理方式)，只留 `constexpr`/`dtype`/`block_type`/`_value`/`tensor` 与本章实际用到的六个标量
dtype 实例(int1/int16/int32/float16/bfloat16/float32)，删除全部与内存层级/copy/fixpipe 无关的
算子重载与类型目录。

## Source Map

| 精简版位置 | 对应真实源码 | 改动 | 原因 |
|---|---|---|---|
| `third_party/.../extension/core.py` `ascend_address_space_base`/`ascend_address_space_group`/`ascend_address_space` | `third_party/ascend/language/cann/extension/core.py:L143-163` | 原样保留 | M1 主角——地址空间对程序员可见的入口，反射生成机制逐字保留 |
| 同上文件 `copy_from_ub_to_l1`/`copy` 前端 | `third_party/.../core.py:L174-199` | 去掉 `@builtin` 装饰器应用(仅装饰器，函数体逐字保留)；deprecation `warn()` 保留 | `@builtin`/`ASCEND_BUILTIN` 双标记机制已在 ch04 讲清，本章不重复；函数体（含弃用告警）是 M3 的一部分 |
| 同上文件 `FixpipeDMAMode`/`FixpipeDualDstMode`/`FixpipePreQuantMode`/`FixpipePreReluMode` | `third_party/.../core.py:L247-270` | 原样保留 | M4 的四组模式枚举，读者需理解 NZ2ND/COLUMN_SPLIT 等具体含义 |
| 同上文件 `fixpipe` 前端 | `third_party/.../core.py:L273-333` | 去掉 `@builtin` 装饰器应用；校验逻辑(910_95 门禁/dtype 判断/对齐算术)逐字保留 | M4/M7 主角——对齐约束的具体算术必须逐字可核对 |
| `third_party/.../extension/semantic.py` `copy_from_ub_to_l1`/`copy`/`fixpipe` | `third_party/.../semantic.py:L94-148` | 原样保留；文件顶部删去 `PIPE`/`create_address_space`/`create_sync_block_*`/`sub_vec_id`/`debug_barrier`(code_spine 本就未圈定这些行区间) | M3/M4 语义实现主体；删除项归 ch08，本章不引入其依赖 |
| `python/triton/extension/buffer/language/core.py` `address_space`/`buffer_type`/`buffer` | `python/.../core.py:L70-184` | 只删类文档字符串(rubric 部分)，逻辑逐字保留 | M1/M2/M5 的类型/值对象骨架 |
| 同上文件 `alloc`/`to_buffer`/`to_tensor` | `python/.../core.py:L190-246` | 原样保留(含 `@builtin` 装饰器——buffer 语言自己的双标记机制未被 ch04 覆盖，本章保留) | M2/M5 主角 |
| 同上文件 `check_subview`/`subview` | `python/.../core.py:L249-363` | 按 subtraction_plan 批准：删去 `check_subview` 的详细对齐推导文档字符串；`subview` 删去 offsets/sizes/strides 的 constexpr/张量规整分支，改为直接接受纯 int，其余骨架(校验非负→`check_subview`→转发 `semantic.subview`)逐字保留。`check_subview` 函数体本身**逐字未改**：length==1 分支仍是真实源码里的 `offset[0]`（形参是复数 `offsets`，`offset` 从未定义——上游真实存在的 bug，rank-1 缓冲的 subview() 调用在真实仓库里会炸成 `NameError`，本章不"顺手修好"它，见下方「已知的结构性留白」）；for 循环里 `if isinstance(offsets[i], tl.tensor): return` 的早退分支也原样保留（这是 `check_subview` 自身的控制流，不在 subtraction_plan 批准删除范围内，即使本精简版 `subview()` 前端已把 offsets 收窄成纯 int、使这条支路只能通过直接调用 `check_subview` 触达） | M5 supporting 细节；被批准删除的是"次要类型规整"而非算法本身 |
| `python/triton/extension/buffer/language/semantic.py` 全部四函数 | `python/.../semantic.py:L35-158` | `subview` 因 core.py 侧的简化，offsets 直接以 int 列表转发给 `builder.subview`(不再 `.handle` 取值)；`alloc`/`to_buffer`/`to_tensor` 逐字保留 | M2/M5 语义实现主体 |
| `python/triton/language/core.py`(tl) | `python/triton/language/core.py`(节选，见文件内各符号行号) | 大幅精简为 `constexpr`(部分方法)+`dtype`(六类型子集)+`block_type`+`_value`+`tensor`(仅 `__init__`/`__str__`)+ 六个标量 dtype 实例 | 供 buffer 语言/al.copy/al.fixpipe 依赖的最小容器子集，本身不是本章机制 |
| `python/triton/language/_utils.py` `validate_block_shape` | `python/triton/language/_utils.py:L17-29` | 原样保留 | `block_type` 构造时的 shape 校验依赖 |

## 已知的结构性留白（非缺陷）

- `third_party/.../extension/core.py` 的 `copy`/`copy_from_ub_to_l1`/`fixpipe` 在本精简版里
  不再套 `@builtin` 装饰器（该机制已在 ch04 讲清，删除属 dossier `subtraction_plan.delete` 第 1
  项批准范围）——测试直接以关键字参数 `_builder=` 调用这三个函数，与真实源码经
  `CodeGenerator.visit_Call` 按双标记路由到 `_builder` 的效果一致，只是没有模拟那条路由链路
  （路由链路归 ch04）。
- `subview` 的 offsets/sizes/strides 只接受纯 int（或已经是 int 的 `tl.constexpr`），不支持真实
  源码里"偏移可以是运行时 `tl.tensor`"这条支路——该支路依赖 `triton.language.semantic.to_tensor`/
  `full`/`broadcast` 等基座算子分发的完整实现，与本章内存层级主线无关，按 dossier 批准的
  "constexpr 规整分支细节"一并删除。**这是 `subview()` 前端的限缩，不是 `check_subview()` 的限
  缩**——`check_subview` 自身的 `isinstance(offsets[i], tl.tensor): return` 分支逐字保留（见上方
  Source Map），只是通过 `subview()` 这条精简后的入口已经无法再触达（需要直接调用
  `check_subview` 才能看到，见 `tests/test_buffer_tensor_bridge.py`）。

## 上游真实缺陷（原样保留，非本章引入）

- **`check_subview` 的 rank-1 分支引用未定义变量 `offset`（应为 `offsets`）**：真实源码
  `python/triton/extension/buffer/language/core.py:L279` 是 `if offset[0] % base_byte != 0:`——
  `check_subview` 的形参是复数 `offsets`，函数体内从未定义过单数 `offset`。这是上游真实存在的
  bug：真实 triton-ascend 仓库里任何 rank-1 缓冲（`len(strides) == 1`）调用 `subview()`/
  `buffer.subview()` 都会在这里炸成 `NameError: name 'offset' is not defined`，而不是走到
  32-byte 对齐校验。本章只解读源码、不代表实现者"顺手修好"上游 bug——精简版逐字保留
  `offset[0]`（而非误改成能正常求值的 `offsets[0]`），并用
  `tests/test_buffer_tensor_bridge.py::test_check_subview_rank1_raises_nameerror_matching_upstream_bug`
  断言这条 `NameError` 会被复现。这不属于 dossier `subtraction_plan` 授权的删减范围——是
  「保真度要求逐字保留，包括上游的真实缺陷」的直接体现，而非本章引入的新问题。

## 验证

- `python3 -m pytest tests/` — 40 passed（纯 Python 单元测试，靠 `conftest.py` 的 `FakeBuilder`
  站在真实 `ir.builder`/`ascendnpu_ir_builder`（C++ 绑定，host 无昇腾 NPU/CANN 工具链故无法拥有）
  位置上，验证的是"地址空间校验/copy 方向门禁/fixpipe 对齐算术/buffer↔tensor 桥"这些 Python 语言
  层可观察行为，不模拟 hivm op 的 IR 语义——IR dump 级别验证需真机，按 INSTANCE.md 约束不在本章
  测试范围）。
- `python3 scripts/lint_fidelity.py {chapter_dir}` — 无 BLOCKING。
