# ch23 HIVM 方言 — trace 素材来源

本章 `trace_source="manual"`。原因:HIVM 属 bishengir 编译器栈,取证需 `bishengir-opt`
可执行文件驱动 `-hivm-infer-mem-scope` pass;本仓仅含 AscendNPU-IR **源码**,未构建出
`bishengir-opt`(`which bishengir-opt` 为空、源码树内无该二进制),host 无法真机跑 pass。

因此 trace 的「地面真相」取自仓库**已提交的 lit 回归夹具**
`infer-hivm-mem-scope.mlir`(本目录副本,源:
`third_party/ascend/AscendNPU-IR/bishengir/test/Dialect/HIVM/infer-hivm-mem-scope.mlir`)。
该夹具的 `// CHECK:` / `// CHECK-SAME:` 断言就是 `bishengir-opt -hivm-infer-mem-scope`
的**期望输出**(CI 每次运行都逐字核对),权威性等价于真机 trace——explainer 表格里每个
address_space 标注(gm/cbuf/cc/ub)都能在这些 CHECK 行里逐字找到。

源码常量(AddressSpace 枚举值、TCoreType、PIPE 绑定等)标 `file:Lxxx`,可回溯到
`HIVMAttrs.td` / `HIVMMacroOps.td` / `HIVMDMAOps.td` / `InferHIVMMemScope.cpp`。
