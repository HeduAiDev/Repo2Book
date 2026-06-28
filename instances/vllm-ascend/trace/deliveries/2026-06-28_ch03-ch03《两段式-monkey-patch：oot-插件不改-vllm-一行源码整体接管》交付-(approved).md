# ch03《两段式 monkey-patch：OOT 插件不改 vLLM 一行源码整体接管》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 03
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T16:05:45Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch03, vllm-ascend, monkey-patch, adapt_patch, two-stage, rebinding, oot-plugin

## What happened

ch03 完成全流水线并经 reviewer APPROVED。讲透 adapt_patch 单一入口 → platform 段（构图前/进程级，pre_register_and_update / _ensure_global_patch + _GLOBAL_PATCH_APPLIED 守卫触发）vs worker 段（NPUWorker.__init__ 触发）两阶段时机；靠 import 副作用执行 patch；5 种重绑定技法（整类替换 patch_multiproc_executor / 工厂注册表替换 patch_mamba_manager / 方法替换 patch_scheduler / 库函数 wrapper + from-import 缓存陷阱双绑 patch_distributed / triton wrapper）；条件加载 is_310p/HAS_TRITON/vllm_version_is。distributed wrapper、multiproc_executor 子类替换、scheduler 方法替换 3 个干净样本解剖。精简版 9 测试 host 全过（纯 Python 重绑定）。

## Why it matters

ch03 是旗舰地基章，确立 OOT 插件“不改 vLLM 源码整体接管”的两段式 patch 总纲与 5 技法分类心智模型，后续昇腾章节凡涉及 patch 处复用该框架；正式回收 ch02 埋下的 f1（两段式 adapt_patch）。

## What to remember

review APPROVED，9 条 issue 全 non-blocking/negotiable（worker import 顺序中性措辞、platform 折叠计数约13→约10、3 处半角标点、spec 缩写注解、_GLOBAL_PATCH_APPLIED 守卫存在理由收口、lint_source_grounding vllm_files 误报、SoC 首现注解、torch broadcast 双路径泛化、reader-comp）。bible 已登记 8 个 ch03 精简版接口；伏笔 f1 已回收(resolved_in ch03)，f2(→ch05) 仍 open。
