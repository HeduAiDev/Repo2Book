# ch05《check_and_update_config：插件改写 VllmConfig 的总闸 + AscendConfig 配置面》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 05
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T17:49:06Z
- **Agents involved**: analyst,  implementer,  tester,  writer,  reviewer,  archivist
- **User present**: False
- **Tags**: check_and_update_config,  AscendConfig,  _fix_incompatible_config,  三级取值,  additional_config,  platform-hook,  foreshadow-f2-payoff

## What happened

本章讲透 NPUPlatform.check_and_update_config 全流程（校验→_fix_incompatible_config 9 段 cascade reset→init_ascend_config→cudagraph/编译/splitting_ops 改写→worker_cls 选择→设 env）与 AscendConfig 单例配置面（开放 dict additional_config→强类型子配置；_get_config_value 三级取值 additional_config→env→default；envs.py lambda 懒求值）。两点设计哲学：①平台=配置改写器（vLLM 钩子约定，构图前最后改写）；②无 schema 配置后门 additional_config 的取舍。Reviewer 判 APPROVED；22 条 issue 全部 non-blocking（reader-comprehension 维度 + 4 处保真/导航小瑕：L27-50 内嵌块静默删 finegrained_tp 两行未加省略标记、超长多句括号旁注、5.5 缺节内路标、三级取值表只追两级等）。回收伏笔 f2（ch02 埋的 check_and_update_config 完整流程，本章在 worker_cls 段显式回扣）。

## Why it matters

兑现 ch02 的 f2 伏笔（完整配置改写流程），把昇腾「不改源码、只改 config 接管」这条主线落到平台钩子契约；为后续章节（worker 选择 ch13+、sleep mode ch07、内存分配策略）建立 AscendConfig/三级取值/env 的共享词汇与接口注册。

## What to remember

ch05 APPROVED 交付，f2 已回收。新接口已入 bible interfaces.json（check_and_update_config 全流程、_fix_incompatible_config 9 段 reset、AscendConfig.__init__、_get_config_value 三级取值、init/get/clear_ascend_config、AscendCompilationConfig **kwargs 后门、envs.py lambda+__getattr__）。22 条非阻断 issue 留待 writer 定点小修（不退章）。
