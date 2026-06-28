# ch02《插件如何被 vLLM 发现并顶替》交付 (APPROVED)

- **Type**: delivery
- **Chapter**: 02
- **Date**: 2026-06-28
- **Timestamp**: 2026-06-28T05:27:19Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch02, vllm-ascend, entry-points, NPUPlatform, qualname, oot-plugin

## What happened

ch02 完成全流水线并经 reviewer APPROVED。讲透 setup.py 两个 entry-point 组 → resolve_current_platform_cls_qualname/current_platform 懒加载 → OOT 平台优先于 builtin、register() 只返回类名字符串不 import；NPUPlatform 身份替换类属性 + 返回 qualname 的工厂钩子家族 (get_attn_backend_cls/get_device_communicator_cls/worker_cls …)；AscendDeviceType/310P 横切线索。4 张图 (roadmap+selection+hooks+lifecycle) 落地，确定性 linter 通过。

## Why it matters

ch02 是全书『插件如何顶替 builtin』的入口章，确立 qualname 字符串态 vs 已 import 态分水岭这一主线心智模型，后续章节复用。

## What to remember

review APPROVED，9 条 issue 全为 non-blocking/negotiable（省略号标记、签名类型标注、conftest 软化、qualname 首现注解、四遍计数口径、§2.6 两拍状态表、key 元组澄清、lifecycle.png 密度）。bible 已登记 9 个精简版接口；伏笔 f1(→ch03)/f2(→ch05) 已埋、ch02 无需回收。
