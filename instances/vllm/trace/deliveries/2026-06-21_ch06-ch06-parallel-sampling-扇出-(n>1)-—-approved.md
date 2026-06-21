# ch06 Parallel Sampling 扇出 (n>1) — APPROVED

- **Type**: delivery
- **Chapter**: 06
- **Date**: 2026-06-21
- **Timestamp**: 2026-06-21T10:39:07Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: ch06, parallel-sampling, n>1, ParentRequest, foreshadow-payoff, APPROVED

## What happened

ParentRequest 把 n>1 请求扇成 n 个独立 n=1 child（唯一 id f"{index}_{request_id}"+确定性种子 seed+index），引擎零特判按普通请求调度；归并 get_outputs 流式逐条转发去重 / FINAL_ONLY 按 index 攒齐才吐，request_id 改回 external。四 linter 全过、14/14 测试过、内嵌源码逐项核对 pin f3fef123 一致。兑现 ch04 延迟的 n>1。

## Why it matters

首个把 ch04 明确延迟的分支正式回收的章节；验证伏笔注册-回收闭环（补登并 resolved f8: ch04->ch06）。bible 新增 3 条 ParentRequest 派生/归并接口。

## What to remember

ParentRequest 把 n>1 请求扇成 n 个独立 n=1 child（唯一 id f"{index}_{request_id}"+确定性种子 seed+index），引擎零特判按普通请求调度；归并 get_outputs 流式逐条转发去重 / FINAL_ONLY 按 index 攒齐才吐，request_id 改回 external。四 linter 全过、14/14 测试过、内嵌源...
