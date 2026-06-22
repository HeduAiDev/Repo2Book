# ch14《抢占与请求生命周期回流》交付 APPROVED

- **Type**: delivery
- **Chapter**: ch14
- **Date**: 2026-06-22
- **Timestamp**: 2026-06-22T14:12:28Z
- **Agents involved**: analyst, implementer, tester, writer, reviewer, archivist
- **User present**: False
- **Tags**: scheduler, preemption, lifecycle, dual-queue, check_stop, spec-decode, async

## What happened

承接 ch13 schedule() 主线，补完两处刹车分支：(1) allocate_slots 失败的抢占循环——while True LIFO 抢 RUNNING 末尾(running.pop())、_preempt_request 六项副作用(free KV/status=PREEMPTED/num_computed_tokens=0/清 spec/num_preemptions++/waiting.prepend_request)、preempted_req==request 终止；丢弃重算而非 swap，重算成本靠前缀缓存兜底(f11→ch15)。(2) if not preempted_reqs 背压守卫本拍跳过 WAITING。(3) waiting/skipped_waiting 双队列防队头阻塞：_is_blocked_waiting_status 三阻塞态隔离、_select FCFS skipped 优先、跳过+step_skipped_waiting 重排；WAITING_FOR_REMOTE_KVS 提升路径留 f12→ch29。(4) update_from_output 回流闭环：回流落点 WAITING/PREEMPTED 二分、spec 回退 num_rejected 回扣 num_computed_tokens/num_output_placeholders、逐 token append+check_stop+截断、先抓 finish_reason 再 _handle_stopped、按 status_before_stop 分流批量摘除、真完成 _free_request。(5) check_stop token 级停止优先级(min_tokens→EOS→stop_token_ids→length→repetition)，澄清 stop string 子串匹配属 ch09 文本层非调度器。(6) RequestStatus IntEnum 顺序排布让 is_finished == status>PREEMPTED 一次整型比较。(7) AsyncScheduler 占位簿记(discard_latest_async_tokens 短路/placeholders 回扣/仅 RUNNING cache_blocks)。四 linter 全 PASS、host 28/28 测试通过、11 项行为逐条核对 pin f3fef123 为忠实子集。

## Why it matters

首次补完调度器抢占·回流分支，闭合 ch13 留下的 allocate_slots/update_from_output 两个黑盒；为 ch15(KV cache/前缀缓存) 与 ch29(PD 分离/远程 KV) 埋下机制伏笔。

## What to remember

ch14 已 APPROVED 归档。bible 新增 10 条 ch14 接口(抢占/双队列/check_stop/AsyncScheduler 覆写)。伏笔 f11→ch15、f12→ch29 已埋 status=open，本章无应回收伏笔、无悬挂。stop string 边界已明确划归 ch09(文本层) 与调度器 check_stop(token 层) 两层，后续章节勿在调度器侧杜撰字符串匹配。启下 ch15：allocate_slots 真身 + 前缀缓存命中(回收 f11)。
