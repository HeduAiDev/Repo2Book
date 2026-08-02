# ch13《schedule(): Token 为中心的连续批处理》交付 APPROVED

- **Type**: delivery
- **Chapter**: 13
- **Date**: 2026-06-22
- **Timestamp**: 2026-06-22T13:31:37Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: scheduler, continuous-batching, async-scheduling, foreshadow-payoff, part-iii

## What happened

ch13 把 EngineCore 循环里的黑盒 schedule()/update_from_output() 拆到底：v1 调度器不分 prefill/decode 相，统一为 num_computed_tokens 追赶 num_tokens_with_spec 一条数轴；批大小=token 数(token_budget)非请求数；两阶段先 RUNNING(追赶公式三重 min + allocate_slots 失败 FCFS 抢占队尾)后 WAITING(if not preempted_reqs 守卫 + chunked prefill 截断)；SchedulerOutput 二分(首次 NewRequestData 全量 / 后续 CachedRequestData 增量 diff)；乐观推进 num_computed_tokens；AsyncScheduler 用 num_output_placeholders 占位让 schedule(N) 与 forward(N-1) 重叠消气泡。四 linter 全 PASS，host 13/13 测试通过，内嵌源码逐字核对 pin f3fef123 一致。bible 新增 4 条 ch13 接口(NewRequestData.from_request / CachedRequestData / SchedulerOutput / create_request_queue)，先前已注册 3 条共 7 条。

## Why it matters

回收 f6(ch03 埋: async_scheduling 推导的实例如何驱动连续批处理)+f10(ch11 埋: schedule() 每拍推哪些请求/多少 token)，二者 arc-map status=resolved/resolved_in=ch13；承上 ch11 黑盒+ch12 过渡，启下 ch14 KV cache 管理器(allocate_slots 真身)。无悬挂应回收伏笔，本章无新埋伏笔。

## What to remember

ch13 把 EngineCore 循环里的黑盒 schedule()/update_from_output() 拆到底：v1 调度器不分 prefill/decode 相，统一为 num_computed_tokens 追赶 num_tokens_with_spec 一条数轴；批大小=token 数(token_budget)非请求数；两阶段先 RUNNING(追赶公式三重 min + allo...
