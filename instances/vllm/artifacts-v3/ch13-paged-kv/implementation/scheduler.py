# SOURCE: vllm/v1/core/sched/scheduler.py
# Scheduler 的**KV 账本消费面**（第 2/5/6/11/12 站的调度侧半边）：入场要块
# （WAITING allocate_slots / RUNNING 循环+抢占环）、过线打包（新请求全量
# 块表 / 在跑请求增量 new_block_ids）、清零账（_get_new_block_ids_to_zero）、
# 终局还块（_free_blocks → manager.free 逆序归还）。
# 完整 schedule()（token 预算/FCFS 队列/chunked prefill）归 ch10、抢占全景
# 归 ch11、async 账本归 ch12——本章按站点抽块（ENGINE SEAM：内联块抽出为
# 方法以便单测，控制流逐字）。
# SUBTRACTED（dossier.delete 批准项的落点）：第 5 条 spec/eagle；第 7 条
#   connector（update_state_after_alloc/connector stats/_skip_zero_block_ids
#   的 async 覆写区跳过集合 → ch16）；第 8 条 full_sequence_must_fit 实参与
#   watermark 传递（→ ch14）；第 12 条 V2 model runner 分支（L1132-L1142）；
#   encoder/LoRA/PP/PRIORITY 抢支与 token 预算细账（L560-L574、L590-L613、
#   L640-L712、L899-L971——ch10/11）；deferred free 栅栏（L2345-L2354 的
#   defer 支——ch12，本章即时还块）；KVCacheConfig 之外的装配面。
from .kv_cache_interface import KVCacheConfig
from .kv_cache_manager import KVCacheManager, KVCacheBlocks
from .output import CachedRequestData, NewRequestData
from .request import Request, RequestStatus


# SOURCE: vllm/v1/core/sched/scheduler.py:L267 Scheduler（KV 账本消费面切面）
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L276 __init__ 的 KVCacheManager
    #   装配段（切面）
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        enable_caching: bool = False,
    ) -> None:
        # SUBTRACTED: VllmConfig 装配面（L268-L275——ch03）、request 队列/
        #   connector/encoder/spec/LORA 观测（L276-L430——各邻章）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L276-L290 建 KVCacheManager
        #   （enable_caching=cache_config.enable_prefix_caching、watermark——
        #   两账位 → ch14；本章 False 支）
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            enable_caching=enable_caching,
        )
        # SOURCE: vllm/v1/core/sched/scheduler.py:L~300 needs_kv_cache_zeroing
        #   装配位（清零通道的调度侧开关——Mamba/混合精度模型为 True）
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        # SOURCE: vllm/v1/core/sched/scheduler.py:L~310 running/waiting 队列
        #   账位（真实为 RequestQueue 优先级队列——ch10/11；切面持列表）
        self.running: list[Request] = []
        self.waiting: list[Request] = []
        self.requests: dict[str, Request] = {}
        # SOURCE: vllm/v1/core/sched/scheduler.py:L~362 finished_req_ids
        self.finished_req_ids: set[str] = set()
        # SUBTRACTED: defer_block_free/sched_step_seq/processed_step_seq/
        #   deferred_frees（async 步序栅栏——ch12：本章即时还块）、
        #   _skip_zero_block_ids（async KV load 覆写区跳过集合 → ch16）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L973-L985 第 2 站：WAITING 侧
    #   allocate_slots 入口（ENGINE SEAM 从 schedule() WAITING 循环抽出，
    #   控制流逐字；拿不到块 None——ch10 只见 break，这里是 None 的出生地）
    def allocate_slots_for_waiting(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
    ) -> KVCacheBlocks | None:
        # SUBTRACTED: num_encoder_tokens/reserved_blocks 实参与 full_sequence_
        #   must_fit=self.scheduler_reserve_full_isl、delay_cache_blocks=
        #   load_kv_async（L966-L983 的预算/connector 参数——第 7/8 条 → ch14/
        #   ch16；decoder-only 无 connector 时全为 0/False）；effective_
        #   lookahead（L948-L951——spec → ch33）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L973-L985
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
        )
        # SUBTRACTED: encoder 缓存回滚与 connector 状态更新（L987-L1014
        #   ——第 4/7 条）；prefix_cache_stats 记账（L1016-L1019——第 3 条）。
        return new_blocks

    # SOURCE: vllm/v1/core/sched/scheduler.py:L575-L629 第 11 站：RUNNING 循环
    #   allocate_slots——decode 每多 block_size 个 token 多要一块；池满 None →
    #   while True 抢占环（ENGINE SEAM 抽出；外部行为 ch11，本章看块侧内景）
    def allocate_slots_for_running(
        self,
        request: Request,
        num_new_tokens: int,
    ) -> KVCacheBlocks | None:
        preempted_reqs: list[Request] = []
        # Schedule newly needed KV blocks for the request.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L576-L577
        while True:
            # SUBTRACTED: num_lookahead_tokens 实参（L581——spec → ch33）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L578-L582
            new_blocks = self.kv_cache_manager.allocate_slots(
                request,
                num_new_tokens,
            )

            if new_blocks is not None:
                # The request can be scheduled.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L584-L586
                break

            # The request cannot be scheduled.
            # Preempt the lowest-priority request.
            # SUBTRACTED: PRIORITY 抢支与 token 预算/账目回滚（L590-L613
            #   ——ch10/11；FCFS 支 = pop 队尾最新者）。
            # SOURCE: vllm/v1/core/sched/scheduler.py:L614-L615
            preempted_req = self.running.pop()

            # SOURCE: vllm/v1/core/sched/scheduler.py:L617-L621 _preempt_request
            #   （drop_stale_output=requires_kv_delivery 删——第 7 条）
            self._preempt_request(preempted_req)
            preempted_reqs.append(preempted_req)
            if preempted_req == request:
                # No more request to preempt. Cannot schedule this request.
                # SOURCE: vllm/v1/core/sched/scheduler.py:L623-L625
                break

        if new_blocks is None:
            # Cannot schedule this request.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L627-L629
            return None
        return new_blocks

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1274 _preempt_request（块侧切面
    #   ——ch11 讲外部行为，本章保留块侧两件事：free 全部块 + computed 归零）
    def _preempt_request(self, request: Request) -> None:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1287-L1289
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1290 free 全部块（哈希保留
        #   在表 → ch15 的物质基础）
        self._free_request_blocks(request)
        # SUBTRACTED: encoder 缓存/inflight 账/spec 清空（L1291-L1296——第
        #   4/5 条）；async stale 标记（L1297-L1308——ch12）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1293-L1294
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1131-L1149 第 5 站：过线打包
    #   （新请求全量块表——ENGINE SEAM 抽出；use_v2_model_runner 分支删）
    def make_new_reqs_data(
        self,
        scheduled_new_reqs: list[Request],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> list[NewRequestData]:
        # SUBTRACTED: use_v2_model_runner 分支（L1132-L1142——第 12 条：V2
        #   runner 多带 all_token_ids/prefill_token_ids，与主分支同一语义）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1144-L1149 全量块表随首帧
        #   过线（block_id 是两进程间唯一共享键）
        new_reqs_data = [
            NewRequestData.from_request(
                req, req_to_new_blocks[req.request_id].get_block_ids()
            )
            for req in scheduled_new_reqs
        ]
        return new_reqs_data

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1400-L1467 _make_cached_request_
    #   data（切面：增量 new_block_ids 半边逐字，token 账面删）
    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1414-L1424 账面声明
        req_ids: list[str] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1426-L1427
        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(running_reqs + resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            # SUBTRACTED: PP token 回传（L1430-L1445——ch17）；use_v2 的
            #   all_token_ids 传播（L1448-L1450——第 12 条）。
            if idx >= num_running_reqs:
                # SOURCE: vllm/v1/core/sched/scheduler.py:L1446-L1447
                resumed_req_ids.add(req_id)
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1451-L1453 增量块表
            #   （allow_none=True：空则 None 不占带宽——worker 侧差量
            #   extend 的依据）
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1454-L1457
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(req.num_output_tokens)

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1459-L1467
        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1260 _get_new_block_ids_to_zero
    #   —— 第 6 站：清零账（排干 → 随 SchedulerOutput 过线）
    def _get_new_block_ids_to_zero(self) -> list[int] | None:
        # Drain new attention block ids every step so the manager-side list
        # does not grow unbounded; only kv-cache zeroing consumes them.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1261-L1265
        new_block_ids_to_zero = self.kv_cache_manager.take_new_block_ids()
        if not self.needs_kv_cache_zeroing:
            return None

        # SUBTRACTED: _skip_zero_block_ids 过滤（L1267-L1270——async KV load
        #   覆写区跳过集合 → ch16）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L1272
        return new_block_ids_to_zero or None

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2329 _free_blocks —— 第 12 站：
    #   终局还块入口
    def _free_blocks(self, request: Request):
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2330-L2332
        assert request.is_finished()
        self._free_request_blocks(request)
        del self.requests[request.request_id]

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2341 _free_request_blocks
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        # SUBTRACTED: defer_block_free 步序栅栏分支（L2345-L2349 的条件与
        #   L2352-L2354 的 deferred_frees 扣留——ch12；本章即时还块）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2350（即时支：manager.free
        #   → 逆序 free_blocks——ref_cnt 归零回池）
        self.kv_cache_manager.free(request)
