# SOURCE: vllm/v1/core/sched/scheduler.py
# Scheduler 的**前缀缓存配合面**（第 3/8/9/10-12 站的调度侧半边）：
# admission_lookup——waiting 循环准入查命中（num_computed_tokens==0 时
# get_computed_blocks，顺手写回 shared_prefix_boundary；被抢占者重排回来
# 也从这再进——F2）；_mamba_block_aligned_split——强制停点（partial-tail
# 哈希边界 + Marconi junction 块对齐下取整）；pack_kv_cache_block_copies
# ——CoW 拷贝过线打包（retained 块挂步序栅栏）；_free_cow_retained_blocks
# ——拷贝两端引用的延迟释放；_preempt_request——F2 起点（free 全部块【哈希
# 保留】+ num_computed_tokens=0 + 回 waiting 队头）。
# 完整 schedule()（token 预算/FCFS 队列/chunked prefill）归 ch10、抢占全景
# 归 ch11、async 步序栅栏的完整面归 ch12——本章按站点抽块（ENGINE SEAM：
# 内联块抽出为方法以便单测，控制流逐字）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 3 条 eagle：last_cache_position 回退一块（L393-L394）；
#   第 5 条 connector：get_computed_blocks_for_connector 分支（L749-L759）与
#     外部命中/仲裁段（L769-L825）、WAITING_FOR_REMOTE_KVS 段（L1022-L1053）、
#     producer partial-tail 手递手（L1165-L1179）；
#   第 10 条 log_stats：record_prefix_cache_stats 调用（L1016-L1020）与
#     PREEMPTED 事件（L1310-L1311）；
#   第 4 条 spec/encoder/stale 协议（L1291-L1308——ch11/ch12/ch33）；
#   ch10 的 token 预算/长 prefill 阈值对齐段（L402-L409）、ch12 的
#     _free_request_blocks defer 支（L2345-L2349——本章即时还块；CoW 保留块
#     的栅栏 _free_cow_retained_blocks 保留）。
from collections import deque

from .cache import CacheConfig
from .kv_cache_interface import KVCacheConfig
from .kv_cache_manager import KVCacheManager, KVCacheBlocks
from .kv_cache_utils import KVCacheBlock
from .output import SchedulerOutput
from .request import Request, RequestStatus


# SOURCE: vllm/v1/core/sched/request_queue.py:L80 FCFSRequestQueue（最小切面：
#   deque + prepend_request——ch10/11 已建全量优先级队列面）
class FCFSRequestQueue(deque):
    """A first-come-first-served queue that supports deque operations."""

    # SOURCE: vllm/v1/core/sched/request_queue.py:L92 prepend_request
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        # SOURCE: vllm/v1/core/sched/request_queue.py:L94（被抢占者回队头——
        #   下一步最先被重新准入，F2 的『重排回来』落点）
        self.appendleft(request)


# SOURCE: vllm/v1/core/sched/scheduler.py:L267 Scheduler（前缀缓存配合面切面）
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L276 __init__ 的 KVCacheManager
    #   装配段（切面）
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        scheduler_block_size: int,
        hash_block_size: int,
        cache_config: CacheConfig | None = None,
        enable_caching: bool = True,
    ) -> None:
        # SUBTRACTED: VllmConfig 装配面（L268-L275——ch03）、request 队列全量/
        #   connector/encoder/spec/LoRA 观测（L276-L430——各邻章）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L276-L290 建 KVCacheManager
        #   （enable_caching=cache_config.enable_prefix_caching）
        if cache_config is None:
            cache_config = CacheConfig(
                block_size=scheduler_block_size, enable_prefix_caching=True
            )
        self.cache_config = cache_config
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            enable_caching=enable_caching,
        )
        # SOURCE: vllm/v1/core/sched/scheduler.py:L283-291 装配账位（block_size/
        #   hash_block_size——分点与停点都用它们）
        self.block_size = cache_config.block_size
        self.hash_block_size = hash_block_size
        # SOURCE: vllm/v1/core/sched/scheduler.py:L309-L310 has_mamba_layers/
        #   needs_kv_cache_zeroing（mamba 停点判定读前者）
        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        # SOURCE: vllm/v1/core/sched/scheduler.py:L314-L323 mamba 停点两旗
        #   （need_mamba_block_aligned_split：mamba+align 才强制停点；
        #   mamba_partial_cache_hit：更细 prefix_match_unit 时加 partial-tail
        #   哈希边界停点）
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.mamba_partial_cache_hit = (
            self.need_mamba_block_aligned_split
            and self.hash_block_size < self.block_size
        )
        # Counts of non-empty steps scheduled / processed. update_from_output
        # is called once per scheduled step in FIFO order, so these stay in sync.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L325-L331 步序栅栏三账（CoW
        #   retained 块的延迟释放用）
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        # FIFO of (fence_seq, blocks): blocks become safe to free once
        # processed_step_seq >= fence_seq.
        self.deferred_frees: deque[tuple[int, list[KVCacheBlock]]] = deque()
        # SOURCE: vllm/v1/core/sched/scheduler.py:L131 defer_block_free 装配位
        #   （async scheduling 开时置 True——ch12；本章默认 False，测试可拨）
        self.defer_block_free = False
        # SOURCE: vllm/v1/core/sched/scheduler.py:L~310 running/waiting 队列
        #   账位（FCFS 最小镜像）+ connector 恒 None（第 5 条 → ch16）
        self.running: list[Request] = []
        self.waiting: FCFSRequestQueue = FCFSRequestQueue()
        self.connector = None
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1313-L1315 回写集合账位
        self.reset_preempted_req_ids: set[str] = set()

    # SOURCE: vllm/v1/core/sched/scheduler.py:L744-L766 第 3 站：waiting 循环
    #   准入查命中（ENGINE SEAM 从 schedule() WAITING 循环抽出，控制流逐字；
    #   connector 分支删——无 connector 走本地 get_computed_blocks，顺手写回
    #   shared_prefix_boundary）
    def admission_lookup(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int, int]:
        # Get already-cached tokens.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L744-L746（num_computed_
        #   tokens==0 才查——被抢占者归零后重排回来也从这再进，F2）
        assert request.num_computed_tokens == 0
        # SUBTRACTED: connector 分支（L749-L759——get_computed_blocks_for_
        #   connector，第 5 条 → ch16）
        # SOURCE: vllm/v1/core/sched/scheduler.py:L760-L766（本地命中 +
        #   Marconi junction 写回 Request——调度器写、cache 读的跨模块协议）
        (
            new_computed_blocks,
            num_new_local_computed_tokens,
            # Marconi shared-prefix junction to pin; 0 if none.
            request.shared_prefix_boundary,
        ) = self.kv_cache_manager.get_computed_blocks(request)
        return new_computed_blocks, num_new_local_computed_tokens, (
            request.shared_prefix_boundary
        )

    # SOURCE: vllm/v1/core/sched/scheduler.py:L362 _mamba_block_aligned_split
    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        """Clip a prefill chunk so it ends where Mamba state must be cached.

        In "align" cache mode reusable SSM states are materialized at block
        boundaries, plus mandatory early stops (the prompt's partial-tail hash
        boundary, a detected shared-prefix junction). If a block is larger
        than the configured prefill chunk limit, intermediate chunks keep
        private running state until they reach the next cacheable position.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L377-L386
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Split only during prefill: `request.num_tokens - 1` extends this to
        # resumed requests replaying their output tokens.
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end:
            return num_new_tokens

        # SOURCE: vllm/v1/core/sched/scheduler.py:L388-L394 last_cache_position
        #   （eagle 回退一块 L393-L394 随第 3 条删）
        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
        # Eagle, FullAttn prunes the last matching block, so back off one
        # block to avoid a Mamba cache miss.
        last_cache_position = request.num_tokens - request.num_tokens % block_size

        end = start + num_new_tokens
        # Invariant: slot p holds the state after exactly (p + 1) * block_size
        # tokens. State is written at chunk ends, so chunk ends must be block
        # aligned. Exempt: the prompt's last chunk, whose slot decode advances
        # to the boundary. A block too wide for one chunk advances sub-block
        # and re-aligns at the next boundary.
        # SUBTRACTED: chunk 预算对齐段（L402-L409——max_num_scheduled_tokens/
        #   long_prefill_token_threshold 的对齐裁剪 → ch10；本章停点语义只看
        #   强制停位）
        # SOURCE: vllm/v1/core/sched/scheduler.py:L411-L437 四停点 + 取最早
        next_block_boundary = (start // block_size + 1) * block_size
        tail_boundary = (
            request.num_prompt_tokens // self.hash_block_size * self.hash_block_size
            if self.mamba_partial_cache_hit
            else 0
        )
        stops = (
            # Same invariant: a chunk starting mid-block stops at the boundary
            # rather than running past it.
            next_block_boundary if start % block_size != 0 else 0,
            # Never run past the last cacheable block boundary mid-chunk.
            last_cache_position,
            # Fine-grained hits: the prompt's partial-tail entry can only be
            # registered by a chunk ending exactly at its last hash boundary.
            tail_boundary
            if last_cache_position < tail_boundary < request.num_prompt_tokens
            else 0,
            # Marconi shared-prefix junction, block-floored (a sub-block
            # junction's state is not separately cacheable): cache its state
            # so sibling requests sharing the prefix can reuse it.
            start + (request.shared_prefix_boundary - start) // block_size * block_size
            if start < request.shared_prefix_boundary < end
            else 0,
        )
        # Stop at the earliest mandatory position strictly inside the chunk.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L435-L437
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1181-L1190 第 9 站：CoW 拷贝
    #   过线打包（ENGINE SEAM 从 schedule() 尾段抽出；producer 手递手段
    #   L1165-L1179 随第 5 条删）
    def pack_kv_cache_block_copies(self) -> SchedulerOutput | None:
        kv_cache_block_copies, cow_retained_blocks = (
            self.kv_cache_manager.take_kv_cache_block_copies()
        )
        if kv_cache_block_copies:
            # The copies run with this step's execution; the first non-empty
            # step at or after it gets seq `sched_step_seq + 1` (0-token steps
            # do not advance the seq), and its completion implies the copies
            # have run.
            # SOURCE: vllm/v1/core/sched/scheduler.py:L1184-L1189（retained 块
            #   的释放挂步序栅栏——拷贝与本拍执行同飞、完成即放）
            self._free_cow_retained_blocks(cow_retained_blocks, self.sched_step_seq + 1)
        pending_kv_cache_block_copies = kv_cache_block_copies or None
        # SUBTRACTED: SchedulerOutput 其余装配面（L1208-L1248——各邻章）。
        # SOURCE: vllm/v1/core/sched/output.py:L259 kv_cache_block_copies 字段
        return SchedulerOutput(kv_cache_block_copies=pending_kv_cache_block_copies)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L1274 _preempt_request（F2 起点：
    #   抢占机制本体归 ch11，本章保留『free 不清哈希』这一条事实的调度侧入口）
    def _preempt_request(
        self, request: Request, timestamp: float = 0.0, drop_stale_output: bool = False
    ) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.

        drop_stale_output: drop (rather than deliver) any in-flight output; used
        by reset_prefix_cache, whose same-step resume would otherwise deliver
        tokens out of order, and for connectors with a pending KV hand-off,
        which the preemption's block free would leave without valid KV.
        """
        # SUBTRACTED: drop_stale_output 参数的 stale 协议消费段（L1297-L1308
        #   ——async 账本 → ch12）；签名保留默认值。
        del timestamp, drop_stale_output
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1287-L1289
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1290 free 全部块——但
        #   free_blocks 只动 ref_cnt 和队列位置，block_hash 原样留在 map 里
        #   （前缀保留 = F2 的物质基础）
        self._free_request_blocks(request)
        # SUBTRACTED: encoder 缓存/inflight 账/spec 清空（L1291-L1296——第
        #   4/5 条）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1293-L1294
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1309
        request.num_preemptions += 1
        # SUBTRACTED: PREEMPTED 事件（L1310-L1311——第 10 条 log_stats）。

        # Put the request back to the waiting queue.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L1313-L1315（回 waiting 队头
        #   ——重排回来重走准入查询）
        self.waiting.prepend_request(request)
        self.reset_preempted_req_ids.add(request.request_id)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2341 _free_request_blocks
    def _free_request_blocks(self, request: Request):
        """Free the request's KV blocks, deferring the return to the block
        pool when an in-flight GPU step may still write them.
        """
        # SUBTRACTED: defer_block_free 步序栅栏分支（L2345-L2349 的条件与
        #   L2352-L2354 的 deferred_frees 扣留——ch12 的请求块栅栏；本章即时
        #   还块。CoW retained 块的栅栏在 _free_cow_retained_blocks，保留）。
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2350（即时支：manager.free
        #   → 逆序 free_blocks——ref_cnt 归零回池、哈希保留）
        self.kv_cache_manager.free(request)

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2356 _free_cow_retained_blocks
    def _free_cow_retained_blocks(
        self, blocks: list[KVCacheBlock], fence_seq: int
    ) -> None:
        """Release CoW copy retentions, deferring their return to the block
        pool while the step that runs the copy may still be in flight.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2362-L2365（fence 过了即还/
        #   没过先进 deferred_frees（倒序入队——出队时再逆序还原成驱逐序））
        if not self.defer_block_free or fence_seq <= self.processed_step_seq:
            self.kv_cache_manager.block_pool.free_blocks(blocks)
            return
        self.deferred_frees.append((fence_seq, blocks[::-1]))

    # SOURCE: vllm/v1/core/sched/scheduler.py:L2367 _drain_deferred_frees
    def _drain_deferred_frees(self):
        """Return deferred blocks whose fence step has completed.

        Fences are appended in near-monotonic order (a CoW retention fence
        can lead request-free fences by one step), so stop at the first
        pending one; any satisfied entry behind it is merely freed later.
        """
        # SOURCE: vllm/v1/core/sched/scheduler.py:L2374-L2380
        while self.deferred_frees:
            fence, _ = self.deferred_frees[0]
            if fence > self.processed_step_seq:
                break
            _, blocks = self.deferred_frees.popleft()
            # Free in reverse order so that the tail blocks are evicted first.
            self.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
