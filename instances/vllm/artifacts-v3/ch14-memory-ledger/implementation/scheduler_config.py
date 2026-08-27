# SOURCE: vllm/config/scheduler.py
# SchedulerConfig——两道门的配置面（m10/m12）：scheduler_reserve_full_isl
# （full-ISL 准入门开关，默认 True）、watermark（水位比例，默认 0.0 关、
# [0,1) 区间）、disable_hybrid_kv_cache_manager（组化回退开关）。
# SUBTRACTED: @config pydantic 装配与 Field 校验面（ge/lt——切面构造期
#   直供值，校验归 ch03 的 config 装配链）；调度策略/chunked-mm/
#   scheduler_cls/async_scheduling/stream_interval 等其余配置位
#   （L26-L119、L143-L157——ch02/10/11/12 各章切面）；
#   prefill_schedule_interval（L143-L146——DP 部署节流）。


# SOURCE: vllm/config/scheduler.py:L26 SchedulerConfig
class SchedulerConfig:
    """Scheduler configuration (本章消费面：两道门 + 组化回退 + token 预算)。"""

    # SOURCE: vllm/config/scheduler.py:L159-L168 default_factory 的 None→默认
    #   解析语义（切面构造期直供；Field(default=8192, ge=1) 的校验面删）
    def __init__(
        self,
        max_num_batched_tokens: int = 8192,
        disable_hybrid_kv_cache_manager: bool | None = None,
        scheduler_reserve_full_isl: bool = True,
        watermark: float = 0.0,
    ):
        # SOURCE: vllm/config/scheduler.py:L49 max_num_batched_tokens
        self.max_num_batched_tokens = max_num_batched_tokens
        # SOURCE: vllm/config/scheduler.py:L122-L128 disable_hybrid_kv_cache_manager
        self.disable_hybrid_kv_cache_manager = disable_hybrid_kv_cache_manager
        # """If set to True, KV cache manager will allocate the same size of
        # KV cache for all attention layers even if there are multiple type of
        # attention layers..."""
        # SOURCE: vllm/config/scheduler.py:L130-L134 scheduler_reserve_full_isl
        self.scheduler_reserve_full_isl = scheduler_reserve_full_isl
        # """If True, the scheduler checks whether the full input sequence
        # length fits in the KV cache before admitting a new request, rather
        # than only checking the first chunk. Prevents over-admission and KV
        # cache thrashing with chunked prefill."""
        # SOURCE: vllm/config/scheduler.py:L136-L141 watermark
        self.watermark = watermark
        # """Fraction of total KV cache blocks to keep free (the watermark)
        # when admitting waiting or preempted requests into the running queue.
        # ... Must be in the range [0.0, 1.0); 0.0 (the default) disables the
        # watermark."""
