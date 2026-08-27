# SOURCE: vllm/v1/core/sched/scheduler.py
# Scheduler 的账本装配 + 入场过门调用点（站 8/9 的调度侧半边——站点抽块）：
#   __init__ 尾段——resolve 后拿 block_size/hash_block_size，KVCacheManager
#   带 watermark 装配（L289），scheduler_reserve_full_isl 绑定（L305-L307）；
#   allocate_slots_for_waiting——WAITING 循环里的入场调用点（L965-L985）：
#   三预算（整序列门/异步在途预约/水位）全在调用点显式传入。
# 站点抽块纪律：从 schedule() 的内联块抽出为方法——抽出而非改写，控制流
#   逐字；整章 schedule() 的 token 预算面归 ch10/11。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 13 条 lookahead/encoder/external 的判定与实参（L945-L963、L976-
#     L981——账位保留 0/缺省值）；load_kv_async 的 _inflight_prefill_
#     reserved_blocks（L965-L971——异步 KV load → ch16，reserved_blocks=0
#     直供、参数位保留）；delay_cache_blocks/num_encoder_tokens 实参；
#   第 9 条 kv_metrics_collector/log_stats/events 贯穿；
#   第 6 条 use_eagle 装配（L256-L270 spec 段 + L281）；第 8 条 dcp/pcp
#     乘子透传；connector 绑定（L291-L294——ch16）；mamba 对齐分裂
#     （L309-L323——邻章）。
from collections import deque

from .kv_cache_interface import KVCacheConfig
from .kv_cache_manager import KVCacheManager
from .request import Request


# SOURCE: vllm/v1/core/sched/scheduler.py:L~100 Scheduler（账本切面——真实
#   类为完整调度器，ch10/11 全文）
class Scheduler:
    # SOURCE: vllm/v1/core/sched/scheduler.py:L~110 __init__（切面装配）
    def __init__(
        self,
        vllm_config,
        kv_cache_config: KVCacheConfig,
        block_size: int | None = None,
        hash_block_size: int | None = None,
        log_stats: bool = False,
    ) -> None:
        # SUBTRACTED: 队列/preemptor/输出处理器/connector/encoder cache/
        #   spec-decode 装配（L110-L255——ch02/10/11/16/33）。
        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.max_model_len = vllm_config.model_config.max_model_len
        self.log_stats = log_stats
        # SOURCE: vllm/v1/core/sched/scheduler.py:L~230 调度队列（站点上下文
        #   的最小装配——WAITING/RUNNING 双队列）
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

        # SOURCE: vllm/v1/core/sched/scheduler.py:L256-L270（spec-decode 的
        #   lookahead 判定——第 6 条 eagle/第 13 条 lookahead 删；账位 0）
        self.num_lookahead_tokens = 0

        # Create the KV cache manager.
        # SOURCE: vllm/v1/core/sched/scheduler.py:L272-L274
        if hash_block_size is None:
            hash_block_size = block_size
        self.hash_block_size = hash_block_size
        # SOURCE: vllm/v1/core/sched/scheduler.py:L276-L290 KVCacheManager
        #   构造（enable_caching/watermark（L289）等参数落地；max_in_flight_
        #   tokens 来自 VllmConfig——准入上限的输入）
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            enable_caching=self.cache_config.enable_prefix_caching,
            scheduler_block_size=block_size,
            hash_block_size=hash_block_size,
            watermark=self.scheduler_config.watermark,
        )
        # SUBTRACTED: connector 绑定（L291-L294——ch16）。

        # SOURCE: vllm/v1/core/sched/scheduler.py:L305-L307（full-ISL 门开关
        #   的调度器绑定）
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        # SUBTRACTED: mamba 对齐分裂与细粒度命中停止位（L309-L323——
        #   邻章）。

    # SOURCE: vllm/v1/core/sched/scheduler.py:L965-L985 allocate_slots 调用
    #   点（站点抽块：schedule() WAITING 循环内联块的逐字抽出；三预算
    #   显式传入——full_sequence_must_fit / reserved_blocks / has_scheduled_reqs）
    def allocate_slots_for_waiting(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks=None,
    ):
        # SUBTRACTED: async KV load 的 lookahead 限幅与 encoder 判定
        #   （L945-L963——第 13 条；lookahead/encoder 恒 0）。
        # SUBTRACTED: load_kv_async 的在途预约计算（L965-L971——→ ch16：
        #   异步 KV load 的预 reserved 防挤死在途 prefill；本章非 async
        #   主线 reserved_blocks=0，参数位保留）。
        reserved_blocks = 0

        # SOURCE: vllm/v1/core/sched/scheduler.py:L973-L985（门参数三件：
        #   full_sequence_must_fit=self.scheduler_reserve_full_isl、
        #   reserved_blocks、has_scheduled_reqs=bool(self.running)）
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            full_sequence_must_fit=self.scheduler_reserve_full_isl,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=bool(self.running),
        )
        # SOURCE: vllm/v1/core/sched/scheduler.py:L987-L994（None 即不可调度
        #   ——break 回 WAITING；encoder 释放随第 13 条删）
        return new_blocks
