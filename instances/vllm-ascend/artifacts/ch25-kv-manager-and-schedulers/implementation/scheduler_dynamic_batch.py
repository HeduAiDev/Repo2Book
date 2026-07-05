# vllm_ascend/core/scheduler_dynamic_batch.py —— subtract-only 精简版
#
# 三个 Scheduler 子类之一。SchedulerDynamicBatch 对 vLLM Scheduler 只有两处实质改动：
#   (1) BudgetRefiner 按当前 running 的 decode 画像查表，动态细化 token_budget；
#   (2) decode-first 重排 self.running。
# 其余 schedule() 是 vLLM Scheduler.schedule 的逐字 inline 复刻——昇腾无法热补父方法
# 内部的循环片段，只能整段 override（见 # SUBTRACTED 标注）。
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. (Apache-2.0)
import os
import time

import pandas as pd
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


# SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L35
class BudgetRefiner:
    """This budget refiner can make dynamic adjustment to the token budget
    in the chunked prefill scheduling strategy."""

    # SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L39
    def __init__(self, default_budget, slo_limit=-1) -> None:
        self.enabled = slo_limit > 0
        if not self.enabled:
            return
        logger.info(
            "Dynamic batch is enabled with SLO limit: %s, and chunked prefill is "
            "forced to be activated because dynamic batch relies on it",
            slo_limit,
        )
        self.lookup: dict[tuple[int, int], int] = {}
        self.context_keys: set[int] = set()
        self.dnum_keys: set[int] = set()
        self.default_budget = default_budget
        self._read_lookup_table(slo_limit)

    # SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L54
    def _read_lookup_table(self, slo_limit):
        """Load the lookup table for dynamic budget."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        table_file_path = os.path.join(base_dir, "profile_table.csv")
        if not os.path.exists(table_file_path):
            # proceed without dynamic batch
            logger.error(
                "The dynamic batching feature requires the lookup table "
                "'profile_table.csv', but it was not found at '%s'.",
                table_file_path,
            )
            self.enabled = False
            return
        # SUBTRACTED: pandas 读 profile_table.csv 并按 (ctx_len, d_num) 分组、取
        #   cost<=slo_limit 内 chunk_size 最大行填充 self.lookup/context_keys/dnum_keys
        #   的 IO 细节（原 L68-L81）。纯数据加载，与查表控制流解耦；测试直接注入 lookup
        #   即可验 _get_max_budget/refine_budget。原 vllm_ascend/core/scheduler_dynamic_batch.py:L68-L81

    # SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L83
    def _align_key(self, value, valid_keys):
        """Align the minimum value within the valid_keys that is greater than the value."""
        for k in valid_keys:
            if k >= value:
                return k
        return None

    # SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L90
    def _get_max_budget(self, num_decode_tokens, num_decode):
        """Get the maximum budget according to the number of decoding tokens and the decoding requests."""
        aligned_ctx = self._align_key(num_decode_tokens, self.context_keys)
        aligned_dnum = self._align_key(num_decode, self.dnum_keys)
        if aligned_ctx is None or aligned_dnum is None:
            return self.default_budget
        budget = self.lookup.get((aligned_ctx, aligned_dnum), None)
        if budget is None:
            logger.warning("Table miss for ctx,dnum%s", (aligned_ctx, aligned_dnum))
            budget = self.default_budget
        return budget

    # SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L107
    def refine_budget(self, running_request, budget):
        """Dynamically refine the token budget according to the running request."""
        if not self.enabled:
            return budget
        # assume all running request will be scheduled.
        num_decode_token_lst = [
            req.num_tokens_with_spec for req in running_request if req.num_computed_tokens >= req.num_prompt_tokens
        ]
        num_decode = len(num_decode_token_lst)
        if num_decode <= 0:
            return budget
        num_decode_tokens = sum(num_decode_token_lst) / num_decode
        return self._get_max_budget(num_decode_tokens, num_decode)


# SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L122
class SchedulerDynamicBatch(Scheduler):
    """This Scheduler extends vllm's original v1 scheduler
    with dynamic batch."""

    # SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L126
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )
        self.running: list[Request] = []
        self.budget_refiner = BudgetRefiner(
            default_budget=self.scheduler_config.max_num_batched_tokens,
            slo_limit=self.scheduler_config.SLO_limits_for_dynamic_batch,
        )

    # SOURCE: vllm_ascend/core/scheduler_dynamic_batch.py:L151
    def schedule(self) -> SchedulerOutput:
        # NOTE: This scheduling algorithm is developed based on "super.schedule()"
        # with the dynamic batch implementation and two modifications:
        # 1. token_budget is dynamically refined via BudgetRefiner;
        # 2. decode-first chunked prefills via reordering self.running.

        scheduled_running_reqs: list[Request] = []
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}

        # >>> ASCEND CHANGE (1): dynamic token budget >>>
        token_budget = self.max_num_scheduled_tokens
        token_budget = self.budget_refiner.refine_budget(self.running, token_budget)
        # <<< ASCEND CHANGE (1) <<<

        # >>> ASCEND CHANGE (2): decode-first reorder of self.running >>>
        # NOTE: We move the prefill requests to the end of the self.running
        # list and keep the relative order unchanged. This rearrangement makes this
        # scheduling algorithm a strict decode-first chunked prefills.
        d_lst = [req for req in self.running if req.num_computed_tokens >= req.num_prompt_tokens]
        p_lst = [req for req in self.running if req.num_computed_tokens < req.num_prompt_tokens]
        self.running = d_lst + p_lst
        # <<< ASCEND CHANGE (2) <<<

        # For logging.
        scheduled_timestamp = time.monotonic()  # noqa: F841

        # SUBTRACTED: schedule() 其余 ~400 行（RUNNING 循环 L191-291、WAITING 循环
        #   L307-498、约束断言与 SchedulerOutput 组装 L504-573）逐字复刻
        #   vllm/v1/core/sched/scheduler.py Scheduler.schedule —— 非昇腾特化。昇腾因
        #   无法热补 super 内部循环片段，只能整段重写；上面两处 ASCEND CHANGE 是仅有的
        #   实质改动。原 vllm_ascend/core/scheduler_dynamic_batch.py:L181-L574
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=self._make_cached_request_data(
                scheduled_running_reqs, [], num_scheduled_tokens, {}, req_to_new_blocks
            ),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * len(self.kv_cache_config.kv_cache_groups),
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
        )
        self._update_after_schedule(scheduler_output)
        return scheduler_output
