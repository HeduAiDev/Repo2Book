# SOURCE: vllm/v1/engine/utils.py
# ch34 切面（站 7）：launch_core_engines 的 DP coordinator 出生段（L1067-L1110）+
# SignalCallback 已在 core.py 切面携带。SUBTRACTED：引擎发射主体（握手/CoreEngine
# 表/ray actor/多节点，L1112-L1300）与 APIServerProcessManager/信号族其余——ch05/
# ch04 域。

from __future__ import annotations

import contextlib

from vllm.logger import init_logger
from vllm.v1.engine.coordinator import DPCoordinator

logger = init_logger(__name__)


# SOURCE: vllm/v1/engine/utils.py:L1054-L1203 launch_core_engines —— 签名/
#   docstring/DP coordinator 出生段（L1087-L1110）逐字；yield 四元组的
#   engines/tensor_queue 两臂以 None 占位（发射主体 L1112-L1200 按删除面裁除）
@contextlib.contextmanager
def launch_core_engines(
    vllm_config,
    executor_class,
    log_stats,
    addresses,
):
    """Launch engine and DP coordinator processes as needed."""
    # SOURCE: vllm/v1/engine/utils.py:L1054-L1203（锚点双置）

    parallel_config = vllm_config.parallel_config
    dp_size = parallel_config.data_parallel_size
    local_engine_count = parallel_config.data_parallel_size_local
    local_start_index = parallel_config.data_parallel_rank_local
    dp_rank = parallel_config.data_parallel_rank
    host = parallel_config.data_parallel_master_ip
    local_engines_only = parallel_config.local_engines_only

    offline_mode = local_start_index is not None

    # SUBTRACTED: multimodal tensor_queue（L1079-L1085）——多模态域。

    # Run the DP Coordinator process with rank 0 when in online DP mode.
    # The coordinator is needed for:
    # 1. Internal/hybrid LB: collecting and publishing queue stats for load balancing
    # 2. MoE models: wave coordination in addition to stats
    run_coordinator = (
        vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
    )

    if run_coordinator:
        coordinator = DPCoordinator(
            parallel_config,
            enable_wave_coordination=vllm_config.model_config.is_moe,
        )

        addresses.coordinator_input, addresses.coordinator_output = (
            coordinator.get_engine_socket_addresses()
        )
        addresses.frontend_stats_publish_address = (
            coordinator.get_stats_publish_address()
        )

        logger.info("Started DP Coordinator process (PID: %d)", coordinator.proc.pid)
    else:
        coordinator = None

    # SUBTRACTED: ray 后端分支（L1112-L1123）与引擎发射主体（握手表/
    #   CoreEngineProcManager，L1125-L1300）——ch05 域。

    yield None, coordinator, addresses, None
