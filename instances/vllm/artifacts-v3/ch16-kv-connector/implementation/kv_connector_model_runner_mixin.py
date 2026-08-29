# SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py
# worker 一拍生命周期的宿主（m7）：_get_kv_connector_output——bind→start_
# load_kv（后台传输，涉及的请求可以与本拍 running 完全不相交，省一次
# collective_rpc）→yield（前向跑在中间，逐层钩子见 kv_transfer_utils）→
# finally：wait_for_save + get_finished + get_block_ids_with_load_errors +
# worker_meta + clear；no_forward（无 token 步也走收发，wait_for_save=False）；
# maybe_get（有组才激活、否则 nullcontext 零开销）；finalize（spec decode
# 推迟收尾）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 12 条 use_uniform_kv_cache/allocate_uniform_kv_caches（L114-L255
#     ——布局族第 5 条）；
#   第 3 条观测面：get_kv_connector_stats/get_kv_connector_kv_cache_events
#     两行（output 的 stats/events 账位随第 3 条删）。
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import TYPE_CHECKING

from .base import KVConnectorBase
from .forward_context import get_forward_context, set_forward_context
from .kv_transfer_state import get_kv_transfer_group, has_kv_transfer_group
from .outputs import KVConnectorOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from .config import VllmConfig
    from .output import SchedulerOutput

import logging

logger = logging.getLogger(__name__)  # LOGGER SEAM


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
# SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L34
#   KVConnectorModelRunnerMixin
class KVConnectorModelRunnerMixin:
    # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L36
    #   kv_connector_no_forward——无 token 步也走收发
    @staticmethod
    def kv_connector_no_forward(
        scheduler_output: "SchedulerOutput", vllm_config: "VllmConfig"
    ) -> ModelRunnerOutput:
        # KV send/recv even if no work to do.
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L39-L48
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ) as kv_connector_output,
        ):
            pass

        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L50
    #   maybe_get_kv_connector_output——零开销旁路入口
    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        defer_finalize: bool = False,
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L55-L61
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, defer_finalize=defer_finalize
            )
            if has_kv_transfer_group()
            else nullcontext()
        )

    # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L63
    #   finalize_kv_connector——spec decode 延迟收尾
    @staticmethod
    def finalize_kv_connector() -> None:
        """Finalize the KV connector: wait_for_save and clear metadata.

        Call after draft model forward when defer_finalize=True was used.
        """
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L69-L72
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.wait_for_save()
            kv_connector.clear_connector_metadata()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L74-L76
    #   _get_kv_connector_output——worker 一拍生命周期本体
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: "SchedulerOutput",
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L83
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L85-L89
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L91-L95
        kv_connector.start_load_kv(get_forward_context())
        try:
            # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L96-L97
            yield output
        finally:
            # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L98-L100
            if wait_for_save and not defer_finalize:
                kv_connector.wait_for_save()

            # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L102-L105
            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            # SUBTRACTED: stats/events 两行（L107-L108——第 3 条观测面）。
            # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L109
            output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

            # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L111-L112
            if not defer_finalize:
                kv_connector.clear_connector_metadata()

    # SUBTRACTED: use_uniform_kv_cache / allocate_uniform_kv_caches
    #   （L114-L255——第 12 条布局族）。
