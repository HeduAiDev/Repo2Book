# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Define KV connector functionality mixin for model runners.

SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py
        + vllm/model_executor/layers/attention/kv_transfer_utils.py
本章主线：_get_kv_connector_output 用一个 context manager 把 WORKER-role connector
的整条生命周期夹在 model forward 两侧，使 KV load 与 compute 重叠。
"""

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import wraps

from .base import KVConnectorBase_V1
from .runtime import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
    get_forward_context,
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
    set_forward_context,
)


# Defined as a kv connector functionality mixin for ModelRunner (GPU, TPU)
class KVConnectorModelRunnerMixin:
    # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L36
    @staticmethod
    def kv_connector_no_forward(scheduler_output, vllm_config) -> ModelRunnerOutput:
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L37-L55
        # KV send/recv even if no work to do.
        with (
            set_forward_context(None, vllm_config),
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, wait_for_save=False
            ) as kv_connector_output,
        ):
            pass

        if kv_connector_output.is_empty():
            return EMPTY_MODEL_RUNNER_OUTPUT

        output = copy.copy(EMPTY_MODEL_RUNNER_OUTPUT)
        output.kv_connector_output = kv_connector_output
        return output

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output,
        defer_finalize: bool = False,
    ) -> AbstractContextManager:
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L57-L68
        return (
            KVConnectorModelRunnerMixin._get_kv_connector_output(
                scheduler_output, defer_finalize=defer_finalize
            )
            if has_kv_transfer_group()
            else nullcontext()
        )

    @staticmethod
    def finalize_kv_connector() -> None:
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L70-L79
        """Finalize the KV connector: wait_for_save and clear metadata.

        Call after draft model forward when defer_finalize=True was used.
        """
        if has_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.wait_for_save()
            kv_connector.clear_connector_metadata()

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output,
        wait_for_save: bool = True,
        defer_finalize: bool = False,
    ) -> Generator[KVConnectorOutput, None, None]:
        # SOURCE: vllm/v1/worker/kv_connector_model_runner_mixin.py:L81-L119
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase_V1)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        kv_connector.start_load_kv(get_forward_context())
        try:
            yield output
        finally:
            if wait_for_save and not defer_finalize:
                kv_connector.wait_for_save()

            output.finished_sending, output.finished_recving = (
                kv_connector.get_finished(scheduler_output.finished_req_ids)
            )
            output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()

            # SUBTRACTED: get_kv_connector_stats / get_kv_connector_kv_cache_events 的回填
            #             （mixin L114-L115）属可观测性扩展点（dossier delete 项），删去
            #             不影响 load/save/finished 闭环。build_connector_worker_meta 保留
            #             —— Offloading store 完成靠它上报 completed_jobs。
            output.kv_connector_worker_meta = kv_connector.build_connector_worker_meta()

            if not defer_finalize:
                kv_connector.clear_connector_metadata()

    # SUBTRACTED: use_uniform_kv_cache / allocate_uniform_kv_caches（mixin L121-L283）—— 跨层
    #             统一 KV layout（cross-layer blocks）是『整块多层一次性传』的物理布局优化，
    #             与 worker 生命周期主线正交（dossier delete 批准项）。


def maybe_transfer_kv_layer(func):
    # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L15-L61
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # SUBTRACTED: inspect.signature 取 layer_name 参数索引的样板（kv_transfer_utils.py
    #             L25-L35，dossier delete 项）—— 与『进层前 wait_for_layer_load、出层后
    #             save_kv_layer』语义无关。精简版用固定签名 wrapper(layer_name, ...) 演示。

    @wraps(func)
    def wrapper(layer_name, *args, **kwargs):
        # SOURCE: vllm/model_executor/layers/attention/kv_transfer_utils.py:L37-L59
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(layer_name, *args, **kwargs)

        # Extract attention context (metadata, layer, kv_cache, layer_slot_mapping)
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        kv_cache = ctx.no_compile_layers.get(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(layer_name, *args, **kwargs)

        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        result = func(layer_name, *args, **kwargs)

        # Save KV cache layer on exit
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

        return result

    return wrapper
