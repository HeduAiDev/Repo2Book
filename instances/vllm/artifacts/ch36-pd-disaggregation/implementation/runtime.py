# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Worker-side runtime scaffolding the KV-connector lifecycle hangs off of.

精简版把 vLLM 散落在多个模块里的运行时支撑（forward context、全局 connector
注册、worker→scheduler 回传载体）收进一个文件，保留它们与 worker 生命周期相关
的形状与控制流，删去与本章无关的字段/实现细节。这样精简版可在 host 直接跑（不
import vllm、不触 CUDA），仍忠实复现 _get_kv_connector_output 夹住 forward 的结构。
"""

from contextlib import contextmanager

# ----------------------------------------------------------------------------
# 全局 KV transfer group（worker 进程里唯一的 WORKER-role connector 单例）
# SOURCE: vllm/distributed/kv_transfer/__init__.py（get_kv_transfer_group /
#         has_kv_transfer_group / is_v1_kv_transfer_group 的 re-export）
# ----------------------------------------------------------------------------
_KV_TRANSFER_GROUP = None


def set_kv_transfer_group(connector):
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:ensure_kv_transfer_initialized
    # SUBTRACTED: 真实初始化经 KVConnectorFactory + KVTransferConfig 构造 WORKER-role
    #             connector（vllm/distributed/kv_transfer/kv_transfer_state.py），精简版
    #             直接注入已构造好的 connector 实例，剥离工厂/配置解析样板。
    global _KV_TRANSFER_GROUP
    _KV_TRANSFER_GROUP = connector


def get_kv_transfer_group():
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:get_kv_transfer_group
    assert _KV_TRANSFER_GROUP is not None
    return _KV_TRANSFER_GROUP


def has_kv_transfer_group() -> bool:
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:has_kv_transfer_group
    return _KV_TRANSFER_GROUP is not None


def is_v1_kv_transfer_group() -> bool:
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:is_v1_kv_transfer_group
    # SUBTRACTED: 真实实现判 connector 是否为 KVConnectorBase_V1 子类；本章三类后端
    #             都是 v1，精简版恒为 True。
    return _KV_TRANSFER_GROUP is not None


# ----------------------------------------------------------------------------
# Forward context（forward 期间的全局上下文，装 attn_metadata 与逐层 KV buffer）
# SOURCE: vllm/forward_context.py:ForwardContext / set_forward_context / get_forward_context
# ----------------------------------------------------------------------------
class ForwardContext:
    # SOURCE: vllm/forward_context.py:ForwardContext
    def __init__(self, attn_metadata, no_compile_layers=None):
        # SUBTRACTED: cudagraph/ubatch/slot_mapping/dp 等参数（vllm/forward_context.py）
        #             与 KV connector 生命周期正交，精简版只留 attn_metadata 与
        #             no_compile_layers（P2P start_load_kv 逐层取 KV buffer 要用它）。
        self.attn_metadata = attn_metadata
        self.no_compile_layers = no_compile_layers or {}


_FORWARD_CONTEXT = None


@contextmanager
def set_forward_context(attn_metadata, vllm_config=None, no_compile_layers=None):
    # SOURCE: vllm/forward_context.py:set_forward_context
    global _FORWARD_CONTEXT
    prev = _FORWARD_CONTEXT
    _FORWARD_CONTEXT = ForwardContext(attn_metadata, no_compile_layers)
    try:
        yield _FORWARD_CONTEXT
    finally:
        _FORWARD_CONTEXT = prev


def get_forward_context() -> ForwardContext:
    # SOURCE: vllm/forward_context.py:get_forward_context
    assert _FORWARD_CONTEXT is not None
    return _FORWARD_CONTEXT


# ----------------------------------------------------------------------------
# worker→scheduler 回传载体
# SOURCE: vllm/v1/outputs.py:KVConnectorOutput
# ----------------------------------------------------------------------------
class KVConnectorOutput:
    # SOURCE: vllm/v1/outputs.py:KVConnectorOutput
    def __init__(self):
        self.finished_sending = None
        self.finished_recving = None
        self.invalid_block_ids = set()
        self.kv_connector_stats = None
        self.kv_cache_events = None
        self.kv_connector_worker_meta = None

    def is_empty(self) -> bool:
        # SOURCE: vllm/v1/outputs.py:KVConnectorOutput.is_empty
        return (
            not self.finished_sending
            and not self.finished_recving
            and not self.invalid_block_ids
            and not self.kv_connector_stats
            and not self.kv_cache_events
            and not self.kv_connector_worker_meta
        )


# SOURCE: vllm/v1/outputs.py:EMPTY_MODEL_RUNNER_OUTPUT
class ModelRunnerOutput:
    # SOURCE: vllm/v1/outputs.py:ModelRunnerOutput
    def __init__(self):
        # SUBTRACTED: sampled_token_ids/logprobs/... 等真实 forward 产物字段，
        #             本章只关心挂在其上的 kv_connector_output。
        self.kv_connector_output = None


EMPTY_MODEL_RUNNER_OUTPUT = ModelRunnerOutput()
