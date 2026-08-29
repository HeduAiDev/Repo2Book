# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py
# worker 侧装配（m1 站 2）：ensure_kv_transfer_initialized 以 role=WORKER
# 再建一份挂全局 _KV_CONNECTOR_AGENT——与调度器侧那份毫无联系；get/
# has_kv_transfer_group/is_v1_kv_transfer_group 三个取用谓词（worker 面的
# 零开销旁路判据）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   _sync_engine_id_across_tp 的 distributed 面（L51-L69——TP/PP 组的
#     broadcast_object：HOST SEAM 单进程恒等。多机 engine_id 对齐 → ch36）。
from typing import TYPE_CHECKING

from .base import KVConnectorBase_V1, KVConnectorRole
from .factory import KVConnectorFactory

if TYPE_CHECKING:
    from .config import VllmConfig
    from .kv_cache_interface import KVCacheConfig

# SOURCE: vllm/distributed/kv_transfer/kv_connector/base.py（类型别名面）
KVConnectorBaseType = KVConnectorBase_V1

# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L16 全局 agent
_KV_CONNECTOR_AGENT: KVConnectorBaseType | None = None


# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L19
#   get_kv_transfer_group——worker 取用全局 agent
def get_kv_transfer_group() -> KVConnectorBaseType:
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L20-L23
    assert _KV_CONNECTOR_AGENT is not None, (
        "disaggregated KV cache transfer parallel group is not initialized"
    )
    return _KV_CONNECTOR_AGENT


# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L26
#   has_kv_transfer_group——零开销旁路判据（无 connector 时 mixin/装饰器
#   直通）
def has_kv_transfer_group() -> bool:
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L27
    return _KV_CONNECTOR_AGENT is not None


# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L30
#   is_v1_kv_transfer_group——v1 connector 判定
def is_v1_kv_transfer_group(connector: KVConnectorBaseType | None = None) -> bool:
    """Check if the KV connector is the v1 connector.
    If the argument is None, it will check the global KV connector

    Args:
        connector: The KV connector to check. If None, it will check the
            global KV connector.
    """
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L42-L48
    if connector is None:
        connector = _KV_CONNECTOR_AGENT

    if connector is None:
        return False

    return isinstance(connector, KVConnectorBase_V1)


# _sync_engine_id_across_tp——engine_id 跨 TP/PP 广播对齐
# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L51
def _sync_engine_id_across_tp(vllm_config: "VllmConfig") -> None:
    """Broadcast engine_id from TP rank 0 so all workers in a
    multi-node TP group share the same value.

    When PP is enabled, also broadcast across PP ranks so all workers in the
    same model-parallel engine share the same value.
    """
    # SUBTRACTED: get_tp_group().broadcast_object / PP 广播（L58-L69——
    #   distributed 运行时面，HOST SEAM 单进程恒等：TP=1 时 rank0 的值
    #   即全组值；多机对齐 → ch36）。
    assert vllm_config.kv_transfer_config is not None
    return


# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L72
#   ensure_kv_transfer_initialized——worker 侧装配主入口
#   （gpu_worker.py:L662 在 KV cache 配置就绪后调用）
def ensure_kv_transfer_initialized(
    vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"
) -> None:
    """
    Initialize KV cache transfer parallel group.
    """

    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L79
    global _KV_CONNECTOR_AGENT

    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L81-L82
    #   （无 kv_transfer_config → 直接返回：无 connector 的零开销路径）
    if vllm_config.kv_transfer_config is None:
        return

    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L84-L94
    #   （is_kv_transfer_instance 门 + 幂等（agent 已建则不重建）+
    #   WORKER role 再建一份——与调度器侧实例零共享）
    if (
        vllm_config.kv_transfer_config.is_kv_transfer_instance
        and _KV_CONNECTOR_AGENT is None
    ):
        _sync_engine_id_across_tp(vllm_config)

        _KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
            config=vllm_config,
            role=KVConnectorRole.WORKER,
            kv_cache_config=kv_cache_config,
        )


# SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L97
#   ensure_kv_transfer_shutdown——进程收尾（shutdown 钩子的调用点）
def ensure_kv_transfer_shutdown() -> None:
    # SOURCE: vllm/distributed/kv_transfer/kv_transfer_state.py:L98-L101
    global _KV_CONNECTOR_AGENT
    if _KV_CONNECTOR_AGENT is not None:
        _KV_CONNECTOR_AGENT.shutdown()
        _KV_CONNECTOR_AGENT = None
