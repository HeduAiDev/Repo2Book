"""测试接缝桩（NOT subtract-only）—— 把 KV 池化链路从 vllm / torch_npu / mooncake / zmq
拉取的运行期符号，在 host（无 NPU/CANN/mooncake）上接住，让纯 Python 控制流
（命中算术 / LoadSpec 节拍 / 两端队列解耦 / lookup 去重 / 后端 put/get/exists 契约）
可跑可断言。

实际池存取 / RDMA 搬运在 host 不可发车——这正是 dossier 明示的边界：
「池调度节拍 / 两端解耦 / 后端契约是纯 Python，可跑；实际池存取与搬运不真跑」。
每个替身都标 # SOURCE 指向它顶替的真实符号。
"""
import logging as _logging

logger = _logging.getLogger("ch11")  # SOURCE: vllm/logger.py:logger


class KVConnectorRole:  # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorRole
    SCHEDULER = "scheduler"
    WORKER = "worker"


class KVConnectorMetadata:  # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorMetadata
    pass


class SupportsHMA:  # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:SupportsHMA
    pass


class KVConnectorBase_V1:  # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorBase_V1
    def __init__(self, vllm_config=None, role=None, kv_cache_config=None):  # SOURCE: base.py:KVConnectorBase_V1.__init__
        self._vllm_config = vllm_config
        self._role = role
        self._connector_metadata = None
        self.connector_scheduler = None
        self.connector_worker = None

    def _get_connector_metadata(self):  # SOURCE: base.py:KVConnectorBase_V1._get_connector_metadata
        return self._connector_metadata

    def bind_connector_metadata(self, meta):  # SOURCE: base.py:KVConnectorBase_V1.bind_connector_metadata
        self._connector_metadata = meta


class BlockHash(bytes):  # SOURCE: vllm/v1/core/kv_cache_utils.py:BlockHash（哈希即 bytes，.hex() 序列化）
    pass


# SOURCE: vllm/v1/core/kv_cache_utils.py:BlockHashList（就是逐块哈希的列表）
BlockHashList = list


def maybe_convert_block_hash(bh):  # SOURCE: vllm/v1/core/kv_cache_utils.py:maybe_convert_block_hash
    return bh


def cdiv(a: int, b: int) -> int:  # SOURCE: vllm/utils/math_utils.py:cdiv
    return -(-a // b)


def make_zmq_socket(ctx, path, socket_type, bind=False):  # SOURCE: vllm/utils/network_utils.py:make_zmq_socket
    # host 不真连 RPC，返回惰性 socket（不 bind/connect 真实端点）。
    return ctx.socket(socket_type)


class MsgpackEncoder:  # SOURCE: vllm/v1/serial_utils.py:MsgpackEncoder（host 用 msgpack 直编，省零拷贝缓冲）
    def encode(self, obj):  # SOURCE: vllm/v1/serial_utils.py:MsgpackEncoder.encode
        import msgpack
        return [msgpack.packb(obj, use_bin_type=True)]


class MsgpackDecoder:  # SOURCE: vllm/v1/serial_utils.py:MsgpackDecoder
    def __init__(self, *args, **kwargs):  # SOURCE: vllm/v1/serial_utils.py:MsgpackDecoder.__init__
        pass

    def decode(self, frames):  # SOURCE: vllm/v1/serial_utils.py:MsgpackDecoder.decode
        import msgpack
        return msgpack.unpackb(bytes(frames[0]), raw=False)


class ParallelConfig:  # SOURCE: vllm/config.py:ParallelConfig（仅作类型标注用）
    pass


class NpuEvent:  # SOURCE: torch.npu.Event（host 无 NPU：record/synchronize 均为 no-op）
    def record(self):  # SOURCE: torch.npu.Event.record
        pass

    def synchronize(self):  # SOURCE: torch.npu.Event.synchronize
        pass


def get_tensor_model_parallel_rank() -> int:  # SOURCE: vllm/distributed/__init__.py:get_tensor_model_parallel_rank
    # host 无分布式：单卡退化 tp_rank=0。
    return 0


def get_tensor_model_parallel_world_size() -> int:  # SOURCE: vllm/distributed/__init__.py:get_tensor_model_parallel_world_size
    # host 无分布式：单卡退化 tp_size=1。
    return 1
