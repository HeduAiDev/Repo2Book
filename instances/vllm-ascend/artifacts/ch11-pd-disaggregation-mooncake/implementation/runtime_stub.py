"""测试接缝桩（NOT subtract-only）—— 把 PD 分离三层真实代码从 vllm / torch_npu /
mooncake 拉取的运行期符号，在 host（无 NPU/CANN/mooncake）上接住，让纯 Python 控制流
（连接器分发选举 / layerwise 角色分发 / 亲和路由决策 / proxy 最少负载分发 / 连续块合并）
可跑可断言。

真实 mooncake P2P 跨节点 KV 搬运在 host 不可发车，这里用「记录式/惰性」替身代替：
TransferEngine 不真注册内存、不真 batch_transfer；只让连接器契约的调度侧逻辑跑通。这正是
dossier 明示的边界——「连接器分发 / 亲和路由决策 / proxy 分发是纯 Python，可跑；实际
mooncake P2P 传输不真跑」。每个替身都标 # SOURCE 指向它顶替的真实符号。
"""
import logging as _logging

logger = _logging.getLogger("ch10")  # SOURCE: vllm/logger.py:logger


class KVConnectorRole:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorRole
    SCHEDULER = "scheduler"
    WORKER = "worker"


class KVConnectorMetadata:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorMetadata
    pass


class SupportsHMA:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:SupportsHMA
    pass


def supports_hma(c) -> bool:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:supports_hma
    return isinstance(c, SupportsHMA)


class KVConnectorBase_V1:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorBase_V1
    def __init__(self, vllm_config=None, role=None, kv_cache_config=None):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/v1/base.py:KVConnectorBase_V1.__init__
        self._vllm_config = vllm_config
        self._role = role


class KVConnectorFactory:
    # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:KVConnectorFactory
    _registry: dict = {}

    @classmethod
    def register_connector(cls, name, module, class_name):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:register_connector
        cls._registry[name] = (module, class_name)

    @classmethod
    def get_connector_class(cls, kv_transfer_config):
        # SOURCE: vllm/distributed/kv_transfer/kv_connector/factory.py:get_connector_class
        return cls._registry[kv_transfer_config]


class SchedulerOutput:
    # SOURCE: vllm/v1/core/sched/output.py:SchedulerOutput（仅类型标注用）
    pass


def get_ip() -> str:
    # SOURCE: vllm/utils/network_utils.py:get_ip（host 固定回环，不探测真实网卡）
    return "0.0.0.0"


def round_down(x: int, y: int) -> int:
    # SOURCE: vllm/utils/math_utils.py:round_down
    return x // y * y


def cdiv(a: int, b: int) -> int:
    # SOURCE: vllm/utils/math_utils.py:cdiv
    return -(-a // b)


def make_zmq_socket(ctx, path, socket_type, bind=False):
    # SOURCE: vllm/utils/network_utils.py:make_zmq_socket（host 不真连 RPC，返回惰性 socket）
    return ctx.socket(socket_type)


class BlockHash(bytes):
    # SOURCE: vllm/v1/core/kv_cache_utils.py:BlockHash（哈希就是 bytes，.hex() 即可序列化）
    pass


class MsgpackEncoder:
    # SOURCE: vllm/v1/serial_utils.py:MsgpackEncoder（host 用 msgpack 直编，省去零拷贝缓冲）
    def encode(self, obj):
        # SOURCE: vllm/v1/serial_utils.py:MsgpackEncoder.encode
        import msgpack
        return [msgpack.packb(obj, use_bin_type=True)]


# --- 以下是 vllm.v1 的请求 / 块对象替身，仅承载 PD 控制流读取的字段 -----------------
class Request:
    # SOURCE: vllm/v1/request.py:Request（仅保留连接器/亲和判定读取的字段）
    def __init__(self, request_id, prompt_token_ids, num_tokens=None,
                 kv_transfer_params=None, block_hashes=None, num_computed_tokens=0):
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.num_tokens = num_tokens if num_tokens is not None else len(prompt_token_ids)
        self.kv_transfer_params = kv_transfer_params
        self.block_hashes = block_hashes or []
        self.num_computed_tokens = num_computed_tokens
        self.all_token_ids = list(prompt_token_ids)


class KVCacheBlocks:
    # SOURCE: vllm/v1/core/kv_cache_manager.py:KVCacheBlocks（只保留 get_block_ids / new_empty）
    def __init__(self, block_ids):
        self._block_ids = list(block_ids)

    def get_block_ids(self):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:KVCacheBlocks.get_block_ids
        return list(self._block_ids)

    def new_empty(self):
        # SOURCE: vllm/v1/core/kv_cache_manager.py:KVCacheBlocks.new_empty
        return KVCacheBlocks([])


class _TransferEngineStub:
    # SOURCE: mooncake.engine.TransferEngine（record-only：不真注册内存/不真 P2P 搬运）
    def initialize(self, hostname, mode, backend, device_name):
        # SOURCE: mooncake.engine.TransferEngine.initialize
        return 0

    def register_memory(self, ptr, size):
        # SOURCE: mooncake.engine.TransferEngine.register_memory
        return 0

    def batch_transfer_sync_write(self, session_id, src, dst, length):
        # SOURCE: mooncake.engine.TransferEngine.batch_transfer_sync_write
        return 0
