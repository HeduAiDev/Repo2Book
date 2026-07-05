# vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py
#   —— subtract-only companion（ch10 第 2 层入口对象：进程级单例 TransferEngine）
#
# GlobalTE：双重检查锁懒初始化一个 mooncake TransferEngine，后端 'ascend' + 'P2PHANDSHAKE'，
#   register_buffer 一次性把 KV 张量内存登记给 RDMA 式 P2P。整进程共享一个，避免重复登记/重复握手端口。
# host 无 mooncake：真实 TransferEngine 由 runtime_stub 的 record-only 替身接住（不真注册/不真搬）。
import threading

from runtime_stub import _TransferEngineStub


# SOURCE: vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py:L4
class GlobalTE:
    # SOURCE: vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py:L5
    def __init__(self):
        self.transfer_engine = None
        self.is_register_buffer: bool = False
        self.transfer_engine_lock = threading.Lock()
        self.register_buffer_lock = threading.Lock()

    # SOURCE: vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py:L11
    def get_transfer_engine(self, hostname: str, device_name: str | None):
        if self.transfer_engine is None:
            with self.transfer_engine_lock:
                # Double-Checked Locking
                if self.transfer_engine is None:
                    # SUBTRACTED: from mooncake.engine import TransferEngine + ImportError 指引
                    #   （L16-L23）—— host 无 mooncake，用 record-only 替身代替真实引擎。
                    #   原 mooncake_transfer_engine.py:L16-L24
                    TransferEngine = _TransferEngineStub
                    self.transfer_engine = TransferEngine()
                    device_name = device_name if device_name is not None else ""
                    ret_value = self.transfer_engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)
                    if ret_value != 0:
                        raise RuntimeError(f"TransferEngine initialization failed with ret_value: {ret_value}")
        return self.transfer_engine

    # SOURCE: vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py:L31
    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        with self.register_buffer_lock:
            assert self.transfer_engine is not None, "Transfer engine must be initialized"
            if self.is_register_buffer:
                return
            for ptr, size in zip(ptrs, sizes):
                ret_value = self.transfer_engine.register_memory(ptr, size)
                if ret_value != 0:
                    raise RuntimeError("Mooncake memory registration failed.")
            self.is_register_buffer = True


# SOURCE: vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py:L43
global_te = GlobalTE()
