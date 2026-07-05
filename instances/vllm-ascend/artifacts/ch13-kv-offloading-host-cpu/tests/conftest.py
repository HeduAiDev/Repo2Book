"""把精简版 implementation/ 放上 sys.path，让纯控制流测试能 import。

host 无 NPU/CANN：测试只验 dossier 明示『可跑』的纯 Python 控制流——分层搬运节拍 /
block 视图重建 / DMA 拷贝调度 / 指针算术。torch.npu 与 torch.ops._C_ascend.swap_blocks_batch
由 implementation/runtime_stub.py 补丁接住（记录调用、不真搬字节）。
"""
import sys
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent / "implementation"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))
