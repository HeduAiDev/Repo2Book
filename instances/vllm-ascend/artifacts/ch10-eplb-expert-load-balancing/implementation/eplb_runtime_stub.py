# 测试接缝桩（NOT subtract-only）—— 把真实 EPLB 代码从 vllm / torch.distributed /
# vllm_ascend 拉取的运行期符号在 host（无 NPU/CANN/分布式后端）上接住，让纯 Python
# 控制流（节拍状态机 / 队列解耦 / 策略多态 / 三态机）可跑可断言。
#
# 真实通信（all_gather / P2POp / batch_isend_irecv）在 host 不可发车，这里用「记录式」
# 替身代替：只记下发了哪些 send/recv、reqs.wait() 可调用，绝不真搬数据。这正是 dossier
# 明示的边界——「真实 P2P 权重搬运与子进程绑 NPU 不真跑，用桩/注释标注」。
import logging
from contextlib import nullcontext


class _Logger:
    # SOURCE: vllm/logger.py:logger（host 用 stdlib logging 接住；warning_once 退化为 warning）
    def __init__(self):  # SOURCE: vllm/logger.py:logger
        self._l = logging.getLogger("eplb")

    def info(self, *a, **k):  # SOURCE: vllm/logger.py:logger.info
        self._l.info(*a, **k)

    def debug(self, *a, **k):  # SOURCE: vllm/logger.py:logger.debug
        self._l.debug(*a, **k)

    def warning(self, *a, **k):  # SOURCE: vllm/logger.py:logger.warning
        self._l.warning(*a, **k)

    def warning_once(self, *a, **k):  # SOURCE: vllm/logger.py:logger.warning_once
        self._l.warning(*a, **k)

    def error(self, *a, **k):  # SOURCE: vllm/logger.py:logger.error
        self._l.error(*a, **k)


logger = _Logger()


def record_function_or_nullcontext(name):
    # SOURCE: vllm/v1/utils.py:record_function_or_nullcontext（profiler 打点；host 视作透明）
    return nullcontext()


class _Envs:
    # SOURCE: vllm/envs.py（仅 init_eplb 读 VLLM_ALLOW_EXPERT_LOAD_COLLECTING；旧版本无此字段→走 except）
    pass


envs = _Envs()


class _FakeReq:
    # SOURCE: torch.distributed Work（isend/irecv 句柄；record-only：wait() 仅置位，不真同步）
    def __init__(self):  # SOURCE: torch.distributed Work
        self.waited = False

    def wait(self):  # SOURCE: torch.distributed Work.wait
        self.waited = True


class _RecordingDist:
    # SOURCE: torch.distributed（record-only 替身：P2POp/batch_isend_irecv 只记录不真发；
    #         get_rank/get_world_size 默认单卡，测试可注入）
    isend = "isend"
    irecv = "irecv"

    def __init__(self):  # SOURCE: torch.distributed（record-only 替身状态）
        self._rank = 0
        self._world = 1

    def set_topology(self, rank, world_size):  # SOURCE: torch.distributed（测试注入拓扑）
        self._rank = rank
        self._world = world_size

    def get_rank(self):  # SOURCE: torch.distributed.get_rank
        return self._rank

    def get_world_size(self):  # SOURCE: torch.distributed.get_world_size
        return self._world

    class P2POp:
        # SOURCE: torch.distributed.P2POp（record-only：仅记 (op, tensor, peer)）
        def __init__(self, op, tensor, peer, group=None):  # SOURCE: torch.distributed.P2POp
            self.op = op
            self.tensor = tensor
            self.peer = peer
            self.group = group

    def batch_isend_irecv(self, op_list):  # SOURCE: torch.distributed.batch_isend_irecv（record-only）
        return [_FakeReq() for _ in op_list]


dist = _RecordingDist()


class _FakeEplbGroup:
    # SOURCE: vllm/distributed/parallel_state.py:GroupCoordinator（ch08 建的 _DYNAMIC_EPLB；
    #         all_gather 单卡退化为 unchanged，device_group 占位）
    device_group = None

    def all_gather(self, tensor, dim):  # SOURCE: vllm/distributed/parallel_state.py:GroupCoordinator.all_gather
        return tensor


_DYNAMIC_EPLB_GROUP = _FakeEplbGroup()


def get_dynamic_eplb_group():
    # SOURCE: vllm_ascend/distributed/parallel_state.py:get_dynamic_eplb_group（f6 回收：ch08 _DYNAMIC_EPLB）
    return _DYNAMIC_EPLB_GROUP


def generate_log2phy_map(expert_map, rank_id):
    # SOURCE: vllm_ascend/eplb/core/eplb_utils.py:generate_log2phy_map
    # （真实实现把每个 logical expert 映到本 rank 的 physical slot；host 桩用 identity 占位，
    #   pack_update_info 只需一个可 .numpy().tolist() 的张量即可验打包结构）
    return expert_map
