# SOURCE: vllm/v1/core/sched/interface.py
# ch34 切面：PauseState 枚举逐字（L23-L32；DPEngineCoreProc.add_request 的
# pause_state 判据）。SchedulerInterface 抽象（L34+）归 ch10 域。

from __future__ import annotations

import enum


# SOURCE: vllm/v1/core/sched/interface.py:L24-L35 PauseState —— 逐字
class PauseState(enum.IntEnum):
    """Scheduler pause state.

    - UNPAUSED: Normal operation
    - PAUSE_NEW: No new requests are scheduled, requests already in
                 running state are scheduled.
    - PAUSE_ALL: No requests are scheduled
    """

    UNPAUSED = 0
    PAUSED_NEW = 1
    PAUSED_ALL = 2
