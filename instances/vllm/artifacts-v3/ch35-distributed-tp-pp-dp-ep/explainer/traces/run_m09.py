# m09 取证：AsyncIntermediateTensors 懒同步。
#   a) 钩子语义（真类 + 假句柄）：构造不等待；碰 .tensors 才 wait 全部句柄 +
#      postprocess；幂等。
#   b) 真两进程结构证据（world=2 PP=2）：sender 延迟 0.5s 才 isend；receiver 的
#      irecv_tensor_dict 在 metadata 门等到 0.5s 返回，句柄外带未 wait；本地准备
#      0.3s 后碰 .tensors 数值正确。
#   c) 传输在飞的等待模型（建模值，标注）：LateHandle(ready_at=0.5) 模拟
#      『数据 0.5s 才到』——串行参照（先 wait 后准备）done=0.8 vs 懒同步（先准备
#      后 wait）done=0.5，省 0.3s。语义（__getattribute__ 钩子、wait 幂等）是真
#      代码路径；时长是 host 建模（gloo 本机传输瞬完，造不出真实在飞延迟）。
from __future__ import annotations

import time

from _driver_common import clean_tmp, dump


def _unit_hook_semantics() -> dict:
    from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

    order = []

    class _Handle:
        def __init__(self, n):
            self.n = n

        def wait(self):
            order.append(f"wait{self.n}")

    def _post_a():
        order.append("post_a")

    a = AsyncIntermediateTensors(
        {"hidden": [[1.0, 2.0]]}, comm_handles=[_Handle(1), _Handle(2)],
        comm_postprocess=[_post_a],
    )
    after_construct = list(order)
    _ = a._comm_waited  # 非 .tensors 属性不触发
    after_attr = list(order)
    _ = a.tensors  # 首次触碰 → wait + postprocess
    after_touch = list(order)
    _ = a.tensors  # 幂等
    after_retouch = list(order)
    return {
        "after_construct": after_construct,
        "after_other_attr": after_attr,
        "after_first_tensors_touch": after_touch,
        "after_second_tensors_touch": after_retouch,
        "n_waits_after_construct": len(
            [x for x in after_construct if x.startswith("wait")]
        ),
        "n_waits_after_other_attr": len([x for x in after_attr if x.startswith("wait")]),
        "n_new_waits_on_retouch": len(
            [x for x in after_retouch if x.startswith("wait")]
        )
        - len([x for x in after_touch if x.startswith("wait")]),
        "wait_called_once": len([x for x in after_retouch if x.startswith("wait")]) == 2,
    }


class _LateHandle:
    """建模件：数据 ready_at 才到位——wait() 阻塞到那一刻（真实句柄语义的
    时长替身；钩子路径是真代码）。"""

    def __init__(self, ready_at: float):
        self.ready_at = ready_at
        self.waited = False

    def wait(self):
        now = time.perf_counter()
        if now < self.ready_at:
            time.sleep(self.ready_at - now)
        self.waited = True


def _modeled_transfer_wait() -> dict:
    from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors

    data_ready_s, prep_s = 0.5, 0.3
    # 串行参照：irecv 即 wait（无懒同步）
    t0 = time.perf_counter()
    h = _LateHandle(t0 + data_ready_s)
    h.wait()
    t_waited = time.perf_counter() - t0
    time.sleep(prep_s)  # 本地准备被迫排在等待之后
    t_serial_done = time.perf_counter() - t0
    # 懒同步：句柄存着 → 先做本地准备 → 碰 .tensors 才 wait
    t0 = time.perf_counter()
    a = AsyncIntermediateTensors(
        {"hidden": [[1.0, 2.0]]}, comm_handles=[_LateHandle(t0 + data_ready_s)], comm_postprocess=[],
    )
    time.sleep(prep_s)
    t_touch = time.perf_counter() - t0
    _ = a.tensors
    t_lazy_done = time.perf_counter() - t0
    return {
        "modeled": True,
        "model_note": "时长为 host 建模（gloo 本机传输瞬完，造不出在飞延迟）；"
        "钩子语义/幂等/等待推迟均为真代码路径（见 unit_hook_semantics 与 lazy_mp）",
        "data_ready_s": data_ready_s,
        "local_prep_s": prep_s,
        "serial_wait_then_prep_done_s": round(t_serial_done, 1),
        "lazy_prep_then_touch_at_s": round(t_touch, 1),
        "lazy_done_s": round(t_lazy_done, 1),
        "saved_s": round(t_serial_done - t_lazy_done, 1),
        "lazy_residual_wait_s": round(t_lazy_done - t_touch, 1),
    }


def main() -> None:
    from _trace_jobs import _fresh_store
    from _trace_pool import TracePool

    doc = {
        "mechanisms": ["m9"],
        "params": "PP=2 两进程；b) sender 延迟 0.5s、本地准备 0.3s；c) 建模同参数",
        "unit_hook_semantics": _unit_hook_semantics(),
        "modeled_transfer_wait": _modeled_transfer_wait(),
    }
    pool = TracePool(2)
    try:
        doc["lazy_mp"] = pool.map(
            "lazy_mp", {"store": _fresh_store(), "sender_delay_s": 0.5, "local_prep_s": 0.3}
        )
    finally:
        pool.close()
    dump("m09", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
