"""Driver for m10 (RequestStatus 单 IntEnum 状态机：全序 1..12，PREEMPTED=6 是
完成分界，is_finished = status > PREEMPTED 一次整数比较；终态→FinishReason
映射含 WAITING_FOR_STREAMING_REQ→STOP 特例) — host run, pin vLLM v0.27.1
(vllm/v1/request.py:L348-L390 + vllm/v1/engine/__init__.py:L43-L62).

Dumps the full enum with integer values, the one-comparison finished test,
and the _FINISHED_REASON_MAP. This is the chapter's map: a request's whole
life is the value of one integer.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.engine import FinishReason  # noqa: E402
from implementation.request import RequestStatus  # noqa: E402


def main():
    order = [
        RequestStatus.WAITING,
        RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
        RequestStatus.WAITING_FOR_REMOTE_KVS,
        RequestStatus.WAITING_FOR_STREAMING_REQ,
        RequestStatus.RUNNING,
        RequestStatus.PREEMPTED,
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
        RequestStatus.FINISHED_ABORTED,
        RequestStatus.FINISHED_IGNORED,
        RequestStatus.FINISHED_ERROR,
        RequestStatus.FINISHED_REPETITION,
    ]
    rows = []
    for s in order:
        rows.append({
            "status": s.name,
            "value": int(s),
            "is_finished": RequestStatus.is_finished(s),
            "finished_reason": (RequestStatus.get_finished_reason(s).name
                                if RequestStatus.get_finished_reason(s) is not None else None),
        })
    out = {
        "driver": "run_m10_status_enum.py",
        "mechanism": "m10 RequestStatus 单 IntEnum 状态机（request.py:L348-L390）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（全状态 + _FINISHED_REASON_MAP 原样保留）",
        "boundary": {
            "preempted_value": int(RequestStatus.PREEMPTED),
            "rule": "is_finished = status > RequestStatus.PREEMPTED（request.py:L369-L371 一次整数比较）",
            "unfinished_max": int(RequestStatus.PREEMPTED),
            "finished_min": int(RequestStatus.FINISHED_STOPPED),
        },
        "finish_reason_enum": {fr.name: int(fr) for fr in FinishReason},
        "statuses": rows,
        "order_invariants": {
            "waiting_lt_blocked": int(RequestStatus.WAITING) < int(RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR),
            "blocked_lt_running": int(RequestStatus.WAITING_FOR_STREAMING_REQ) < int(RequestStatus.RUNNING),
            "running_lt_preempted": int(RequestStatus.RUNNING) < int(RequestStatus.PREEMPTED),
            "preempted_lt_all_finished": all(
                int(RequestStatus.PREEMPTED) < int(s) for s in order
                if s.name.startswith("FINISHED")),
            "note": "枚举数值顺序=隐式 API：新状态必须插在 PREEMPTED 的正确一侧，插错不报错、只静默改变 is_finished 语义（全仓无断言保护）",
        },
    }
    assert out["boundary"]["preempted_value"] == 6
    assert out["boundary"]["finished_min"] == 7
    assert sum(1 for r in rows if r["is_finished"]) == 6
    special = next(r for r in rows if r["status"] == "WAITING_FOR_STREAMING_REQ")
    assert special["is_finished"] is False and special["finished_reason"] == "STOP"

    dest = Path(__file__).with_name("m10_status_enum.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for r in rows:
        print(f"{r['value']:>2} {r['status']:<42} finished={r['is_finished']!s:<5} reason={r['finished_reason']}")


if __name__ == "__main__":
    main()
