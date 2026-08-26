"""Driver for m13 (check_stop 五连判，固定顺序即优先级：min_tokens → EOS →
stop_token_ids → 长度封顶（num_tokens≥max_model_len 或 output≥max_tokens）→
重复检测) — host run, pin vLLM v0.27.1 (vllm/v1/core/sched/utils.py:L94-L130).

Direct unit calls of check_stop on hand-built requests (prompt 4 tokens,
block_size hashing not needed here). Cases:
  J1 min_tokens gate: with min_tokens=3 and eos=5, outputs [5] and [5,5] are
     BLOCKED (return False) even though the EOS is there — the gate precedes
     the EOS judgment; outputs [5,5,5] passes the gate and the EOS fires.
  J2 EOS -> FINISHED_STOPPED, stop_reason None.
  J3 stop_token_ids -> FINISHED_STOPPED, stop_reason = the token id.
  J4a length: num_tokens (4 prompt + 4 outputs = 8) >= max_model_len 8.
  J4b length: num_output_tokens (2) >= max_tokens 2.
  J5 repetition: [1,2,1,2,1,2] with (max_pattern=2, min_pattern=1, count=3)
     -> FINISHED_REPETITION, stop_reason "repetition_detected".
  NEG no stop -> False.
Note: stop STRING matching is NOT here — it lives in the frontend
detokenizer (Part II); check_stop only sees integer token ids.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.request import (  # noqa: E402
    RepetitionDetectionParams,
    Request,
    RequestStatus,
    SamplingParams,
)
from implementation.utils import check_stop  # noqa: E402


def build(output_tokens, *, min_tokens=0, eos=None, stop_ids=None,
          max_tokens=64, repetition=None):
    req = Request(request_id="t", prompt_token_ids=[0, 1, 2, 3],
                  sampling_params=SamplingParams(
                      max_tokens=max_tokens, min_tokens=min_tokens,
                      eos_token_id=eos, stop_token_ids=stop_ids,
                      repetition_detection=repetition))
    for t in output_tokens:
        req.append_output_token_ids(t)
    return req


def run(case, req, max_model_len):
    stopped = check_stop(req, max_model_len)
    return {
        "case": case,
        "num_tokens": req.num_tokens,
        "num_output_tokens": req.num_output_tokens,
        "stopped": stopped,
        "status": req.status.name,
        "stop_reason": req.stop_reason,
    }


def main():
    out = {
        "driver": "run_m13_check_stop.py",
        "mechanism": "m13 check_stop 五连判（utils.py:L94-L130；顺序即优先级）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（utils.py 130 行无删除）",
        "order": ["1 min_tokens 门槛", "2 EOS", "3 stop_token_ids", "4 长度封顶",
                  "5 重复检测"],
        "cases": [],
    }
    cases = out["cases"]

    # J1: the gate precedes EOS
    cases.append(run("J1a min_tokens=3 拦 EOS（输出 [5]，1<3）",
                     build([5], min_tokens=3, eos=5), 8192))
    cases.append(run("J1b min_tokens=3 拦 EOS（输出 [5,5]，2<3）",
                     build([5, 5], min_tokens=3, eos=5), 8192))
    cases.append(run("J1c 3≥3 放行 → EOS 判定生效（输出 [5,5,5]）",
                     build([5, 5, 5], min_tokens=3, eos=5), 8192))
    # J2 EOS
    cases.append(run("J2 EOS=9 命中", build([1, 9], eos=9), 8192))
    # J3 stop token id
    cases.append(run("J3 stop_token_ids=[7] 命中", build([1, 7], stop_ids=[7]), 8192))
    # J4 length caps
    cases.append(run("J4a num_tokens=8 ≥ max_model_len=8", build([4, 5, 6, 7]), 8))
    cases.append(run("J4b output=2 ≥ max_tokens=2",
                     build([1, 2], max_tokens=2), 8192))
    # J5 repetition
    det = RepetitionDetectionParams(max_pattern_size=2, min_pattern_size=1,
                                    min_count=3)
    cases.append(run("J5 [1,2]×3 重复检测", build([1, 2, 1, 2, 1, 2],
                                                repetition=det), 8192))
    # NEG
    cases.append(run("NEG 无命中", build([5]), 8192))

    assert not cases[0]["stopped"] and not cases[1]["stopped"]
    assert cases[2]["stopped"] and cases[2]["status"] == "FINISHED_STOPPED"
    assert cases[3]["stopped"] and cases[3]["stop_reason"] is None
    assert cases[4]["stopped"] and cases[4]["stop_reason"] == 7
    assert cases[5]["status"] == "FINISHED_LENGTH_CAPPED"
    assert cases[6]["status"] == "FINISHED_LENGTH_CAPPED"
    assert cases[7]["status"] == "FINISHED_REPETITION"
    assert cases[7]["stop_reason"] == "repetition_detected"
    assert not cases[8]["stopped"]
    out["order_note"] = ("五连判固定顺序即优先级：J1 证明 min_tokens 在 EOS 之前"
                         "（门槛拦下时纵使最后一 token 是 EOS 也不停）；"
                         "stop string 不在 check_stop——字符串子串匹配在前端 detokenizer 文本空间")

    dest = Path(__file__).with_name("m13_check_stop.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for c in cases:
        print(f"{c['case']:<40} stopped={c['stopped']!s:<5} {c['status']:<22} reason={c['stop_reason']}")


if __name__ == "__main__":
    main()
