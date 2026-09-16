# ch37《P/D 分离》explainer 取证脚手架：复用 tests/_pd_harness 的装配件，
# 跑 implementation/ 精简版取真实数值轨迹（trace_source="run"）。
# 运行：cd instances/vllm/artifacts-v3/ch37-pd-disaggregation && python explainer/traces/run_mXX.py
from __future__ import annotations

import json
import sys
from pathlib import Path

_TRACES = Path(__file__).resolve().parent
_CH = _TRACES.parents[1]  # ch37-pd-disaggregation/
_IMPL = _CH / "implementation"
_TESTS = _CH / "tests"
_PROXY = _IMPL / "tests" / "v1" / "kv_connector" / "nixl_integration"
for _p in (str(_TESTS), str(_IMPL), str(_PROXY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def dump(name: str, doc: dict) -> None:
    out = _TRACES / f"{name}.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {out}")


# ─────────────── 与测试同款的请求/块替身（tests/test_pd_disaggregation.py 复用件） ───────────────


def make_p_request(request_id: str, num_tokens: int = 16):
    """P 腿请求：do_remote_decode=True——P 由此知道自己只算 prefill。"""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(
            max_tokens=1,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_engine_id": None,
                    "remote_block_ids": None,
                    "remote_host": None,
                    "remote_port": None,
                }
            },
        ),
    )


def make_d_request(request_id: str, receipt: dict, num_tokens: int = 16, max_tokens: int = 8):
    """D 腿请求：proxy 把 P 的回执信封原样附加（do_remote_prefill=True）。"""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    params = dict(receipt)
    params["do_remote_prefill"] = True
    params["do_remote_decode"] = False
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(
            max_tokens=max_tokens, extra_args={"kv_transfer_params": params}
        ),
    )


class _Block:
    def __init__(self, block_id, block_hash=None):
        self.block_id = block_id
        self.block_hash = block_hash
        self.is_null = False


class KVCacheBlocksStub:
    """KVCacheBlocks 的测试替身：只提供 connector 用到的查询面。"""

    def __init__(self, groups):
        self.blocks = tuple([_Block(i) for i in group] for group in groups)

    def get_unhashed_block_ids_all_groups(self):
        from vllm.v1.core.kv_cache_manager import KVCacheBlocks

        return KVCacheBlocks.get_unhashed_block_ids_all_groups(self)


def receipt_from_p(p, request_id: str, blocks=(0, 1), num_computed_tokens: int = 8):
    """P 终局产出的回执信封（控制面搬运的那只）。"""
    from vllm.v1.request import RequestStatus

    req = make_p_request(request_id)
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.num_computed_tokens = num_computed_tokens
    return p.scheduler.request_finished(req, (list(blocks),))[1]
