"""ch06 测试：验证精简版复现真实 vLLM parallel sampling 扇出/归并的可观察行为。

纯单元测试，不 import vllm。对照真实行为见 dossier.code_spine / design_decisions。
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from implementation.async_llm import AsyncLLM  # noqa: E402
from implementation.input_processor import InputProcessor  # noqa: E402
from implementation.output_processor import OutputProcessor  # noqa: E402
from implementation.parallel_sampling import ParentRequest  # noqa: E402
from implementation.types import (  # noqa: E402
    CompletionOutput,
    EngineCoreRequest,
    RequestOutputKind,
    SamplingParams,
)


def make_request(external_id: str, n: int, seed=None,
                 output_kind=RequestOutputKind.DELTA) -> EngineCoreRequest:
    params = SamplingParams(n=n, seed=seed, output_kind=output_kind)
    return EngineCoreRequest(request_id=external_id, sampling_params=params)


# --- assign_request_id -------------------------------------------------------

def test_assign_request_id_internalizes_and_randomizes():
    req = make_request("R", n=1)
    InputProcessor.assign_request_id(req)
    # 外部 id 存入 external_req_id
    assert req.external_req_id == "R"
    # 内部 id = external-<8hex>
    assert req.request_id.startswith("R-")
    suffix = req.request_id.split("-", 1)[1]
    assert len(suffix) == 8
    int(suffix, 16)  # 必须是 hex


def test_assign_request_id_rejects_preset_external():
    req = EngineCoreRequest(request_id="R", external_req_id="already",
                            sampling_params=SamplingParams())
    with pytest.raises(ValueError):
        InputProcessor.assign_request_id(req)


# --- get_child_info / 唯一 id 与种子递进 --------------------------------------

def test_child_ids_are_index_prefixed_and_unique():
    req = make_request("R", n=3)
    InputProcessor.assign_request_id(req)
    parent = ParentRequest(req)
    ids = [parent.get_child_info(i)[0] for i in range(3)]
    # f"{index}_{request_id}"
    assert ids == [f"{i}_{req.request_id}" for i in range(3)]
    assert len(set(ids)) == 3  # 全局唯一
    # 全部登记进 child_requests
    assert parent.child_requests == set(ids)


def test_child_params_forced_n1_with_seed_progression():
    req = make_request("R", n=3, seed=42)
    InputProcessor.assign_request_id(req)
    parent = ParentRequest(req)
    seeds = []
    for i in range(3):
        _, child_params = parent.get_child_info(i)
        assert child_params.n == 1  # child 强制 n=1
        seeds.append(child_params.seed)
    # 确定性递进 seed+index，n 路互异
    assert seeds == [42, 43, 44]
    assert len(set(seeds)) == 3


def test_no_seed_reuses_cached_params_object():
    req = make_request("R", n=3, seed=None)
    InputProcessor.assign_request_id(req)
    parent = ParentRequest(req)
    p0 = parent.get_child_info(0)[1]
    p1 = parent.get_child_info(1)[1]
    p2 = parent.get_child_info(2)[1]
    # 无 seed：所有 child 复用同一缓存 params 对象
    assert p0 is p1 is p2
    assert p0.n == 1
    assert p0.seed is None


def test_seed_does_not_cache_distinct_objects():
    req = make_request("R", n=2, seed=7)
    InputProcessor.assign_request_id(req)
    parent = ParentRequest(req)
    p0 = parent.get_child_info(0)[1]
    p1 = parent.get_child_info(1)[1]
    # 有 seed：每个 child 是独立 clone（不缓存复用）
    assert p0 is not p1
    assert parent.cached_child_sampling_params is None


# --- get_outputs 归并：streaming vs FINAL_ONLY -------------------------------

def test_get_outputs_streaming_forwards_each_increment():
    req = make_request("R", n=2, output_kind=RequestOutputKind.DELTA)
    InputProcessor.assign_request_id(req)
    parent = ParentRequest(req)
    cid0, _ = parent.get_child_info(0)
    cid1, _ = parent.get_child_info(1)

    # 未完成的增量直接转发
    out, finished = parent.get_outputs(cid0, CompletionOutput(index=0, text="a"))
    assert [o.text for o in out] == ["a"]
    assert finished is False

    # child0 完成：转发其终态并从集合移除
    out, finished = parent.get_outputs(cid0, CompletionOutput(index=0, finish_reason="stop"))
    assert len(out) == 1
    assert finished is False  # child1 仍在
    assert cid0 not in parent.child_requests

    # child1 完成：finished=True
    out, finished = parent.get_outputs(cid1, CompletionOutput(index=1, finish_reason="stop"))
    assert len(out) == 1
    assert finished is True


def test_get_outputs_streaming_dedups_already_returned_child():
    req = make_request("R", n=1, output_kind=RequestOutputKind.DELTA)
    InputProcessor.assign_request_id(req)
    parent = ParentRequest(req)
    cid, _ = parent.get_child_info(0)
    # 第一次完成
    parent.get_outputs(cid, CompletionOutput(index=0, finish_reason="stop"))
    # 再次收到已完成已返还的 child：不重复吐
    out, finished = parent.get_outputs(cid, CompletionOutput(index=0, finish_reason="stop"))
    assert out == []
    assert finished is True


def test_get_outputs_final_only_aggregates_by_index():
    req = make_request("R", n=3, output_kind=RequestOutputKind.FINAL_ONLY)
    InputProcessor.assign_request_id(req)
    parent = ParentRequest(req)
    cids = [parent.get_child_info(i)[0] for i in range(3)]
    # output_aggregator 预分配 n 个槽
    assert len(parent.output_aggregator) == 3

    # 乱序到达：child2、child0 先完成，未攒齐 -> 返回 []
    out, finished = parent.get_outputs(cids[2], CompletionOutput(index=2, text="c", finish_reason="stop"))
    assert out == [] and finished is False
    out, finished = parent.get_outputs(cids[0], CompletionOutput(index=0, text="a", finish_reason="stop"))
    assert out == [] and finished is False

    # 最后 child1 完成：攒齐 3 路，按 index 归位一次性吐出
    out, finished = parent.get_outputs(cids[1], CompletionOutput(index=1, text="b", finish_reason="stop"))
    assert finished is True
    assert [o.text for o in out] == ["a", "b", "c"]  # index 顺序，不受到达顺序影响


# --- AsyncLLM.add_request 扇出端到端 ----------------------------------------

def test_add_request_n1_fast_path_no_parent():
    llm = AsyncLLM()
    req = make_request("R", n=1)
    asyncio.run(llm.add_request(req))
    # 只下发 1 个请求，无 parent 登记
    assert len(llm.engine_core.received) == 1
    assert llm.output_processor.parent_requests == {}


def test_add_request_fans_out_n_independent_children():
    llm = AsyncLLM()
    req = make_request("R", n=3, seed=42)
    queue = asyncio.run(llm.add_request(req))

    # 引擎侧收到 3 个独立 child（各自 n=1）
    recv = llm.engine_core.received
    assert len(recv) == 3
    for r in recv:
        assert r.sampling_params.n == 1
    # 3 个 child id 唯一且 index 前缀
    ids = [r.request_id for r in recv]
    assert len(set(ids)) == 3
    assert all("_" in i for i in ids)
    # 种子确定性递进
    assert sorted(r.sampling_params.seed for r in recv) == [42, 43, 44]

    # OutputProcessor 侧：3 个独立 RequestState + 1 个 parent + external 反查表
    op = llm.output_processor
    assert len(op.request_states) == 3
    assert len(op.parent_requests) == 1
    # external_req_id -> [3 个 internal id]
    assert len(op.external_req_ids["R"]) == 3
    # n 路共享同一 queue
    assert queue is not None


def test_add_request_shares_external_id_and_merges_to_single_request_output():
    llm = AsyncLLM()
    req = make_request("R", n=2, output_kind=RequestOutputKind.FINAL_ONLY)
    asyncio.run(llm.add_request(req))
    op = llm.output_processor
    internal_ids = op.external_req_ids["R"]

    # 模拟两路 child 完成，经各自 RequestState 归并
    rs0 = op.request_states[internal_ids[0]]
    rs1 = op.request_states[internal_ids[1]]
    out0 = rs0.make_request_output(CompletionOutput(index=rs0.request_index, text="x", finish_reason="stop"))
    assert out0 is None  # FINAL_ONLY 未攒齐 -> None
    out1 = rs1.make_request_output(CompletionOutput(index=rs1.request_index, text="y", finish_reason="stop"))
    assert out1 is not None
    # 归并后对外 request_id 被改回 external_req_id
    assert out1.request_id == "R"
    assert out1.finished is True
    assert len(out1.outputs) == 2


# --- 级联 abort --------------------------------------------------------------

def test_abort_parent_cascades_to_children():
    llm = AsyncLLM()
    req = make_request("R", n=3)
    asyncio.run(llm.add_request(req))
    op = llm.output_processor
    parent_id = next(iter(op.parent_requests))

    aborted = op.abort_requests([parent_id], internal=True)
    # 父 abort -> 级联 abort 全部未完成 child
    assert len(aborted) == 3
    assert op.parent_requests == {}


def test_abort_external_id_aborts_all_children():
    llm = AsyncLLM()
    req = make_request("R", n=2)
    asyncio.run(llm.add_request(req))
    op = llm.output_processor

    aborted = op.abort_requests(["R"], internal=False)
    # 外部 id 经反查表 abort 全部 internal child
    assert len(aborted) == 2
    assert "R" not in op.external_req_ids
    assert op.request_states == {}
