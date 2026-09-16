# ch31 主电池八：spec 草稿回传通道（m17）+ UniProc 转发面。
# 基准：vllm/v1/worker/gpu/spec_decode/utils.py:L11-L70 /
# vllm/v1/executor/uniproc_executor.py:L26-L137。
from __future__ import annotations

import numpy as np
import pytest
import torch


class FakeInputBatch:
    def __init__(self, req_ids, has_structured_output_reqs):
        self.req_ids = list(req_ids)
        self.has_structured_output_reqs = has_structured_output_reqs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="copy_stream 需 CUDA")
class TestDraftTokensHandler:
    def test_no_structured_reqs_skips_return(self):
        from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

        h = DraftTokensHandler(device=torch.device("cuda"))
        ib = FakeInputBatch(["r1"], has_structured_output_reqs=False)
        drafts = torch.randint(0, 100, (1, 3), device="cuda")
        h.set_draft_tokens(ib, drafts)
        assert h.draft_tokens_np is None  # 无语法约束 → 整批跳过回传
        assert h.num_draft_tokens == 3

    def test_d2h_roundtrip_with_structured_reqs(self):
        from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

        h = DraftTokensHandler(device=torch.device("cuda"))
        ib = FakeInputBatch(["r1", "r2"], has_structured_output_reqs=True)
        drafts = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda")
        h.set_draft_tokens(ib, drafts)
        assert h.draft_tokens_np is not None

        got = h.get_draft_tokens()
        assert got.req_ids == ["r1", "r2"]
        assert got.draft_token_ids == [[1, 2, 3], [4, 5, 6]]

    def test_record_stream_called_for_async_safety(self):
        # record_stream 防分配器提前复用（copy_stream 上读主流临时分配）
        from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

        h = DraftTokensHandler(device=torch.device("cuda"))
        ib = FakeInputBatch(["r1"], has_structured_output_reqs=True)
        drafts = torch.tensor([[7, 8]], device="cuda")
        recorded = []
        drafts.record_stream = lambda stream: recorded.append(stream)
        h.set_draft_tokens(ib, drafts)
        assert recorded == [h.copy_stream]


class TestAsyncOutputFuture:
    """AsyncOutputFuture：只在 result() 才等 D2H 事件（uniproc L26-L42）。"""

    def test_result_resolves_once(self):
        from concurrent.futures import Future as PyFuture

        from vllm.v1.executor.uniproc_executor import AsyncOutputFuture

        class FakeAsyncOutput:
            def __init__(self):
                self.gets = 0

            def get_output(self):
                self.gets += 1
                return "MRO"

        fut = AsyncOutputFuture(FakeAsyncOutput(), single_value=True)
        assert isinstance(fut, PyFuture)
        assert not fut.done()
        assert fut.result() == "MRO"
        assert fut.result() == "MRO"  # 幂等：get_output 只调一次
        assert fut.async_output.gets == 1
        assert fut.done()


class TestUniProcDelegation:
    """UniProcExecutor 三方法转发（execute_model/sample_tokens/
    take_draft_token_ids——L108-L137）。"""

    def test_delegation_and_futures(self):
        from vllm.v1.executor.uniproc_executor import UniProcExecutor

        class FakeWorker:
            def __init__(self):
                self.calls = []

            def execute_model(self, scheduler_output):
                self.calls.append(("execute_model", scheduler_output))
                return None  # 两段式：返回 None ⇒ 紧跟 sample_tokens

            def sample_tokens(self, grammar_output):
                self.calls.append(("sample_tokens", grammar_output))
                return "MRO"

            def take_draft_token_ids(self):
                self.calls.append(("take_draft_token_ids",))
                from vllm.v1.outputs import DraftTokenIds

                return DraftTokenIds(["r1"], [[1, 2]])

        ex = UniProcExecutor.__new__(UniProcExecutor)
        ex.driver_worker = FakeWorker()

        fut = ex.execute_model("SO", non_block=True)
        # non_block + 同步返回 None → 立即解析的 Future（不是 AsyncOutputFuture）
        assert fut.done() and fut.result() is None

        out = ex.sample_tokens("GO")
        assert out == "MRO"

        drafts = ex.take_draft_token_ids()
        assert drafts.req_ids == ["r1"]
        assert ex.driver_worker.calls[0] == ("execute_model", "SO")
        assert ex.driver_worker.calls[1] == ("sample_tokens", "GO")
        assert ex.driver_worker.calls[2] == ("take_draft_token_ids",)
