# ch31 主电池五：两段式 execute GPU 窗口（m01/m02/m13）——
# worker 两方法契约 + ExecuteModelState 暂存态 + step 编排 + UniProc 转发。
# 基准：vllm/v1/worker/worker_base.py:L142-L157 / gpu_model_runner.py:L437-L450
# / L4165-L4175 / L4484-L4485 / L4516-L4535 / L4552-L4589 / engine/core.py:L584-L614
# / executor/uniproc_executor.py:L26-L137。
from __future__ import annotations

from concurrent.futures import Future

import numpy as np
import pytest
import torch

from conftest import FakeGrammarOutput, FakeSchedulerOutput, cdiv


V = 64


class FakeModel:
    """compute_logits 消费面（真实 LM head 归 ch23——本章以注入面承载）。"""

    def __init__(self, out=None):
        self.calls = []
        self._out = out

    def compute_logits(self, hidden_states):
        self.calls.append(hidden_states)
        if self._out is not None:
            return self._out
        return torch.zeros(hidden_states.shape[0], V)


class FakeSampler:
    def __init__(self):
        self.calls = []

    def __call__(self, *, logits, sampling_metadata):
        self.calls.append(logits)
        return ("sampled", logits)


class RecordingSampler:
    """记录『拿到的是被掩码改过的 logits』的采样器替身。"""

    def __init__(self, order):
        self.order = order
        self.seen = None

    def __call__(self, *, logits, sampling_metadata):
        self.order.append("sample")
        self.seen = logits
        return ("sampled", logits)


class FakeInputBatchFull:
    def __init__(self, req_ids, logits_indices):
        self.req_ids = list(req_ids)
        self.logits_indices = torch.tensor(logits_indices)
        self.sampling_metadata = "MD"

    def update_async_output_token_ids(self):
        pass


def make_runner(req_ids=("r1",), n_tokens=3):
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.execute_model_state = None
    hidden = torch.arange(n_tokens * 4, dtype=torch.float32).reshape(n_tokens, 4)
    runner.model = FakeModel()
    runner.sampler = FakeSampler()
    runner.input_batch = FakeInputBatchFull(req_ids, [n_tokens - 1])
    runner._seam_hidden_states = hidden  # ENGINE SEAM：前向产物注入位
    return runner


class TestExecuteModelState:
    """NamedTuple 十元组：两幕之间的一次性传递暂存态。"""

    def test_state_fields(self):
        from vllm.v1.worker.gpu_model_runner import ExecuteModelState

        fields = ExecuteModelState._fields
        assert fields == (
            "scheduler_output",
            "logits",
            "spec_decode_metadata",
            "spec_decode_common_attn_metadata",
            "hidden_states",
            "sample_hidden_states",
            "aux_hidden_states",
            "ec_connector_output",
            "cudagraph_stats",
            "slot_mappings",
        )


class TestWorkerTwoAct:
    """第一幕 return None + 第二幕 sample_tokens（先掩码后采样）。"""

    def test_execute_model_returns_none_after_packaging(self):
        runner = make_runner()
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
        assert runner.execute_model(so) is None
        assert runner.execute_model_state is not None
        st = runner.execute_model_state
        assert st.scheduler_output is so
        assert torch.equal(st.logits, torch.zeros(1, V))  # compute_logits 产物
        # logits 来自采样位切片 hidden_states[logits_indices]
        assert runner.model.calls[0].shape == (1, 4)

    def test_state_defense_no_double_execute(self):
        # 配对防御：sample_tokens 没来不许再发车（L4171-L4175）
        runner = make_runner()
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
        runner.execute_model(so)
        with pytest.raises(RuntimeError, match="State error"):
            runner.execute_model(so)

    def test_sample_tokens_unpacks_and_clears_single_slot(self):
        runner = make_runner()
        so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
        runner.execute_model(so)
        out = runner.sample_tokens(None)
        assert runner.execute_model_state is None  # 解包即清（单槽自清）
        assert out == ("sampled", runner.sampler.calls[0])

    def test_mask_applied_before_sample_and_inplace(self):
        # 『先掩码后采样』：masker 先于 sampler，且改写的是同一块 logits 张量
        # （gpu_model_runner.py:L211 将 apply_grammar_bitmask 导入自身命名空间）
        import vllm.v1.worker.gpu_model_runner as mr

        order = []
        real_apply = mr.apply_grammar_bitmask

        def spy_apply(scheduler_output, grammar_output, input_batch, logits):
            order.append("apply")
            logits.fill_(-1.0)  # 原地改写（in-place 所有权契约 m13）

        mr.apply_grammar_bitmask = spy_apply
        try:
            runner = make_runner()
            sampler = RecordingSampler(order)
            runner.sampler = sampler
            so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
            runner.execute_model(so)
            go = FakeGrammarOutput(["r1"], np.ones((1, cdiv(V, 32)), np.int32))
            out = runner.sample_tokens(go)
            assert order == ["apply", "sample"]
            assert torch.all(out[1] == -1.0)  # sampler 拿到的是被掩码改过的张量
        finally:
            mr.apply_grammar_bitmask = real_apply

    def test_no_grammar_output_skips_mask(self):
        import vllm.v1.worker.gpu_model_runner as mr

        order = []
        real_apply = mr.apply_grammar_bitmask

        def spy_apply(*a, **k):
            order.append("apply")

        mr.apply_grammar_bitmask = spy_apply
        try:
            runner = make_runner()
            runner.sampler = RecordingSampler(order)
            so = FakeSchedulerOutput(num_scheduled_tokens={"r1": 3})
            runner.execute_model(so)
            runner.sample_tokens(None)
            assert order == ["sample"]  # grammar_output 为 None 不触掩码
        finally:
            mr.apply_grammar_bitmask = real_apply


class TestWorkerBaseContract:
    """worker 两方法契约基类（全硬件后端统一，docstring 自注技术债）。"""

    def test_base_methods_not_implemented(self):
        from vllm.v1.worker.worker_base import WorkerBase

        with pytest.raises(NotImplementedError):
            WorkerBase.execute_model(self=None, scheduler_output=None)
        with pytest.raises(NotImplementedError):
            WorkerBase.sample_tokens(self=None, grammar_output=None)


class TestEngineCoreStep:
    """step 四段排布：② non_block 发车 → ③ 掩码藏进前向窗口 → ④ 补采样。"""

    def test_step_orders_acts(self):
        from vllm.v1.engine.core import EngineCore

        core = EngineCore.__new__(EngineCore)
        core.scheduler = SpyScheduler(pending=False)
        core.model_executor = SpyExecutor(core.scheduler.log)
        outputs, model_executed = core.step()
        # ③ 掩码计算发生在 ② 发车之后（发车标记先于 bitmask 标记）
        assert core.scheduler.log == [
            "dispatch",        # ② execute_model(non_block=True) 发车即返
            "bitmask",         # ③ get_grammar_bitmask（藏进前向窗口的 CPU 活）
            "sample_tokens",   # ④ future.result()=None → sample_tokens(grammar_output)
            "update_from_output",
        ]
        assert core.model_executor.grammar_output_passed is not None
        assert model_executed is True

    def test_step_no_requests_early_return(self):
        from vllm.v1.engine.core import EngineCore

        core = EngineCore.__new__(EngineCore)

        class Empty:
            def has_requests(self):
                return False

        core.scheduler = Empty()
        core.model_executor = SpyExecutor([])
        assert core.step() == ({}, False)
        assert core.model_executor.grammar_output_passed is None


class SpyExecutor:
    """UniProcExecutor 消费面替身：non_block 发车 → Future(None)。"""

    def __init__(self, log):
        self.log = log
        self.grammar_output_passed = None

    def execute_model(self, scheduler_output, non_block=False):
        assert non_block is True  # L596/L656：发车即返
        self.log.append("dispatch")
        fut = Future()
        fut.set_result(None)  # worker 未就地采样 → 返回 None
        return fut

    def sample_tokens(self, grammar_output, non_block=False):
        self.grammar_output_passed = grammar_output
        self.log.append("sample_tokens")
        fut = Future()
        fut.set_result(("model_output", None))
        return fut

    def take_draft_token_ids(self):
        self.log.append("take_draft")
        return None


class SpyScheduler:
    def __init__(self, pending=False):
        self.log = []
        self.pending = pending
        self.update_called = False

    def has_requests(self):
        return True

    def schedule(self, throttle=False):
        return FakeSchedulerOutput(
            num_scheduled_tokens={"r1": 3},
            has_structured_output_requests=True,
            pending_structured_output_tokens=self.pending,
        )

    def get_grammar_bitmask(self, scheduler_output):
        # ③ 拍：此刻 ② 的发车标记必须在场（CPU 填表藏进 GPU 前向窗口期）
        assert "dispatch" in self.log
        self.log.append("bitmask")
        return FakeGrammarOutput(["r1"], np.ones((1, 2), np.int32))

    def update_from_output(self, scheduler_output, model_output):
        self.update_called = True
        self.log.append("update_from_output")
        return {}
