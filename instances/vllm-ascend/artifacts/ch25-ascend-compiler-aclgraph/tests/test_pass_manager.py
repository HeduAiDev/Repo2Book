"""ch25 — GraphFusionPassManager 串 pass 的真实控制流（对位 vLLM PostGradPassManager）。

__call__ 遍历 self.passes，只对 is_applicable_for_range(compile_range) 通过的 pass 执行
pass_(graph)，最后 graph.recompile()。add() 强制 isinstance(pass_, VllmInductorPass)。
"""
import pytest

import _ch25_pass_manager as pm
from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass


class RecordingPass(VllmInductorPass):
    def __init__(self, applicable=True):
        self.applicable = applicable
        self.ran_on = []

    def is_applicable_for_range(self, compile_range):
        return self.applicable

    def __call__(self, graph):
        self.ran_on.append(graph)


class FakeGraph:
    def __init__(self):
        self.recompiled = 0

    def recompile(self):
        self.recompiled += 1


def test_call_runs_applicable_passes_then_recompiles():
    mgr = pm.GraphFusionPassManager()
    p_yes = RecordingPass(applicable=True)
    p_no = RecordingPass(applicable=False)
    mgr.add(p_yes)
    mgr.add(p_no)

    g = FakeGraph()
    out = mgr(g)

    assert out is g
    assert p_yes.ran_on == [g]      # applicable pass ran
    assert p_no.ran_on == []        # non-applicable pass skipped
    assert g.recompiled == 1        # graph.recompile() called once at the end


def test_passes_run_in_registration_order():
    order = []

    class OrderPass(RecordingPass):
        def __init__(self, tag):
            super().__init__(True)
            self.tag = tag

        def __call__(self, graph):
            order.append(self.tag)

    mgr = pm.GraphFusionPassManager()
    mgr.add(OrderPass("a"))
    mgr.add(OrderPass("b"))
    mgr.add(OrderPass("c"))
    mgr(FakeGraph())
    assert order == ["a", "b", "c"]


def test_add_rejects_non_vllm_inductor_pass():
    mgr = pm.GraphFusionPassManager()
    with pytest.raises(AssertionError):
        mgr.add(object())


def test_empty_manager_still_recompiles():
    mgr = pm.GraphFusionPassManager()
    g = FakeGraph()
    mgr(g)
    assert g.recompiled == 1
