"""对位真实行为：四步生命周期骨架齐备，且 execute_model 把活儿派发给 NPUModelRunner。

四步：init_device → determine_available_memory → compile_or_warm_up_model → execute_model
（vllm_ascend/worker/worker.py）。execute_model 本身很薄——真正前向在 NPUModelRunner。
"""
import types

import worker


def test_four_step_lifecycle_methods_present():
    for name in (
        "init_device", "_init_device", "determine_available_memory",
        "compile_or_warm_up_model", "execute_model", "_warm_up_atb",
        "_init_worker_distributed_environment",
    ):
        assert callable(getattr(worker.NPUWorker, name)), name


def test_execute_model_dispatches_to_model_runner(monkeypatch):
    class FakeModelRunnerOutput:
        pass

    monkeypatch.setattr(worker, "ModelRunnerOutput", FakeModelRunnerOutput, raising=False)
    monkeypatch.setattr(worker, "AsyncModelRunnerOutput", FakeModelRunnerOutput, raising=False)

    produced = FakeModelRunnerOutput()
    seen = {}

    def fake_execute_model(scheduler_output, intermediate_tensors):
        seen["sched"] = scheduler_output
        seen["intermediate"] = intermediate_tensors
        return produced

    w = worker.NPUWorker.__new__(worker.NPUWorker)
    w.profiler = None
    w.model_runner = types.SimpleNamespace(execute_model=fake_execute_model)
    sched = types.SimpleNamespace(total_num_scheduled_tokens=4)

    out = w.execute_model(sched)

    # 派发给 model_runner.execute_model，并原样返回它的 ModelRunnerOutput。
    assert out is produced
    assert seen["sched"] is sched
    assert seen["intermediate"] is None  # 单机首 rank：无 PP 中间张量


def test_warm_up_atb_uses_ascend_matmul_add(monkeypatch):
    # _warm_up_atb 打一发 ATB matmul_add 预热。host 无 npu：桩掉 .npu() 与 torch_npu，验调用形状。
    calls = {}

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def npu(self):
            return self

    monkeypatch.setattr(worker.torch, "rand", lambda shape, dtype=None: FakeTensor(shape), raising=False)
    monkeypatch.setattr(
        worker, "torch_npu",
        types.SimpleNamespace(
            _npu_matmul_add_fp32=lambda x, weight, c: calls.update(x=x.shape, w=weight.shape, c=c.shape)
        ),
        raising=False,
    )

    w = worker.NPUWorker.__new__(worker.NPUWorker)
    w._warm_up_atb()

    assert calls == {"x": (2, 4), "w": (2, 4), "c": (4, 4)}
