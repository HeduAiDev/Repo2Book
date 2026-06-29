"""对位真实行为：NPUWorker 与 GPU Worker 是 WorkerBase 的两个『平级实现』，不是继承关系。

对照真实源码：
  - vllm/v1/worker/worker_base.py：四步生命周期方法体全是 raise NotImplementedError；
  - vllm/v1/worker/gpu_worker.py:L239-L309：init_device 把整段包在 if device_type=='cuda'，
    非 cuda 直接 `else: raise RuntimeError(Not support device type)`；
  - vllm_ascend/worker/worker.py:L81：class NPUWorker(WorkerBase)，直接派生抽象基类。
"""
import types

import pytest

import gpu_worker
import worker
import worker_base


def test_npuworker_derives_abstract_workerbase_not_gpu_worker():
    # NPUWorker 与 GPU 的 Worker 都直接派生 WorkerBase（平级兄弟）。
    assert issubclass(worker.NPUWorker, worker_base.WorkerBase)
    assert issubclass(gpu_worker.Worker, worker_base.WorkerBase)
    # 关键：NPUWorker 不是 GPU Worker 的子类——它重写而非继承。
    assert not issubclass(worker.NPUWorker, gpu_worker.Worker)
    assert worker.NPUWorker.__bases__ == (worker_base.WorkerBase,)


def test_workerbase_lifecycle_methods_are_abstract():
    # 四步生命周期在抽象层全是 raise NotImplementedError——派生类必须自己实现。
    wb = worker_base.WorkerBase.__new__(worker_base.WorkerBase)
    for name in ("init_device", "compile_or_warm_up_model"):
        with pytest.raises(NotImplementedError):
            getattr(wb, name)()
    with pytest.raises(NotImplementedError):
        wb.execute_model(scheduler_output=None)


def test_workerbase_init_spreads_vllm_config_into_fields():
    # super().__init__ 复用的公共逻辑：把整份 vllm_config 摊开成各 config 字段。
    cfg = types.SimpleNamespace(
        model_config="MC", cache_config="CC", lora_config="LC", load_config="LDC",
        parallel_config=types.SimpleNamespace(), scheduler_config="SC", device_config="DC",
        speculative_config="SPC", observability_config="OC", kv_transfer_config="KVC",
        compilation_config="COMP",
    )
    wb = worker_base.WorkerBase(cfg, local_rank=3, rank=7, distributed_init_method="env://")
    assert wb.model_config == "MC"
    assert wb.cache_config == "CC"
    assert wb.compilation_config == "COMP"
    assert wb.local_rank == 3 and wb.rank == 7
    assert wb.parallel_config.rank == 7  # __init__ 回写 parallel_config.rank
    assert wb.device is None and wb.model_runner is None


def test_gpu_worker_init_device_raises_on_non_cuda():
    # 『不能继承只能重写』的硬证据：GPU Worker.init_device 对非 cuda 设备直接 raise。
    w = gpu_worker.Worker.__new__(gpu_worker.Worker)
    w.device_config = types.SimpleNamespace(device_type="npu", device="npu:0")
    with pytest.raises(RuntimeError, match="Not support device type"):
        w.init_device()
