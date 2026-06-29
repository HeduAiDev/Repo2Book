"""对照基座：GPU 的 Worker.init_device 把设备层钉死在 'cuda'（subtract-only 忠实摘录）。

『为何不能继承只能重写』的硬证据：Worker.init_device 整段实现包在
`if device_config.device_type == "cuda"` 里，任何非 cuda 设备直接走 `else: raise
RuntimeError`。昇腾若继承它，唯一能走到的就是 raise 分支——所以 NPUWorker 只能整
方法重写，而非薄改继承。

非 cuda 路径（host 上传 device_type='npu'）可运行验证 → 必抛 RuntimeError。
cuda 路径里的真实设备调用（torch.accelerator / current_platform）host 跑不到，
仅作可读骨架。
"""
# SUBTRACTED: 文件头 SPDX + 大量 import + _maybe_get_memory_pool_context 等辅助（原 L1-L105）。
from worker_base import WorkerBase


# SOURCE: vllm/v1/worker/gpu_worker.py:L106
class Worker(WorkerBase):
    # SUBTRACTED: __init__（原 L107-L238）——super().__init__ + float32 matmul 精度 +
    #   ElasticEP executor 等，与『设备层钉死 cuda』要点无关。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L239-L309
    def init_device(self):
        if self.device_config.device_type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            # SUBTRACTED: DP 本地 rank 调整 + visible_device_count 断言（原 L241-L272）。
            self.device = torch.device(f"cuda:{self.local_rank}")  # noqa: F821
            torch.accelerator.set_device_index(self.device)  # noqa: F821

            current_platform.check_if_supports_dtype(self.model_config.dtype)  # noqa: F821

            # Initialize the distributed environment BEFORE taking memory snapshot.
            # This ensures NCCL buffers are allocated before we measure available
            # memory —— 注意顺序：基座先初始化分布式、后拍快照，故 NCCL 通信 buffer
            # 被计入快照基线（昇腾相反，见 vllm_ascend/worker/worker.py:_init_device）。
            init_worker_distributed_environment(  # noqa: F821
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,  # noqa: F821
            )
            # Now take memory snapshot after NCCL is initialized.
            self.init_snapshot = MemorySnapshot(device=self.device)  # noqa: F821
            self.requested_memory = request_memory(self.init_snapshot, self.cache_config)  # noqa: F821
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # SUBTRACTED: init_workspace_manager + 构造 GPUModelRunner（原 L310 起）——与
        #   『设备层钉死 cuda』要点无关，保留 if cuda / else raise 骨架即可看清不能继承的原因。

    # SUBTRACTED: determine_available_memory / compile_or_warm_up_model / execute_model
    #   等（原 L354-L900+）——NPUWorker 与之同构但换设备层，正文以昇腾侧 worker.py 为主线对照。
