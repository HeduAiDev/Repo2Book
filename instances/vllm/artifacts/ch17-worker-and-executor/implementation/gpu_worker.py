# 只做减法的忠实精简版 —— 镜像 vllm/v1/worker/gpu_worker.py 的 Worker（pin f3fef123）。
# 与 vLLM 同名、同结构、同控制流；只删不增。
#
# 本章把 Worker 当**生命周期锚点**：只保留 __init__ / init_device / load_model / execute_model
# 的主干骨架，展示 WorkerWrapperBase→Worker→model_runner 的调用链。GPU 设备/分布式环境/内存
# 快照/前向(PP 通信)/KV 内存剖析(determine_available_memory) 等实体属 ch18/ch19/ch20，在此
# 全部作 SUBTRACTED 锚点——不杜撰任何前向计算。
#
# SUBTRACTED: 模块顶部 SPDX 版权头与 torch / current_platform / 各 vllm.distributed /
#   model_runner / WeightTransferEngineFactory / ElasticEPScalingExecutor 等真实 import
#   （vllm/v1/worker/gpu_worker.py:L1-L105 的 import 区）—— 全是 CUDA/torch/分布式实体。

from worker_base import WorkerBase


# SOURCE: vllm/v1/worker/gpu_worker.py:L106-L861
class Worker(WorkerBase):
    # SOURCE: vllm/v1/worker/gpu_worker.py:L107-L156
    def __init__(
        self,
        vllm_config,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        # SUBTRACTED: VLLM_FLOAT32_MATMUL_PRECISION 设定、ElasticEPScalingExecutor、
        #   _sleep_saved_buffers、WeightTransferEngine、profiler、use_v2_model_runner、
        #   _pp_send_work 等构造期字段（vllm/v1/worker/gpu_worker.py:L123-L156）—— 精度/弹性 EP/
        #   权重迁移/profiler/PP 发送队列，均非本章生命周期主线，留作锚点。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L236-L333
    def init_device(self) -> None:
        # SUBTRACTED: 真实 init_device 做 torch.cuda 设备选择、init_worker_distributed_environment
        #   建分布式组、torch.cuda 内存快照、构造 GPUModelRunner 等（vllm/v1/worker/gpu_worker.py
        #   :L236-L333）—— 全是 CUDA/分布式实体，属 ch18+。这里把『构造 model_runner』作锚点。
        self.model_runner = None

    # SOURCE: vllm/v1/worker/gpu_worker.py:L160-L170
    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        # SUBTRACTED: 真实 load_model 调 self.model_runner.load_model(...) 把权重搬上设备——
        #   模型加载本身属 ch19；本章只点到生命周期里有这一步。
        return

    # SOURCE: vllm/v1/worker/gpu_worker.py:L772-L861
    def execute_model(self, scheduler_output):
        # SUBTRACTED: 真实 execute_model 包 self.model_runner.execute_model(scheduler_output)，
        #   其中含 PP irecv/isend 通信与采样（vllm/v1/worker/gpu_worker.py:L772-L861）—— 前向/PP
        #   通信属 ch18；本章只保留『Worker.execute_model 转调 model_runner』的骨架。
        if self.model_runner is None:
            raise NotImplementedError(
                "model_runner forward is out of scope for this chapter (see ch18)."
            )
        return self.model_runner.execute_model(scheduler_output)
