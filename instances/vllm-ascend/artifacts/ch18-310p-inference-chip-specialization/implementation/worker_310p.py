# vllm_ascend/_310p/worker_310p.py —— subtract-only 精简版（ch17：整条 310 栈的入口 worker）
#
# NPUWorker310(NPUWorker) 是再继承一层的薄子类，本章只需两个动作：
#   (1) init_device 末尾把 self.model_runner 换成 NPUModelRunner310——这是整条 310 栈
#       被链式拉起的起点（platform.py 把 worker_cls 指向 NPUWorker310 → init_device →
#       NPUModelRunner310 → NPUInputBatch310 → 310 BlockTable）；
#   (2) save_sharded_state 走 ShardedStateLoader310（单 part + parameters_type_map.json）。
#
# 这些都触 torch_npu/CANN，host 不真跑；精简版只展示"worker 子类如何接上
# NPUModelRunner310 与 ShardedStateLoader310"的装配控制流。
import torch
import torch_npu
from vllm.logger import logger

from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
from vllm_ascend.worker.worker import NPUWorker, init_workspace_manager

# SUBTRACTED: _IS_RC_DEVICE 缓存与 _is_rc_device()（worker_310p.py:L31-L45）—— RC（共享
#   host/device 内存）设备探测靠 lspci，是 worker 启动细节，与"子类化"立意正交。


# SOURCE: vllm_ascend/_310p/worker_310p.py:L48
class NPUWorker310(NPUWorker):
    # SOURCE: vllm_ascend/_310p/worker_310p.py:L49-L56
    def init_device(self):
        self.device = self._init_device()
        torch_npu.npu.set_compile_mode(jit_compile=False)

        init_workspace_manager(self.device, num_ubatches=1)

        # 把 model_runner 换成 NPUModelRunner310——再继承一层的起点，整条 310 栈由此拉起。
        self.model_runner = NPUModelRunner310(self.vllm_config, self.device)
        logger.info_once("Using NPUWorker310 and NPUModelRunner310.")

    # SOURCE: vllm_ascend/_310p/worker_310p.py:L58-L77
    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm_ascend._310p.sharded_state_loader_310p import ShardedStateLoader310

        # 310 权重保存：单 part + 额外量化描述 JSON。
        ShardedStateLoader310.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

        ShardedStateLoader310.generate_quant_description(
            self.model_runner.model,
            path,
            self.vllm_config.quant_config,
        )

    # SUBTRACTED: determine_available_memory / _warm_up_atb / _init_device
    #   （worker_310p.py:L79-L192）—— RC 设备显存探测（torch.npu.mem_get_info / psutil）、
    #   跳过 310P 不支持的 _npu_matmul_add_fp32 atb 预热、分布式环境初始化，都是 worker
    #   启动期设备细节，与本章四大主线（输入批/runner 设备路径/权重/KV 清零）正交。
