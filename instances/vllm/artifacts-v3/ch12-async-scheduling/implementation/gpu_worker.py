# SOURCE: vllm/v1/worker/gpu_worker.py
# GPUWorker —— executor 与 model_runner 之间的 worker 壳（三层归 ch17 全文）。
# 本章切面：execute_model / sample_tokens 的 @with_gpu_sync_check 包裹位
# （L1010-L1021，m19 tripwire 的落点）；warmup 尾部的 enable_gpu_sync_check
# 翻闸（L846-L848）以注释存证（warmup 本体归 ch17）。
from __future__ import annotations

from typing import Any

import torch

from .gpu_model_runner import GPUModelRunner
from .gpu_sync_debug import with_gpu_sync_check


# SOURCE: vllm/v1/worker/gpu_worker.py GPUWorker（本章切面）
class GPUWorker:
    # SOURCE: vllm/v1/worker/gpu_worker.py GPUWorker.__init__（装配切面：
    # 真实走 init_device/load_model/determine_available_memory——ch17）
    def __init__(self, vllm_config: Any, *args, **kwargs):
        # SOURCE: vllm/v1/worker/gpu_worker.py GPUWorker.__init__
        # SUBTRACTED: 设备初始化/分布式/显存剖析（ch17）。
        self.model_runner = GPUModelRunner(vllm_config)
        # SUBTRACTED: _pp_send_work 的 PP 在途句柄（L1023-L1026 消费——PP 面，
        #   dossier.delete 第 5 条批准）。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L846-L848 warmup 尾部翻闸（m19：
    #   门在 warmup/首编译完成后才开——此前 setup 期的 sync 放行）。warmup
    #   本体归 ch17，此处 compile_or_warm_up_model 以注释存证：
    #       # Warmup / first-compile is done — activate the `VLLM_GPU_SYNC_CHECK`
    #       # gate so subsequent `execute_model` / `sample_tokens` calls enforce it.
    #       enable_gpu_sync_check()
    # SUBTRACTED: compile_or_warm_up_model 全文（L700-L852——ch17）。

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1010-L1015 sample_tokens（tripwire
    # 包裹位，逐字）
    @torch.inference_mode()
    @with_gpu_sync_check
    def sample_tokens(self, grammar_output):
        # SOURCE: vllm/v1/worker/gpu_worker.py:L1013-L1015
        return self.model_runner.sample_tokens(grammar_output)

    # SOURCE: vllm/v1/worker/gpu_worker.py:L1017-L1021 execute_model（tripwire
    # 包裹位）
    @torch.inference_mode()
    @with_gpu_sync_check
    def execute_model(self, scheduler_output):
        # SUBTRACTED: PP 在途 send 等待（L1022-L1026——第 5 条）与
        #   intermediate_tensors/forward_pass 深水（L1028-L1050s——ch17/PP）。
        # SOURCE: vllm/v1/worker/gpu_worker.py 直调 model_runner
        return self.model_runner.execute_model(scheduler_output)

    # SOURCE: vllm/v1/worker/gpu_worker.py take_draft_token_ids 转发位（spec
    # drafter 面已随 dossier.delete 第 6 条删——无 drafter 恒 None，
    # post_step 的 None 守卫原样成立）
    def take_draft_token_ids(self):
        # SOURCE: vllm/v1/worker/gpu_worker.py take_draft_token_ids
        return None
