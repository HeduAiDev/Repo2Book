# SOURCE: vllm/v1/worker/gpu_model_runner.py
# worker 侧两件副作用的**KV 拷贝切面**（m14 管线终点半边）：新块清零（防陈旧
# NaN，L1219-L1222——零清通道本体归 ch13）+ copy_kv_cache_blocks_inplace 在
# GPU 上执行 CoW 块拷贝（L1223-L1228）——拷贝排在清零之后、forward 之前。
# HOST SEAM：CPU 张量等价复现（无 CUDA 面）；容器内验真 GPU。
# SUBTRACTED: _update_states 的请求生命周期/块表先行拷贝/positions 等其余
#   段（ch07/ch13/ch17/ch18 各章切面）与 _zero_block_ids 内景（ch13）。
from .worker_utils import copy_kv_cache_blocks_inplace


# SOURCE: vllm/v1/worker/gpu_model_runner.py:~L1195 apply_scheduler_output_side_
#   effects（ENGINE SEAM：从 _update_states 尾段抽出 L1219-L1228 的两件副作用）
def apply_scheduler_output_side_effects(
    scheduler_output,
    kv_caches: list,
    num_blocks: int,
) -> None:
    # Zero GPU memory for freshly allocated cache blocks to prevent
    # stale NaN/data from corrupting attention or SSM computation.
    # SUBTRACTED: 新块清零分支（L1221-L1222——_zero_block_ids 走 KVBlockZeroer
    #   的 kernel → ch13；CPU 镜像无陈旧 NaN 风险，账位以注释保留）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1223-L1228（CoW 真拷贝：
    #   调度器只给块号对，worker 在 GPU 上执行）
    if scheduler_output.kv_cache_block_copies:
        copy_kv_cache_blocks_inplace(
            kv_caches,
            num_blocks,
            scheduler_output.kv_cache_block_copies,
        )
