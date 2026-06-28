# 章 ch05 精简版 —— vllm_ascend/platform.py（subtract-only）
#
# 本章第一条主线：平台 = 配置改写器。vLLM 在 VllmConfig.__post_init__ 末段把完整 VllmConfig 按引用
# 交给 NPUPlatform.check_and_update_config 做构图前最后改写。本精简版保留其编排骨架：
# 守卫早退 → 一致性校验 → _fix_incompatible_config（cascade reset GPU/ROCm 专属参数）→
# init_ascend_config（解析 additional_config）→ cudagraph/编译改写 → worker_cls 落定 → 设环境变量。
import os

# SUBTRACTED: 真源码顶部 `from vllm.config import CompilationMode`（L444 函数内）、
#             `from vllm.config.compilation import CUDAGraphMode`、
#             `from vllm.v1.attention.backends.registry import AttentionBackendEnum`（L35）、
#             `from vllm.logger import logger`、is_310p/get_ascend_device_type/refresh_block_size
#             (vllm_ascend/utils.py)、init_ascend_config(vllm_ascend/ascend_config.py)——
#             host 无 vLLM/CANN，统一改从本精简版同目录的替身/真实精简模块导入。
from _support import (
    AttentionBackendEnum,
    CompilationMode,
    CUDAGraphMode,
    is_310p,
    logger,
    refresh_block_size,
)
from ascend_config import init_ascend_config

if False:  # TYPE_CHECKING
    from vllm.config import VllmConfig


# SOURCE: vllm_ascend/platform.py:L134 (class NPUPlatform(Platform))
class NPUPlatform:
    # SUBTRACTED: 原 NPUPlatform 继承 vLLM Platform 并有大量设备/后端职责方法
    #             (get_device_*、get_attn_backend_cls、apply_config_platform_defaults、各 get_*_cls 工厂…)，
    #             分散在别章；本章只围绕 check_and_update_config / _fix_incompatible_config。
    device_name: str = "npu"
    device_type: str = "npu"

    # SOURCE: vllm_ascend/platform.py:L413-L714
    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        # SOURCE: vllm_ascend/platform.py:L413-L714
        # SUBTRACTED: from vllm_ascend.quantization.utils import maybe_auto_detect_quantization（L415）

        device_config = getattr(vllm_config, "device_config", None)
        if device_config is not None and getattr(device_config, "device_type", cls.device_type) != cls.device_type:
            logger.debug(
                "Skipping Ascend-specific config updates for device type %s.",
                device_config.device_type,
            )
            return

        if vllm_config.model_config is None:
            logger.warning("Model config is missing. Skipping Ascend-specific config updates.")
            return

        # SUBTRACTED: maybe_auto_detect_quantization(vllm_config)（L429，量化探测需 CANN）

        cls._validate_layer_sharding_config(vllm_config)
        cls._validate_draft_decode_context_parallel_config(vllm_config)

        # initialize ascend config from vllm additional_config
        cls._fix_incompatible_config(vllm_config)

        ascend_config = init_ascend_config(vllm_config)

        # SUBTRACTED: kv_transfer engine_id uuid 补丁、ascend_compilation/fusion 回写 additional_config、
        #             update_compile_ranges_split_points 等周边(L439-468)，各属别章子系统钩入点。
        compilation_config = vllm_config.compilation_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        enforce_eager = getattr(model_config, "enforce_eager", False)

        # SUBTRACTED: xlite_graph_config full/decode-only 模式下对 cudagraph_mode 的预改写(L473-481)

        if enforce_eager:
            logger.info("Compilation disabled, using eager mode by default")
            compilation_config.mode = CompilationMode.NONE
            if compilation_config.splitting_ops is None:
                compilation_config.splitting_ops = []

        compilation_config.cudagraph_num_of_warmups = 1

        if compilation_config.mode not in [CompilationMode.NONE, CompilationMode.VLLM_COMPILE]:
            logger.warning(
                "NPU does not support compilation mode. mode=%s, action: setting CUDAGraphMode to NONE.",
                compilation_config.mode,
            )
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        # SUBTRACTED: 据 cudagraph_mode 分支改写 splitting_ops（piecewise→set_splitting_ops_for_v1+extend
        #             mla/dsa_forward；full→splitting_ops=[]）、_set_cudagraph_sizes、SP sizes 重算、
        #             encoder-decoder PIECEWISE 回退、oot_compiler/use_inductor、ASCEND_LAUNCH_BLOCKING
        #             校验(L498-600)——同属『平台改写编译配置』，但控制流密、各为专章细节。

        if parallel_config and parallel_config.worker_cls == "auto":
            # TODO: this is a tricky way to disable `use_sequence_parallel_moe` in vllm.
            if not vllm_config.compilation_config.pass_config.enable_sp:
                parallel_config.all2all_backend = "flashinfer_all2allv"
            if is_310p():
                parallel_config.worker_cls = "vllm_ascend._310p.worker_310p.NPUWorker310"
            elif ascend_config.xlite_graph_config.enabled:
                logger.info("openEuler Xlite enabled. See: https://atomgit.com/openeuler/GVirt/tree/master/xlite")
                parallel_config.worker_cls = "vllm_ascend.xlite.xlite_worker.XliteWorker"
            else:
                parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"

        refresh_block_size(vllm_config)

        # SUBTRACTED: custom_ops=['all'](非310P)、enable_balance_scheduling/recompute_scheduler/dynamic_batch
        #             /profiling_chunk 的 scheduler_cls 切换、各种 PD/sparse/mc2-hierarchy 互斥 raise 校验
        #             (L616-695)——其它子系统钩入点与一致性校验，非本章两条主线。

        # Set "PYTORCH_NPU_ALLOC_CONF=expandable_segments:True" by default to optimize NPU memory management.
        # NOTE: We should not set this environment variable in RL (sleep mode) scenarios.
        if model_config and not model_config.enable_sleep_mode:
            npu_alloc_configs = os.getenv("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
            # This environment variable may have more than one key-value pairs.
            # We should append ",expandable_segments:True" to the current configs.
            # For example: "page_size:1g" + ",expandable_segments:True".
            # NOTE: `max_split_size_mb` or `garbage_collection_threshold` cannot
            # be enabled together with `expandable_segments=True`.
            if (
                "expandable_segments" not in npu_alloc_configs
                and "max_split_size_mb" not in npu_alloc_configs
                and "garbage_collection_threshold" not in npu_alloc_configs
            ):
                npu_alloc_configs += ",expandable_segments:True"
            os.environ["PYTORCH_NPU_ALLOC_CONF"] = npu_alloc_configs
            logger.info("Set PYTORCH_NPU_ALLOC_CONF=%s", npu_alloc_configs)

    # SOURCE: vllm_ascend/platform.py:L979-L1180
    @staticmethod
    def _fix_incompatible_config(vllm_config: "VllmConfig") -> None:
        """
        Check and correct parameters in VllmConfig that are incompatible with Ascend NPU.
        If GPU-specific or currently unsupported parameters are set by the user,
        log a warning and reset them to safe values.
        """
        # SOURCE: vllm_ascend/platform.py:L979-L1180
        model_config = vllm_config.model_config
        # ==================== 1. Model Config ====================
        if model_config:
            # Disable Cascade Attention (GPU feature)
            if getattr(model_config, "disable_cascade_attn", False):
                logger.warning(
                    "GPU-specific parameter is not supported on Ascend. "
                    "parameter=disable_cascade_attn, value=True, action: resetting to False."
                )
                model_config.disable_cascade_attn = False

        # ==================== 2. Cache Config ====================
        if vllm_config.cache_config:
            # Check and reset cpu_kvcache_space_bytes
            if getattr(vllm_config.cache_config, "cpu_kvcache_space_bytes", False):
                logger.warning(
                    "Parameter is tied to incompatible backend. "
                    "parameter=cpu_kvcache_space_bytes, action: resetting to None for Ascend."
                )
                vllm_config.cache_config.cpu_kvcache_space_bytes = None

        # SUBTRACTED: 段3 MultiModal（mm_encoder_attn_backend→None, L1007-1016）、
        #             段6 Speculative（quantization→None, L1038-1047）、
        #             段7 KV Transfer（kv_buffer_size→1e9 / enable_permute_local_kv→False, L1049-1068）。
        #             三段与保留各段同构（getattr 探测→warn→写回安全默认），删减计划批准。

        # ==================== 4. Observability Config ====================
        if vllm_config.observability_config:
            # NVTX tracing is NVIDIA specific
            if getattr(vllm_config.observability_config, "enable_layerwise_nvtx_tracing", False):
                logger.warning(
                    "Parameter relies on NVIDIA-specific tools. "
                    "parameter=enable_layerwise_nvtx_tracing, action: resetting to False."
                )
                vllm_config.observability_config.enable_layerwise_nvtx_tracing = False

        # ==================== 5. Scheduler Config ====================
        if vllm_config.scheduler_config:
            # Partial prefills are specific to ROCm optimization
            if getattr(vllm_config.scheduler_config, "max_num_partial_prefills", 1) != 1:
                logger.warning(
                    "Parameter is optimized for incompatible platform. "
                    "parameter=max_num_partial_prefills, action: resetting to default (1). "
                )
                vllm_config.scheduler_config.max_num_partial_prefills = 1

        # ==================== 8. Attention Config ====================
        if vllm_config.attention_config:
            att_config = vllm_config.attention_config

            # Boolean flags that must be False on Ascend (typically NVIDIA-specific)
            force_false_flags = [
                "use_prefill_decode_attention",
                "use_cudnn_prefill",
                "use_trtllm_ragged_deepseek_prefill",
                "use_trtllm_attention",
                "disable_flashinfer_prefill",
                "disable_flashinfer_q_quantization",
            ]
            for flag in force_false_flags:
                if getattr(att_config, flag, False):
                    logger.warning(
                        "Ignored GPU-specific parameter. parameter=%s, action: resetting to False. ",
                        flag,
                    )
                    setattr(att_config, flag, False)

            # Reset specific values to None as Ascend uses its own internal logic
            if getattr(att_config, "flash_attn_version", None) is not None:
                logger.warning(
                    "Ignored parameter. Ascend uses its own attention backend. "
                    "parameter=flash_attn_version, action: resetting to None. "
                )
                att_config.flash_attn_version = None

            # Notify user that the backend will be managed by Ascend plugins,
            # and for training-inference consistency, when att_config.backend
            # == AttentionBackendEnum.FLASH_ATTN,it is NOT reset to None
            if (
                getattr(att_config, "backend", None) is not None
                and att_config.backend != AttentionBackendEnum.FLASH_ATTN
            ):
                logger.info(
                    "User specified attention backend '%s'. Note that Ascend NPU "
                    "will use its registered plugin backend instead. Resetting to None.",
                    att_config.backend,
                )
                att_config.backend = None

            # CUDA Graph specific split points are not applicable
            if getattr(att_config, "flash_attn_max_num_splits_for_cuda_graph", 32) != 32:
                logger.warning(
                    "Parameter is ignored on Ascend. "
                    "parameter=flash_attn_max_num_splits_for_cuda_graph, action: resetting to default (32). "
                )
                att_config.flash_attn_max_num_splits_for_cuda_graph = 32

        # ==================== 9. Parallel Config ====================
        if vllm_config.parallel_config:
            # SUBTRACTED: ray_workers_use_nsight→False(L1125-1130)、numa_bind_cpus→None(L1159-1165)、
            #             enable_dbo→False(L1167-1171)、ubatch_size→0(L1173-1180) 等结构同构的『丢弃/归零』
            #             分支，删减计划批准。保留 numa_bind：它是唯一『改写非丢弃』特例，串起两条主线。

            # --numa-bind relies on GPU-to-NUMA topology detection which is
            # not supported on Ascend NPU.  Seamlessly replace with the
            # Ascend-native CPU binding via additional_config.
            if getattr(vllm_config.parallel_config, "numa_bind", False):
                vllm_config.parallel_config.numa_bind = False
                if vllm_config.additional_config is None:
                    vllm_config.additional_config = {}
                vllm_config.additional_config.setdefault("enable_cpu_binding", True)
                logger.info(
                    "'--numa-bind' is not supported on Ascend NPU (GPU-to-"
                    "NUMA topology detection unavailable). Automatically "
                    "converted to --additional-config "
                    "'{\"enable_cpu_binding\": true}' for Ascend-native "
                    "CPU-core binding."
                )

            if getattr(vllm_config.parallel_config, "numa_bind_nodes", None):
                logger.info(
                    "'--numa-bind-nodes' is ignored on Ascend NPU. The "
                    "Ascend-native CPU binding automatically performs "
                    "topo-affinity core allocation."
                )
                vllm_config.parallel_config.numa_bind_nodes = None

    # SOURCE: vllm_ascend/platform.py (NPUPlatform._validate_layer_sharding_config)
    @classmethod
    def _validate_layer_sharding_config(cls, vllm_config: "VllmConfig") -> None:
        # SOURCE: vllm_ascend/platform.py (NPUPlatform._validate_layer_sharding_config)
        # SUBTRACTED: 真实现校验 layer_sharding 与 FLASHCOMM2/DSA-CP 的兼容性（一致性校验，非本章主线）。
        return None

    # SOURCE: vllm_ascend/platform.py (NPUPlatform._validate_draft_decode_context_parallel_config)
    @classmethod
    def _validate_draft_decode_context_parallel_config(cls, vllm_config: "VllmConfig") -> None:
        # SOURCE: vllm_ascend/platform.py (NPUPlatform._validate_draft_decode_context_parallel_config)
        # SUBTRACTED: 真实现校验 draft-decode 与 context-parallel 组合（一致性校验，非本章主线）。
        return None
