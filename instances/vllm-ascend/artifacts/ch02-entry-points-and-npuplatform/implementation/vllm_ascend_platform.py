"""Subtract-only companion — NPUPlatform：OOT 平台的‘运行期分发总台’。

规范源码：vllm_ascend/platform.py

NPUPlatform 继承 vLLM 的 Platform，做两件事：
  (1) 用‘身份替换类属性’宣告‘我是 npu 设备’（device_name/dispatch_key/_enum=OOT …）；
  (2) 用一批返回 qualname 字符串的 classmethod 工厂钩子，把 attention/communicator/
      worker/compile 等具体类顶替成昇腾实现——和平台选择同一招‘写字符串、推迟 import’。

注：跨文件 import 路径已由真实的 `from vllm.platforms import Platform, PlatformEnum`
等改写成本目录的扁平模块名，符号与控制流保持一致。
"""
# 真实为 `from vllm.platforms import Platform, PlatformEnum`
from vllm_interface import Platform, PlatformEnum, AttentionBackendEnum
# 真实为 `from vllm_ascend.utils import (ASCEND_QUANTIZATION_METHOD, …, is_310p)`
from vllm_ascend_utils import (
    ASCEND_QUANTIZATION_METHOD,
    COMPRESSED_TENSORS_METHOD,
    FP8_METHOD,
    is_310p,
)


# SOURCE: vllm_ascend/ascend_config.py（init_ascend_config）
def init_ascend_config(vllm_config):
    # SUBTRACTED: 原函数从 vllm_config.additional_config 解析出 AscendConfig 单例（含
    #   xlite_graph_config 等）。完整解析属 ch10/config 章；精简版只取出 worker_cls 分流需要
    #   的那个对象，由调用方挂在 vllm_config.ascend_config 上。原 vllm_ascend/ascend_config.py
    return vllm_config.ascend_config


# SOURCE: vllm_ascend/platform.py:L134-L151
class NPUPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "npu"
    device_type: str = "npu"
    simple_compile_backend: str = "eager"  # Disable torch.compile()
    ray_device_key: str = "NPU"
    device_control_env_var: str = "ASCEND_RT_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
    ]
    dispatch_key: str = "PrivateUse1"

    supported_quantization: list[str] = [
        ASCEND_QUANTIZATION_METHOD,
        COMPRESSED_TENSORS_METHOD,
        FP8_METHOD,
        "deepseek_v4_fp8",
    ]

    # SUBTRACTED: is_sleep_mode_available / pass_key / pre_register_and_update 等‘身份/能力’
    #   方法，以及 npu-smi/HBM/get_device_* 等需真 NPU 的方法体；check_and_update_config 上半段
    #   （ACL graph 校验/init_ascend_config 全量解析）属 ch10。原 vllm_ascend/platform.py:L152~

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        # SOURCE: vllm_ascend/platform.py:L164-L172
        """
        Get the pass manager class for this platform.
        """
        return "vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager"

    @classmethod
    def get_compile_backend(self) -> str:
        # SOURCE: vllm_ascend/platform.py:L174-L180
        """
        Get the custom compile backend.
        """
        return "vllm_ascend.compilation.compiler_interface.AscendCompiler"

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        # SOURCE: vllm_ascend/platform.py:L602-L612（截取自 check_and_update_config 方法体）
        # SUBTRACTED: 方法体绝大部分（ACL graph / ASCEND_LAUNCH_BLOCKING 校验、refresh_block_size、
        #   内存与并行配置等）属 ch10/config 章。本章只取 worker_cls 从 'auto' 改写成 NPUWorker
        #   qualname 这一处——同样是‘写 qualname 字符串、推迟 import’，但落点在 config 而非工厂方法。
        ascend_config = init_ascend_config(vllm_config)
        parallel_config = vllm_config.parallel_config
        if parallel_config and parallel_config.worker_cls == "auto":
            # TODO: this is a tricky way to disable `use_sequence_parallel_moe` in vllm.
            if not vllm_config.compilation_config.pass_config.enable_sp:
                parallel_config.all2all_backend = "flashinfer_all2allv"
            if is_310p():
                parallel_config.worker_cls = "vllm_ascend._310p.worker_310p.NPUWorker310"
            elif ascend_config.xlite_graph_config.enabled:
                # SUBTRACTED: logger.info("openEuler Xlite enabled. ...")
                parallel_config.worker_cls = "vllm_ascend.xlite.xlite_worker.XliteWorker"
            else:
                parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                             num_heads: int | None = None):
        # SOURCE: vllm_ascend/platform.py:L738-L764
        use_compress = getattr(attn_selector_config, "use_compress", False)
        key = (attn_selector_config.use_mla, attn_selector_config.use_sparse)

        if selected_backend == AttentionBackendEnum.FLASH_ATTN and cls._validate_fa3_backend(key, attn_selector_config):
            return "vllm_ascend.attention.fa3_v1.AscendFABackend"

        backend_map = {
            (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
            (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            (True, True, False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
            (True, False, True): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
        }
        backend_map_310 = {
            (
                False,
                False,
            ): "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310",
            # SUBTRACTED: 原有两行 TODO 注释（MLA/SFA 的 310P 实现待补）。
        }

        if is_310p():
            return backend_map_310.get(key, backend_map_310[(False, False)])

        return backend_map[(attn_selector_config.use_mla, attn_selector_config.use_sparse, use_compress)]

    @classmethod
    def _validate_fa3_backend(cls, key, attn_selector_config):
        # SOURCE: vllm_ascend/platform.py:L766-L793
        # SUBTRACTED: 完整校验体（use_batch_invariant 判定、flash_attn_npu_v3 探测/import_module、
        #   flash_attn_with_kvcache 检查）。FA3 是训推一致特例，依赖外部包 flash_attn_npu_v3
        #   （host 无）。降为返回 False 占位：让 get_attn_backend_cls 走查表主干。
        #   原 vllm_ascend/platform.py:L766-L793
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # SOURCE: vllm_ascend/platform.py:L795-L796
        return "vllm_ascend.lora.punica_npu.PunicaWrapperNPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # SOURCE: vllm_ascend/platform.py:L803-L805
        return "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        # SOURCE: vllm_ascend/platform.py:L815-L820
        """
        Get piecewise backend class for piecewise graph.
        """
        return "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"  # noqa
