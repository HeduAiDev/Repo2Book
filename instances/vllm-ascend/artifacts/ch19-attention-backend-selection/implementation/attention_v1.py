# vllm_ascend/attention/attention_v1.py —— subtract-only 精简版（OOT 后端契约的昇腾实现）
#
# 本章主角：AscendAttentionBackend —— 标准 MHA 后端，也是「OOT 后端怎么实现/伪装 vLLM
# 契约点」的样板。本文件只摘 L73-L140 这段「选择与契约方法」，逐点对账：
#   (2) 注册：@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND") 占住 CUSTOM 槽。
#   (3) 伪装：get_name() 在 V2 model-runner 下**故意返回 "FLASH_ATTN"** 绕过 vLLM 按名字判定后端的断言。
#   (4) 静态契约：get_kv_cache_shape（v1 @abstractmethod）+ get_supported_kernel_block_sizes（覆写基类默认）。
#       swap_blocks/copy_blocks 是昇腾自带、v0 遗留方法，**非** v1 基类契约要求。
#   (5) 运行期分流：get_impl_cls/get_builder_cls 按 enable_cp() 二选一并延迟 import CP 实现。
#
# 真实文件顶部有 torch_npu/acl_graph/flashcomm2/kvcomp/device_op 等数十行 import 与
# AscendAttentionBackendImpl(L357+)、AscendMetadata 等算子/元数据实体 —— 那些是 ch19 MHA 的内容，
# 本章只要契约骨架，整体折叠。get_impl/builder 返回的 Impl/Builder 这里用「占位类替身」承接控制流。
import torch
import vllm.envs as envs_vllm

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from vllm_ascend.attention.utils import enable_cp

# SUBTRACTED: torch_npu 与 acl_graph/flashcomm2/kvcomp/device_op/attention_mask 等数十行 import
#   （attention_v1.py:L18-L66）—— 全是 ch19 MHA 算子/元数据的依赖，本章契约骨架不触达。


# SUBTRACTED: AscendAttentionBackendImpl（attention_v1.py:L357+，含 forward 算子）与
#   AscendAttentionMetadataBuilder 的真实定义 —— ch19 MHA 内容；这里用占位类承接
#   get_impl_cls/get_builder_cls 的返回，仅保留「按 enable_cp 二选一」的控制流。
class AscendAttentionBackendImpl:  # 占位替身：真身（forward 算子）留 ch19。
    # SOURCE: vllm_ascend/attention/attention_v1.py:L357（占位替身，真身留 ch19 MHA）
    pass


class AscendAttentionMetadataBuilder:  # 占位替身：真身留 ch19。
    # SOURCE: vllm_ascend/attention/attention_v1.py:L357+（占位替身，真身留 ch19 MHA）
    pass


# SUBTRACTED: AscendAttentionCPImpl / AscendAttentionCPMetadataBuilder 的真实定义
#   （vllm_ascend/attention/context_parallel/attention_cp.py）—— 上下文并行子系统（ch15 延伸）；
#   这里用占位替身，仅保留 get_impl/builder 在 enable_cp() 为 True 时「延迟 import 并返回 CP 实现」的控制流。
class AscendAttentionCPImpl:  # 占位替身：CP 内核留后续。
    # SOURCE: vllm_ascend/attention/context_parallel/attention_cp.py（占位替身，CP 内核留后续）
    pass


class AscendAttentionCPMetadataBuilder:  # 占位替身：CP 内核留后续。
    # SOURCE: vllm_ascend/attention/context_parallel/attention_cp.py（占位替身，CP 内核留后续）
    pass


# SOURCE: vllm_ascend/attention/attention_v1.py:L73-L140
@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    # SOURCE: vllm_ascend/attention/attention_v1.py:L73-L140
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L77-L82
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L84-L90
        if enable_cp():
            # 运行期才 import：CP 实现重且仅 CP 场景需要，命中才付加载成本。
            # SUBTRACTED: 原为 `from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl`
            #   —— 精简版用本文件顶部的占位替身（CP 内核留后续），仅保留延迟选择的控制流。
            return AscendAttentionCPImpl

        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L92-L98
        if enable_cp():
            # SUBTRACTED: 原为 `from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder`
            #   —— 同上，占位替身承接控制流。
            return AscendAttentionCPMetadataBuilder

        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L100-L108
        # 首维 2 = key/value 两半合存一张；swap/copy_blocks 都据此取 [0]/[1]。
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    # ↓↓↓ swap_blocks/copy_blocks：昇腾自带、v0 遗留接口，**非** v1 基类契约要求 ↓↓↓
    # （基座 vllm/v1/attention/ 全目录无此二方法的 def，也无任何调用）
    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L110-L122
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        # SUBTRACTED: 原右侧带 `.to(dst_key_cache.device)`（attention_v1.py:L121-L122）——
        #   真实 NPU 设备搬运 host 不可跑；精简版用 CPU 张量同设备索引复制验证「按 (2,...) 布局块级搬运」的控制流。
        dst_key_cache[dst_indices] = src_key_cache[src_indices]
        dst_value_cache[dst_indices] = src_value_cache[src_indices]

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L124-L136
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # SOURCE: vllm_ascend/attention/attention_v1.py:L138-L140
        # 覆写基类默认（基座 [MultipleOf(1)]）→ 昇腾返回 [128]。
        return [128]
