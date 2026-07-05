# vllm_ascend/device/device_op.py —— subtract-only 精简版（设备算子门面 = 注意力各章共用的多态层）
#
# 章节立意(4)：reshape_and_cache / 稀疏内核选择 / KV cache 解包等算子，在不同昇腾代际（A2/A3 vs A5）
# 上签名不同。device_op 用「门面 + 按设备代际多态」把这些差异封进一处：
#   - BaseDeviceAdaptor：A2/A3 主路实现；
#   - A5DeviceAdaptor(BaseDeviceAdaptor)：A5 代际对部分方法的 override；
#   - 模块级 DeviceOperator = get_device_adaptor() 在 import 期一次性按 AscendDeviceType 选好类，
#     于是 sfa_v1 / dsa_v1 / mla_v1 都写设备无关的控制流，调 DeviceOperator.xxx 即按代际派发。
#
# 本章只保留 SFA/DSA 稀疏注意力会经门面调到的方法（top-k 选择、稀疏 flash、KV cache 解包/scatter）。
# host 无 CANN/torch_npu：真实 torch.ops._C_ascend.* / torch_npu.* 算子由测试的「记录调用」替身承接，
# 只验派发/入参（sparse_count / sparse_mode / sparse_indices 等），不真算（昇腾才有内核）。
#
# SUBTRACTED: 本文件原 1670 行覆盖 MoE 路由 / MLA prolog / RMSNorm / rope / flash 等几十个门面方法；
#   只保留 ch21 稀疏注意力链路触达的方法。原 vllm_ascend/device/device_op.py:L42-L1670。
from typing import Any

import torch
import torch_npu

# SUBTRACTED: import F / triton kernels / QUANT_DTYPES / QuantType（服务已减的量化与 triton 旁路）——
#   原 device_op.py:L21,L23,L25-L33,L36-L39。triton_q_rms 在 host 无 triton 时为 None（见 apply_dsa_q_rms）。
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

triton_q_rms = None  # SUBTRACTED: HAS_TRITON 分支——host 无 triton，走 apply_dsa_q_rms 的纯 torch 回退


# SOURCE: vllm_ascend/device/device_op.py:L42
class BaseDeviceAdaptor:
    # ===== KV Cache 写入（注意力各章共用，回指 ch19/ch20）=====
    @classmethod
    # SOURCE: vllm_ascend/device/device_op.py:L44
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu._npu_reshape_and_cache(
            key=key, value=value, key_cache=key_cache, value_cache=value_cache, slot_indices=slot_mapping
        )

    # ===== SFA：Lightning Indexer 选 top-k（sparse_count=2048）=====
    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L371
    def indexer_select_post_process(
        sfa_impl,
        q_li: torch.Tensor,
        q_li_scale: "torch.Tensor | None",
        q_li_shape_ori: "tuple[Any, ...] | None",
        weights: torch.Tensor,
        kv_cache: tuple,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        use_sparse_c8_indexer: bool,
        use_torch_npu_lightning_indexer: bool,
    ) -> torch.Tensor:
        # SUBTRACTED: use_sparse_c8_indexer(npu_lightning_indexer_quant) 与 use_torch_npu_lightning_indexer
        #   (torch_npu.npu_lightning_indexer) 两条等价旁支——仅为不同代际/INT8 量化索引；保 default 主路。
        #   原 device_op.py:L387-L420。三条都 sparse_count=2048, sparse_mode=3。
        topk_indices, _ = torch.ops._C_ascend.npu_lightning_indexer(
            query=q_li,
            key=kv_cache[2],
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=attn_metadata.block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )
        return topk_indices

    # ===== SFA：稀疏 flash 注意力（只对 top-k 个 KV 算）=====
    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L437
    def execute_sparse_flash_attention_process(
        sfa_impl,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple,
        topk_indices: torch.Tensor,
        attn_metadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        block_table = attn_metadata.block_table
        kv = kv_cache[0]
        key_rope = kv_cache[1]

        # 核心：sparse_indices=topk_indices —— 内核只对索引器选出的 top-k 个 KV 位置算全精度注意力。
        attn_output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            scale_value=sfa_impl.scale,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=key_rope,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
        )
        return attn_output

    # ===== DSA：稀疏注意力元数据 / 内核 选择器 =====
    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L504
    def get_dsa_sparse_attn_metadata_op():
        """Returns the metadata-building operator for sparse attention."""
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L509
    def get_dsa_sparse_attn_metadata_kwargs(device):
        """Returns kwargs for sparse attention metadata builder."""
        return {"device": str(device)}

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L514
    def get_dsa_sparse_attn_op():
        """Returns the sparse attention operator."""
        return torch.ops._C_ascend.npu_sparse_attn_sharedkv

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L519
    def get_dsa_sparse_attn_base_kwargs():
        """Returns base kwargs for sparse attention (extended by caller)."""
        return {}

    # ===== DSA：SWA / Compressor / Indexer KV scatter =====
    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L526
    def dsa_kv_compress_scatter(cache, x, slot_mapping):
        """Scatter KV into cache. Non-A5: simple scatter of pre-quantized tensor."""
        torch.ops._C_ascend.npu_scatter_nd_update_v2(cache, slot_mapping, x)

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L541
    def indexer_quant_scatter(q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, slot_mapping):
        """Quantize q and scatter kv into indexer cache.
        Non-A5: int8 quant + 2x scatter_nd_update_v2 for k_cache and scale_cache."""
        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=torch.int8)
        q_scale = q_scale.to(torch.float16)

        kv_out = kv
        kv_scale_out = None
        if kv is not None:
            kv_out, kv_scale_out = torch_npu.npu_dynamic_quant(kv, dst_type=torch.int8)
            kv_scale_out = kv_scale_out.unsqueeze(-1).to(torch.float16)
            if kv_scale_out.ndim < 4:
                kv_scale_out = kv_scale_out.unsqueeze(-1)
            torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_k_cache, slot_mapping, kv_out)
            torch.ops._C_ascend.npu_scatter_nd_update_v2(indexer_scale_cache, slot_mapping, kv_scale_out)

        return q, q_scale, kv_out, kv_scale_out

    # ===== DSA：Lightning Indexer 入参 dtype 预处理 =====
    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L597
    def prepare_dsa_indexer_weights(weights):
        """Non-A5: cast indexer weights to float16."""
        return weights.to(torch.float16)

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L602
    def prepare_dsa_indexer_query_scale(q_scale):
        """Non-A5: q_scale already float16, pass through."""
        return q_scale

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L607
    def prepare_dsa_indexer_key_scale(indexer_scale_cache):
        """Non-A5: cast key dequant scale to float16."""
        return indexer_scale_cache.squeeze(-2).to(torch.float16)

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L614
    def apply_dsa_q_rms(q, eps, q_norm_without_weight=None):
        """Apply Q RMS norm. Non-A5: triton_q_rms.
        A5: uses q_norm_without_weight callable when provided."""
        if triton_q_rms is not None:
            return triton_q_rms(q, eps)
        else:
            dtype = q.dtype
            q = q.float()
            variance = q.square().mean(-1, keepdim=True)
            q = q * torch.rsqrt(variance + eps)
            return q.to(dtype)

    # ===== DSA：KV cache 解包 =====
    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L629
    def unpack_dsa_indexer_kv_cache(kv_cache):
        """Unpack indexer kv_cache tuple.
        Non-A5: returns (state_cache, k_cache, scale_cache, None)."""
        _, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache = kv_cache
        return indexer_state_cache, indexer_k_cache, indexer_scale_cache, None

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L637
    def unpack_dsa_forward_kv_cache(kv_cache, compress_ratio):
        """Unpack kv_cache for forward pass.
        Returns 6-tuple: (compress_kv_cache, swa_kv_cache, state_cache,
        indexer_k_cache, indexer_scale_cache, indexer_full_cache)."""
        idx_full = 6  # 7th element (indexer_full_cache), A5 only
        full_cache = kv_cache[idx_full] if len(kv_cache) > idx_full else None
        if compress_ratio == 4:
            # [0]=compress, [1]=swa, [2]=state, [3]=unused, [4]=ik, [5]=isc
            return (kv_cache[0], kv_cache[1], kv_cache[2], kv_cache[4], kv_cache[5], full_cache)
        elif compress_ratio == 128:
            return (kv_cache[0], kv_cache[1], kv_cache[2], None, None, full_cache)
        else:
            return (None, kv_cache[1], None, None, None, full_cache)

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L660
    def format_dsa_slot_mapping(slot_mapping, block_size):
        """Format slot_mapping for metadata storage.
        Non-A5: 2D [block_idx, offset]; A5: 1D pass-through."""
        return torch.stack([slot_mapping // block_size, slot_mapping % block_size], axis=-1)

    @staticmethod
    # SOURCE: vllm_ascend/device/device_op.py:L672
    def add_dsa_sparse_attn_extra_kwargs(extra_kwargs, **kwargs_to_add):
        """Non-A5: add extra kwargs for sparse attention. A5: no-op."""
        extra_kwargs.update(kwargs_to_add)


# SOURCE: vllm_ascend/device/device_op.py:L785
class A5DeviceAdaptor(BaseDeviceAdaptor):
    # A5 代际：同一门面方法换成 A5 专属算子（这里以 reshape_and_cache 为例，其余 override 同理）。
    @classmethod
    # SOURCE: vllm_ascend/device/device_op.py:L787
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        torch_npu.npu_scatter_pa_kv_cache(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping.contiguous(),
            cache_mode="Norm",
        )

    # SUBTRACTED: A5DeviceAdaptor 其余几十个 override（indexer_quant_scatter / unpack_* / apply_dsa_q_rms …）——
    #   A5 代际算子分叉，与「门面按代际多态」立意无关的体量；正文一句话点到。原 device_op.py:L797-L1660。


# SOURCE: vllm_ascend/device/device_op.py:L1663
def get_device_adaptor() -> type["BaseDeviceAdaptor"]:
    ascend_device_type = get_ascend_device_type()
    if ascend_device_type == AscendDeviceType.A5:
        return A5DeviceAdaptor
    return BaseDeviceAdaptor


# 模块级单例：import 期一次性按当前设备代际选定门面类。sfa/dsa/mla 都拿这个 DeviceOperator 派发。
# SOURCE: vllm_ascend/device/device_op.py:L1670
DeviceOperator: type["BaseDeviceAdaptor"] = get_device_adaptor()
