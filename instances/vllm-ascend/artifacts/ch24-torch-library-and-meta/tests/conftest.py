"""Host scaffold for ch24 runnable tests (NOT part of the subtract-only source).

The two implementation files are faithful copies of vllm_ascend/meta_registration.py and
vllm_ascend/ops/register_custom_ops.py. To exercise them on a plain host (no NPU, no vllm /
torch_npu installed), we inject the modules they import as lightweight stubs into sys.modules
BEFORE importing them. The stub `direct_register_custom_op` mirrors the *base vLLM* behaviour
(vllm/utils/torch_utils.py: infer_schema -> define -> impl -> _register_fake) so registration
and fake shape-inference genuinely happen via torch.library — that is the observable behaviour
the tests assert.
"""
import sys
import types
from pathlib import Path

import torch
from torch.library import Library, infer_schema

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL_DIR))


def _mod(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# ---- base-vLLM-faithful direct_register_custom_op (vllm.utils.torch_utils) ----
_vllm_lib = Library("vllm", "FRAGMENT")


def direct_register_custom_op(
    op_name, op_func, mutates_args=None, fake_impl=None, target_lib=None, dispatch_key=None, tags=()
):
    if mutates_args is None:
        mutates_args = []
    if dispatch_key is None:
        dispatch_key = "PrivateUse1"
    schema_str = infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or _vllm_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)


# ---- stub: torch_npu ----
_torch_npu = _mod("torch_npu")


def _npu_quantize(in_tensor, scale, offset, dtype, axis, flag):
    return torch.empty(in_tensor.shape, dtype=torch.int8, device=in_tensor.device)


_torch_npu.npu_quantize = _npu_quantize
_npu_ns = types.SimpleNamespace(current_stream=lambda: None)
_torch_npu.npu = _npu_ns

# ---- stub: vllm.* ----
_mod("vllm")
_dist = _mod("vllm.distributed")
for _n in (
    "get_dp_group",
    "get_ep_group",
    "get_tensor_model_parallel_rank",
    "tensor_model_parallel_all_gather",
    "tensor_model_parallel_reduce_scatter",
):
    setattr(_dist, _n, lambda *a, **k: None)
_dist.get_tensor_model_parallel_world_size = lambda *a, **k: 1
_dist.get_tensor_model_parallel_rank = lambda *a, **k: 0
_dist.tensor_model_parallel_all_reduce = lambda x: x

_fc = _mod("vllm.forward_context")


def _get_forward_context():
    raise AssertionError("no forward context on host")


_fc.get_forward_context = _get_forward_context

_mod("vllm.utils")
_tu = _mod("vllm.utils.torch_utils")
_tu.direct_register_custom_op = direct_register_custom_op

# ---- stub: vllm_ascend.* ----
_mod("vllm_ascend")
_va_utils = _mod("vllm_ascend.utils")
_va_utils.is_310p = lambda: False
_va_utils.enable_sp_by_pass = lambda: False
_va_utils.is_vl_model = lambda: False
_va_utils.npu_stream_switch = lambda *a, **k: None
_va_utils.prefetch_stream = lambda *a, **k: None

_afc = _mod("vllm_ascend.ascend_forward_context")
_afc._EXTRA_CTX = types.SimpleNamespace(
    flash_comm_v1_enabled=False, pad_size=0, padded_length=0, moe_comm_type=None
)
_afc.MoECommType = types.SimpleNamespace(ALLTOALL="alltoall", MC2="mc2", FUSED_MC2="fused_mc2")

_mod("vllm_ascend.ops")
_rope = _mod("vllm_ascend.ops.rotary_embedding")


def rope_forward_oot(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


_rope.rope_forward_oot = rope_forward_oot

_mod("vllm_ascend.ops.triton")
_muls = _mod("vllm_ascend.ops.triton.muls_add")


def muls_add_triton(x: torch.Tensor, y: torch.Tensor, scale: float) -> torch.Tensor:
    return x + y * scale


_muls.muls_add_triton = muls_add_triton

_wp = _mod("vllm_ascend.ops.weight_prefetch")
_wp.maybe_npu_prefetch = lambda *a, **k: None

# ---- simulate the C++ .so having DEF'd the _C_ascend ops that meta_registration补 meta ----
# (on a real device these come from `import vllm_ascend.vllm_ascend_C` loading the compiled .so)
_C_ascend_def = Library("_C_ascend", "DEF")
_C_ascend_def.define(
    "get_masked_input_and_mask(Tensor input, int org_vocab_start_index, int org_vocab_end_index, "
    "int num_org_vocab_padding, int added_vocab_start_index, int added_vocab_end_index) "
    "-> (Tensor masked_input, Tensor mask)"
)
_C_ascend_def.define(
    "bgmv_expand(Tensor x, Tensor weight, Tensor indices, Tensor y, int slice_offset, int slice_size) -> Tensor"
)
_C_ascend_def.define(
    "sgmv_expand(Tensor x, Tensor weight, Tensor lora_indices, Tensor seq_len, Tensor y, "
    "int slice_offset, int slice_size) -> Tensor"
)
