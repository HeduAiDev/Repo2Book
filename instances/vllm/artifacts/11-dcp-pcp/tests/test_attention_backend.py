"""Tests for attention_backend_dcp_pcp.py — AttentionImplBase.__new__ discovery.

Source: vllm/v1/attention/backend.py:L685-L757
"""

from __future__ import annotations

import pytest

from implementation import parallel_state_dcp_pcp as ps
from implementation.attention_backend_dcp_pcp import (
    AttentionImplBase,
    FlashAttn3MlaBackend,
    FlashAttnBackend,
    FlashInferBackend,
    RocmAiterMlaBackend,
)


@pytest.fixture(autouse=True)
def _reset_singletons():
    ps.reset_cp_singletons()
    yield
    ps.reset_cp_singletons()


# --------------------------------------------------------------------------
# Default flag values per backend (REFERENCE: backend.py:L703 supports_pcp)
# --------------------------------------------------------------------------


def test_supports_pcp_default_false():
    """REFERENCE: vllm/v1/attention/backend.py:L703 — default False."""
    assert AttentionImplBase.supports_pcp is False


def test_supports_mtp_with_cp_default_false():
    """REFERENCE: vllm/v1/attention/backend.py:L705-L706 — default False."""
    assert AttentionImplBase.supports_mtp_with_cp_non_trivial_interleave_size is False


def test_can_return_lse_for_decode_default_false():
    assert AttentionImplBase.can_return_lse_for_decode is False


def test_flashattn3_mla_can_return_lse():
    """MLA backend: can_return_lse_for_decode = True."""
    assert FlashAttn3MlaBackend.can_return_lse_for_decode is True


def test_flashattn_does_not_return_lse():
    assert FlashAttnBackend.can_return_lse_for_decode is False


def test_flashinfer_can_return_lse():
    assert FlashInferBackend.can_return_lse_for_decode is True


def test_rocm_aiter_can_return_lse():
    assert RocmAiterMlaBackend.can_return_lse_for_decode is True


# --------------------------------------------------------------------------
# __new__ discovery WITHOUT CP groups initialised → falls back to size-1, rank-0
# REFERENCE: vllm/v1/attention/backend.py:L731-L757 (try/except AssertionError)
# --------------------------------------------------------------------------


def _make_base():
    """Bypass __init__ which is not defined on AttentionImplBase itself."""
    return AttentionImplBase.__new__(AttentionImplBase)


def test_new_falls_back_to_dcp_size_1_when_uninitialized():
    self = _make_base()
    assert self.dcp_world_size == 1
    assert self.dcp_rank == 0


def test_new_falls_back_to_pcp_size_1_when_uninitialized():
    self = _make_base()
    assert self.pcp_world_size == 1
    assert self.pcp_rank == 0


def test_new_total_cp_world_size_is_1_when_uninitialized():
    self = _make_base()
    assert self.total_cp_world_size == 1
    assert self.total_cp_rank == 0


def test_new_need_to_return_lse_for_decode_false_when_dcp_1():
    """need_to_return_lse_for_decode = (dcp > 1 AND can_return)."""
    self = _make_base()
    assert self.need_to_return_lse_for_decode is False


# --------------------------------------------------------------------------
# __new__ discovery WITH CP groups initialised
# --------------------------------------------------------------------------


def test_new_picks_up_dcp_world_size_after_init():
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    self = _make_base()
    assert self.dcp_world_size == 2


def test_new_picks_up_dcp_rank_after_init():
    ps.initialize_model_parallel(
        rank=3,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    self = _make_base()
    # Rank 3 in DCP sub-group [2, 3] is at index 1.
    assert self.dcp_rank == 1


def test_new_picks_up_pcp_world_size_after_init():
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
    )
    self = _make_base()
    assert self.pcp_world_size == 2


def test_new_picks_up_pcp_rank_after_init():
    ps.initialize_model_parallel(
        rank=2,
        world_size=4,
        tensor_model_parallel_size=2,
        prefill_context_model_parallel_size=2,
    )
    self = _make_base()
    # Rank 2 in PCP group [0, 2] is at index 1.
    assert self.pcp_rank == 1


# --------------------------------------------------------------------------
# total_cp_world_size = pcp_world_size * dcp_world_size (REFERENCE: backend.py:L751)
# --------------------------------------------------------------------------


def test_total_cp_world_size_after_full_init():
    """tp=4, pcp=2, dcp=2 → total_cp = 4."""
    ps.initialize_model_parallel(
        rank=0,
        world_size=8,
        tensor_model_parallel_size=4,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    self = _make_base()
    assert self.total_cp_world_size == 4


def test_total_cp_rank_formula_pcp_major():
    """REFERENCE: vllm/v1/attention/backend.py:L752

    total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
    """
    ps.initialize_model_parallel(
        rank=0,
        world_size=8,
        tensor_model_parallel_size=4,
        prefill_context_model_parallel_size=2,
        decode_context_model_parallel_size=2,
    )
    self = _make_base()
    assert self.total_cp_rank == self.pcp_rank * self.dcp_world_size + self.dcp_rank


# --------------------------------------------------------------------------
# need_to_return_lse_for_decode = (dcp > 1 AND can_return_lse)
# REFERENCE: vllm/v1/attention/backend.py:L754-L756
# --------------------------------------------------------------------------


def test_need_to_return_lse_when_dcp_2_and_can_return():
    """MLA backend with dcp>1 → need_to_return_lse_for_decode True."""
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    assert backend.need_to_return_lse_for_decode is True


def test_no_need_to_return_lse_when_dcp_1():
    """dcp=1 → no LSE needed for decode (no cross-rank combine)."""
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
    )
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    assert backend.need_to_return_lse_for_decode is False


def test_no_need_to_return_lse_when_backend_cant_return():
    """FlashAttnBackend can_return=False → no need even at dcp>1."""
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    backend = AttentionImplBase.__new__(FlashAttnBackend)
    assert backend.need_to_return_lse_for_decode is False


# --------------------------------------------------------------------------
# MLA num_heads_q = num_heads * dcp_world_size (REFERENCE: flashattn_mla.py:L175)
# --------------------------------------------------------------------------


def test_mla_num_heads_q_replicated_at_dcp_1():
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    assert backend.num_heads_q == 8


def test_mla_num_heads_q_replicated_at_dcp_2():
    """REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L175

    Q replicated num_heads * dcp_world_size when dcp>1.
    """
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    assert backend.num_heads_q == 8 * 2


# --------------------------------------------------------------------------
# MLA kernel call signature exposes the right wired fields
# REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L353-L355
# --------------------------------------------------------------------------


def test_mla_kernel_signature_exposes_cp_world_size():
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    sig = backend.kernel_call_signature()
    assert sig["cp_world_size"] == 2


def test_mla_kernel_signature_exposes_cp_rank():
    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
    )
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    sig = backend.kernel_call_signature()
    assert sig["cp_rank"] == 0


def test_mla_kernel_signature_fa_version_3():
    """REFERENCE: flashattn_mla.py:L350 — only V3 supports DCP."""
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    sig = backend.kernel_call_signature()
    assert sig["fa_version"] == 3


def test_mla_kernel_signature_return_softmax_lse_field():
    backend = FlashAttn3MlaBackend(num_heads=8, head_dim=128)
    sig = backend.kernel_call_signature()
    assert "return_softmax_lse" in sig


# --------------------------------------------------------------------------
# All backends have supports_pcp = False at this commit
# REFERENCE: knowledge fact D10 (PCP wiring TBD on most backends)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [
    FlashAttnBackend,
    FlashAttn3MlaBackend,
    FlashInferBackend,
    RocmAiterMlaBackend,
])
def test_no_backend_supports_pcp_yet(cls):
    """Per knowledge D10: supports_pcp default False; only some backends override later."""
    assert cls.supports_pcp is False


# --------------------------------------------------------------------------
# Subclass instantiation through __new__ (subclasses inherit the mechanism)
# --------------------------------------------------------------------------


def test_flashinfer_inherits_new_discovery():
    backend = AttentionImplBase.__new__(FlashInferBackend)
    assert backend.dcp_world_size == 1
    assert backend.total_cp_world_size == 1


def test_rocm_aiter_inherits_new_discovery():
    backend = AttentionImplBase.__new__(RocmAiterMlaBackend)
    assert backend.total_cp_rank == 0


def test_flashattn_inherits_new_discovery():
    backend = AttentionImplBase.__new__(FlashAttnBackend)
    assert backend.pcp_world_size == 1


# --------------------------------------------------------------------------
# Discovery is per-instance — each __new__ call snapshots singleton state
# --------------------------------------------------------------------------


def test_discovery_reflects_state_at_new_call_time():
    """If singletons change between __new__ calls, the new instance reflects new state."""
    self_uninit = _make_base()
    assert self_uninit.dcp_world_size == 1

    ps.initialize_model_parallel(
        rank=0,
        world_size=4,
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=4,
    )
    self_init = _make_base()
    assert self_init.dcp_world_size == 4
