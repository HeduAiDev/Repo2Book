"""Pedagogical mirror of vLLM's per-attention-backend CP discovery.

In vLLM, every attention backend extends ``AttentionImplBase`` and
inherits the ``__new__`` method that discovers DCP and PCP rank info
from the singletons. The pattern lets attention backends written
*before* DCP/PCP existed work unchanged when CP is enabled — they
read ``self.dcp_world_size`` etc. without needing to know about the
groups themselves.

Source pattern (verbatim semantics):

.. code-block:: python

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        try:
            from vllm.distributed.parallel_state import get_dcp_group
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            self.dcp_world_size = 1
            self.dcp_rank = 0
        # ... same for PCP ...
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        return self

The ``except AssertionError`` lets unit tests instantiate attention
without setting up the distributed groups.
"""

from __future__ import annotations

from .parallel_state_dcp_pcp import get_dcp_group, get_pcp_group


class AttentionImplBase:
    """Pedagogical mirror of ``vllm/v1/attention/backend.py:L685-L757``.

    Subclasses set ``supports_pcp`` (per-backend feature flag) and inherit
    ``__new__`` discovery. When CP groups are not initialised (testing),
    DCP/PCP fall back to size-1 / rank-0.
    """

    # REFERENCE: vllm/v1/attention/backend.py:L703
    supports_pcp: bool = False

    # REFERENCE: vllm/v1/attention/backend.py:L705-L706
    # Cross-link to Ch10 MTP — explicit knob for MTP+CP interaction.
    supports_mtp_with_cp_non_trivial_interleave_size: bool = False

    # REFERENCE: vllm/v1/attention/backend.py:L700
    can_return_lse_for_decode: bool = False

    # REFERENCE: vllm/v1/attention/backend.py:L722-L729
    dcp_world_size: int
    dcp_rank: int
    pcp_world_size: int
    pcp_rank: int
    total_cp_world_size: int
    total_cp_rank: int
    need_to_return_lse_for_decode: bool

    def __new__(cls, *args, **kwargs):
        # REFERENCE: vllm/v1/attention/backend.py:L731-L757
        self = super().__new__(cls)

        # Discover DCP from singleton (or fall back).
        try:
            dcp = get_dcp_group()
            self.dcp_world_size = dcp.world_size
            self.dcp_rank = dcp.rank_in_group
        except AssertionError:
            # DCP not initialized in unit tests.
            self.dcp_world_size = 1
            self.dcp_rank = 0

        # Discover PCP from singleton (or fall back).
        try:
            pcp = get_pcp_group()
            self.pcp_world_size = pcp.world_size
            self.pcp_rank = pcp.rank_in_group
        except AssertionError:
            self.pcp_world_size = 1
            self.pcp_rank = 0

        # REFERENCE: vllm/v1/attention/backend.py:L751
        # Composed CP world size and rank — wired into every attention
        # backend in the codebase.
        self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        # REFERENCE: vllm/v1/attention/backend.py:L752
        self.total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

        # REFERENCE: vllm/v1/attention/backend.py:L754-L756
        # LSE return is only required when DCP > 1 AND backend can return it.
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.can_return_lse_for_decode
        )
        return self


class FlashAttn3MlaBackend(AttentionImplBase):
    """Mirror of the MLA backend's CP fields.

    REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L125-L355

    Real backend wires ``cp_world_size``, ``cp_rank``, and
    ``cp_tot_seqused_k`` into the FlashAttention V3 kernel call at
    L353-L355. We don't run the kernel; we expose the wiring so the
    chapter can quote the exact field names.
    """

    # REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L125 et seq.
    can_return_lse_for_decode = True
    supports_pcp = False  # MLA does not yet expose PCP wiring directly.

    def __init__(self, num_heads: int, head_dim: int) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        # REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L175
        # When DCP > 1, Q is replicated num_heads * dcp_world_size times.
        self.num_heads_q = num_heads * self.dcp_world_size

    def kernel_call_signature(self) -> dict:
        """Show the per-call args wired to FlashAttention V3.

        REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L353-L355
        REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L349 (return_softmax_lse=self.need_to_return_lse_for_decode)
        REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L350 (fa_version=3 only V3 supports DCP)
        """
        return {
            "cp_world_size": self.dcp_world_size,
            "cp_rank": self.dcp_rank,
            "return_softmax_lse": self.need_to_return_lse_for_decode,
            "fa_version": 3,
        }


class FlashAttnBackend(AttentionImplBase):
    """A backend that supports neither DCP-aware kernels nor PCP.

    REFERENCE: vllm/v1/attention/backends/flash_attn.py (DCP path gated at decode_context_parallel_size > 1)
    """

    can_return_lse_for_decode = False
    supports_pcp = False


class FlashInferBackend(AttentionImplBase):
    """Mirror of the flashinfer backend.

    REFERENCE: vllm/v1/attention/backends/flashinfer.py:L213 (BatchDCPPrefillWrapper — only DCP-prefixed class)
    REFERENCE: vllm/v1/attention/backends/flashinfer.py:L444 (BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper)
    REFERENCE: vllm/v1/attention/backends/flashinfer.py:L763-L766 (_prefill_wrapper = BatchDCPPrefillWrapper(...))

    The flashinfer-specific helper class ``BatchDCPPrefillWrapper`` at
    L213 is the **only** DCP-prefixed class anywhere in vLLM. It is a
    backend-internal helper, NOT a top-level CP orchestrator. This is
    the "no class X" reframe (5th instance).
    """

    can_return_lse_for_decode = True
    supports_pcp = False


class RocmAiterMlaBackend(AttentionImplBase):
    """ROCm AITER MLA backend — DCP integration verified.

    REFERENCE: vllm/v1/attention/backends/mla/rocm_aiter_mla.py:L213 (DCP wiring)
    REFERENCE: vllm/v1/attention/backends/mla/rocm_aiter_mla.py:L311 (DCP forward path)
    """

    can_return_lse_for_decode = True
    supports_pcp = False
