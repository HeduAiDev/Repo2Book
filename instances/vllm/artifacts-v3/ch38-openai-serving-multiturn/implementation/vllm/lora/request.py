# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/lora/request.py —— 逐字承载（73 行全量，无删改）：
# LoRARequest 经 _check_model/_maybe_get_adapters（BaseServing）与
# abort_requests 的 lora_states 清理路径消费。
import msgspec


# SOURCE: vllm/lora/request.py:L8-L73
class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    Request for a LoRA adapter.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.

    load_inplace: If True, forces reloading the adapter even if one
        with the same lora_int_id already exists in the cache. This replaces
        the existing adapter in-place. If False (default), only loads if the
        adapter is not already loaded.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False
    is_3d_lora_weight: bool = False
    """Whether this adapter's MoE weights are stored in the 3D fused
    `gate_up_proj` / `down_proj` layout (one fused tensor per expert) or the
    2D per-expert split layout (separate `gate_proj` / `up_proj` /
    `down_proj` tensors per expert). Only consulted when the engine is
    started with `enable_mixed_moe_lora_format=True`; otherwise the
    on-disk format is inferred from the base model."""

    # SOURCE: vllm/lora/request.py:L43-L49
    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        # Ensure lora_path is not empty
        assert self.lora_path, "lora_path cannot be empty"

    # SOURCE: vllm/lora/request.py:L51-L52
    @property
    def adapter_id(self):
        return self.lora_int_id

    # SOURCE: vllm/lora/request.py:L54-L56
    @property
    def name(self):
        return self.lora_name

    # SOURCE: vllm/lora/request.py:L58-L60
    @property
    def path(self):
        return self.lora_path

    # SOURCE: vllm/lora/request.py:L62-L69
    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare LoRARequest
        instances based on lora_name. This allows for identification
        and comparison lora adapter across engines.
        """
        return isinstance(value, self.__class__) and self.lora_name == value.lora_name

    # SOURCE: vllm/lora/request.py:L71-L73
    def __hash__(self) -> int:
        """
        Overrides the hash method to hash LoRARequest instances
        based on lora_name. This ensures that LoRARequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.lora_name)
