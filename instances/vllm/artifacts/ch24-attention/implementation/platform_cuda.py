"""ch24 精简版 — CUDA 平台选后端逻辑（只做减法）。

对应 vllm/platforms/cuda.py 的 _get_backend_priorities / get_valid_backends /
get_attn_backend_cls。两条路：① 用户显式指定 backend → 只校验它，不合法直接报错；
② 未指定 → 按 compute capability 给出的优先级列表逐个 validate_configuration 过滤、
取优先级最高的合法者。命名/控制流与真实一致。
"""

from __future__ import annotations

from registry import AttentionBackendEnum
from selector import AttentionSelectorConfig


# SOURCE: vllm/platforms/cuda.py (DeviceCapability — 简化为可比较的 major/minor)
class DeviceCapability:
    # SUBTRACTED: 真实 DeviceCapability 是 vllm.platforms.interface 里的 NamedTuple，含
    # to_int/as_version_str 等；本章只用它参与 >= 比较与 .major 读取，给最小等价实现。
    def __init__(self, major: int, minor: int = 0):  # SOURCE: vllm/platforms/interface.py (DeviceCapability.__init__)
        self.major = major
        self.minor = minor

    def _as_tuple(self):  # SOURCE: vllm/platforms/interface.py (DeviceCapability.to_int helper)
        return (self.major, self.minor)

    def __ge__(self, other: "DeviceCapability") -> bool:  # SOURCE: vllm/platforms/interface.py (DeviceCapability.__ge__)
        return self._as_tuple() >= other._as_tuple()

    def __lt__(self, other: "DeviceCapability") -> bool:  # SOURCE: vllm/platforms/interface.py (DeviceCapability.__lt__)
        return self._as_tuple() < other._as_tuple()


# SOURCE: vllm/platforms/cuda.py:L79 (_get_backend_priorities)
def _get_backend_priorities(
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
    kv_cache_dtype: str | None = None,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""
    # SUBTRACTED: use_mla 分支（cuda.py:L86-L126）给 MLA 模型一套 FLASH_ATTN_MLA/FLASHMLA/
    # FLASHINFER_MLA/... 优先级列表（含 Blackwell sm100 的 sparse 变体）；本章主线非 MLA，
    # 删 MLA 分支。
    if device_capability.major == 10:
        # Blackwell(sm100): FlashInfer 提到第一
        return [
            AttentionBackendEnum.FLASHINFER,
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
        ]
    else:
        # Hopper 及以下: FLASH_ATTN 优先
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.FLASHINFER,
            AttentionBackendEnum.TRITON_ATTN,
        ]
    # SUBTRACTED: 真实列表还含 FLEX_ATTENTION / TURBOQUANT 兜底项（cuda.py:L131-L143）——同构
    # 的「次优后端」，删之不影响 FLASH_ATTN 在标准配置下取胜。


# SOURCE: vllm/platforms/cuda.py (CudaPlatform — 仅选后端相关方法)
class CudaPlatform:
    device_name = "cuda"

    # SUBTRACTED: 真实 get_device_capability 探测物理 GPU（cuda.py 经 pynvml/torch.cuda）；
    # host 无 CUDA，本精简版用类属性注入一个 capability，供选后端逻辑读取（默认 Hopper sm90）。
    _device_capability = DeviceCapability(9, 0)

    @classmethod
    def get_device_capability(cls) -> DeviceCapability:  # SOURCE: vllm/platforms/cuda.py (get_device_capability)
        return cls._device_capability

    @classmethod
    # SOURCE: vllm/platforms/cuda.py:L248 (get_valid_backends)
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ):
        valid_backends_priorities = []
        invalid_reasons: dict = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
            attn_selector_config.kv_cache_dtype,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = backend.get_class()
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = (priority, invalid_reasons_i)
            else:
                valid_backends_priorities.append((backend, priority))

        return valid_backends_priorities, invalid_reasons

    @classmethod
    # SOURCE: vllm/platforms/cuda.py:L282 (get_attn_backend_cls)
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum | None,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str:
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = selected_backend.get_class()
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                return selected_backend.get_path()

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, all_invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        # SUBTRACTED: reasons_str 拼接、debug 日志、--block-size 误排除更高优先级后端的 warning
        # （cuda.py:L319-L367）——纯诊断信息，删之不改选择结果。
        if len(valid_backends_priorities) == 0:
            raise ValueError("No valid attention backend found")

        # We have found some valid backends. Select the one with the
        # highest priority (lowest priority index).
        sorted_indices = sorted(
            range(len(valid_backends_priorities)),
            key=lambda i: valid_backends_priorities[i][1],
        )
        selected_index = sorted_indices[0]
        selected_backend = valid_backends_priorities[selected_index][0]

        return selected_backend.get_path()
