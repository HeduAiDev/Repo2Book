"""ch18 —— 昇腾如何接进 vLLM 的注意力后端选择（OOT 插件契约全貌）。

测的是精简版**复现真实仓的可观察行为**（非自洽）：
  (1) 后端路由：三元 key → 4 昇腾后端的点分路径；
  (2) 注册占位：@register_backend(CUSTOM,"ASCEND") → _ATTN_OVERRIDES，装饰器 no-op；
  (3) 伪装 HACK：get_name 在 V2 model-runner 下返回 "FLASH_ATTN"（冒充 FA 样板）；
  (4) 静态契约：4 个 @abstractmethod 强制 + get_supported_kernel_block_sizes 可覆写 +
      swap/copy_blocks 是昇腾自带、非 v1 契约；
  (5) 运行期分流：get_impl/builder_cls 按 enable_cp() 二选一；
  外加 selector 点分路径 → resolve_obj_by_qualname → 后端类的端到端解析。
"""
import inspect

import pytest
import torch

from conftest import make_cfg


# ----------------------------- (1) 后端路由：三元 key → 4 后端 ----------------------------- #
@pytest.mark.parametrize(
    "use_mla,use_sparse,use_compress,expected",
    [
        (True, False, False, "vllm_ascend.attention.mla_v1.AscendMLABackend"),
        (False, False, False, "vllm_ascend.attention.attention_v1.AscendAttentionBackend"),
        (True, True, False, "vllm_ascend.attention.sfa_v1.AscendSFABackend"),
        (True, False, True, "vllm_ascend.attention.dsa_v1.AscendDSABackend"),
    ],
)
def test_routing_table_maps_triple_key_to_four_backends(env, use_mla, use_sparse, use_compress, expected):
    mods, _ = env
    cfg = make_cfg(use_mla=use_mla, use_sparse=use_sparse, use_compress=use_compress)
    got = mods.platform.NPUPlatform.get_attn_backend_cls(None, cfg)
    assert got == expected


def test_use_compress_defaults_false_via_getattr(env):
    """基座 AttentionSelectorConfig 不声明 use_compress → getattr 默认 False → 走 (.,.,False) 主路。"""
    mods, _ = env
    cfg = make_cfg(use_mla=False, use_sparse=False, use_compress=None)  # 无 use_compress 字段
    assert not hasattr(cfg, "use_compress")
    got = mods.platform.NPUPlatform.get_attn_backend_cls(None, cfg)
    assert got == "vllm_ascend.attention.attention_v1.AscendAttentionBackend"


def test_routing_returns_dotted_string_not_class(env):
    """平台返回的是点分路径字符串（交给 selector 延迟解析），不是类对象。"""
    mods, _ = env
    got = mods.platform.NPUPlatform.get_attn_backend_cls(None, make_cfg(use_mla=True))
    assert isinstance(got, str)


# ----------------------------- (2) 注册占位：register_backend ----------------------------- #
def test_register_backend_writes_override_for_custom_slot(env):
    mods, _ = env
    reg = mods.registry
    assert reg.AttentionBackendEnum.CUSTOM.value is None  # OOT 占位槽
    assert reg._ATTN_OVERRIDES[reg.AttentionBackendEnum.CUSTOM] == "ASCEND"


def test_register_backend_with_class_path_is_noop_decorator(env):
    """class_path 非 None → 返回 no-op 装饰器，类本身不被改写。"""
    mods, _ = env
    cls = mods.attention_v1.AscendAttentionBackend
    assert inspect.isclass(cls)
    assert cls.__name__ == "AscendAttentionBackend"


# ----------------------------- (3) 伪装 HACK：get_name ----------------------------- #
def test_get_name_impersonates_flash_attn_under_v2_runner(env):
    mods, knobs = env
    knobs.use_v2_model_runner = True
    assert mods.attention_v1.AscendAttentionBackend.get_name() == "FLASH_ATTN"


def test_get_name_is_custom_without_v2_runner(env):
    mods, knobs = env
    knobs.use_v2_model_runner = False
    assert mods.attention_v1.AscendAttentionBackend.get_name() == "CUSTOM"


def test_impersonation_matches_flash_attn_sample(env):
    """伪装返回的名字必须等于被冒充的 FlashAttentionBackend.get_name()。"""
    mods, knobs = env
    knobs.use_v2_model_runner = True
    assert (
        mods.attention_v1.AscendAttentionBackend.get_name()
        == mods.flash_attn.FlashAttentionBackend.get_name()
        == "FLASH_ATTN"
    )


# ----------------------------- (4) 静态契约 ----------------------------- #
def test_four_abstractmethods_are_the_contract(env):
    mods, _ = env
    abstracts = mods.backend.AttentionBackend.__abstractmethods__
    assert abstracts == frozenset(
        {"get_name", "get_impl_cls", "get_builder_cls", "get_kv_cache_shape"}
    )


def test_incomplete_backend_cannot_instantiate(env):
    """缺任一 @abstractmethod 的后端无法实例化 —— 这正是契约的强制力。"""
    mods, _ = env

    class HalfBackend(mods.backend.AttentionBackend):
        @staticmethod
        def get_name():
            return "HALF"

    with pytest.raises(TypeError):
        HalfBackend()


def test_ascend_backend_satisfies_full_contract(env):
    mods, _ = env
    cls = mods.attention_v1.AscendAttentionBackend
    # 4 个契约方法全实现 → 抽象方法集为空 → 可实例化
    assert cls.__abstractmethods__ == frozenset()
    assert cls() is not None


def test_get_kv_cache_shape_layout(env):
    """昇腾 KV cache 形状 (2, num_blocks, block_size, num_kv_heads, head_size)，首维 2 = key/value。"""
    mods, _ = env
    shape = mods.attention_v1.AscendAttentionBackend.get_kv_cache_shape(8, 16, 4, 64)
    assert shape == (2, 8, 16, 4, 64)


def test_get_supported_kernel_block_sizes_override(env):
    """基类默认 [MultipleOf(1)]（可覆写、非 @abstractmethod）；昇腾覆写返回 [128]。"""
    mods, _ = env
    base_default = mods.backend.AttentionBackend.get_supported_kernel_block_sizes()
    assert len(base_default) == 1 and base_default[0].base == 1
    assert mods.attention_v1.AscendAttentionBackend.get_supported_kernel_block_sizes() == [128]


def test_swap_copy_blocks_not_in_v1_contract(env):
    """swap_blocks/copy_blocks 不在 v1 基类契约里（基座 AttentionBackend 无此二方法），是昇腾自带 v0 遗留。"""
    mods, _ = env
    assert not hasattr(mods.backend.AttentionBackend, "swap_blocks")
    assert not hasattr(mods.backend.AttentionBackend, "copy_blocks")
    # 但昇腾后端自带它们
    assert hasattr(mods.attention_v1.AscendAttentionBackend, "swap_blocks")
    assert hasattr(mods.attention_v1.AscendAttentionBackend, "copy_blocks")


def test_swap_blocks_moves_by_block_index(env):
    """按 (2,...) 布局做块级索引搬运：src 块 → dst 块（key/value 各自一半）。"""
    mods, _ = env
    nb, bs, h, d = 4, 2, 1, 2
    src_key = torch.arange(nb * bs * h * d).reshape(nb, bs, h, d).float()
    src_val = src_key + 1000
    dst_key = torch.zeros(nb, bs, h, d)
    dst_val = torch.zeros(nb, bs, h, d)
    src_to_dst = torch.tensor([[0, 3], [1, 2]])  # src块0→dst块3, src块1→dst块2
    mods.attention_v1.AscendAttentionBackend.swap_blocks(
        [src_key, src_val], [dst_key, dst_val], src_to_dst
    )
    assert torch.equal(dst_key[3], src_key[0])
    assert torch.equal(dst_key[2], src_key[1])
    assert torch.equal(dst_val[3], src_val[0])
    assert torch.equal(dst_key[0], torch.zeros(bs, h, d))  # 未映射块保持原样


def test_copy_blocks_within_same_cache(env):
    mods, _ = env
    nb, bs, h, d = 4, 2, 1, 2
    key = torch.arange(nb * bs * h * d).reshape(nb, bs, h, d).float()
    val = key + 1000
    kv_cache = torch.stack([key, val])  # 形状 (2, nb, bs, h, d)
    expected_src0 = kv_cache[0][1].clone()
    src_to_dists = torch.tensor([[1, 0]])  # 块1 → 块0
    mods.attention_v1.AscendAttentionBackend.copy_blocks([kv_cache], src_to_dists)
    assert torch.equal(kv_cache[0][0], expected_src0)  # key 半区块0 被块1覆盖


# ----------------------------- (5) 运行期分流：enable_cp ----------------------------- #
def test_impl_builder_cls_normal_path(env):
    mods, knobs = env
    knobs.cp_enabled = False
    be = mods.attention_v1
    assert be.AscendAttentionBackend.get_impl_cls() is be.AscendAttentionBackendImpl
    assert be.AscendAttentionBackend.get_builder_cls() is be.AscendAttentionMetadataBuilder


def test_impl_builder_cls_cp_path(env):
    """enable_cp() 为 True → 运行期切换到 CP 实现/构造器。"""
    mods, knobs = env
    knobs.cp_enabled = True
    be = mods.attention_v1
    assert be.AscendAttentionBackend.get_impl_cls() is be.AscendAttentionCPImpl
    assert be.AscendAttentionBackend.get_builder_cls() is be.AscendAttentionCPMetadataBuilder


# ----------------------------- selector 端到端解析 ----------------------------- #
def test_selector_resolves_dotted_path_to_backend_class(env):
    """selector → platform.get_attn_backend_cls(点分路径) → resolve_obj_by_qualname → 后端类。"""
    mods, _ = env
    sel = mods.selector
    sel._cached_get_attn_backend.cache_clear()
    cls = sel._cached_get_attn_backend(None, make_cfg(use_mla=False, use_sparse=False))
    assert cls is mods.attention_v1.AscendAttentionBackend


def test_resolve_obj_by_qualname_imports_class(env):
    mods, _ = env
    obj = mods.selector.resolve_obj_by_qualname(
        "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
    )
    assert obj is mods.attention_v1.AscendAttentionBackend
