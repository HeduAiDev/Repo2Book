"""ch25 — register_dummy_fusion_op 的占位算子锚点 + platform.py 三个编译钩子的字符串。

(1) register_dummy_fusion_op 给 torch.ops._C_ascend 挂一组同名 dummyFusionOp，让融合 pattern
    的 search_fn 引用 torch.ops._C_ascend.* 时有可匹配的锚点对象。
(2) NPUPlatform 的 get_compile_backend / get_static_graph_wrapper_cls / get_pass_manager_cls
    返回 NPU 版类路径字符串，整体顶替 vLLM 的 InductorAdaptor / CUDAGraphWrapper /
    PostGradPassManager；pass_key 返回 COMPILATION_PASS_KEY。
"""
import torch

import _ch25_ops as ops
import _ch25_platform_hooks as ph

EXPECTED_DUMMY_OPS = [
    "rms_norm",
    "fused_add_rms_norm",
    "static_scaled_fp8_quant",
    "dynamic_scaled_fp8_quant",
    "dynamic_per_token_scaled_fp8_quant",
    "rms_norm_static_fp8_quant",
    "fused_add_rms_norm_static_fp8_quant",
    "rms_norm_dynamic_per_token_quant",
]


def test_register_dummy_fusion_op_attaches_all_anchors():
    ops.register_dummy_fusion_op()
    for name in EXPECTED_DUMMY_OPS:
        attached = getattr(torch.ops._C_ascend, name)
        assert isinstance(attached, ops.dummyFusionOp)
        assert attached.name == name


def test_dummy_fusion_op_has_default_none():
    op = ops.dummyFusionOp(name="x")
    assert op.default is None
    assert op.name == "x"


def test_compile_backend_hook_returns_ascend_compiler_path():
    assert ph.NPUPlatform.get_compile_backend() == "vllm_ascend.compilation.compiler_interface.AscendCompiler"


def test_static_graph_wrapper_hook_returns_aclgraph_wrapper_path():
    assert ph.NPUPlatform.get_static_graph_wrapper_cls() == "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"


def test_pass_manager_hook_returns_graph_fusion_pass_manager_path():
    assert (
        ph.NPUPlatform.get_pass_manager_cls()
        == "vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager"
    )


def test_pass_key_is_compilation_pass_key():
    assert ph.NPUPlatform().pass_key == "graph_fusion_manager"
