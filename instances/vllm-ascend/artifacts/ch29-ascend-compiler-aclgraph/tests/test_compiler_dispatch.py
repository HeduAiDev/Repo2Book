"""ch25 — AscendCompiler.compile() 按 enable_npugraph_ex 二选一的真实 dispatch。

compile() 先 deepcopy 图 + 按 fake_mode 重绑 example_inputs，再读
get_ascend_config().ascend_compilation_config.enable_npugraph_ex：
  True  → npugraph_ex_compile(...)（含 vllm_config / cache_dir 透传）
  False → fusion_pass_compile(...)
这里 spy 掉两个模块级编译函数，断言走对分支、参数透传正确。
"""
import _ch25_compiler_interface as ci
from vllm_ascend.ascend_config import get_ascend_config


def _patch_branches(monkeypatch):
    calls = {"npugraph_ex": None, "fusion": None}

    def fake_npugraph_ex(graph, example_inputs, compiler_config, vllm_config,
                         ascend_compilation_config, compile_range, key, cache_dir):
        calls["npugraph_ex"] = dict(key=key, cache_dir=cache_dir, vllm_config=vllm_config)
        return ("npugraph_ex_fn", (key, cache_dir))

    def fake_fusion(graph, example_inputs, compiler_config, compile_range, key):
        calls["fusion"] = dict(key=key)
        return ("fusion_fn", None)

    monkeypatch.setattr(ci, "npugraph_ex_compile", fake_npugraph_ex)
    monkeypatch.setattr(ci, "fusion_pass_compile", fake_fusion)
    return calls


def test_compile_dispatches_to_fusion_when_npugraph_ex_disabled(monkeypatch):
    calls = _patch_branches(monkeypatch)
    get_ascend_config().ascend_compilation_config.enable_npugraph_ex = False

    comp = ci.AscendCompiler()
    fn, handle = comp.compile(graph=object(), example_inputs=[], compiler_config={},
                              compile_range=None, key="k0")
    assert fn == "fusion_fn" and handle is None
    assert calls["fusion"] == {"key": "k0"}
    assert calls["npugraph_ex"] is None


def test_compile_dispatches_to_npugraph_ex_when_enabled(monkeypatch):
    calls = _patch_branches(monkeypatch)
    get_ascend_config().ascend_compilation_config.enable_npugraph_ex = True

    comp = ci.AscendCompiler()
    comp.vllm_config = "VC"          # set by compute_hash in real flow; required by assert in compile()
    comp.initialize_cache("/tmp/cache", disable_cache=False)
    fn, handle = comp.compile(graph=object(), example_inputs=[], compiler_config={},
                              compile_range=None, key="k1")
    assert fn == "npugraph_ex_fn"
    assert calls["fusion"] is None
    assert calls["npugraph_ex"]["key"] == "k1"
    assert calls["npugraph_ex"]["cache_dir"] == "/tmp/cache"
    assert calls["npugraph_ex"]["vllm_config"] == "VC"


def test_disable_cache_forces_cache_dir_none(monkeypatch):
    calls = _patch_branches(monkeypatch)
    get_ascend_config().ascend_compilation_config.enable_npugraph_ex = True

    comp = ci.AscendCompiler()
    comp.vllm_config = "VC"
    comp.initialize_cache("/tmp/cache", disable_cache=True)
    comp.compile(graph=object(), example_inputs=[], compiler_config={}, compile_range=None, key="k2")
    assert calls["npugraph_ex"]["cache_dir"] is None
