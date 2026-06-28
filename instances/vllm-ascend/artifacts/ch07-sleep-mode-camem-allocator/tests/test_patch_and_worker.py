"""验证接入点 patch_camem_allocator 的 hasattr 守护 fallback + worker 调用方摘录可导入。

对位真实行为 vllm_ascend/patch/platform/patch_camem_allocator.py:L20-L28：
  - _patched_is_cumem_allocator_available() 恒返回 True；
  - 仅当 vllm.config.model 有 is_cumem_allocator_available 属性时才替换（前向兼容 fallback），
    否则 no-op（当前 pin 的 v0.21.0 base 上正是 no-op）。
"""
import importlib
import sys
import types


def test_patched_returns_true_and_noop_on_host():
    sys.modules.pop("patch_camem_allocator", None)
    mod = importlib.import_module("patch_camem_allocator")
    assert mod._patched_is_cumem_allocator_available() is True
    # host 无 vllm → 可选 import 失败 → model_config_module is None → patch no-op
    assert mod.model_config_module is None


def _inject_fake_vllm_model(monkeypatch, with_attr):
    fake_vllm = types.ModuleType("vllm")
    fake_config = types.ModuleType("vllm.config")
    fake_model = types.ModuleType("vllm.config.model")
    if with_attr:
        fake_model.is_cumem_allocator_available = lambda: False  # 上游原始校验（只认 CUDA/ROCm）
    fake_vllm.config = fake_config
    fake_config.model = fake_model
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", fake_config)
    monkeypatch.setitem(sys.modules, "vllm.config.model", fake_model)
    return fake_model


def test_guard_replaces_when_attr_present(monkeypatch):
    fake_model = _inject_fake_vllm_model(monkeypatch, with_attr=True)
    sys.modules.pop("patch_camem_allocator", None)
    mod = importlib.import_module("patch_camem_allocator")
    # 上游有校验函数 → 被替换为 ascend 版（恒 True）
    assert fake_model.is_cumem_allocator_available is mod._patched_is_cumem_allocator_available
    assert fake_model.is_cumem_allocator_available() is True
    sys.modules.pop("patch_camem_allocator", None)


def test_guard_noop_when_attr_absent(monkeypatch):
    fake_model = _inject_fake_vllm_model(monkeypatch, with_attr=False)
    sys.modules.pop("patch_camem_allocator", None)
    importlib.import_module("patch_camem_allocator")
    # 上游无该函数（如 v0.21.0 base）→ 不新增、不替换，保持 no-op
    assert not hasattr(fake_model, "is_cumem_allocator_available")
    sys.modules.pop("patch_camem_allocator", None)


def test_worker_excerpt_imports_and_has_callsites():
    sys.modules.pop("worker_excerpt", None)
    we = importlib.import_module("worker_excerpt")
    for name in ("sleep", "wake_up", "load_model", "initialize_from_config"):
        assert callable(getattr(we.NPUWorker, name))
