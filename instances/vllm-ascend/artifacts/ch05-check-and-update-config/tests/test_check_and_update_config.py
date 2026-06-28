"""ch05 精简版测试 —— 复现 vllm_ascend 真实可观察行为（host 纯 dict 控制流，无需 NPU/CANN）。

覆盖两条主线：
  1) 平台=配置改写器：_fix_incompatible_config 的 cascade reset + check_and_update_config 编排骨架；
  2) 无 schema 配置后门：AscendConfig 解析 additional_config + _get_config_value 三级取值 + envs 懒求值 + 单例。
"""
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

IMPL = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import ascend_config as ac  # noqa: E402
import envs as ascend_envs  # noqa: E402
from _support import AttentionBackendEnum, CompilationMode  # noqa: E402

# 精简版文件名忠实保留为 platform.py，与 stdlib `platform` 同名；用 importlib 按路径加载避开冲突。
_spec = importlib.util.spec_from_file_location("npu_platform_impl", IMPL / "platform.py")
npu_platform_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(npu_platform_impl)
NPUPlatform = npu_platform_impl.NPUPlatform


# ----------------------------- fixtures -----------------------------

def make_vllm_config(**overrides):
    """构造一个 VllmConfig 替身（SimpleNamespace 充当各子配置；真源码全程 getattr 鸭子取值）。"""
    cfg = SimpleNamespace(
        device_config=None,
        model_config=SimpleNamespace(
            enforce_eager=False,
            enable_sleep_mode=False,
            disable_cascade_attn=False,
        ),
        cache_config=SimpleNamespace(cpu_kvcache_space_bytes=False),
        observability_config=SimpleNamespace(enable_layerwise_nvtx_tracing=False),
        scheduler_config=SimpleNamespace(max_num_partial_prefills=1),
        speculative_config=None,
        kv_transfer_config=None,
        attention_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(
            worker_cls="auto",
            numa_bind=False,
            numa_bind_nodes=None,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            splitting_ops=None,
            cudagraph_mode=None,
            cudagraph_num_of_warmups=0,
            pass_config=SimpleNamespace(enable_sp=True),
        ),
        additional_config=None,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(autouse=True)
def _reset_singleton_and_env():
    ac.clear_ascend_config()
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
    ac.clear_ascend_config()


# ----------------- 主线1：_fix_incompatible_config cascade reset -----------------

def test_disable_cascade_attn_force_false():
    cfg = make_vllm_config()
    cfg.model_config.disable_cascade_attn = True
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.model_config.disable_cascade_attn is False


def test_cpu_kvcache_space_bytes_reset_none():
    cfg = make_vllm_config()
    cfg.cache_config.cpu_kvcache_space_bytes = 123456
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.cache_config.cpu_kvcache_space_bytes is None


def test_nvtx_and_partial_prefills_reset():
    cfg = make_vllm_config()
    cfg.observability_config.enable_layerwise_nvtx_tracing = True
    cfg.scheduler_config.max_num_partial_prefills = 4
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.observability_config.enable_layerwise_nvtx_tracing is False
    assert cfg.scheduler_config.max_num_partial_prefills == 1


def test_force_false_flags_batch_reset():
    cfg = make_vllm_config()
    # use_cudnn_prefill / use_trtllm_attention 等 GPU 专属布尔位被批量归零
    cfg.attention_config.use_cudnn_prefill = True
    cfg.attention_config.use_trtllm_attention = True
    cfg.attention_config.disable_flashinfer_prefill = True
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.attention_config.use_cudnn_prefill is False
    assert cfg.attention_config.use_trtllm_attention is False
    assert cfg.attention_config.disable_flashinfer_prefill is False


def test_attention_backend_flash_attn_kept_others_reset():
    # backend == FLASH_ATTN：训推一致，保留不动
    cfg = make_vllm_config()
    cfg.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.attention_config.backend == AttentionBackendEnum.FLASH_ATTN
    # backend 为其它值：reset 为 None
    cfg2 = make_vllm_config()
    cfg2.attention_config.backend = "TRITON_ATTN"
    NPUPlatform._fix_incompatible_config(cfg2)
    assert cfg2.attention_config.backend is None


def test_flash_attn_version_and_splits_reset():
    cfg = make_vllm_config()
    cfg.attention_config.flash_attn_version = 3
    cfg.attention_config.flash_attn_max_num_splits_for_cuda_graph = 16
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.attention_config.flash_attn_version is None
    assert cfg.attention_config.flash_attn_max_num_splits_for_cuda_graph == 32


def test_numa_bind_rewritten_not_dropped():
    # 关键特例：numa_bind 不是丢弃，而是无损改写成 additional_config['enable_cpu_binding']=True
    cfg = make_vllm_config()
    cfg.parallel_config.numa_bind = True
    cfg.additional_config = None
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.parallel_config.numa_bind is False
    assert cfg.additional_config == {"enable_cpu_binding": True}


def test_numa_bind_setdefault_respects_existing():
    cfg = make_vllm_config()
    cfg.parallel_config.numa_bind = True
    cfg.additional_config = {"enable_cpu_binding": False}
    NPUPlatform._fix_incompatible_config(cfg)
    # setdefault 不覆盖用户已显式给的值
    assert cfg.additional_config["enable_cpu_binding"] is False


def test_numa_bind_nodes_ignored_to_none():
    cfg = make_vllm_config()
    cfg.parallel_config.numa_bind_nodes = [0, 1]
    NPUPlatform._fix_incompatible_config(cfg)
    assert cfg.parallel_config.numa_bind_nodes is None


# ----------------- 主线2：三级取值 / 解析 / 单例 / envs -----------------

def test_get_config_value_additional_config_overrides():
    val = ac.AscendConfig._get_config_value({"k": 7}, "k", "ENV_K", 0)
    assert val == 7  # additional_config 命中即用（最高优先级）


def test_get_config_value_falls_back_to_env_value():
    val = ac.AscendConfig._get_config_value({}, "k", "ENV_K", 99)
    assert val == 99  # 未命中 additional_config → 返回已塌缩的 env_value（含 default）


def test_envs_lazy_evaluation_reflects_env_change():
    os.environ.pop("VLLM_ASCEND_ENABLE_FLASHCOMM1", None)
    assert ascend_envs.VLLM_ASCEND_ENABLE_FLASHCOMM1 is False  # default 0 → False
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "1"
    assert ascend_envs.VLLM_ASCEND_ENABLE_FLASHCOMM1 is True  # 懒求值：每次访问重读环境


def test_envs_unknown_attr_raises():
    with pytest.raises(AttributeError):
        _ = ascend_envs.NOT_A_REAL_ENV_VAR


def test_ascend_config_parses_additional_config_to_typed():
    cfg = make_vllm_config(additional_config={
        "ascend_compilation_config": {"enable_static_kernel": False, "fuse_norm_quant": False},
    })
    conf = ac.AscendConfig(cfg)
    assert isinstance(conf.ascend_compilation_config, ac.AscendCompilationConfig)
    assert conf.ascend_compilation_config.fuse_norm_quant is False


def test_ascend_config_none_additional_config_ok():
    cfg = make_vllm_config(additional_config=None)
    conf = ac.AscendConfig(cfg)
    assert conf.xlite_graph_config.enabled is False


def test_scalar_three_level_additional_config_wins_over_env():
    os.environ["VLLM_ASCEND_ENABLE_FLASHCOMM1"] = "0"
    cfg = make_vllm_config(additional_config={"enable_flashcomm1": True})
    conf = ac.AscendConfig(cfg)
    assert conf.enable_flashcomm1 is True  # additional_config 压过 env


def test_compilation_config_kwargs_backdoor_swallows_unknown():
    # **kwargs 静默吞未知键（向前兼容），且 fuse_muls_add 从 kwargs 取
    c = ac.AscendCompilationConfig(totally_unknown_key=123, fuse_muls_add=False)
    assert c.fuse_muls_add is False
    assert not hasattr(c, "totally_unknown_key")


def test_compilation_config_static_kernel_requires_npugraph_ex():
    with pytest.raises(AssertionError):
        ac.AscendCompilationConfig(enable_static_kernel=True, enable_npugraph_ex=False)


def test_init_ascend_config_singleton_same_config_cached():
    cfg = make_vllm_config()
    a = ac.init_ascend_config(cfg)
    b = ac.init_ascend_config(cfg)
    assert a is b  # 同一 vllm_config + 完整初始化 → 命中缓存
    assert ac.get_ascend_config() is a


def test_init_ascend_config_refresh_rebuilds():
    cfg = make_vllm_config()
    a = ac.init_ascend_config(cfg)
    cfg.additional_config = {"refresh": True}
    b = ac.init_ascend_config(cfg)
    assert a is not b  # refresh=True 强制重建


def test_init_ascend_config_different_config_rebuilds():
    a = ac.init_ascend_config(make_vllm_config())
    b = ac.init_ascend_config(make_vllm_config())
    assert a is not b  # 不同 vllm_config 对象 → 不复用缓存


def test_get_ascend_config_before_init_raises():
    ac.clear_ascend_config()
    with pytest.raises(RuntimeError):
        ac.get_ascend_config()


# ----------------- 总闸：check_and_update_config 编排骨架 -----------------

def test_check_and_update_config_device_type_mismatch_early_return():
    cfg = make_vllm_config(device_config=SimpleNamespace(device_type="cuda"))
    cfg.parallel_config.worker_cls = "auto"
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == "auto"  # 早退，未改写


def test_check_and_update_config_model_config_none_early_return():
    cfg = make_vllm_config(model_config=None)
    cfg.parallel_config.worker_cls = "auto"
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == "auto"


def test_check_and_update_config_worker_cls_auto_to_npuworker():
    # ch02 伏笔 f2 回收点：'auto' → 具体 NPUWorker qualname
    cfg = make_vllm_config()
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == "vllm_ascend.worker.worker.NPUWorker"


def test_check_and_update_config_enforce_eager_disables_compilation():
    from _support import CompilationMode
    cfg = make_vllm_config()
    cfg.model_config.enforce_eager = True
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.compilation_config.mode == CompilationMode.NONE
    assert cfg.compilation_config.splitting_ops == []
    assert cfg.compilation_config.cudagraph_num_of_warmups == 1


def test_check_and_update_config_sets_npu_alloc_env():
    os.environ.pop("PYTORCH_NPU_ALLOC_CONF", None)
    cfg = make_vllm_config()
    NPUPlatform.check_and_update_config(cfg)
    assert os.environ["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"


def test_check_and_update_config_sleep_mode_skips_env():
    os.environ.pop("PYTORCH_NPU_ALLOC_CONF", None)
    cfg = make_vllm_config()
    cfg.model_config.enable_sleep_mode = True
    NPUPlatform.check_and_update_config(cfg)
    assert "PYTORCH_NPU_ALLOC_CONF" not in os.environ


def test_check_and_update_config_npu_alloc_env_no_double_append():
    os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:128"
    cfg = make_vllm_config()
    NPUPlatform.check_and_update_config(cfg)
    # 与 max_split_size_mb 互斥保护：不追加 expandable_segments
    assert os.environ["PYTORCH_NPU_ALLOC_CONF"] == "max_split_size_mb:128"


def test_check_and_update_config_runs_fix_and_numa_rewrite():
    # 端到端：总闸内部确实跑了 _fix_incompatible_config（numa_bind 被改写）
    cfg = make_vllm_config()
    cfg.parallel_config.numa_bind = True
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.numa_bind is False
    assert cfg.additional_config.get("enable_cpu_binding") is True
