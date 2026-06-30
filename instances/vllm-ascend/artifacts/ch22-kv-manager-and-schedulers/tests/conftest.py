"""ch22 测试脚手架：host 无 vLLM/NPU，在 sys.modules 桩掉 vllm.* / vllm_ascend.* /
pandas 等重运行时依赖，再把（已减法的）implementation/*.py 按规范模块名注册进去，让它们
互相解析到精简版。

可在 host 验证、与真仓一致的纯 Python 控制流：
  (1) get_manager_for_kv_cache_spec 重映射：压缩 MLA(compress_ratio>1)→CompressAttentionManager
      并设 max_admission_blocks_per_request=cdiv(max_model_len//cr, block)+1；非压缩→spec_manager_map。
  (2) CompressAttentionManager：get_num_blocks_to_allocate 先 //=compress_ratio 再调父类；
      find_longest_cache_hit 按 logical_block_size=block_size×compress_ratio 命中粒度。
  (3) BudgetRefiner：未配 SLO→refine_budget 恒等；配置后 _get_max_budget/_align_key 查表。
  (4) ChunkSizePredictor.predict：解二次方程求 chunk size；ProfilingChunkManager 就绪门控。
  (5) RecomputeSchedulerConfig 按 async_scheduling 选类；register_ascend_mla_spec_in_manager 补登记；
      update_from_output 把 recomputed_reqs 以 stop_reason='recomputed' 回吐；AsyncRecomputeScheduler MRO。
"""
import importlib.util
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _mod(dotted, added):
    parts = dotted.split(".")
    for i in range(len(parts)):
        name = ".".join(parts[: i + 1])
        if name not in sys.modules:
            m = types.ModuleType(name)
            sys.modules[name] = m
            added.append(name)
            if i > 0:
                setattr(sys.modules[".".join(parts[:i])], parts[i], m)
    return sys.modules[dotted]


# ---- 复用 vLLM 的 manager 基类（原样继承的对照线）的最小替身 ---- #
class _FullAttentionManager:
    """vLLM FullAttentionManager 替身：记录收到的 num_tokens/kwargs，便于验缩放。"""

    def __init__(self, kv_cache_spec, block_pool=None, **kwargs):
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self._kwargs = kwargs
        # SingleTypeKVCacheManager 基类常驻字段（精简替身）
        self.block_size = getattr(kv_cache_spec, "block_size", 16)
        self.req_to_blocks = {}
        self.num_cached_block = {}
        self.enable_caching = True
        self.kv_cache_group_id = kwargs.get("kv_cache_group_id", 0)

    def get_num_blocks_to_allocate(
        self, request_id, num_tokens, new_computed_blocks, total_computed_tokens, num_tokens_main_model,
        apply_admission_cap=False,
    ):
        # 回显父类收到的 num_tokens —— 测试据此断言子类已 //=compress_ratio
        return num_tokens


class _SingleTypeKVCacheManager:
    pass


# ---- spec 类型（vLLM 原生，用于 spec_manager_map 查表）---- #
class _KVCacheSpec:
    block_size = 16


class _FullAttentionSpec(_KVCacheSpec):
    pass


class _MLAAttentionSpec(_KVCacheSpec):
    def __init__(self, compress_ratio=1, block_size=16):
        self.compress_ratio = compress_ratio
        self.block_size = block_size


class _SlidingWindowSpec(_KVCacheSpec):
    def __init__(self, cap=7):
        self._cap = cap

    def max_admission_blocks_per_request(self, max_num_batched_tokens, max_model_len):
        return self._cap


class _ChunkedLocalAttentionSpec(_KVCacheSpec):
    def max_admission_blocks_per_request(self, max_num_batched_tokens, max_model_len):
        return 3


class _BlockHashListWithBlockSize:
    """vLLM 替身：按 logical_block_size 重切块的哈希链——精简版只需可迭代。"""

    def __init__(self, block_hashes, block_size, logical_block_size):
        self.block_hashes = list(block_hashes)
        self.block_size = block_size
        self.logical_block_size = logical_block_size

    def __iter__(self):
        return iter(self.block_hashes)


@pytest.fixture
def env():
    saved = dict(sys.modules)
    added = []

    # ---------- vllm.utils.math_utils.cdiv（真实 ceil 除）---------- #
    mu = _mod("vllm.utils.math_utils", added)
    mu.cdiv = lambda a, b: -(-a // b)

    # ---------- pandas（仅 import，被减法跳过实际使用）---------- #
    _mod("pandas", added)

    # ---------- numpy 已在 host，留真实 ---------- #

    # ---------- vllm.logger ---------- #
    lg = _mod("vllm.logger", added)
    lg.logger = types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None,
        error=lambda *a, **k: None, debug=lambda *a, **k: None,
    )

    # ---------- vllm.config ---------- #
    cfg = _mod("vllm.config", added)

    @dataclass
    class SchedulerConfig:
        async_scheduling: bool = False
        max_model_len: int = 0
        is_encoder_decoder: bool = False
        max_num_batched_tokens: int = 0

    cfg.SchedulerConfig = SchedulerConfig
    cfg.VllmConfig = type("VllmConfig", (), {})

    # ---------- vllm.multimodal ---------- #
    mm = _mod("vllm.multimodal", added)
    mm.MULTIMODAL_REGISTRY = object()
    mm.MultiModalRegistry = type("MultiModalRegistry", (), {})

    # ---------- vllm.v1.core.* ---------- #
    bp = _mod("vllm.v1.core.block_pool", added)
    bp.BlockPool = type("BlockPool", (), {})

    khu = _mod("vllm.v1.core.kv_cache_utils", added)
    khu.BlockHashList = type("BlockHashList", (), {})
    khu.BlockHashListWithBlockSize = _BlockHashListWithBlockSize
    khu.KVCacheBlock = type("KVCacheBlock", (), {})

    stm = _mod("vllm.v1.core.single_type_kv_cache_manager", added)
    stm.FullAttentionManager = _FullAttentionManager
    stm.SingleTypeKVCacheManager = _SingleTypeKVCacheManager
    stm.spec_manager_map = {
        _FullAttentionSpec: _FullAttentionManager,
        _MLAAttentionSpec: _FullAttentionManager,
        _SlidingWindowSpec: _FullAttentionManager,
        _ChunkedLocalAttentionSpec: _FullAttentionManager,
    }

    kci = _mod("vllm.v1.kv_cache_interface", added)
    kci.ChunkedLocalAttentionSpec = _ChunkedLocalAttentionSpec
    kci.FullAttentionSpec = _FullAttentionSpec
    kci.KVCacheSpec = _KVCacheSpec
    kci.MLAAttentionSpec = _MLAAttentionSpec
    kci.SlidingWindowSpec = _SlidingWindowSpec
    kci.KVCacheConfig = type("KVCacheConfig", (), {})

    kcm = _mod("vllm.v1.core.kv_cache_manager", added)
    kcm.KVCacheBlocks = type("KVCacheBlocks", (), {})

    sout = _mod("vllm.v1.core.sched.output", added)
    sout.SchedulerOutput = _make_scheduler_output_cls()
    sout.NewRequestData = type("NewRequestData", (), {})

    sch = _mod("vllm.v1.core.sched.scheduler", added)
    sch.Scheduler = type("Scheduler", (), {})

    asc = _mod("vllm.v1.core.sched.async_scheduler", added)
    asc.AsyncScheduler = type("AsyncScheduler", (), {})

    iface = _mod("vllm.v1.core.sched.interface", added)
    iface.PauseState = types.SimpleNamespace(PAUSED_ALL="PAUSED_ALL", UNPAUSED="UNPAUSED")

    rq = _mod("vllm.v1.core.sched.request_queue", added)
    rq.SchedulingPolicy = types.SimpleNamespace(PRIORITY="PRIORITY", FCFS="FCFS")
    rq.create_request_queue = lambda policy: None

    # ---------- vllm.v1.engine ---------- #
    eng = _mod("vllm.v1.engine", added)
    eng.EngineCoreEventType = types.SimpleNamespace(QUEUED="QUEUED", SCHEDULED="SCHEDULED", PREEMPTED="PREEMPTED")
    eng.FinishReason = types.SimpleNamespace(STOP="STOP")

    @dataclass
    class EngineCoreOutput:
        request_id: str = ""
        new_token_ids: list = field(default_factory=list)
        finish_reason: object = None
        stop_reason: object = None

    @dataclass
    class EngineCoreOutputs:
        outputs: list = field(default_factory=list)
        finished_requests: object = None
        scheduler_stats: object = None

    eng.EngineCoreOutput = EngineCoreOutput
    eng.EngineCoreOutputs = EngineCoreOutputs

    out = _mod("vllm.v1.outputs", added)
    out.ModelRunnerOutput = type("ModelRunnerOutput", (), {})

    so_mod = _mod("vllm.v1.structured_output", added)
    so_mod.StructuredOutputManager = type("StructuredOutputManager", (), {})

    req = _mod("vllm.v1.request", added)
    req.Request = type("Request", (), {})
    req.RequestStatus = types.SimpleNamespace(
        WAITING="WAITING", RUNNING="RUNNING", PREEMPTED="PREEMPTED",
        WAITING_FOR_STREAMING_REQ="WAITING_FOR_STREAMING_REQ",
        FINISHED_ABORTED="FINISHED_ABORTED",
    )
    req.StreamingUpdate = type("StreamingUpdate", (), {"from_request": staticmethod(lambda r: None)})

    rs = _mod("vllm.v1.sample.rejection_sampler", added)
    rs.PLACEHOLDER_TOKEN_ID = -1

    vutil = _mod("vllm.v1.utils", added)
    vutil.ConstantList = list

    class _NullCtx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    vutil.record_function_or_nullcontext = lambda name: _NullCtx()

    # ---------- vllm_ascend.* ---------- #
    _mod("vllm_ascend.core", added)
    ac = _mod("vllm_ascend.ascend_config", added)
    ac.get_ascend_config = lambda: types.SimpleNamespace()
    ac.init_ascend_config = lambda cfg: None

    # ---------- 加载精简版（profiling_chunk_predictor 先于 scheduler_profiling_chunk）---------- #
    kvm = _load("single_type_kv_cache_manager.py", "vllm_ascend.core.single_type_kv_cache_manager")
    dyn = _load("scheduler_dynamic_batch.py", "vllm_ascend.core.scheduler_dynamic_batch")
    rec = _load("recompute_scheduler.py", "vllm_ascend.core.recompute_scheduler")
    pred = _load("profiling_chunk_predictor.py", "vllm_ascend.core.profiling_chunk_predictor")
    prof = _load("scheduler_profiling_chunk.py", "vllm_ascend.core.scheduler_profiling_chunk")

    bundle = types.SimpleNamespace(
        kvm=kvm, dyn=dyn, rec=rec, pred=pred, prof=prof,
        FullAttentionManager=_FullAttentionManager,
        MLAAttentionSpec=_MLAAttentionSpec,
        FullAttentionSpec=_FullAttentionSpec,
        SlidingWindowSpec=_SlidingWindowSpec,
        spec_manager_map=stm.spec_manager_map,
        SchedulerConfig=SchedulerConfig,
        EngineCoreOutput=EngineCoreOutput,
        cdiv=mu.cdiv,
        math=math,
    )
    yield bundle

    sys.modules.clear()
    sys.modules.update(saved)


def _make_scheduler_output_cls():
    @dataclass
    class SchedulerOutput:
        scheduled_new_reqs: list = field(default_factory=list)
        scheduled_cached_reqs: object = None
        num_scheduled_tokens: dict = field(default_factory=dict)
        total_num_scheduled_tokens: int = 0
        scheduled_spec_decode_tokens: dict = field(default_factory=dict)
        scheduled_encoder_inputs: dict = field(default_factory=dict)
        num_common_prefix_blocks: list = field(default_factory=list)
        preempted_req_ids: set = field(default_factory=set)
        finished_req_ids: set = field(default_factory=set)
        free_encoder_mm_hashes: object = None

    return SchedulerOutput
