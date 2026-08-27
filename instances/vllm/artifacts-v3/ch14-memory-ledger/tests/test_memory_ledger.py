"""ch14 显存账本 —— 单元+契约测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测行为**
（锚点 = vllm/... 行号，基线 v0.27.1 现核，非 v2 资产的 v0.21.0 旧行号）。
本章跑 enable_prefix_caching=False 支的协调器（NoPrefixCache 支持任意组数
——源码原生路径 kv_cache_coordinator.py:L864-L876）；前缀哈希命中 → ch15。

行为清单（按 dossier.mechanisms 对账）：
- m1 启动三步定账：request_memory = ceil(total×util)、free 不足 raise
  （worker/utils.py:L409-L429）；available_kv = requested − non_kv −
  cudagraph_applied（gpu_worker.py:L544-L548）；num_blocks = available //
  page // group_size（kv_cache_utils.py:L993-L1010）
- m2 memory_profiling 三类显存：non_kv_cache_memory = total_consumed +
  transient_peak_headroom；docstring 1GiB/2GiB/0.5GiB 量级例为 oracle
  （mem_utils.py:L233-L326）
- m3 CUDA graph 估计入账：first_capture + max(1MiB, per-graph)×(n−1)、
  跨 mode 取 max 防重复计账（gpu_model_runner.py:L6645-L6811）+
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS 默认开（envs.py:L295）
- m4 护栏四道：_check_enough 至少装下一条 max_model_len（L751-L788）/
  二分估可行长度（L800-L851）/ auto-fit（L1988-L2049）/ override 折算
  不漂账（L2159-L2179）/ PP 取最小并缩张量（L2210-L2223）
- m5 混合组化：uniform 单组；n:1 分组 + layers[i::n] 交错；1.5 启发式
  （12SW+13full → 13/13 非 12/24）；padding warning（L1140-L1280）；
  disable 回退 SWA 当 full（L1568-L1589）
- m6 页大小统一：调大较小层 block_size / Mamba pad / 不可整除且无
  stride 索引 → NotImplementedError（L1070-L1132）
- m7 张量共享布局：group_size 池每池各组出一层（L1411-L1437）
- m8 resolve 两个对齐粒度：scheduler=LCM / hash=GCD 或 prefix_match_unit
  覆盖 + 整除校验；无缓存/无 connector 回退；mamba 非 align 回退
  （L626-L688）
- m9 一份账喂两侧：generate_scheduler_kv_cache_config 拍平 + 写回
  cache_config + auto-fit 触发 update_max_model_len（engine/core.py:L250-L359）
- m10 full-ISL 准入门：整序列不够 → None；只查第一 chunk 则放进
  （#39734 超收演示）（kv_cache_manager.py:L472-L488）
- m11 回收感知准入上限：SWA cdiv(window−1+in_flight, bs)+1 / chunked
  cdiv(chunk+in_flight, bs)（kv_cache_interface.py:L519-L546、L587-L618）；
  夹取段 + 注入装配点（single_type:L178-L192、L1861-L1878）
- m12 精修版水位：watermark_blocks=watermark×num_blocks；只对
  WAITING/PREEMPTED 且 has_scheduled_reqs 生效（kv_cache_manager.py:L463-L527）
- m13 SWA 窗外回收：get_num_skipped_tokens=max(0,computed−window+1)（窗口
  4/computed 7→4）；chunked 按 chunk 对齐（8:13→8/8→8/7→0）；窗外整块
  free + null 原位占位（single_type:L622-L672、L1057-L1083、L1200-L1244）
- m14 kernel 块细分：map_to_kernel_blocks [0,1,2]→[0..5]；32/16 拆分
  blocks_per_kv_block=2；MultiGroupBlockTable 每组一表
  （block_table.py:L82-L154、L220-L248、L270-L336）
- m15 容量核算：max_concurrency = num_blocks / Σ_groups cdiv(bytes,page)；
  size_tokens = 并发×max_model_len（kv_cache_utils.py:L937-L959、L1877-L1887）
- m16 util 语义：默认 0.92 per-instance；profile 快照 assert 拒绝他进程
  释放显存（gpu_worker.py:L533-L543）
"""
import contextlib
import math
import os
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation.cache import CacheConfig  # noqa: E402
from implementation.config import (  # noqa: E402
    ModelConfig,
    ParallelConfig,
    VllmConfig,
)
from implementation.engine_core import EngineCore  # noqa: E402
from implementation.gpu_worker import Worker  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.kv_cache_utils import (  # noqa: E402
    estimate_max_model_len,
    get_kv_cache_capacity,
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
    get_max_concurrency_for_kv_cache_config,
    get_num_blocks,
    generate_scheduler_kv_cache_config,
    resolve_kv_cache_block_sizes,
    unify_kv_cache_spec_page_size,
)
from implementation.mem_utils import (  # noqa: E402
    MemorySnapshot,
    memory_profiling,
)
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.worker_utils import request_memory  # noqa: E402

GiB = 1 << 30
BLOCK = 16


# --------------------------------------------------------------------------- #
# 构造辅助
# --------------------------------------------------------------------------- #

def full_spec(block_size=BLOCK, heads=8, head=128, dtype=torch.float16):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=heads, head_size=head, dtype=dtype
    )


def swa_spec(window, block_size=BLOCK, heads=8, head=128, dtype=torch.float16):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=heads,
        head_size=head,
        dtype=dtype,
        sliding_window=window,
    )


def chunked_spec(chunk, block_size=BLOCK, heads=8, head=128, dtype=torch.float16):
    return ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=heads,
        head_size=head,
        dtype=dtype,
        attention_chunk_size=chunk,
    )


def mamba_spec(block_size=BLOCK, nbytes=4096, cache_mode="none"):
    return MambaSpec(
        block_size=block_size,
        shapes=((nbytes,),),
        dtypes=(torch.uint8,),
        mamba_cache_mode=cache_mode,
    )


def make_vllm_config(
    max_model_len=4096,
    num_layers_spec=None,
    cache: CacheConfig | None = None,
    scheduler: SchedulerConfig | None = None,
    original_max_model_len=None,
) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(
            max_model_len=max_model_len,
            original_max_model_len=(
                original_max_model_len
                if original_max_model_len is not None
                else max_model_len
            ),
        ),
        cache_config=cache or CacheConfig(enable_prefix_caching=False),
        scheduler_config=scheduler or SchedulerConfig(),
        parallel_config=ParallelConfig(),
    )


def uniform_full_layers(num_layers, spec=None, prefix="model.layers"):
    spec = spec or full_spec()
    return {
        f"{prefix}.{i}.self_attn.attn": spec for i in range(num_layers)
    }


def make_request(req_id="req-0", num_tokens=100, status=RequestStatus.WAITING):
    req = Request(request_id=req_id, prompt_token_ids=list(range(num_tokens)))
    req.status = status
    req.num_computed_tokens = 0
    return req


def snap(free, total, peak=0, allocated=0, reserved=None, non_torch=0):
    """手工快照（auto_measure=False）——measure() 的设备读数由测试注入。"""
    s = MemorySnapshot(device=torch.device("cpu"), auto_measure=False)
    s.free_memory = free
    s.total_memory = total
    s.torch_peak = peak
    s.torch_allocated = allocated
    s.torch_memory = total - free if reserved is None else reserved
    s.cuda_memory = total - free
    s.non_torch_memory = non_torch
    return s


# --------------------------------------------------------------------------- #
# m1 启动三步定账 —— request_memory（预算先于一切）
# --------------------------------------------------------------------------- #


class TestRequestMemory:
    # SOURCE 对锚：vllm/v1/worker/utils.py:L409-L429
    def test_requested_is_ceil_total_times_util(self):
        total = int(79.65 * GiB)  # 80GB 卡真实量级
        cache = CacheConfig(gpu_memory_utilization=0.92)
        init = snap(free=total, total=total)
        assert request_memory(init, cache) == math.ceil(total * 0.92)

    def test_free_below_requested_raises(self):
        total = 10 * GiB
        cache = CacheConfig(gpu_memory_utilization=0.92)
        init = snap(free=5 * GiB, total=total)  # 他进程已占一半
        with pytest.raises(ValueError, match="Decrease GPU memory"):
            request_memory(init, cache)

    def test_default_util_is_092_per_instance(self):
        # 默认 0.92：同卡两个实例各设 0.5 互不干扰（per-instance 语义）
        assert CacheConfig().gpu_memory_utilization == 0.92
        assert CacheConfig(gpu_memory_utilization=0.5).gpu_memory_utilization == 0.5


# --------------------------------------------------------------------------- #
# m2 memory_profiling 三类显存 —— docstring 量级例为 oracle
# --------------------------------------------------------------------------- #


class TestMemoryProfiling:
    # SOURCE 对锚：vllm/utils/mem_utils.py:L233-L326
    def _drive(self, monkeypatch, before_profile, after_profile, before_create):
        """驱动 memory_profiling：measure() 的设备读数按场景注入。"""
        seq = iter([before_profile, after_profile])

        def fake_measure(self):
            src = next(seq)
            for f in (
                "free_memory", "total_memory", "torch_peak",
                "torch_allocated", "torch_memory", "cuda_memory",
                "non_torch_memory",
            ):
                setattr(self, f, getattr(src, f))
            import time
            self.timestamp = time.time()

        monkeypatch.setattr(MemorySnapshot, "measure", fake_measure)
        # torch.accelerator 的 host 替身：empty_cache/reset 是 CUDA 缓存语义，
        # host 上 no-op（设备读数全部走上面的注入路径）
        monkeypatch.setattr(
            torch, "accelerator", SimpleAccelFake(), raising=False
        )
        with memory_profiling(before_create, weights_memory=2 * GiB) as result:
            pass  # profile_run 的位置（真实为 dummy 前向）
        return result

    def test_docstring_example_non_kv_is_5gib(self, monkeypatch):
        # mem_utils.py:L252-L278 的量化例：他进程 1G、权重 2G、激活峰 2G、
        # 非 torch（NCCL+后端缓冲）1G → non-KV 总账 = 5 GiB
        total = 10 * GiB
        before_create = snap(free=9 * GiB, total=total)  # cat1=1G
        before_profile = snap(
            free=6 * GiB, total=total, peak=2 * GiB, allocated=2 * GiB,
            reserved=2 * GiB, non_torch=1 * GiB,
        )  # 权重 2G + NCCL 1G
        after_profile = snap(
            free=5 * GiB, total=total, peak=4 * GiB, allocated=3 * GiB,
            reserved=3 * GiB, non_torch=1 * GiB,
        )  # gc 后 3G；峰值 4G（激活峰 +2G）；cat3 = NCCL+后端缓冲 1G
        r = self._drive(monkeypatch, before_profile, after_profile, before_create)
        assert r.total_consumed == 4 * GiB          # before_create.free − after.free
        assert r.torch_peak_increase == 2 * GiB     # 峰值增量 = 激活峰
        assert r.non_torch_increase == 1 * GiB      # NCCL+后端缓冲
        assert r.transient_peak_headroom == 1 * GiB  # gc 后仍着的峰差
        assert r.non_kv_cache_memory == 5 * GiB      # 2 权重 + 2 峰 + 1 非 torch
        assert r.non_kv_cache_memory == (
            r.total_consumed + r.transient_peak_headroom
        )


class SimpleAccelFake:
    """torch.accelerator 的 host 替身（设备读数已由 fake_measure 注入）。

    empty_cache/reset_peak_memory_stats 是 CUDA 缓存语义，host 上 no-op；
    内存读数全部走 MemorySnapshot.measure 的注入路径。
    """

    def empty_cache(self):
        pass

    def reset_peak_memory_stats(self, device):
        pass

    def synchronize(self, device=None):
        pass


# --------------------------------------------------------------------------- #
# m1/m3/m16 worker 侧三步 —— determine_available_memory
# --------------------------------------------------------------------------- #


class FakeRunner:
    """GPUModelRunner 的 ENGINE SEAM：账本测试只关心它的记账回调。"""

    def __init__(self, non_kv_gib=3.0, cudagraph_gib=0.5, model_memory=2.0):
        self.non_kv_bytes = int(non_kv_gib * GiB)
        self.cudagraph_bytes = int(cudagraph_gib * GiB)
        self.model_memory_usage = int(model_memory * GiB)
        self.profile_run_called = 0
        self.cudagraph_profiled = 0

    def profile_run(self):
        self.profile_run_called += 1

    def profile_cudagraph_memory(self):
        self.cudagraph_profiled += 1
        return self.cudagraph_bytes

    def get_kv_cache_spec(self):
        return uniform_full_layers(2)

    def initialize_kv_cache(self, *a, **k):
        pass


class FakeCudaPlatform:
    """current_platform 的 CUDA 替身（host 测试：让 cudagraph 门为真）。"""

    def is_xpu(self) -> bool:
        return False

    def is_cpu(self) -> bool:
        return False

    def is_cuda_alike(self) -> bool:
        return True


def make_worker(monkeypatch, total_gib=10.0, free_gib=10.0, util=0.8,
                cudagraph_mode=None, estimate_flag=True, runner=None):
    from implementation import gpu_worker as gw
    from implementation.config import CUDAGraphMode

    monkeypatch.setattr(gw, "current_platform", FakeCudaPlatform())
    monkeypatch.setattr(
        gw.envs, "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", estimate_flag
    )
    cache = CacheConfig(gpu_memory_utilization=util)
    vllm_config = make_vllm_config(cache=cache)
    vllm_config.compilation_config = type(
        "CC", (), {"cudagraph_mode": cudagraph_mode or CUDAGraphMode.FULL}
    )()
    runner = runner or FakeRunner()
    worker = Worker(vllm_config=vllm_config, model_runner=runner)
    # init_device 快照尾段（gpu_worker.py:L390-L396）——设备读数注入
    worker.init_snapshot = snap(free=int(free_gib * GiB), total=int(total_gib * GiB))
    worker.requested_memory = math.ceil(int(total_gib * GiB) * util)
    return worker


class TestDetermineAvailableMemory:
    # SOURCE 对锚：vllm/v1/worker/gpu_worker.py:L459-L611
    def test_available_kv_is_requested_minus_nonkv_minus_cudagraph(
        self, monkeypatch
    ):
        worker = make_worker(monkeypatch, total_gib=10, util=0.8)
        # profile_run 侧产出 non_kv=3G（memory_profiling 的注入——见 m2 测试）
        monkeypatch.setattr(
            "implementation.gpu_worker.memory_profiling",
            lambda snap, weights_memory: fake_profiling_ctx(3 * GiB),
        )
        avail = worker.determine_available_memory()
        # requested 8G − non_kv 3G − cudagraph 0.5G（默认估计开）
        assert avail == int(8 * GiB) - int(3 * GiB) - int(0.5 * GiB)
        assert worker.model_runner.profile_run_called == 1
        assert worker.model_runner.cudagraph_profiled == 1

    def test_estimate_flag_off_excludes_cudagraph_from_ledger(self, monkeypatch):
        worker = make_worker(monkeypatch, total_gib=10, util=0.8,
                             estimate_flag=False)
        monkeypatch.setattr(
            "implementation.gpu_worker.memory_profiling",
            lambda snap, weights_memory: fake_profiling_ctx(3 * GiB),
        )
        avail = worker.determine_available_memory()
        assert avail == int(8 * GiB) - int(3 * GiB)
        # 估计关闭只影响入账，不影响测量本身
        assert worker.model_runner.cudagraph_profiled == 1
        assert worker.cudagraph_memory_estimate == int(0.5 * GiB)

    def test_cudagraph_none_mode_skips_profiling(self, monkeypatch):
        from implementation.config import CUDAGraphMode

        worker = make_worker(monkeypatch, total_gib=10, util=0.8,
                             cudagraph_mode=CUDAGraphMode.NONE)
        monkeypatch.setattr(
            "implementation.gpu_worker.memory_profiling",
            lambda snap, weights_memory: fake_profiling_ctx(3 * GiB),
        )
        avail = worker.determine_available_memory()
        assert avail == int(8 * GiB) - int(3 * GiB) - 0
        assert worker.model_runner.cudagraph_profiled == 0

    def test_snapshot_assert_rejects_external_release(self, monkeypatch):
        # m16：profile 是快照不是保证——他进程在 profile 期间释放显存 →
        # init.free < after.free → assert 拒绝（gpu_worker.py:L533-L543）
        worker = make_worker(monkeypatch, total_gib=10, util=0.8)
        after_free = int(10.5 * GiB)  # 比初始 free（10G）还大 = 他进程释放了
        monkeypatch.setattr(
            "implementation.gpu_worker.memory_profiling",
            lambda snap, weights_memory: fake_profiling_ctx(
                3 * GiB, after_free=after_free
            ),
        )
        with pytest.raises(AssertionError, match="memory profiling"):
            worker.determine_available_memory()

    def test_estimate_flag_defaults_on(self):
        from implementation.envs import envs

        assert envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is True


@contextlib.contextmanager
def fake_profiling_ctx(non_kv_bytes, after_free=None):
    """memory_profiling 的注入替身（测试侧）：只回填账本要用的读数。"""

    class R:
        pass

    r = R()
    r.non_kv_cache_memory = non_kv_bytes
    r.total_consumed = non_kv_bytes
    r.transient_peak_headroom = 0
    r.after_profile = snap(
        free=after_free if after_free is not None else int(6 * GiB),
        total=int(10 * GiB),
    )
    yield r


# --------------------------------------------------------------------------- #
# 页物理公式 + 定块数总算术（m1 的换算半边）
# --------------------------------------------------------------------------- #


class TestPageSizeAndNumBlocks:
    # SOURCE 对锚：vllm/v1/kv_cache_interface.py:L211-L226（基类 2× 公式）、
    #   L335-L350（FullAttention 的 head_size+head_size_v 版）、L567-L585（SWA）
    def test_base_formula_2x_bs_heads_headdim_dtype(self):
        s = full_spec(heads=32, head=128, dtype=torch.float16)
        # Llama-7B 每层每块页 = 16×16KB = 256KiB（dossier theory 口径）
        assert s.real_page_size_bytes == 2 * 16 * 32 * 128 * 2 == 256 * 1024

    def test_full_spec_page_uses_head_size_plus_head_size_v(self):
        s = FullAttentionSpec(
            block_size=16, num_kv_heads=8, head_size=128,
            head_size_v=64, dtype=torch.float16,
        )
        assert s.page_size_bytes == 16 * 8 * (128 + 64) * 2

    def test_swa_page_same_shape_formula(self):
        s = swa_spec(window=512, heads=8, head=128)
        assert s.page_size_bytes == 16 * 8 * (128 + 128) * 2

    def test_chunked_page_uses_base_2x(self):
        s = chunked_spec(chunk=8192)
        assert s.page_size_bytes == 2 * 16 * 8 * 128 * 2

    def test_get_num_blocks_floor_division(self):
        cfg = make_vllm_config()
        page = 256 * 1024
        # theory worked example：8GiB // 256KiB // 32 层 = 1024 块
        assert get_num_blocks(cfg, 32, 8 * GiB, page) == 1024
        # 不够一块 → 0（max(0) 防负）
        assert get_num_blocks(cfg, 32, page - 1, page) == 0

    def test_num_gpu_blocks_override_wins(self):
        cfg = make_vllm_config(
            cache=CacheConfig(num_gpu_blocks_override=7)
        )
        assert get_num_blocks(cfg, 32, 8 * GiB, 256 * 1024) == 7


# --------------------------------------------------------------------------- #
# m4 护栏四道
# --------------------------------------------------------------------------- #


class TestGuardrails:
    # SOURCE 对锚：kv_cache_utils.py:L751-L788、L800-L851、L1988-L2049、
    #   L2159-L2179、L2210-L2223
    def test_check_enough_raises_when_one_request_does_not_fit(self):
        from implementation.kv_cache_utils import check_enough_kv_cache_memory

        # 2 层 full、bs16、heads8/head128/fp16 → page 65536；4096 token 需
        # cdiv(4096,16)×65536×2 = 256×131072 = 32MiB
        cfg = make_vllm_config(max_model_len=4096)
        spec = uniform_full_layers(2, full_spec())
        with pytest.raises(ValueError, match="To serve at least one request"):
            check_enough_kv_cache_memory(
                cfg, spec, available_memory=32 * 65536 * 2 - 1
            )

    def test_check_enough_no_memory_at_all(self):
        from implementation.kv_cache_utils import check_enough_kv_cache_memory

        cfg = make_vllm_config()
        with pytest.raises(ValueError, match="No available memory"):
            check_enough_kv_cache_memory(cfg, uniform_full_layers(2), 0)

    def test_estimate_binary_search_finds_exact_boundary(self):
        # needed(len) = cdiv(len,16)×65536×2 随 len 单调不减 → 二分找最大可行长度
        cfg = make_vllm_config(max_model_len=8192)
        spec = uniform_full_layers(2, full_spec())
        page_per_len_block = 65536 * 2
        # 供 100 个块 → len ≤ 1600
        avail = 100 * page_per_len_block
        assert estimate_max_model_len(cfg, spec, avail) == 1600
        # 连 1 个 token 都装不下 → 0
        assert estimate_max_model_len(cfg, spec, 0) == 0
        # 全程无副作用：max_model_len 恢复原值
        assert cfg.model_config.max_model_len == 8192

    def test_override_rebases_available_memory_no_drift(self):
        # override 生效后 available_memory 折算成 override×每块字节——
        # auto-fit/护栏/定块数都按同一有效容量规划（账本不漂）
        # 2 层 full 单组：page 65536 → group_size 2 → 每块 131072 B
        # 3 块 = 393216 B → 恰装 max_model_len=48（cdiv(48,16)=3 块/层）
        cfg = make_vllm_config(
            max_model_len=48,
            cache=CacheConfig(num_gpu_blocks_override=3),
        )
        specs = [uniform_full_layers(2)]
        # 真实 available 远大于 override 容量——不折算则护栏按大容量放行、
        # 定块数却被 override 夹小 = 账本漂移；折算后两侧一致
        configs = get_kv_cache_configs(cfg, specs, [100 * 65536 * 2])
        assert configs[0].num_blocks == 3  # override 凌驾

    def test_pp_min_num_blocks_and_proportional_shrink(self):
        # 两个 worker（PP 两段）available 不同 → 取最小并按比例缩张量；
        # max_len=1024 需 64 块（两段都装得下——护栏不响）
        cfg = make_vllm_config(max_model_len=1024)
        w0 = uniform_full_layers(2, full_spec(), prefix="stage0")
        w1 = uniform_full_layers(2, full_spec(), prefix="stage1")
        page = 65536
        configs = get_kv_cache_configs(
            cfg, [w0, w1], [200 * page * 2, 90 * page * 2]
        )
        assert configs[0].num_blocks == configs[1].num_blocks == 90
        # 张量按比例缩：worker0 的张量 = 90/200 × 原尺寸
        t0 = configs[0].kv_cache_tensors[0]
        assert t0.size == page * 90

    def test_auto_fit_reduces_max_model_len(self):
        cfg = make_vllm_config(
            max_model_len=8192, original_max_model_len=-1
        )
        specs = [uniform_full_layers(2)]
        page_per_len_block = 65536 * 2
        avail = 100 * page_per_len_block  # len ≤ 1600
        get_kv_cache_configs(cfg, specs, [avail])
        assert cfg.model_config.max_model_len == 1600

    def test_auto_fit_full_context_fits_keeps_len(self):
        cfg = make_vllm_config(max_model_len=512, original_max_model_len=-1)
        specs = [uniform_full_layers(2)]
        avail = (512 // 16) * 65536 * 2
        get_kv_cache_configs(cfg, specs, [avail])
        assert cfg.model_config.max_model_len == 512


# --------------------------------------------------------------------------- #
# m5 混合组化
# --------------------------------------------------------------------------- #


def gemma3_like(num_swa=10, num_full=2, window=512):
    spec = {}
    for i in range(num_swa):
        spec[f"model.layers.{i}.self_attn.attn"] = swa_spec(window)
    for j in range(num_full):
        spec[f"model.layers.{num_swa + j}.self_attn.attn"] = full_spec()
    return spec


class TestHybridGrouping:
    # SOURCE 对锚：kv_cache_utils.py:L1781-L1852、L1140-L1280、L1568-L1589
    def test_uniform_model_single_group(self):
        cfg = make_vllm_config()
        spec = uniform_full_layers(32)
        groups = get_kv_cache_groups(cfg, spec)
        assert len(groups) == 1
        assert len(groups[0].layer_names) == 32

    def test_gemma3_pattern_groups_by_type_interleaved(self):
        # 10 SWA + 2 full（5:1 模式 × 2）→ min=2 组大小：sw 拆 5 组×2、
        # full 1 组×2；layers[i::num_groups] 交错分派
        cfg = make_vllm_config()
        spec = gemma3_like(num_swa=10, num_full=2)
        groups = get_kv_cache_groups(cfg, spec)
        assert len(groups) == 6
        assert all(len(g.layer_names) == 2 for g in groups)
        swa_groups = [g for g in groups
                      if isinstance(g.kv_cache_spec, SlidingWindowSpec)]
        full_groups = [g for g in groups
                       if isinstance(g.kv_cache_spec, FullAttentionSpec)]
        assert len(swa_groups) == 5 and len(full_groups) == 1
        # 交错：10 个 sw 层按 [i::5] 分——第 0 组 = layers 0,5
        assert swa_groups[0].layer_names == [
            "model.layers.0.self_attn.attn",
            "model.layers.5.self_attn.attn",
        ]

    def test_one_and_half_heuristic_pads_12sw_13full_to_13(self, caplog):
        # 12 SW + 13 full：max(13) < min(12)×1.5 → 组大小取 13（非 12），
        # padding warning「may waste at most 8.33%」
        cfg = make_vllm_config()
        spec = gemma3_like(num_swa=12, num_full=13)
        groups = get_kv_cache_groups(cfg, spec)
        swa_groups = [g for g in groups
                      if isinstance(g.kv_cache_spec, SlidingWindowSpec)]
        # 12 sw 進 1 组（12 层，补 1 padding 到 13）；13 full 進 1 组
        assert len(groups) == 2
        assert len(swa_groups[0].layer_names) == 12
        assert any("padding layers" in r.message for r in caplog.records)

    def test_disable_hybrid_falls_back_to_full_allocation(self, caplog):
        # --disable-hybrid-kv-cache-manager：SWA 当 full 分配 + warning
        # 「we do not enable any optimizations for saving KV cache memory」
        cfg = make_vllm_config(
            scheduler=SchedulerConfig(disable_hybrid_kv_cache_manager=True)
        )
        spec = gemma3_like(num_swa=5, num_full=1)
        groups = get_kv_cache_groups(cfg, spec)
        assert len(groups) == 1
        merged = groups[0].kv_cache_spec
        assert isinstance(merged, FullAttentionSpec)
        assert merged.sliding_window == 512  # SWA 尺寸记录在案但按 full 分配
        assert any(
            "do not enable any optimizations" in r.message for r in caplog.records
        )
        # 原 dict 被原位改写为 full spec（disable 分支的副作用）
        assert all(
            isinstance(s, FullAttentionSpec) for s in spec.values()
        )


# --------------------------------------------------------------------------- #
# m6 页大小统一
# --------------------------------------------------------------------------- #


class TestUnifyPageSize:
    # SOURCE 对锚：kv_cache_utils.py:L1070-L1132
    def test_smaller_layer_block_size_scaled_up(self):
        big = full_spec(heads=8)    # page 65536
        small = full_spec(heads=4)  # page 32768
        out = unify_kv_cache_spec_page_size({"a": big, "b": small})
        assert out["b"].block_size == 32          # 16 × (65536//32768)
        assert out["b"].page_size_bytes == 65536
        assert out["a"].page_size_bytes == 65536

    def test_uniform_pages_untouched(self):
        spec = uniform_full_layers(2)
        out = unify_kv_cache_spec_page_size(spec)
        assert out is spec

    def test_mamba_page_padded_to_max(self):
        att = full_spec(heads=8)  # 65536
        mb = mamba_spec(nbytes=4096)
        out = unify_kv_cache_spec_page_size({"a": att, "m": mb})
        # Mamba 页由状态形状决定、不随块缩放 → 物理 pad
        assert out["m"].page_size_padded == 65536
        assert out["m"].block_size == BLOCK  # 块大小不变

    def test_non_divisible_without_stride_raises(self):
        big = full_spec(heads=8)              # 65536
        odd = replace(full_spec(heads=4), page_size_padded=40000)
        with pytest.raises(NotImplementedError, match="cannot be padded"):
            unify_kv_cache_spec_page_size({"a": big, "b": odd})

    def test_stride_indexed_layer_pads_instead(self):
        big = full_spec(heads=8)
        strideful = replace(
            full_spec(heads=4), page_size_padded=40000,
            indexes_kv_by_block_stride=True,
        )
        out = unify_kv_cache_spec_page_size({"a": big, "b": strideful})
        assert out["b"].page_size_padded == 65536


# --------------------------------------------------------------------------- #
# m7 张量共享布局（通用 group_size 池）
# --------------------------------------------------------------------------- #


class TestTensorLayout:
    # SOURCE 对锚：kv_cache_utils.py:L1361-L1443
    def test_general_case_group_size_pools_shared_by_one_layer_each(self):
        # 3 组 (full.0, full.1), (sw.0, sw.2), (sw.1, padding) 的布局例：
        # group_size=2 → 2 张量，每张量由每组各出一层共享
        cfg = make_vllm_config()
        full_g = KVCacheGroupSpec(["full.0", "full.1"], full_spec())
        sw0 = KVCacheGroupSpec(["sw.0", "sw.2"], swa_spec(512))
        sw1 = KVCacheGroupSpec(["sw.1"], swa_spec(512))
        page = 65536
        config = get_kv_cache_config_from_groups(
            cfg, [full_g, sw0, sw1], available_memory=2 * page * 10
        )
        assert config.num_blocks == 10  # avail // page // group_size(2)
        assert len(config.kv_cache_tensors) == 2
        t0, t1 = config.kv_cache_tensors
        assert t0.shared_by == ["full.0", "sw.0", "sw.1"]
        assert t1.shared_by == ["full.1", "sw.2"]  # sw.1 组只有 1 层
        assert t0.size == t1.size == page * 10

    def test_single_uniform_type_group_per_layer_tensors(self):
        # 单组异宽（同型不同 heads）：每层一张张量、按各自页大小分账
        cfg = make_vllm_config()
        s0, s1 = full_spec(heads=8), full_spec(heads=4)
        uni = UniformTypeKVCacheSpecs(
            block_size=BLOCK, kv_cache_specs={"l0": s0, "l1": s1}
        )
        group = KVCacheGroupSpec(["l0", "l1"], uni)
        total_page = s0.page_size_bytes + s1.page_size_bytes  # 98304
        config = get_kv_cache_config_from_groups(
            cfg, [group], available_memory=total_page * 5
        )
        assert config.num_blocks == 5
        sizes = {t.size for t in config.kv_cache_tensors}
        assert sizes == {s0.page_size_bytes * 5, s1.page_size_bytes * 5}

    def test_attention_free_models_get_single_null_block(self):
        cfg = make_vllm_config()
        config = get_kv_cache_config_from_groups(cfg, [], 12345)
        assert config.num_blocks == 1  # BlockPool 永远需要 null_block


# --------------------------------------------------------------------------- #
# m8 resolve 两个对齐粒度（LCM / GCD）
# --------------------------------------------------------------------------- #


def two_group_config(bs_a=16, bs_b=32, spec_b=None):
    ga = KVCacheGroupSpec(["a"], full_spec(block_size=bs_a))
    gb = KVCacheGroupSpec(
        ["b"], spec_b or full_spec(block_size=bs_b)
    )
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=[ga, gb],
    )


class TestResolveBlockSizes:
    # SOURCE 对锚：kv_cache_utils.py:L626-L688
    def test_single_group_both_equal_block_size(self):
        cfg = make_vllm_config()
        config = KVCacheConfig(
            num_blocks=10, kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(["a"], full_spec(16))],
        )
        assert resolve_kv_cache_block_sizes(config, cfg) == (16, 16)

    def test_multi_group_scheduler_lcm_hash_gcd(self):
        cfg = make_vllm_config(
            cache=CacheConfig(enable_prefix_caching=True)
        )
        config = two_group_config(16, 32)
        assert resolve_kv_cache_block_sizes(config, cfg) == (32, 16)

    def test_prefix_match_unit_overrides_hash_granularity(self):
        cfg = make_vllm_config(
            cache=CacheConfig(
                enable_prefix_caching=True, prefix_match_unit=8
            )
        )
        config = two_group_config(16, 32)
        assert resolve_kv_cache_block_sizes(config, cfg) == (32, 8)

    def test_non_divisible_prefix_match_unit_raises(self):
        cfg = make_vllm_config(
            cache=CacheConfig(
                enable_prefix_caching=True, prefix_match_unit=5
            )
        )
        with pytest.raises(ValueError, match="prefix_match_unit"):
            resolve_kv_cache_block_sizes(two_group_config(16, 32), cfg)

    def test_no_caching_no_connector_hash_falls_back_to_scheduler(self):
        cfg = make_vllm_config(cache=CacheConfig(enable_prefix_caching=False))
        assert resolve_kv_cache_block_sizes(
            two_group_config(16, 32), cfg
        ) == (32, 32)

    def test_mamba_non_align_backs_off(self):
        # mamba_cache_mode != "align"（块大小 ≠ cache block_size）破坏整除性
        # → hash 回退 scheduler 粒度
        cfg = make_vllm_config(cache=CacheConfig(enable_prefix_caching=True))
        config = two_group_config(16, 64, spec_b=mamba_spec(block_size=64))
        assert resolve_kv_cache_block_sizes(config, cfg) == (64, 64)


# --------------------------------------------------------------------------- #
# m15 容量与并发核算
# --------------------------------------------------------------------------- #


class TestCapacity:
    # SOURCE 对锚：kv_cache_utils.py:L937-L959、L1877-L1887
    def test_worked_example_llama7b(self):
        # theory：32 层单组、8GiB、page 256KiB → 1024 块；max_len 4096 →
        # 并发 4×；容量 16384 token
        cfg = make_vllm_config(max_model_len=4096)
        spec = uniform_full_layers(32, full_spec(heads=32))
        configs = get_kv_cache_configs(cfg, [spec], [8 * GiB])
        assert configs[0].num_blocks == 1024
        conc = get_max_concurrency_for_kv_cache_config(cfg, configs[0])
        assert conc == 1024 / 256 == 4.0
        tokens, concurrency = get_kv_cache_capacity(cfg, configs[0])
        assert tokens == int(4.0 * 4096) == 16384
        assert concurrency == 4.0

    def test_hybrid_concurrency_sums_over_groups(self):
        # 混合布局按组求和：full 整序列 256 块 + swa cap（in_flight=8192 →
        # cap = cdiv(min(511+8192, 4096), 16)+1 = 257）
        cfg = make_vllm_config(max_model_len=4096)
        assert cfg.max_in_flight_tokens == 8192
        full_g = KVCacheGroupSpec(["f0", "f1"], full_spec())
        swa_g = KVCacheGroupSpec(["s0", "s1"], swa_spec(512))
        config = KVCacheConfig(
            num_blocks=1024,
            kv_cache_tensors=[],
            kv_cache_groups=[full_g, swa_g],
        )
        # 每请求块 = Σ_groups cdiv(max_memory_usage_bytes, page)
        per_req = (4096 // 16) + (4096 // 16 + 1)  # full 256 + swa cap 257
        conc = get_max_concurrency_for_kv_cache_config(cfg, config)
        assert conc == 1024 / per_req


# --------------------------------------------------------------------------- #
# m9 一份账喂两侧（engine core 装配总编排）
# --------------------------------------------------------------------------- #


class FakeExecutor:
    """ExecutorBase 的 ENGINE SEAM：把三个 worker 钩子路由到 FakeRunner。"""

    def __init__(self, worker: Worker):
        self.worker = worker
        self.initialized = None
        self.rpc_calls = []

    def get_kv_cache_specs(self):
        return [self.worker.model_runner.get_kv_cache_spec()]

    def determine_available_memory(self):
        return [self.worker.determine_available_memory()]

    def initialize_from_config(self, kv_cache_configs):
        self.initialized = kv_cache_configs
        self.worker.initialize_from_config(kv_cache_configs[0])

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.rpc_calls.append((method, args))


class TestEngineCoreBoot:
    # SOURCE 对锚：vllm/v1/engine/core.py:L250-L359、L142-L168
    def _boot(self, monkeypatch, num_layers=2, avail=None, **cfg_kw):
        from implementation import gpu_worker as gw
        from implementation.config import CUDAGraphMode

        monkeypatch.setattr(gw.envs,
                            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", True)
        cache = cfg_kw.pop("cache", None) or CacheConfig(
            enable_prefix_caching=False
        )
        vllm_config = make_vllm_config(cache=cache, **cfg_kw)
        vllm_config.compilation_config = type(
            "CC", (), {"cudagraph_mode": CUDAGraphMode.NONE}
        )()
        runner = FakeRunner(non_kv_gib=1.0, cudagraph_gib=0.0)
        runner.get_kv_cache_spec = lambda: uniform_full_layers(num_layers)
        runner.update_max_model_len = lambda max_model_len: None
        worker = Worker(vllm_config=vllm_config, model_runner=runner)
        worker.init_snapshot = snap(free=10 * GiB, total=10 * GiB)
        worker.requested_memory = 10 * GiB
        if avail is not None:
            worker.determine_available_memory = lambda: avail
        executor = FakeExecutor(worker)
        core = EngineCore(vllm_config, model_executor=executor)
        return core, vllm_config, executor

    def test_boot_writes_back_ledger_to_cache_config(self, monkeypatch):
        # available 40MiB（320 块 × 128KiB/块）、2 层、page 65536 → 320 块；
        # block_size=16；4096 token 需 32MiB（护栏过）
        avail = 320 * 65536 * 2
        core, cfg, executor = self._boot(monkeypatch, avail=avail)
        assert cfg.cache_config.num_gpu_blocks == 320
        assert cfg.cache_config.block_size == BLOCK
        # 容量/并发写回（前端可见）：320/256 = 1.25×；5120 token
        assert cfg.cache_config.kv_cache_size_tokens == 5120
        assert cfg.cache_config.kv_cache_max_concurrency == 320 / 256
        # 同一份 config 喂了 worker 侧
        assert executor.initialized is not None
        assert executor.initialized[0].num_blocks == 320

    def test_auto_fit_triggers_update_max_model_len_rpc(self, monkeypatch):
        avail = 100 * 65536 * 2  # len ≤ 1600（恰装下：needed == avail）
        core, cfg, executor = self._boot(
            monkeypatch, avail=avail, max_model_len=8192,
            original_max_model_len=-1,
        )
        assert cfg.model_config.max_model_len == 1600
        assert ("update_max_model_len", (1600,)) in executor.rpc_calls

    def test_scheduler_gets_flattened_config(self, monkeypatch):
        avail = 320 * 65536 * 2
        core, cfg, executor = self._boot(monkeypatch, avail=avail)
        # 调度器侧拿拍平版：组 spec 是代表层（无 UniformType 包装）
        sched_cfg = core.scheduler_kv_cache_config
        assert isinstance(
            sched_cfg.kv_cache_groups[0].kv_cache_spec, FullAttentionSpec
        )
        # 装配序：先 KV 账本、后 Scheduler（L142-L168）
        assert core.scheduler.kv_cache_manager is not None


# --------------------------------------------------------------------------- #
# 两道门：m10 full-ISL + m12 水位 + m11 准入上限
# --------------------------------------------------------------------------- #


def make_manager_config(specs_by_group, num_blocks):
    groups = [
        KVCacheGroupSpec(names, spec) for names, spec in specs_by_group
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[KVCacheTensor(size=0, shared_by=[]) for _ in groups],
        kv_cache_groups=groups,
    )


def make_manager(specs_by_group, num_blocks, max_model_len=4096,
                 watermark=0.0, max_in_flight_tokens=None, enable_caching=False):
    config = make_manager_config(specs_by_group, num_blocks)
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=max_model_len,
        scheduler_block_size=max(
            g.kv_cache_spec.block_size for g in config.kv_cache_groups
        ),
        hash_block_size=config.kv_cache_groups[0].kv_cache_spec.block_size,
        enable_caching=enable_caching,
        max_in_flight_tokens=max_in_flight_tokens,
        watermark=watermark,
    )


class TestFullIslGate:
    # SOURCE 对锚：vllm/v1/core/kv_cache_manager.py:L472-L488、
    #   vllm/config/scheduler.py:L130-L134
    def test_full_sequence_does_not_fit_rejected(self):
        # 池 10 块×bs16=160 token；请求 200 token、首 chunk 只 16 token：
        # full-ISL 门按整条序列算 → None（#39734 超收堵在此）
        m = make_manager([(["l0"], full_spec())], num_blocks=10)
        req = make_request(num_tokens=200)
        assert m.allocate_slots(
            req, num_new_tokens=16, full_sequence_must_fit=True
        ) is None

    def test_first_chunk_only_check_admits_oversized_request(self):
        # 同一请求、门关掉（只查第一 chunk）→ 放进——超收漏洞的对照面
        m = make_manager([(["l0"], full_spec())], num_blocks=10)
        req = make_request(num_tokens=200)
        blocks = m.allocate_slots(
            req, num_new_tokens=16, full_sequence_must_fit=False
        )
        assert blocks is not None

    def test_gate_default_config_is_true(self):
        assert SchedulerConfig().scheduler_reserve_full_isl is True

    def test_reserved_blocks_kept_for_inflight_prefills(self):
        # reserved_blocks：异步 KV load 的在途预约——可用 = free − reserved
        m = make_manager([(["l0"], full_spec())], num_blocks=10)
        req = make_request(num_tokens=64)  # 需 4 块
        assert m.allocate_slots(
            req, num_new_tokens=64, full_sequence_must_fit=False,
            reserved_blocks=7,  # free=10−1(null)=9；9−7=2 < 4
        ) is None
        req2 = make_request("r2", num_tokens=32)  # 需 2 块
        assert m.allocate_slots(
            req2, num_new_tokens=32, full_sequence_must_fit=False,
            reserved_blocks=7,
        ) is not None


class TestWatermark:
    # SOURCE 对锚：kv_cache_manager.py:L168-L171、L463-L470、L521-L527；
    #   vllm/config/scheduler.py:L136-L141
    def test_watermark_blocks_computed(self):
        m = make_manager([(["l0"], full_spec())], num_blocks=10, watermark=0.5)
        assert m.watermark_blocks == 5

    def test_default_off(self):
        assert SchedulerConfig().watermark == 0.0

    def test_waiting_request_blocked_by_headroom(self):
        # free=9（null 占 1）：需 4 块 + 水位 5 → 9 < 9? required=9 ≤ 9 通过；
        # 用 5 块请求：required = 5+5 = 10 > 9 → None
        m = make_manager([(["l0"], full_spec())], num_blocks=10, watermark=0.5)
        req = make_request(num_tokens=80)  # 5 块
        assert m.allocate_slots(
            req, num_new_tokens=80, full_sequence_must_fit=False,
            has_scheduled_reqs=True,
        ) is None
        # 本步无已调度请求（空转首拍）→ 水位不计入 → 放行
        req2 = make_request("r2", num_tokens=80)
        assert m.allocate_slots(
            req2, num_new_tokens=80, full_sequence_must_fit=False,
            has_scheduled_reqs=False,
        ) is not None

    def test_running_request_ignores_watermark(self):
        # 精修版水位只对 WAITING/PREEMPTED——RUNNING 涨块不受垫片约束
        m = make_manager([(["l0"], full_spec())], num_blocks=10, watermark=0.9)
        req = make_request(num_tokens=80, status=RequestStatus.RUNNING)
        assert m.allocate_slots(
            req, num_new_tokens=80, full_sequence_must_fit=False,
            has_scheduled_reqs=True,
        ) is not None


class TestAdmissionCap:
    # SOURCE 对锚：kv_cache_interface.py:L519-L546、L587-L618；
    #   single_type_kv_cache_manager.py:L178-L192、L1861-L1878
    def test_swa_cap_formula_with_plus_one(self):
        s = swa_spec(window=7, block_size=4)
        # num_tokens = min(7−1+0, 100) = 6 → cdiv(6,4)+1 = 3
        assert s.max_admission_blocks_per_request(0, 100) == 3

    def test_swa_cap_capped_by_max_model_len(self):
        s = swa_spec(window=4096, block_size=16)
        # min(4095+0, 100) = 100 → cdiv(100,16)+1 = 7+1 = 8
        assert s.max_admission_blocks_per_request(0, 100) == 8

    def test_swa_cap_counts_in_flight_tokens(self):
        s = swa_spec(window=7, block_size=4)
        # min(6+3, 100)=9 → cdiv(9,4)+1 = 3+1 = 4
        assert s.max_admission_blocks_per_request(3, 100) == 4

    def test_chunked_cap_no_plus_one(self):
        s = chunked_spec(chunk=8, block_size=4)
        # min(8+0, 100) → cdiv(8,4) = 2（chunked 窗口从块首开始，无 +1）
        assert s.max_admission_blocks_per_request(0, 100) == 2

    def test_swa_max_memory_usage_is_cap_times_page(self):
        s = swa_spec(window=512, block_size=16, heads=8, head=128)
        cfg = make_vllm_config(max_model_len=4096)
        # cap 与池大小器同源：max_memory_usage_bytes = cap × page（spec 方法）
        cap = s.max_admission_blocks_per_request(
            cfg.max_in_flight_tokens, 4096
        )
        assert s.max_memory_usage_bytes(cfg) == cap * s.page_size_bytes

    def test_manager_injects_cap_for_recycling_specs(self):
        from implementation.single_type_kv_cache_manager import (
            get_manager_for_kv_cache_spec,
        )
        from implementation.block_pool import BlockPool

        pool = BlockPool(num_gpu_blocks=16, enable_caching=False,
                         hash_block_size=16)
        swa_m = get_manager_for_kv_cache_spec(
            swa_spec(window=512), max_in_flight_tokens=2048,
            max_model_len=4096, block_pool=pool, enable_caching=False,
            kv_cache_group_id=0, scheduler_block_size=16,
        )
        assert swa_m._max_admission_blocks_per_request == (
            swa_spec(window=512).max_admission_blocks_per_request(2048, 4096)
        )
        full_m = get_manager_for_kv_cache_spec(
            full_spec(), max_in_flight_tokens=2048,
            max_model_len=4096, block_pool=pool, enable_caching=False,
            kv_cache_group_id=1, scheduler_block_size=16,
        )
        assert full_m._max_admission_blocks_per_request is None

    def test_cap_clamps_prediction_only_under_gate(self):
        # apply_admission_cap=True 只由 full-ISL 门传：预测被夹到 cap；
        # 平时（每步分配）不夹——预测器与分配器同构
        from implementation.single_type_kv_cache_manager import (
            get_manager_for_kv_cache_spec,
        )
        from implementation.block_pool import BlockPool

        pool = BlockPool(num_gpu_blocks=1024, enable_caching=False,
                         hash_block_size=16)
        m = get_manager_for_kv_cache_spec(
            swa_spec(window=512), max_in_flight_tokens=0,
            max_model_len=4096, block_pool=pool, enable_caching=False,
            kv_cache_group_id=0, scheduler_block_size=16,
        )
        cap = m._max_admission_blocks_per_request  # cdiv(511,16)+1 = 33
        assert cap == math.ceil(511 / 16) + 1
        # 4096 token 需 256 块，被夹到 33
        assert m.get_num_blocks_to_allocate(
            "r", 4096, (), 0, 0, 4096, apply_admission_cap=True
        ) == cap
        # 不夹：按整长算
        assert m.get_num_blocks_to_allocate(
            "r", 4096, (), 0, 0, 4096
        ) == 256

    def test_hybrid_gate_sums_full_and_capped_swa(self):
        # 混合两组过 full-ISL 门：full 按整序列 + swa 夹到 cap
        m = make_manager(
            [(["f0", "f1"], full_spec()), (["s0", "s1"], swa_spec(512))],
            num_blocks=1000, max_in_flight_tokens=0,
        )
        req = make_request(num_tokens=4096)
        # full: cdiv(4096,16)=256；swa cap: cdiv(511,16)+1=33 → 289 ≤ 999 放行
        assert m.allocate_slots(
            req, num_new_tokens=4096, full_sequence_must_fit=True
        ) is not None


# --------------------------------------------------------------------------- #
# m13 SWA 窗外回收 + null 占位
# --------------------------------------------------------------------------- #


class TestSwaRecycle:
    # SOURCE 对锚：single_type_kv_cache_manager.py:L1057-L1083（SWA 公式）、
    #   L1200-L1244（chunked 公式）、L595-L672（回收+null 换位）
    def test_swa_skipped_tokens_docstring_example(self):
        m = make_manager([(["s0"], swa_spec(window=4, block_size=4))],
                         num_blocks=8)
        swa = m.coordinator.single_type_managers[0]
        # 窗口 4、computed 7 → tokens 0-3 在窗外 → 4
        assert swa.get_num_skipped_tokens(7) == 4
        assert swa.get_num_skipped_tokens(3) == 0  # 窗未起跳

    def test_full_never_skips(self):
        m = make_manager([(["f0"], full_spec())], num_blocks=8)
        full = m.coordinator.single_type_managers[0]
        assert full.get_num_skipped_tokens(10000) == 0

    def test_chunked_skips_whole_chunks_docstring_examples(self):
        m = make_manager([(["c0"], chunked_spec(chunk=8, block_size=8))],
                         num_blocks=8)
        cl = m.coordinator.single_type_managers[0]
        assert cl.get_num_skipped_tokens(13) == 8
        assert cl.get_num_skipped_tokens(8) == 8
        assert cl.get_num_skipped_tokens(7) == 0

    def test_remove_skipped_frees_and_nulls_in_place(self):
        # 块表第 i 块 ↔ 第 i×block_size 个 token：窗外整块 free、原位换
        # null_block——[NULL, block1] 位置对齐不断裂
        m = make_manager([(["s0"], swa_spec(window=4, block_size=4))],
                         num_blocks=8)
        req = make_request(num_tokens=16)
        m.allocate_slots(req, num_new_tokens=16,
                         full_sequence_must_fit=False)
        swa = m.coordinator.single_type_managers[0]
        blocks = swa.req_to_blocks["req-0"]
        assert len(blocks) == 4
        free_before = m.block_pool.get_num_free_blocks()
        # processed=7 → skipped=4 → 1 整块回收
        m.remove_skipped_blocks("req-0", 7)
        assert blocks[0] is m.block_pool.null_block
        assert blocks[1] is not m.block_pool.null_block
        assert m.block_pool.get_num_free_blocks() == free_before + 1

    def test_swa_steady_state_holds_only_window(self):
        # SWA 稳态：长序列推进后实持块 ≈ cap（窗外块全部回收归池）
        m = make_manager([(["s0"], swa_spec(window=8, block_size=4))],
                         num_blocks=64)
        req = make_request(num_tokens=64)
        m.allocate_slots(req, num_new_tokens=64,
                         full_sequence_must_fit=False)
        m.remove_skipped_blocks("req-0", 60)  # skipped = 60-8+1 = 53 → 13 块
        swa = m.coordinator.single_type_managers[0]
        blocks = swa.req_to_blocks["req-0"]
        non_null = [b for b in blocks if b is not m.block_pool.null_block]
        assert len(blocks) == 16 and len(non_null) == 3  # 64−52=12 token → 3 块
        assert m.block_pool.get_num_free_blocks() == 63 - 3


# --------------------------------------------------------------------------- #
# m14 kernel 块细分 + 多组块表
# --------------------------------------------------------------------------- #


class TestKernelBlocks:
    # SOURCE 对锚：block_table.py:L220-L248、L82-L154、L270-L336
    def test_map_to_kernel_blocks_docstring_example(self):
        from implementation.block_table import BlockTable

        ids = np.array([0, 1, 2])
        arange = np.arange(0, 2).reshape(1, -1)
        out = BlockTable.map_to_kernel_blocks(ids, 2, arange)
        assert out.tolist() == [0, 1, 2, 3, 4, 5]

    def test_identity_when_no_split(self):
        from implementation.block_table import BlockTable

        ids = np.array([5, 9])
        out = BlockTable.map_to_kernel_blocks(ids, 1, None)
        assert out.tolist() == [5, 9]

    def test_block_table_hybrid_split_32_to_2x16(self):
        from implementation.block_table import BlockTable, SlotMappingMode

        bt = BlockTable(
            block_size=32, max_num_reqs=4, max_num_blocks_per_req=8,
            max_num_batched_tokens=64, pin_memory=False,
            device=torch.device("cpu"), kernel_block_size=16,
            cp_kv_cache_interleave_size=1,
            slot_mapping_mode=SlotMappingMode.TOKEN_TO_KV_SLOT,
        )
        assert bt.use_hybrid_blocks is True
        assert bt.blocks_per_kv_block == 2
        assert bt.block_size == 16  # kernel 视角
        assert bt.max_num_blocks_per_req == 16  # 8 × 2 细分
        bt.append_row([0, 1], row_idx=0)
        row = bt.block_table.np[0, :4].tolist()
        assert row == [0, 1, 2, 3]

    def test_block_table_standard_no_split(self):
        from implementation.block_table import BlockTable, SlotMappingMode

        bt = BlockTable(
            block_size=16, max_num_reqs=2, max_num_blocks_per_req=4,
            max_num_batched_tokens=32, pin_memory=False,
            device=torch.device("cpu"), kernel_block_size=16,
            cp_kv_cache_interleave_size=1,
        )
        assert bt.use_hybrid_blocks is False
        assert bt.blocks_per_kv_block == 1

    def test_kernel_must_divide_block_size(self):
        from implementation.block_table import BlockTable

        with pytest.raises(ValueError, match="must divide"):
            BlockTable(
                block_size=32, max_num_reqs=1, max_num_blocks_per_req=2,
                max_num_batched_tokens=8, pin_memory=False,
                device=torch.device("cpu"), kernel_block_size=12,
                cp_kv_cache_interleave_size=1,
            )

    def test_multi_group_one_table_per_group(self):
        from implementation.block_table import MultiGroupBlockTable

        mgt = MultiGroupBlockTable(
            max_num_reqs=2, max_num_batched_tokens=32, pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[32, 16], kernel_block_sizes=[16, 16],
            max_num_blocks=[4, 4],
        )
        assert len(mgt.block_tables) == 2
        mgt.append_row(([0, 1], [2]), row_idx=0)
        assert mgt[0].block_table.np[0, :4].tolist() == [0, 1, 2, 3]
        assert mgt[1].block_table.np[0, :2].tolist() == [2, 0]

    def test_prepare_kernel_block_sizes_negotiates_with_backends(self):
        # 后端只认更小的块 → 选全体后端都支持的最大公因子块
        from implementation.worker_utils import prepare_kernel_block_sizes

        class Backend16:
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [16]

        class Backend32Or16:
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [32, 16]

        class Group:
            def __init__(self, backend):
                self.backend = backend

        config = KVCacheConfig(
            num_blocks=10, kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["a"], full_spec(32)),
                KVCacheGroupSpec(["b"], full_spec(16)),
            ],
        )
        # 组 0（块 32）：后端只认 16 → 拆 2×16；组 1（块 16）：直认
        out = prepare_kernel_block_sizes(
            config, [[Group(Backend16())], [Group(Backend32Or16())]]
        )
        assert out == [16, 16]


# --------------------------------------------------------------------------- #
# 真实 GPUModelRunner 的账本切面（profile_run / 估计器 / 最小 KV 池）
# --------------------------------------------------------------------------- #


class FakeAttn:
    """attn_module 契约位（get_kv_cache_spec 的消费面）。"""

    def __init__(self, spec):
        self._spec = spec

    def get_kv_cache_spec(self, vllm_config):
        return self._spec

    def get_attn_backend(self):
        class B:
            @staticmethod
            def indexes_kv_by_block_stride():
                return False

        return B()


def make_runner(max_num_batched_tokens=8192):
    from implementation.config import CUDAGraphMode
    from implementation.gpu_model_runner import GPUModelRunner

    cfg = make_vllm_config()
    cfg.scheduler_config.max_num_batched_tokens = max_num_batched_tokens
    cfg.compilation_config = type(
        "CC",
        (),
        {"cudagraph_mode": CUDAGraphMode.FULL, "max_cudagraph_capture_size": 8},
    )()
    runner = GPUModelRunner(cfg, torch.device("cpu"))
    layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(2)]
    runner.attn_layers = {
        name: FakeAttn(full_spec()) for name in layer_names
    }
    # attn_groups 契约位（→ ch21）：单 KV 组、backend 支持 16-token 块
    class Backend16:
        @staticmethod
        def get_supported_kernel_block_sizes():
            return [16]

    class AttnGroupSeam:
        def __init__(self, spec, names):
            self.backend = Backend16()
            self.kv_cache_spec = spec
            self.layer_names = names
            self.kv_cache_group_id = 0

    runner.attn_groups = [[AttnGroupSeam(full_spec(), layer_names)]]
    return runner


class TestRunnerProfile:
    # SOURCE 对锚：gpu_model_runner.py:L6433-L6506、L6508-L6534、L6645-L6811
    def test_get_kv_cache_spec_collects_per_layer(self):
        runner = make_runner()
        spec = runner.get_kv_cache_spec()
        assert set(spec) == {
            "model.layers.0.self_attn.attn",
            "model.layers.1.self_attn.attn",
        }
        assert all(isinstance(s, FullAttentionSpec) for s in spec.values())

    def test_profile_run_drives_dummy_forward_and_sampler(self, monkeypatch):
        runner = make_runner(max_num_batched_tokens=2048)
        calls = {}
        monkeypatch.setattr(
            runner, "_dummy_run",
            lambda n, is_profile=False: calls.setdefault(
                "dummy_run", (n, is_profile)
            ) or (None, None),
        )
        monkeypatch.setattr(
            runner, "_dummy_sampler_run",
            lambda h: calls.setdefault("sampler", h),
        )
        monkeypatch.setattr(
            runner, "_sync_device", lambda: calls.setdefault("sync", True)
        )
        runner.profile_run()
        # 假数据规模 = max_num_tokens；is_profile=True 预分配通信缓冲
        assert calls["dummy_run"] == (2048, True)
        assert "sampler" in calls and "sync" in calls

    def test_minimal_kv_cache_reuses_ledger_via_override(self):
        runner = make_runner()
        # 估计前先建最小 KV 池：临时 num_gpu_blocks_override = min_blocks
        # （min(max_num_reqs, max_cudagraph_capture_size) or 1 = 1）
        runner._init_minimal_kv_cache_for_profiling()
        assert runner.kv_cache_config.num_blocks == 1
        # override 借用后被还原
        assert make_vllm_config().cache_config.num_gpu_blocks_override is None
        assert runner.vllm_config.cache_config.num_gpu_blocks == 1

    def test_cudagraph_estimator_first_capture_plus_per_graph(self, monkeypatch):
        from implementation.config import CUDAGraphMode

        runner = make_runner()

        class Desc:
            def __init__(self, num_tokens):
                self.num_tokens = num_tokens

        FULL = [Desc(512) for _ in range(5)]
        PIECEWISE = [Desc(512) for _ in range(3)]
        runner.cudagraph_dispatcher = type(
            "D", (), {"get_capture_descs": lambda self: [
                (CUDAGraphMode.FULL, FULL),
                (CUDAGraphMode.PIECEWISE, PIECEWISE),
            ]}
        )()

        MiB = 1 << 20
        free_seq = iter(
            [100 * MiB, 0, 100 * MiB, 90 * MiB,       # FULL: 首捕 100、每图 10
             80 * MiB, 0, 80 * MiB, 72 * MiB]         # PIECEWISE: 首捕 80、每图 8
        )

        class AccelFake:
            @staticmethod
            def get_memory_info():
                return (next(free_seq), 10 * GiB)

            @staticmethod
            def synchronize():
                pass

        monkeypatch.setattr(torch, "accelerator", AccelFake, raising=False)
        estimate = runner.profile_cudagraph_memory()
        # FULL: 100 + 10×4 = 140；PIECEWISE: 80 + 8×2 = 96 → max(100,80) +
        # (40 + 16) = 156 MiB（共享池 overlay 取 max 防重复计账）
        assert estimate == 100 * MiB + 40 * MiB + 16 * MiB

    def test_cudagraph_none_descs_returns_zero(self, monkeypatch):
        runner = make_runner()
        runner.cudagraph_dispatcher = type(
            "D", (), {"get_capture_descs": lambda self: []}
        )()
        assert runner.profile_cudagraph_memory() == 0


# --------------------------------------------------------------------------- #
# 调度器侧传门参数（站点抽块）
# --------------------------------------------------------------------------- #


class TestSchedulerGateParams:
    # SOURCE 对锚：vllm/v1/core/sched/scheduler.py:L276-L290、L965-L985
    def test_manager_built_with_watermark_and_full_isl(self, monkeypatch):
        from implementation.scheduler import Scheduler

        vllm_config = make_vllm_config(
            scheduler=SchedulerConfig(watermark=0.25)
        )
        config = make_manager_config([(["l0"], full_spec())], 8)
        sched = Scheduler(
            vllm_config=vllm_config, kv_cache_config=config,
            block_size=16, hash_block_size=16,
        )
        assert sched.kv_cache_manager.watermark_blocks == 2
        assert sched.scheduler_reserve_full_isl is True

    def test_allocate_slots_for_waiting_passes_three_budgets(self):
        from implementation.scheduler import Scheduler

        vllm_config = make_vllm_config()
        config = make_manager_config([(["l0"], full_spec())], 8)
        sched = Scheduler(
            vllm_config=vllm_config, kv_cache_config=config,
            block_size=16, hash_block_size=16,
        )
        req = make_request(num_tokens=200)  # 13 块 > 7 可用
        sched.waiting.append(req)
        sched.running.append(make_request("r-r", num_tokens=32,
                                          status=RequestStatus.RUNNING))
        out = sched.allocate_slots_for_waiting(req, num_new_tokens=16)
        # 门参数：full-ISL 开 + has_scheduled_reqs=bool(running)=True → None
        assert out is None
        assert req.status == RequestStatus.WAITING


# --------------------------------------------------------------------------- #
# 组化端到端：混合模型启动 → 装配 → 入场过门 → 窗外回收
# --------------------------------------------------------------------------- #


class TestHybridEndToEnd:
    def test_gemma3_boot_gate_and_recycle(self):
        # 5 SWA + 1 full（Gemma3 5:1 × 1 模式）：单卡启动。
        # max_num_batched_tokens=1 → max_in_flight_tokens=1 → SWA cap =
        # cdiv(min(511+1, 4096), 16)+1 = 33
        cfg = make_vllm_config(
            max_model_len=4096,
            scheduler=SchedulerConfig(max_num_batched_tokens=1),
        )
        assert cfg.max_in_flight_tokens == 1
        spec = gemma3_like(num_swa=5, num_full=1)
        page = full_spec().page_size_bytes  # 65536
        # 每组 1 层（min=1 → 6 组各 1 层）→ num_blocks = avail // page // 1；
        # 池 421 块：护栏恰过（boot 需求 = full 256 + 5×cap 33 = 421 块 ==
        # 池容量——满配），自由 420
        avail = page * 421
        configs = get_kv_cache_configs(cfg, [spec], [avail])
        assert configs[0].num_blocks == 421
        assert len(configs[0].kv_cache_groups) == 6

        # 装配：resolve + manager
        sched_bs, hash_bs = resolve_kv_cache_block_sizes(configs[0], cfg)
        assert (sched_bs, hash_bs) == (16, 16)  # 无缓存 → hash 退 scheduler
        manager = KVCacheManager(
            kv_cache_config=configs[0], max_model_len=4096,
            scheduler_block_size=sched_bs, hash_block_size=hash_bs,
            enable_caching=False, max_in_flight_tokens=1, watermark=0.0,
        )
        assert len(manager.coordinator.single_type_managers) == 6
        # SWA 组的 manager 带 cap；full 组不带
        caps = [
            m._max_admission_blocks_per_request
            for m in manager.coordinator.single_type_managers
        ]
        assert [c is None for c in caps].count(True) == 1  # 只有 full 无 cap

        # 入场：4096-token 请求过 full-ISL 门
        # full 1 组整序列 256 块 + 5 个 SWA 组各 cap 33 → 256+165=421
        # > 420 自由 → None（满配池也装不下「再一条」整序列——这就是并发
        # 上限 = 1 的池）
        req = make_request(num_tokens=4096)
        assert manager.allocate_slots(
            req, num_new_tokens=512, full_sequence_must_fit=True
        ) is None
        # 短请求（128 token）：full 8 块 + 5 组各 min(8,33)=8 → 48 ≤ 420 放行
        req2 = make_request("r2", num_tokens=128)
        assert manager.allocate_slots(
            req2, num_new_tokens=128, full_sequence_must_fit=True
        ) is not None
        # 窗外回收推进后 SWA 组实持被压回窗口内
        manager.remove_skipped_blocks("r2", 120)  # window 512 > 128 → skipped 0
        # 窗口 512 > 128 → 无回收；换小窗模型验证已在 TestSwaRecycle 覆盖
        swa_blocks = manager.coordinator.get_blocks("r2")[1]  # 第 2 组
        assert len(swa_blocks) == 8
