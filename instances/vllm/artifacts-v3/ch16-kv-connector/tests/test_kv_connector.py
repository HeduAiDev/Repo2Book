"""ch16 KVConnector —— 单元+契约测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测行为**
（锚点 = vllm/... 行号，基线 v0.27.1 现核，非 v2 资产的 v0.21.0 旧行号）。

行为清单（按 dossier.mechanisms 对账）：
- m1 双面契约与 role-split：同一个 KVConnectorBase_V1 按 SCHEDULER/WORKER
  两角色分开构建零共享（factory.py:L43-L75 NOTE 原话）、kv_transfer_config
  必设（base.py:L208-L211）、worker 全局 agent（kv_transfer_state.py:L72-L94）
- m2 外部命中查询：外部缓存当第二个前缀缓存查、None=『稍后再问』进
  skipped 不堵队头（scheduler.py:L783-L789）、ExampleConnector 磁盘版
  （example_connector.py:L251-L298）
- m3 双命中仲裁：block_aligned_local 呈给 connector、远端严格更长
  truncate 砍尾免 CoW、否则保尾不加载、hit_diverged 回退
  （scheduler.py:L791-L821 / kv_cache_manager.py:L297-L342 / L777-L794）
- m4 异步加载：allocate_slots(ext, delay)『已分配未缓存』、
  WAITING_FOR_REMOTE_KVS、num_computed_tokens 先行、_skip_zero_block_ids
  （scheduler.py:L1023-L1053 / kv_cache_manager.py:L549-L552）
- m5 护轨：reserved_blocks 只许 fits in (free − 在途预约)
  （scheduler.py:L965-L971 / L2614-L2633）
- m6 元数据面：build_connector_meta 产不透明计划过线、调用即重置
  （scheduler.py:L1233-L1258）
- m7 worker 一拍生命周期：bind→start_load_kv→yield→wait_for_save→
  get_finished→clear、no_forward、finalize（mixin:L76-L112）
- m8 逐层重叠：maybe_transfer_kv_layer 层前 wait 层后 save、无组直通
  （kv_transfer_utils.py:L15-L61）
- m9 完成回收：finished_recving→补缓存+全命中退一 token→回 WAITING/
  PREEMPTED（scheduler.py:L2635-L2693）
- m10 失败回滚：第一个坏块截断、共享坏块只重算一次、双策 fail/recompute、
  record_blocks_for_zeroing（scheduler.py:L2743-L2914 /
  kv_cache_manager.py:L817-L829）
- m11 终局接管：request_finished→True 块不释放、get_finished 报
  finished_sending 才放块、SupportsHMA 逐组交接
  （scheduler.py:L2300-L2327 / L2577-L2612 / base.py:L85-L121）
- m12 defer_block_free 步序栅栏（scheduler.py:L2341-L2380）
- m13 requires_kv_delivery 抢占护栏（scheduler.py:L614-L625）
- m14 worker 直写池：inject/extract 的 slot 寻址（example_connector.py）
- m15 producer partial-tail offload 钉住（kv_cache_manager.py:L848-L874）
- m16 配置门：kv_role 三态谓词、failure policy（config/kv_transfer.py）
- m17 观测与保活：connector_prefix_cache_stats、has_pending_push_work
  （scheduler.py:L1006-L1014 / L2394-L2416）

运行配置：enable_prefix_caching=True（默认开——本地前缀缓存与外部缓存
双查的主线）、单组 FullAttentionSpec（block_size=16、hash_block_size=16；
子块尾仲裁场景用 hash_block_size=8 制造块内边界命中）。
"""
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation import kv_transfer_state  # noqa: E402
from implementation.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
    supports_hma,
)
from implementation.config import ModelConfig, VllmConfig  # noqa: E402
from implementation.example_connector import (  # noqa: E402
    ExampleConnector,
    ExampleConnectorMetadata,
    ReqMeta,
    align_to_block_size,
)
from implementation.factory import KVConnectorFactory  # noqa: E402
from implementation.forward_context import (  # noqa: E402
    ForwardContext,
    override_forward_context,
    set_forward_context,
)
from implementation.hashing import sha256  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from implementation.kv_cache_manager import KVCacheBlocks, KVCacheManager  # noqa: E402
from implementation.kv_connector_model_runner_mixin import (  # noqa: E402
    KVConnectorModelRunnerMixin,
)
from implementation.kv_transfer import KVTransferConfig  # noqa: E402
from implementation.kv_transfer_utils import maybe_transfer_kv_layer  # noqa: E402
from implementation.output import SchedulerOutput  # noqa: E402
from implementation.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
from implementation.request import Request, RequestStatus  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.cache import CacheConfig  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.stats import PrefixCacheStats  # noqa: E402

os.environ.setdefault("PYTHONHASHSEED", "0")

from implementation.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)

init_none_hash(sha256)
HASHER16 = get_request_block_hasher(16, sha256)
HASHER8 = get_request_block_hasher(8, sha256)

BLOCK = 16
STORAGE = os.path.join(os.path.dirname(__file__), ".kv_store_tmp")


# --------------------------------------------------------------------------- #
# 构造辅助：真实装配的最小镜像
# --------------------------------------------------------------------------- #


def full_spec(block_size: int = BLOCK, dtype=torch.float16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=8, dtype=dtype
    )


def mamba_align_spec(block_size: int = BLOCK) -> MambaSpec:
    # ch15 m15/m16 的 partial-hit 粒度配置：align 组 + block > hash
    return MambaSpec(
        block_size=block_size,
        shapes=((8, 8),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def _blk(pool, block_hash, group_id=0):
    """哈希表反查块（种子的共享块定位）。"""
    return pool.cached_block_hash_to_block.get_one_block(
        make_block_hash_with_group_id(block_hash, group_id)
    )


def kv_config(num_blocks: int = 64, mixed_precision: bool = False) -> KVCacheConfig:
    """真实构造：needs_kv_cache_zeroing 是派生属性（mamba 或混合精度）——
    单组 fp16 默认关；混合精度（两组不同 dtype）按真实谓词打开零清。"""
    if mixed_precision:
        groups = [
            KVCacheGroupSpec(["layer.0"], full_spec()),
            KVCacheGroupSpec(["layer.1"], full_spec(dtype=torch.float32)),
        ]
    else:
        groups = [KVCacheGroupSpec(["layer.0", "layer.1"], full_spec())]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


def vllm_config(
    kv_connector: str | None = None,
    kv_role: str | None = None,
    module_path: str | None = None,
    max_concurrent_batches: int = 1,
    failure_policy: str = "fail",
    extra: dict | None = None,
    watermark: float = 0.0,
    enable_caching: bool = True,
    disable_hma: bool = True,
) -> VllmConfig:
    cfg = VllmConfig(
        model_config=ModelConfig(max_model_len=512),
        cache_config=CacheConfig(enable_prefix_caching=enable_caching),
        scheduler_config=SchedulerConfig(watermark=watermark),
    )
    cfg.max_concurrent_batches = max_concurrent_batches
    cfg.scheduler_config.disable_hybrid_kv_cache_manager = disable_hma
    if kv_connector is not None:
        cfg.kv_transfer_config = KVTransferConfig(
            kv_connector=kv_connector,
            kv_role=kv_role,
            kv_load_failure_policy=failure_policy,
            kv_connector_extra_config=extra or {},
        )
        if module_path is not None:
            cfg.kv_transfer_config.kv_connector_module_path = module_path
    return cfg


def make_request(rid: str, tokens, hasher=HASHER16, **kwargs) -> Request:
    kwargs.setdefault("block_hasher", hasher)
    return Request(rid, list(tokens), **kwargs)


def make_scheduler(
    num_blocks: int = 64,
    hash_bs: int = BLOCK,
    connector_name: str | None = None,
    kv_role: str | None = None,
    module_path: str | None = None,
    needs_zeroing: bool = False,
    enable_caching: bool = True,
    watermark: float = 0.0,
    failure_policy: str = "fail",
    max_concurrent_batches: int = 1,
    max_model_len: int = 512,
) -> Scheduler:
    cfg = vllm_config(
        kv_connector=connector_name,
        kv_role=kv_role,
        module_path=module_path,
        max_concurrent_batches=max_concurrent_batches,
        failure_policy=failure_policy,
        watermark=watermark,
        enable_caching=enable_caching,
    )
    # 零清场景走真实派生谓词：混合精度（两组不同 dtype）→
    # needs_kv_cache_zeroing=True；此时关缓存走 NoPrefixCache（原生路径，
    # 支持任意组数）
    return Scheduler(
        vllm_config=cfg,
        kv_cache_config=kv_config(num_blocks, mixed_precision=needs_zeroing),
        block_size=BLOCK,
        hash_block_size=hash_bs,
        max_model_len=max_model_len,
    )


def free_count(s: Scheduler) -> int:
    return s.kv_cache_manager.block_pool.get_num_free_blocks()


def full_free(s: Scheduler) -> int:
    """全空闲基线 = num_blocks - 1（null 块恒占用）。"""
    return s.kv_cache_manager.block_pool.num_gpu_blocks - 1


def run_and_cache_prefix(s: Scheduler, rid: str, tokens, hasher=HASHER16) -> None:
    """真实路径缓存一段前缀：准入→分配→写回→归还（哈希留表）。"""
    req = make_request(rid, tokens, hasher=hasher)
    s.add_request(req)
    out = s.schedule()
    assert rid in out.num_scheduled_tokens
    s.kv_cache_manager.free(req)
    s.requests.pop(rid)
    s.running.clear()


def sampled_output(tokens_by_req: dict[str, list[int]]) -> ModelRunnerOutput:
    """真实 ModelRunnerOutput 构造（sampled_token_ids 为 list[list[int]]）。"""
    req_ids = list(tokens_by_req)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[tokens_by_req[r] for r in req_ids],
    )


def worker_output(
    finished_recving=None, finished_sending=None, invalid=None
) -> ModelRunnerOutput:
    ko = KVConnectorOutput()
    ko.finished_recving = finished_recving
    ko.finished_sending = finished_sending
    ko.invalid_block_ids = set(invalid or ())
    return ModelRunnerOutput.with_kv_conn_output_only(ko)


# --------------------------------------------------------------------------- #
# 契约的测试替身：外部命中/收发完成可编程，调用序可观测。
# 经 factory 的外部模块路径装配（kv_connector_module_path——真实机制，
# factory.py:L105-L123 优先于内置注册表）。
# --------------------------------------------------------------------------- #


class OpaqueMeta(KVConnectorMetadata):
    pass


class AggMeta(KVConnectorWorkerMetadata):
    def aggregate(self, other):
        return self


class HarnessConnector(KVConnectorBase_V1):
    # SOURCE 语义（base.py:L465-L539）：两半抽象的可编程实现
    ext_answer: list  # [(tokens, async) | None] 逐次弹出
    finish_answer: bool = False
    events: list

    def __init__(self, vllm_config, role, kv_cache_config, **kwargs):
        super().__init__(vllm_config, role, kv_cache_config)
        self.ext_answer = []
        self.finish_answer = False
        self.events = []
        self.bound_pools = []

    def bind_gpu_block_pool(self, gpu_block_pool):
        # SOURCE 语义（base.py:L455-L463）：默认 no-op；替身记账以核验
        # 调度器侧调用点
        self.bound_pools.append(gpu_block_pool)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if not self.ext_answer:
            return 0, False
        item = self.ext_answer.pop(0)
        return item  # None → 稍后再问

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.events.append(
            ("update_state_after_alloc", request.request_id, num_external_tokens)
        )

    def build_connector_meta(self, scheduler_output):
        self.events.append(("build_connector_meta",))
        return OpaqueMeta()

    def request_finished(self, request, block_ids):
        self.events.append(("request_finished", request.request_id, list(block_ids)))
        return self.finish_answer, None

    def start_load_kv(self, forward_context, **kwargs):
        self.events.append(("start_load_kv",))

    def wait_for_layer_load(self, layer_name):
        self.events.append(("wait_for_layer_load", layer_name))

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        self.events.append(("save_kv_layer", layer_name))

    def wait_for_save(self):
        self.events.append(("wait_for_save",))

    def get_finished(self, finished_req_ids):
        self.events.append(("get_finished", frozenset(finished_req_ids)))
        return None, None

    def get_block_ids_with_load_errors(self):
        return set()

    def build_connector_worker_meta(self):
        return None


class HarnessHMA(HarnessConnector, SupportsHMA):
    def request_finished_all_groups(self, request, block_ids):
        self.events.append(
            ("request_finished_all_groups", request.request_id,
             tuple(list(g) for g in block_ids))
        )
        return self.finish_answer, None


KVConnectorFactory.register_connector(
    "HarnessConnector", "test_kv_connector", "HarnessConnector"
)
KVConnectorFactory.register_connector("HarnessHMA", "test_kv_connector", "HarnessHMA")
MODULE_PATH = "test_kv_connector"


def harness_scheduler(script: list, hma: bool = False, **kw) -> Scheduler:
    name = "HarnessHMA" if hma else "HarnessConnector"
    kw.setdefault("kv_role", "kv_consumer")
    s = make_scheduler(
        connector_name=name, module_path=MODULE_PATH, **kw
    )
    s.connector.ext_answer = list(script)
    return s


# --------------------------------------------------------------------------- #
# m1/m16：契约解剖与 role-split、配置门
# --------------------------------------------------------------------------- #


class TestRoleSplitAndConfig:
    def test_kv_role_required_when_connector_set(self):
        # config/kv_transfer.py:L102-L106：既设 connector 则必设 role
        with pytest.raises(ValueError, match="kv_role"):
            KVTransferConfig(kv_connector="ExampleConnector")

    def test_kv_role_invalid_rejected(self):
        # config/kv_transfer.py:L96-L100
        with pytest.raises(ValueError, match="Unsupported kv_role"):
            KVTransferConfig(kv_connector="X", kv_role="kv_bogus")

    def test_role_predicates(self):
        # config/kv_transfer.py:L108-L118 三谓词
        both = KVTransferConfig(kv_connector="X", kv_role="kv_both")
        assert both.is_kv_transfer_instance
        assert both.is_kv_producer and both.is_kv_consumer
        prod = KVTransferConfig(kv_connector="X", kv_role="kv_producer")
        assert prod.is_kv_producer and not prod.is_kv_consumer
        # 真实配置门：connector 设了而 role 没设 → post_init 直接拒；
        # role 设了而 connector 没设 → 不是 transfer instance（谓词第二半边）
        role_no_conn = KVTransferConfig(kv_role="kv_consumer")
        assert not role_no_conn.is_kv_transfer_instance
        no_conn = KVTransferConfig()
        assert not no_conn.is_kv_producer and not no_conn.is_kv_consumer

    def test_failure_policy_default_fail(self):
        # config/kv_transfer.py:L69-L72 默认 fail
        cfg = KVTransferConfig(kv_connector="X", kv_role="kv_consumer")
        assert cfg.kv_load_failure_policy == "fail"

    def test_role_enum_split(self):
        # base.py:L124-L130：SCHEDULER=0 / WORKER=1
        assert KVConnectorRole.SCHEDULER.value == 0
        assert KVConnectorRole.WORKER.value == 1

    def test_factory_builds_two_roles_same_class(self):
        # factory.py:L67-L75 NOTE：build separately to enforce strict separation
        cfg = vllm_config(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            extra={"shared_storage_path": STORAGE},
        )
        sched_side = KVConnectorFactory.create_connector(
            cfg, KVConnectorRole.SCHEDULER, kv_config()
        )
        worker_side = KVConnectorFactory.create_connector(
            cfg, KVConnectorRole.WORKER, kv_config()
        )
        assert type(sched_side) is type(worker_side)
        assert sched_side.role is KVConnectorRole.SCHEDULER
        assert worker_side.role is KVConnectorRole.WORKER
        assert sched_side is not worker_side  # 零共享状态

    def test_factory_requires_kv_transfer_config(self):
        # factory.py:L49-L51
        cfg = vllm_config()  # kv_transfer_config = None
        with pytest.raises(ValueError, match="kv_transfer_config must be set"):
            KVConnectorFactory.create_connector(
                cfg, KVConnectorRole.SCHEDULER, kv_config()
            )

    def test_hma_gate(self):
        # factory.py:L54-L60：HMA 开而 connector 不支持 → 拒
        cfg = vllm_config(
            kv_connector="ExampleConnector",
            kv_role="kv_consumer",
            disable_hma=False,
            extra={"shared_storage_path": STORAGE},
        )
        with pytest.raises(ValueError, match="does not support HMA"):
            KVConnectorFactory.create_connector(
                cfg, KVConnectorRole.SCHEDULER, kv_config()
            )

    def test_supports_hma_predicate(self):
        # base.py:L117-L121：类与实例两态判定
        h = HarnessConnector(
            vllm_config(kv_connector="X", kv_role="kv_consumer", module_path=MODULE_PATH),
            KVConnectorRole.SCHEDULER,
            kv_config(),
        )
        assert supports_hma(HarnessHMA) and supports_hma(h) is False
        assert supports_hma(HarnessConnector) is False

    def test_base_requires_kv_transfer_config(self):
        # base.py:L208-L211：kv_transfer_config 必设
        with pytest.raises(ValueError, match="kv_transfer_config must be set"):
            HarnessConnector(vllm_config(), KVConnectorRole.SCHEDULER, kv_config())

    def test_worker_side_agent_global(self):
        # kv_transfer_state.py:L72-L94：WORKER role 再建一份挂全局 agent
        kv_transfer_state._KV_CONNECTOR_AGENT = None
        cfg = vllm_config(
            kv_connector="ExampleConnector",
            kv_role="kv_consumer",
            extra={"shared_storage_path": STORAGE},
        )
        assert not kv_transfer_state.has_kv_transfer_group()
        kv_transfer_state.ensure_kv_transfer_initialized(cfg, kv_config())
        assert kv_transfer_state.has_kv_transfer_group()
        agent = kv_transfer_state.get_kv_transfer_group()
        assert agent.role is KVConnectorRole.WORKER
        assert kv_transfer_state.is_v1_kv_transfer_group(agent)
        kv_transfer_state.ensure_kv_transfer_shutdown()
        assert not kv_transfer_state.has_kv_transfer_group()

    def test_worker_agent_gate_not_transfer_instance(self):
        # is_kv_transfer_instance=False（connector 未设——connector+空 role 的
        # 组合在 config 的 post_init 就被拒，这是谓词的第二半边）→ 不装配
        kv_transfer_state._KV_CONNECTOR_AGENT = None
        cfg = vllm_config()  # kv_transfer_config = None → 早退
        kv_transfer_state.ensure_kv_transfer_initialized(cfg, kv_config())
        assert not kv_transfer_state.has_kv_transfer_group()
        cfg2 = vllm_config()
        cfg2.kv_transfer_config = KVTransferConfig(kv_role="kv_consumer")
        kv_transfer_state.ensure_kv_transfer_initialized(cfg2, kv_config())
        assert not kv_transfer_state.has_kv_transfer_group()

    def test_scheduler_four_flags(self):
        # scheduler.py:L125-L158：四旗标
        s = make_scheduler(
            connector_name="ExampleConnector",
            kv_role="kv_consumer",
            failure_policy="recompute",
        )
        assert s.connector is not None
        assert s.connector.role is KVConnectorRole.SCHEDULER
        assert s.recompute_kv_load_failures  # policy=recompute
        assert not s.defer_block_free  # max_concurrent_batches=1
        assert not s.requires_kv_delivery  # consumer：best-effort
        s2 = make_scheduler(connector_name="ExampleConnector", kv_role="kv_producer")
        assert s2.requires_kv_delivery  # base.py:L184-L194 默认 producer

    def test_defer_block_free_when_consumer_and_async(self):
        # scheduler.py:L150-L156：异步调度+consumer → defer
        s = harness_scheduler([], max_concurrent_batches=2)
        assert s.defer_block_free
        s2 = harness_scheduler([], max_concurrent_batches=1)
        assert not s2.defer_block_free

    def test_bind_gpu_block_pool_called(self):
        # scheduler.py:L291-L294：建完 manager 后 bind（默认 no-op——测试
        # 替身覆写记账以核验调用点）
        s = harness_scheduler([])
        assert s.connector.bound_pools[-1] is s.kv_cache_manager.block_pool

    def test_get_from_extra_config(self):
        # config/kv_transfer.py:L120-L121
        cfg = KVTransferConfig(
            kv_connector="X",
            kv_role="kv_consumer",
            kv_connector_extra_config={"k": 7},
        )
        assert cfg.get_from_extra_config("k", 0) == 7
        assert cfg.get_from_extra_config("absent", "d") == "d"


# --------------------------------------------------------------------------- #
# m2：外部命中查询（None=稍后再问）+ ExampleConnector 磁盘版 + 统计
# --------------------------------------------------------------------------- #


class TestExternalHitQuery:
    def test_none_goes_to_skipped_queue(self):
        # scheduler.py:L783-L789：返回 (None, _) → pop 进 step_skipped_waiting
        s = harness_scheduler([(None, False)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert req in list(s.skipped_waiting)
        assert req not in list(s.waiting)
        assert "r1" not in out.num_scheduled_tokens  # 本步没调度它
        # 稍后再问：下一拍 connector 已有答案 → 正常调度
        s.connector.ext_answer = [(0, False)]
        out = s.schedule()
        assert "r1" in out.num_scheduled_tokens

    def test_ext_hit_sync_reduces_new_tokens(self):
        # 外部命中（同步）→ num_computed = 本地+外部、new token 少算
        s = harness_scheduler([(32, False)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert out.num_scheduled_tokens["r1"] == 64 - 32  # 只算外部未覆盖的
        assert req.num_computed_tokens == 64  # update_after_schedule 后
        assert req.status == RequestStatus.RUNNING
        # update_state_after_alloc 拿到 num_external_tokens=32
        ev = [e for e in s.connector.events if e[0] == "update_state_after_alloc"]
        assert ev == [("update_state_after_alloc", "r1", 32)]

    def test_example_connector_miss_returns_zero(self):
        # example_connector.py:L276-L277：磁盘无文件 → (0, False)
        cfg = vllm_config(
            kv_connector="ExampleConnector",
            kv_role="kv_consumer",
            extra={"shared_storage_path": STORAGE},
        )
        conn = ExampleConnector(cfg, KVConnectorRole.SCHEDULER, kv_config())
        req = make_request("r1", range(64))
        assert conn.get_num_new_matched_tokens(req, 0) == (0, False)

    def test_example_connector_hit_after_store(self):
        # 磁盘版参考实现：worker save 落文件 → 调度器侧命中
        if os.path.exists(STORAGE):
            shutil.rmtree(STORAGE)
        cfg = vllm_config(
            kv_connector="ExampleConnector",
            kv_role="kv_consumer",
            extra={"shared_storage_path": STORAGE},
        )
        sched_side = ExampleConnector(cfg, KVConnectorRole.SCHEDULER, kv_config())
        worker_side = ExampleConnector(cfg, KVConnectorRole.WORKER, kv_config())
        tokens = list(range(40))
        worker_side.bind_connector_metadata(
            _StoreMeta(tokens, [0, 1, 2])
        )
        worker_side.save_kv_layer("layer.0", torch.randn(3, 2, BLOCK, 8), None)
        got = sched_side.get_num_new_matched_tokens(make_request("r9", tokens), 0)
        num_check = align_to_block_size(len(tokens) - 1, BLOCK)
        assert got == (num_check, False)
        shutil.rmtree(STORAGE, ignore_errors=True)

    def test_example_update_state_after_alloc_gate(self):
        # example_connector.py:L288-L298：num_external_tokens>0 才登记待加载
        cfg = vllm_config(
            kv_connector="ExampleConnector",
            kv_role="kv_consumer",
            extra={"shared_storage_path": STORAGE},
        )
        conn = ExampleConnector(cfg, KVConnectorRole.SCHEDULER, kv_config())
        req = make_request("r1", range(64))
        conn.update_state_after_alloc(req, None, 0)
        assert conn._requests_need_load == {}
        conn.update_state_after_alloc(req, None, 32)
        assert "r1" in conn._requests_need_load

    def test_align_to_block_size(self):
        # example_connector.py:L442-L444：(n-1)//bs*bs
        assert align_to_block_size(40, 16) == 32
        assert align_to_block_size(17, 16) == 16
        assert align_to_block_size(16, 16) == 0  # (n-1)//bs*bs：留最后 token
        assert align_to_block_size(1, 16) == 0

    def test_req_meta_slot_mapping(self):
        # example_connector.py:L41-L64：slot = block_id*bs + offset 展平
        meta = ReqMeta.make_meta(list(range(40)), [3, 7, 9], 16, False, [])
        sm = meta.slot_mapping
        assert sm[:4].tolist() == [48, 49, 50, 51]  # 块 3 开头
        assert sm[16].item() == 7 * 16  # 第二块开头
        assert len(sm) == 32  # 40 对齐到 32

    def test_connector_prefix_cache_stats_recorded(self):
        # scheduler.py:L1006-L1014：准入时记 queries/hits（未调度不计数）
        s = harness_scheduler([(32, False)])
        s.connector_prefix_cache_stats = PrefixCacheStats()
        req = make_request("r1", range(64))
        s.add_request(req)
        s.schedule()
        st = s.connector_prefix_cache_stats
        assert st.queries == 64 and st.hits == 32
        assert st.requests == 1 and st.preempted_requests == 0


def _StoreMeta(tokens, block_ids, is_store=True):
    """ExampleConnectorMetadata 的真实构造（worker save/load 路径驱动用）。"""
    meta = ExampleConnectorMetadata()
    meta.add_request(tokens, block_ids, BLOCK, is_store, [])
    return meta


# --------------------------------------------------------------------------- #
# m3：双命中仲裁（远端严格更长砍本地尾免 CoW / 否则保尾）
# --------------------------------------------------------------------------- #


class TestDualHitArbitration:
    def _hybrid_scheduler(self, script):
        """子块尾场景的真实配置（ch15 m15/m16 的 partial-hit 粒度）：
        full(16) + mamba-align(16) 两组、hash 8 → enable_partial_hash_hits=True
        ——单组配置下 Unitary 断言 hash==block，块内边界命中只在此形态出生。"""
        cfg = vllm_config(
            kv_connector="HarnessConnector",
            kv_role="kv_consumer",
            module_path=MODULE_PATH,
        )
        kv_cfg = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["layer.0", "layer.1"], full_spec()),
                KVCacheGroupSpec(["layer.2"], mamba_align_spec()),
            ],
        )
        s = Scheduler(
            vllm_config=cfg,
            kv_cache_config=kv_cfg,
            block_size=BLOCK,
            hash_block_size=8,
            max_model_len=512,
        )
        s.connector.ext_answer = list(script)
        return s

    def test_remote_strictly_longer_truncates_local_tail(self):
        # 本地命中 40（块内边界：hash8/block16，full 组细粒度）、远端 16 >
        # partial_tail 8 → truncate 到块对齐 32 + 外部（scheduler.py:L791-L802）
        s = self._hybrid_scheduler([])
        run_and_cache_prefix(s, "seed", range(40), hasher=HASHER8)
        pool = s.kv_cache_manager.block_pool
        s.connector.ext_answer = [(16, False)]
        b = make_request(
            "b", list(range(40)) + list(range(100, 116)), hasher=HASHER8
        )
        s.add_request(b)
        out = s.schedule()
        # truncate 后准入值 = 块对齐 32 + 外部 16 = 48；本拍只算 56−48=8
        assert out.num_scheduled_tokens["b"] == 56 - (32 + 16)
        b_ids = s.kv_cache_manager.get_block_ids("b")[0]
        # 种子块号：seed 已 free 但哈希表仍指向它（惰性驱逐）
        seed_b0 = _blk(pool, b.block_hashes[1]).block_id  # 16-token 边界
        seed_b1 = _blk(pool, b.block_hashes[3]).block_id  # 32-token 边界
        # 前 2 块 = seed 的共享块（命中采用）；子块尾块没被采用（truncate 掉）
        assert b_ids[:2] == [seed_b0, seed_b1]
        assert pool.blocks[seed_b0].ref_cnt == 1  # b 采用（seed 已 free）
        seed_partial = _blk(pool, b.block_hashes[4])  # 40-token 子块尾条目
        assert seed_partial.block_id not in b_ids  # 子块尾块未被采用
        assert seed_partial.ref_cnt == 0  # 免 CoW：没人引用、留在 free 队

    def test_remote_not_longer_reconciles_divergent(self):
        # 远端 8 == partial_tail（不严格更长）→ 保尾外部 0；且 hit_diverged
        # （mamba 组命中 32 < full 组 40）无外部撑腰 → 回退全组一致边界 32
        # （scheduler.py:L803-L809 + L813-L821 的混合回退分支）
        s = self._hybrid_scheduler([])
        run_and_cache_prefix(s, "seed", range(40), hasher=HASHER8)
        s.connector.ext_answer = [(8, False)]
        b = make_request(
            "b", list(range(40)) + list(range(100, 116)), hasher=HASHER8
        )
        s.add_request(b)
        out = s.schedule()
        # 调和到全组一致边界 32：本拍算 56−32=24（无外部可加载）
        assert out.num_scheduled_tokens["b"] == 56 - 32
        ev = [e for e in s.connector.events if e[0] == "update_state_after_alloc"]
        assert ev[-1][2] == 0  # num_external_computed_tokens == 0

    def test_remote_longer_extends_hit(self):
        # 无子块尾（整块命中）：else 支直接采用 ext（L810-L811）
        s = harness_scheduler([])
        run_and_cache_prefix(s, "seed", range(32))
        s.connector.ext_answer = [(16, False)]
        b = make_request("b", list(range(32)) + list(range(100, 132)))
        s.add_request(b)
        out = s.schedule()
        # 准入时 32 本地 + 16 外部 = 48；本拍再算 16（b 共 64 token）
        assert out.num_scheduled_tokens["b"] == 16
        assert b.num_computed_tokens == 64  # update_after_schedule 后

    def test_truncate_computed_blocks_pure_slicing(self):
        # kv_cache_manager.py:L777-L794：纯切片、ref 不动、块对齐断言
        s = harness_scheduler([])
        req = make_request("seed", range(48))
        s.add_request(req)
        s.schedule()  # 活请求：3 块在表上
        blocks = s.kv_cache_manager.get_blocks("seed")
        refs_before = [blk.ref_cnt for grp in blocks.blocks for blk in grp]
        truncated = s.kv_cache_manager.truncate_computed_blocks(blocks, 32)
        assert len(truncated.blocks[0]) == 2
        assert len(blocks.blocks[0]) == 3  # 原视图不受影响
        assert [blk.ref_cnt for grp in blocks.blocks for blk in grp] == refs_before
        # 非块对齐 → 断言拒绝
        with pytest.raises(AssertionError):
            s.kv_cache_manager.truncate_computed_blocks(blocks, 33)


# --------------------------------------------------------------------------- #
# m4/m5/m6：异步加载路径 + 护轨 + 元数据过线
# --------------------------------------------------------------------------- #


class TestAsyncLoadPath:
    def test_async_load_state_and_blocks(self):
        # (ext, True) → WAITING_FOR_REMOTE_KVS、先行记账、『已分配未缓存』
        s = harness_scheduler([(32, True)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert req in list(s.skipped_waiting)
        assert req.num_computed_tokens == 32  # 先行设置、无人消费
        assert "r1" not in out.num_scheduled_tokens  # 零前向
        assert len(s.kv_cache_manager.get_block_ids("r1")[0]) == 2  # ext 2 块
        mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
        # 『已分配未缓存』：账上记 0、哈希入表要等传输完成的补缓存
        assert mgr.num_cached_block.get("r1", 0) == 0
        assert mgr.req_to_blocks["r1"][0].block_hash is None

    def test_skip_zero_block_ids(self):
        # scheduler.py:L1043-L1052：异步加载将覆写的块登记跳过清零
        # （_skip_zero_block_ids 每步即焚——可观测面 = 清零账滤掉了 ext 块）
        s = harness_scheduler([(0, False), (32, True)], needs_zeroing=True)
        r0 = make_request("r0", range(64))  # 普通请求：4 块进清零账
        r1 = make_request("r1", range(100, 164))  # async：2 块 ext 块
        s.add_request(r0)
        s.add_request(r1)
        out = s.schedule()
        ext_blocks = set().union(*s.kv_cache_manager.get_block_ids("r1"))
        r0_blocks = set().union(*s.kv_cache_manager.get_block_ids("r0"))
        assert ext_blocks and r0_blocks
        assert not (ext_blocks & r0_blocks)
        # 清零账包含 r0 的新块、滤掉了 r1 的 ext 块（清零会与远端写入竞争）
        assert set(out.new_block_ids_to_zero) == r0_blocks
        assert s._skip_zero_block_ids == set()  # 每步即焚

    def test_guardrail_reserved_blocks(self):
        # 在途 async load 的预约让第二个 async load 进不来（防死锁/抢占）
        s = harness_scheduler([], num_blocks=16)  # 16 块池（null 外 15）
        r1 = make_request("r1", range(128))  # 全程 8 块
        s.connector.ext_answer = [(64, True)]
        s.add_request(r1)
        s.schedule()
        assert r1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert r1 in s._inflight_prefills
        # r1 已占 4 块（ext 64/16），还需 (128-64)/16=4 块 → 预约 ≥ 1
        assert s._inflight_prefill_reserved_blocks() >= 1
        s.connector.ext_answer = [(128, True)]
        r2 = make_request("r2", range(100, 244))  # 144 token：ext 需 8 块
        s.add_request(r2)
        out = s.schedule()
        # 护轨拒绝 r2（free − reserved 不够）→ 留 waiting、不入 skipped
        assert r2.status == RequestStatus.WAITING
        assert r2 not in list(s.skipped_waiting)
        assert "r2" not in out.num_scheduled_tokens

    def test_metadata_crossing_opaque(self):
        # build_connector_meta → SchedulerOutput.kv_connector_metadata（m6）
        s = harness_scheduler([(32, False)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert isinstance(out.kv_connector_metadata, OpaqueMeta)
        assert ("build_connector_meta",) in s.connector.events
        # scheduler_output 的字段没被 connector 改动（L533 契约）
        assert out.num_scheduled_tokens["r1"] == 32


# --------------------------------------------------------------------------- #
# m7/m8：worker 一拍生命周期 + 逐层钩子
# --------------------------------------------------------------------------- #


def _with_worker_agent(script=None, with_meta=True):
    kv_transfer_state._KV_CONNECTOR_AGENT = None
    cfg = vllm_config(
        kv_connector="HarnessConnector",
        kv_role="kv_consumer",
        module_path=MODULE_PATH,
    )
    kv_transfer_state.ensure_kv_transfer_initialized(cfg, kv_config())
    conn = kv_transfer_state.get_kv_transfer_group()
    conn.ext_answer, conn.events = list(script or []), []
    return cfg, conn


def _out_with_meta(**kw) -> SchedulerOutput:
    """真实形态：connector 在场时 scheduler 总是挂上不透明计划。"""
    out = SchedulerOutput(**kw)
    out.kv_connector_metadata = OpaqueMeta()
    return out


class TestWorkerLifecycle:
    def test_contextmanager_order(self):
        # mixin:L76-L112：bind → start_load_kv → yield → wait_for_save →
        # get_finished(finished_req_ids) → errors → clear
        cfg, conn = _with_worker_agent()
        out = _out_with_meta(finished_req_ids={"dead"})
        with set_forward_context(None, cfg):
            with KVConnectorModelRunnerMixin._get_kv_connector_output(
                out, wait_for_save=True
            ) as ko:
                assert isinstance(ko, KVConnectorOutput)
                assert conn.has_connector_metadata()  # bind 已收
                assert [e[0] for e in conn.events] == ["start_load_kv"]
        names = [e[0] for e in conn.events]
        assert names == ["start_load_kv", "wait_for_save", "get_finished"]
        assert conn.events[-1][1] == frozenset({"dead"})
        assert not conn.has_connector_metadata()  # clear 已收尾
        kv_transfer_state.ensure_kv_transfer_shutdown()

    def test_wait_for_save_skipped_when_no_forward(self):
        # no_forward：wait_for_save=False（mixin:L36-L48）无 token 也走收发
        cfg, conn = _with_worker_agent()
        mro = KVConnectorModelRunnerMixin.kv_connector_no_forward(
            _out_with_meta(), cfg
        )
        # 真实行为：connector 无可报（finished 空）→ with_kv_conn_output_only
        # 返回空实例；有回传才有载荷
        assert mro.kv_connector_output is None
        conn.get_finished = lambda ids: (None, {"rx-done"})
        mro2 = KVConnectorModelRunnerMixin.kv_connector_no_forward(
            _out_with_meta(), cfg
        )
        assert mro2.kv_connector_output is not None
        assert mro2.kv_connector_output.finished_recving == {"rx-done"}
        names = [e[0] for e in conn.events]
        assert "start_load_kv" in names and "wait_for_save" not in names
        kv_transfer_state.ensure_kv_transfer_shutdown()

    def test_maybe_get_nullcontext_without_group(self):
        # mixin:L50-L61：无 kv_transfer_group → nullcontext 零开销
        kv_transfer_state._KV_CONNECTOR_AGENT = None
        from contextlib import nullcontext

        got = KVConnectorModelRunnerMixin.maybe_get_kv_connector_output(
            SchedulerOutput()
        )
        assert isinstance(got, nullcontext)

    def test_finalize_after_defer(self):
        # mixin:L63-L72：defer_finalize=True 时 wait_for_save+clear 推迟
        cfg, conn = _with_worker_agent()
        with set_forward_context(None, cfg):
            with KVConnectorModelRunnerMixin._get_kv_connector_output(
                _out_with_meta(), defer_finalize=True
            ):
                pass  # 模拟 draft forward
        assert conn.has_connector_metadata()  # 未 clear
        KVConnectorModelRunnerMixin.finalize_kv_connector()
        assert not conn.has_connector_metadata()
        assert "wait_for_save" in [e[0] for e in conn.events]
        kv_transfer_state.ensure_kv_transfer_shutdown()

    def test_layer_decorator_order(self):
        # kv_transfer_utils.py:L37-L59：层前 wait_for_layer_load、层后 save_kv_layer
        cfg, conn = _with_worker_agent()
        calls = []

        @maybe_transfer_kv_layer
        def attn_forward(layer_name: str, x):
            calls.append(("attn", layer_name))
            return x + 1

        conn.bind_connector_metadata(OpaqueMeta())
        kv_cache = torch.zeros(1, 2, BLOCK, 8)

        @dataclass
        class _Layer:
            kv_cache: Any = None

        fc = ForwardContext(
            no_compile_layers={"l0": _Layer(kv_cache)},
            attn_metadata={"l0": "md"},
            slot_mapping={"l0": torch.zeros(1)},
        )
        with override_forward_context(fc):
            assert attn_forward("l0", 1) == 2
        assert [e[0] for e in conn.events] == [
            "wait_for_layer_load", "save_kv_layer"
        ]
        assert calls == [("attn", "l0")]
        # 无 metadata → 直通（零开销旁路之二）
        conn.clear_connector_metadata()
        conn.events.clear()
        with override_forward_context(fc):
            assert attn_forward("l0", 1) == 2
        assert conn.events == []
        kv_transfer_state.ensure_kv_transfer_shutdown()

    def test_layer_decorator_requires_layer_name(self):
        # kv_transfer_utils.py:L30-L35：无 layer_name 参数 → TypeError
        with pytest.raises(TypeError, match="layer_name"):

            @maybe_transfer_kv_layer
            def bad(x):
                return x

    def test_layer_decorator_passthrough_without_group(self):
        kv_transfer_state._KV_CONNECTOR_AGENT = None

        @maybe_transfer_kv_layer
        def ok(layer_name: str):
            return layer_name

        assert ok("x") == "x"


# --------------------------------------------------------------------------- #
# m9：完成回收（补缓存 + 全命中退一 token + 回队）与 producer 终局
# --------------------------------------------------------------------------- #


class TestRecycle:
    def test_finished_recving_promotes_and_caches(self):
        # 全流程：async 占块 → worker 报 finished_recving → 下拍提升+补缓存
        s = harness_scheduler([(48, True)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
        # 『已分配未缓存』：add_local 记 0、真实哈希入表要等补缓存
        assert mgr.num_cached_block.get("r1", 0) == 0
        assert mgr.req_to_blocks["r1"][0].block_hash is None
        s.update_from_output(out, worker_output(finished_recving={"r1"}))
        assert "r1" in s.finished_recving_kv_req_ids
        # 下一拍 schedule 提升：_update_waiting_for_remote_kv 补缓存 + 回
        # WAITING，同一拍继续准入调度剩余 16 token
        out2 = s.schedule()
        assert req.status == RequestStatus.RUNNING
        # 补缓存 3 块（传输完成的 48 token）→ 同拍续算 16 → 4 块全入表
        assert mgr.num_cached_block.get("r1") == 4
        assert mgr.req_to_blocks["r1"][0].block_hash is not None  # 入表
        assert out2.num_scheduled_tokens["r1"] == 64 - 48

    def test_full_hit_retreat_one_token(self):
        # 外部命中覆盖全 prompt → 退一 token 重算（要 logits）
        # scheduler.py:L2671-L2674
        s = harness_scheduler([(64, True)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert req.num_computed_tokens == 64  # 先行记账=全命中
        s.update_from_output(out, worker_output(finished_recving={"r1"}))
        out2 = s.schedule()
        # 退一 token 的可观测面：本拍多调 1 个 token（补算 logits）
        assert out2.num_scheduled_tokens["r1"] == 1

    def test_preempted_during_wait_goes_back_preempted(self):
        # num_preemptions>0 → 提升回 PREEMPTED（L2689-L2692）
        s = harness_scheduler([(48, True)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        req.num_preemptions = 1
        s.update_from_output(out, worker_output(finished_recving={"r1"}))
        # 提升判定本体（schedule 内联调用；同拍续算会立刻掩盖瞬态 PREEMPTED）
        assert s._try_promote_blocked_waiting_request(req) is True
        assert req.status == RequestStatus.PREEMPTED

    def test_finished_sending_frees_producer_blocks(self):
        # producer：request_finished=True 接管 → finished_sending 才放块
        s = harness_scheduler([(32, True)])
        s.connector.finish_answer = True
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        req.status = RequestStatus.FINISHED_STOPPED
        used_before = 64 - free_count(s)
        s._free_request(req)
        # 接管：块不释放、请求仍登记（has_finished_requests 知道还有账）
        assert "r1" in s.requests
        assert 64 - free_count(s) == used_before
        assert s.has_finished_requests()
        # worker 报 finished_sending → _free_blocks（L2738-L2741）
        s.update_from_output(out, worker_output(finished_sending={"r1"}))
        assert "r1" not in s.requests
        assert free_count(s) == full_free(s)

    def test_hma_all_groups_handoff(self):
        # SupportsHMA：request_finished_all_groups 拿逐组块表
        s = harness_scheduler([(32, True)], hma=True)
        s.connector.finish_answer = True
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        req.status = RequestStatus.FINISHED_STOPPED
        s._free_request(req)
        ev = [e for e in s.connector.events if e[0] == "request_finished_all_groups"]
        assert len(ev) == 1
        group_ids = ev[0][2]
        assert isinstance(group_ids, tuple) and len(group_ids[0]) >= 1

    def test_non_hma_single_group_path(self):
        # 非 HMA：request_finished(block_ids[0]) 单组路径（L2604-L2610）
        s = harness_scheduler([(32, True)], hma=False)
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        req.status = RequestStatus.FINISHED_STOPPED
        s._free_request(req)
        ev = [e for e in s.connector.events if e[0] == "request_finished"]
        assert len(ev) == 1 and ev[0][2]  # 块表非空
        assert "r1" not in s.requests  # False → 立即放块
        assert free_count(s) == full_free(s)

    def test_has_pending_push_work_keeps_engine_alive(self):
        # scheduler.py:L2406-L2416：push 型传输保活
        s = harness_scheduler([])
        r = make_request("r1", range(64))
        s.add_request(r)
        assert s.has_requests()
        s.finish_requests("r1", RequestStatus.FINISHED_STOPPED)  # 公共路径
        s.schedule()  # 冲账（finished_req_ids 在 _update_after_schedule 清）
        assert not s.has_requests()
        s.connector.has_pending_push_work = lambda: True
        assert s.has_requests()  # push 未排空 → 引擎继续步进
        s.connector.has_pending_push_work = lambda: False
        assert not s.has_requests()

    def test_handoff_block_table_clipped_to_computed(self):
        # _connector_finished：交接块表按 num_computed_tokens 裁剪
        s = harness_scheduler([(32, True)], hma=False)
        req = make_request("r1", range(80))
        s.add_request(req)
        out = s.schedule()
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        req.num_computed_tokens = 32  # 交接时已算 32（表上恰 2 个 ext 块）
        req.status = RequestStatus.FINISHED_STOPPED
        s._free_request(req)
        ev = [e for e in s.connector.events if e[0] == "request_finished"]
        assert len(ev[0][2]) == 2  # 32/16


# --------------------------------------------------------------------------- #
# m10：失败回滚
# --------------------------------------------------------------------------- #


class TestFailureRollback:
    def test_fail_policy_fails_request(self):
        # 默认 fail：invalid blocks → 整请求 FINISHED_ERROR（L2894-L2903）
        s = harness_scheduler([(48, False)])
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        bad = s.kv_cache_manager.get_block_ids("r1")[0][1]  # 第二块坏
        s.update_from_output(out, worker_output(invalid={bad}))
        assert req.is_finished()
        assert req.status == RequestStatus.FINISHED_ERROR
        assert "r1" not in s.requests
        assert free_count(s) == full_free(s)

    def test_recompute_truncates_at_first_bad_block(self):
        # recompute：num_computed_tokens 截到最长有效前缀（块对齐，L2818-L2824）
        s = harness_scheduler([(48, False)], failure_policy="recompute")
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        req.num_computed_tokens = 64  # 模拟 update_after_schedule 推进
        bad = s.kv_cache_manager.get_block_ids("r1")[0][2]
        failed = s._handle_invalid_blocks({bad}, {})
        assert "r1" in failed
        assert req.num_computed_tokens == 2 * BLOCK

    def test_shared_bad_block_recomputed_once(self):
        # 共享坏块：第一个截断重算、共享者按已标记处理不再截断（L2802-L2840）
        s = harness_scheduler([], failure_policy="recompute")
        seed = make_request("seed", range(48))
        s.add_request(seed)
        s.schedule()  # seed 先算完 48、3 块入哈希表
        assert seed.num_computed_tokens == 48
        b = make_request("b", list(range(48)))
        s.add_request(b)
        s.schedule()  # b 命中 seed 的共享块（本地 32）
        assert b.status == RequestStatus.RUNNING
        assert b.num_computed_tokens == 48  # 32 命中 + 16 新算
        ids = s.kv_cache_manager.get_block_ids("b")[0]
        assert len(ids) == 3
        bad = ids[1]
        assert bad in s.kv_cache_manager.get_block_ids("seed")[0]  # 共享
        failed = s._handle_invalid_blocks({bad}, {})
        # 扫描序 = running 序：seed 在前 → 截断到第一个坏块前的合法前缀
        assert seed.num_computed_tokens == 1 * BLOCK
        # b 的坏块已被 seed 标记：b 把它当已算——回退到自己的 cached 计数
        assert "b" in failed and b.num_computed_tokens == 3 * BLOCK

    def test_async_failure_marks_failed_recving(self):
        # 异步失败：failed_recving_kv_req_ids 等重试 + 补缓存有效前缀
        # （L2645-L2665；补登记清零在 test_record_blocks_for_zeroing 单测——
        # 真实源码对混合组的 invalid 扫描是 TODO，单组为准）
        s = harness_scheduler([(48, True)], failure_policy="recompute")
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        bad = s.kv_cache_manager.get_block_ids("r1")[0][2]
        s.update_from_output(out, worker_output(invalid={bad}))
        assert "r1" in s.failed_recving_kv_req_ids
        assert req.num_computed_tokens == 2 * BLOCK  # 已截断
        # worker 报完成 → 下拍提升走失败分支：补缓存 2 块有效前缀，
        # 截断区 32 token 本拍重算（m10 的可观测面）
        s.update_from_output(out, worker_output(finished_recving={"r1"}))
        out2 = s.schedule()
        mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
        assert mgr.num_cached_block.get("r1") == 4  # 2 补缓存 + 2 重算入表
        assert out2.num_scheduled_tokens["r1"] == 64 - 32  # 重算区

    def test_async_failure_no_valid_prefix_frees_all(self):
        # 无有效 token → free 全部块（L2658-L2663）
        s = harness_scheduler([(48, True)], failure_policy="recompute")
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        bad0 = s.kv_cache_manager.get_block_ids("r1")[0][0]
        s.update_from_output(out, worker_output(invalid={bad0}))
        assert req.num_computed_tokens == 0
        s.update_from_output(out, worker_output(finished_recving={"r1"}))
        out2 = s.schedule()  # 提升走失败分支：free 全部块 → 立即重试
        assert out2.num_scheduled_tokens["r1"] == 64  # 从头重算
        assert req.num_computed_tokens == 64

    def test_record_blocks_for_zeroing_block_aligned(self):
        # kv_cache_manager.py:L817-L829：start_token 须块对齐（断言）
        # 零清开关走真实派生谓词（混合精度组）；组 0 为 full → 记账生效
        s = harness_scheduler([], needs_zeroing=True, enable_caching=False)
        req = make_request("r1", range(64))
        s.add_request(req)
        s.schedule()  # NoPrefixCache 支路：分配 4 块
        mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
        mgr.new_block_ids.clear()
        with pytest.raises(AssertionError):
            s.kv_cache_manager.record_blocks_for_zeroing("r1", 17)
        s.kv_cache_manager.record_blocks_for_zeroing("r1", 32)
        blocks = s.kv_cache_manager.get_block_ids("r1")[0]
        assert set(blocks[2:]) <= set(mgr.new_block_ids)

# --------------------------------------------------------------------------- #
# m12/m13/m15：边界三例外
# --------------------------------------------------------------------------- #


class TestBoundaries:
    def test_deferred_free_waits_for_fence(self):
        # 在途步可能还在写 → pop 押 deferred_frees，过栅栏才逆序归还
        s = harness_scheduler([], max_concurrent_batches=2)
        assert s.defer_block_free
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        assert s.sched_step_seq == 1
        req.status = RequestStatus.FINISHED_STOPPED
        req.last_sched_seq = 1  # 在途步（> processed_step_seq=0）
        s._free_request(req)
        assert s.deferred_frees  # 押着没还
        mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
        assert "r1" not in mgr.req_to_blocks  # 账已摘（账实分离）
        # 步处理完：processed_step_seq 前进 → drain 归还
        s.update_from_output(out, sampled_output({"r1": []}))
        assert not s.deferred_frees
        assert free_count(s) == full_free(s)

    def test_deferred_free_direct_when_fence_passed(self):
        # last_sched_seq <= processed_step_seq（正常完成）→ 直接 free
        s = harness_scheduler([], max_concurrent_batches=2)
        req = make_request("r1", range(64))
        s.add_request(req)
        out = s.schedule()
        s.processed_step_seq = s.sched_step_seq  # 步已处理
        req.status = RequestStatus.FINISHED_STOPPED
        req.last_sched_seq = s.processed_step_seq
        s._free_request(req)
        assert not s.deferred_frees
        assert free_count(s) == full_free(s)

    def test_preempt_producer_drops_stale_output(self):
        # requires_kv_delivery=True → drop_stale_output 丢弃在途产出
        s = harness_scheduler([], kv_role="kv_producer")
        assert s.requires_kv_delivery
        req = make_request("r1", range(64))
        s.add_request(req)
        s.schedule()
        assert req.status == RequestStatus.RUNNING
        req.num_in_flight_tokens = 8
        s._preempt_request(req, 0.0, drop_stale_output=s.requires_kv_delivery)
        assert req.drop_stale_output
        assert req.num_stale_output_tokens == 8
        assert req.status == RequestStatus.PREEMPTED
        assert req.num_preemptions == 1
        assert req.num_computed_tokens == 0

    def test_partial_tail_offload_pins_block(self):
        # kv_cache_manager.py:L848-L874：交接块 touch 钉住、随释放路径解钉
        s = harness_scheduler([])
        req = make_request("seed", range(48))
        s.add_request(req)
        s.schedule()  # 活请求：3 块在表上
        mgr = s.kv_cache_manager.coordinator.single_type_managers[0]
        block = mgr.req_to_blocks["seed"][2]
        before = block.ref_cnt
        # mamba align 组才贡献；这里模拟其登记（真实由 align 分配写入）
        mgr._pending_partial_tail_offloads.append(("seed", 0, block, 40))
        offloads = s.kv_cache_manager.take_partial_tail_offloads()
        assert offloads == {"seed": [(0, block.block_id, 40)]}
        assert block.ref_cnt == before + 1  # touch 钉住
        assert mgr._pending_partial_tail_offloads == []  # drain
        # 钉随请求释放路径带上（pop_blocks_for_free 前置 pins）
        blocks = s.kv_cache_manager.pop_blocks_for_free(s.requests["seed"])
        assert block in blocks
        s.kv_cache_manager.block_pool.free_blocks(reversed(blocks))
        assert free_count(s) == full_free(s)

    def test_take_partial_tail_empty_for_non_mamba(self):
        s = harness_scheduler([])
        assert s.kv_cache_manager.take_partial_tail_offloads() == {}


# --------------------------------------------------------------------------- #
# m14：worker 直写池（ExampleConnector inject/extract）
# --------------------------------------------------------------------------- #


@dataclass
class _Layer:
    kv_cache: Any = None


@dataclass
class _FakeForward:
    no_compile_layers: dict
    attn_metadata: dict


class TestExampleWorkerDirectWrite:
    def _worker(self) -> ExampleConnector:
        if os.path.exists(STORAGE):
            shutil.rmtree(STORAGE)
        cfg = vllm_config(
            kv_connector="ExampleConnector",
            kv_role="kv_consumer",
            extra={"shared_storage_path": STORAGE},
        )
        return ExampleConnector(cfg, KVConnectorRole.WORKER, kv_config())

    def test_save_then_inject_round_trip(self):
        # slot 寻址直写池：extract 存盘 → inject 按 slot 写回（m14）
        from safetensors.torch import load_file

        conn = self._worker()
        tokens = list(range(32))
        store_meta = _StoreMeta(tokens, [1, 2], is_store=True)
        conn.bind_connector_metadata(store_meta)
        paged = torch.randn(4, 2, BLOCK, 8)
        slot = store_meta.requests[0].slot_mapping
        expected = paged[slot // BLOCK, :, slot % BLOCK].clone()
        conn.save_kv_layer("layer.0", paged, None)
        fname = conn._generate_filename_debug(
            "layer.0", store_meta.requests[0].token_ids, []
        )
        saved = load_file(fname)["kv_cache"]
        assert torch.equal(saved, expected.cpu())  # extract：按 slot 抽出
        # inject：load 元数据（is_store=False）驱动注回
        load_meta = _StoreMeta(tokens, [1, 2], is_store=False)
        conn.bind_connector_metadata(load_meta)
        paged.zero_()
        layer = _Layer(paged)
        conn.start_load_kv(_FakeForward({"layer.0": layer}, {"layer.0": "md"}))
        restored = paged[slot // BLOCK, :, slot % BLOCK]
        assert torch.equal(restored, expected)
        shutil.rmtree(STORAGE, ignore_errors=True)

    def test_start_load_kv_without_attn_metadata_warns_noop(self):
        # example_connector.py:L155-L158：attn_metadata=None → 告警直返
        conn = self._worker()
        conn.bind_connector_metadata(_StoreMeta(list(range(16)), [0]))
        conn.start_load_kv(_FakeForward({}, None))
        shutil.rmtree(STORAGE, ignore_errors=True)
