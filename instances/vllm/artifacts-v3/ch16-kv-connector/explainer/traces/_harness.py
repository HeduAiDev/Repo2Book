# ch16 explainer 驱动共享 harness——契约的可编程测试替身 + 最小装配镜像。
# 与 tests/test_kv_connector.py 的替身同构（经 factory 的 kv_connector_module_path
# 装配，真实机制 factory.py:L105-L123）；差异：查询入参也被记录（m3 要证
# block_aligned_local 呈给 connector）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch  # noqa: E402

from implementation import kv_transfer_state  # noqa: E402
from implementation.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    SupportsHMA,
)
from implementation.cache import CacheConfig  # noqa: E402
from implementation.config import ModelConfig, VllmConfig  # noqa: E402
from implementation.factory import KVConnectorFactory  # noqa: E402
from implementation.hashing import sha256  # noqa: E402
from implementation.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from implementation.kv_cache_utils import (  # noqa: E402
    get_request_block_hasher,
    init_none_hash,
    make_block_hash_with_group_id,
)
from implementation.outputs import KVConnectorOutput, ModelRunnerOutput  # noqa: E402
from implementation.request import Request  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.kv_transfer import KVTransferConfig  # noqa: E402

init_none_hash(sha256)
HASHER16 = get_request_block_hasher(16, sha256)
HASHER8 = get_request_block_hasher(8, sha256)

BLOCK = 16


# --------------------------------------------------------------------------- #
# 契约测试替身：外部命中/收发完成可编程，调用与入参可观测。
# --------------------------------------------------------------------------- #

class ScriptMeta(KVConnectorMetadata):
    pass


class ScriptAggMeta(KVConnectorWorkerMetadata):
    def aggregate(self, other):
        return self


class ScriptConnector(KVConnectorBase_V1):
    # SOURCE 语义（base.py:L465-L539）：两半抽象的可编程实现
    def __init__(self, vllm_config, role, kv_cache_config, **kwargs):
        super().__init__(vllm_config, role, kv_cache_config)
        self.ext_answer = []          # [(tokens, async) | None] 逐次弹出
        self.finish_answer = False    # request_finished 的接管答案
        self.events = []              # 调用序 + 入参账本
        self.bound_pools = []

    def bind_gpu_block_pool(self, gpu_block_pool):
        self.bound_pools.append(gpu_block_pool)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # 记录入参——m3 的 block_aligned_local 呈 connector 的证据
        self.events.append(
            ("query", request.request_id, num_computed_tokens)
        )
        if not self.ext_answer:
            return 0, False
        return self.ext_answer.pop(0)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.events.append(
            ("update_state_after_alloc", request.request_id, num_external_tokens)
        )

    def build_connector_meta(self, scheduler_output):
        self.events.append(("build_connector_meta",))
        return ScriptMeta()

    def request_finished(self, request, block_ids):
        self.events.append(
            ("request_finished", request.request_id, list(block_ids))
        )
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
        self.events.append(("get_finished", sorted(finished_req_ids)))
        return None, None

    def get_block_ids_with_load_errors(self):
        return set()

    def build_connector_worker_meta(self):
        return None


class ScriptHMA(ScriptConnector, SupportsHMA):
    def request_finished_all_groups(self, request, block_ids):
        self.events.append(
            ("request_finished_all_groups", request.request_id,
             tuple(list(g) for g in block_ids))
        )
        return self.finish_answer, None


KVConnectorFactory.register_connector("ScriptConnector", "_harness", "ScriptConnector")
KVConnectorFactory.register_connector("ScriptHMA", "_harness", "ScriptHMA")
MODULE_PATH = "_harness"


# --------------------------------------------------------------------------- #
# 构造辅助：真实装配的最小镜像（与 tests 同源）
# --------------------------------------------------------------------------- #

def full_spec(block_size: int = BLOCK, dtype=torch.float16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=8, dtype=dtype
    )


def mamba_align_spec(block_size: int = BLOCK) -> MambaSpec:
    return MambaSpec(
        block_size=block_size,
        shapes=((8, 8),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
    )


def kv_config(num_blocks: int = 64, mixed_precision: bool = False) -> KVCacheConfig:
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


def hybrid_kv_config(num_blocks: int = 64) -> KVCacheConfig:
    """m3 子块尾场景：full(16) + mamba-align(16) 两组（ch15 m15/m16 配置）。"""
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer.0", "layer.1"], full_spec()),
            KVCacheGroupSpec(["layer.2"], mamba_align_spec()),
        ],
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
    return Scheduler(
        vllm_config=cfg,
        kv_cache_config=kv_config(num_blocks, mixed_precision=needs_zeroing),
        block_size=BLOCK,
        hash_block_size=hash_bs,
        max_model_len=max_model_len,
    )


def script_scheduler(script: list, hma: bool = False, **kw) -> Scheduler:
    name = "ScriptHMA" if hma else "ScriptConnector"
    kw.setdefault("kv_role", "kv_consumer")
    s = make_scheduler(connector_name=name, module_path=MODULE_PATH, **kw)
    s.connector.ext_answer = list(script)
    return s


def hybrid_scheduler(script: list, **kw) -> Scheduler:
    """m3 的混合两组调度器（full+mamba-align、hash 8 → 块内边界命中）。"""
    cfg = vllm_config(
        kv_connector="ScriptConnector",
        kv_role="kv_consumer",
        module_path=MODULE_PATH,
    )
    s = Scheduler(
        vllm_config=cfg,
        kv_cache_config=hybrid_kv_config(kw.pop("num_blocks", 64)),
        block_size=BLOCK,
        hash_block_size=8,
        max_model_len=512,
    )
    s.connector.ext_answer = list(script)
    return s


def free_count(s: Scheduler) -> int:
    return s.kv_cache_manager.block_pool.get_num_free_blocks()


def full_free(s: Scheduler) -> int:
    """全空闲基线 = num_blocks − 1（null 块恒占用）。"""
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


def blk_by_hash(pool, block_hash, group_id=0):
    return pool.cached_block_hash_to_block.get_one_block(
        make_block_hash_with_group_id(block_hash, group_id)
    )


def sampled_output(tokens_by_req: dict[str, list[int]]) -> ModelRunnerOutput:
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


def dump(name: str, obj) -> str:
    """trace 落盘（LF、ensure_ascii=False）；返回相对路径。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return os.path.relpath(path, os.path.join(os.path.dirname(path), "..", ".."))
