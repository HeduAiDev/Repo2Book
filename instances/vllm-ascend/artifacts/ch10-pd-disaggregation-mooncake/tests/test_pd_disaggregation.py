"""ch10 — PD 分离三层 + KV 亲和调度：可读控制流测试。

测的是「复现昇腾/基座真实源码的可观察行为」（dossier 记录的连接器分发 / 亲和路由决策 /
proxy 分发 / 块合并 / 角色分发），而非精简版自洽。host 无 NPU/CANN/mooncake：真实
mooncake P2P 跨节点搬运不真跑（由 runtime_stub 接住），只验三层里的纯 Python 控制流。
"""
import sys
from pathlib import Path

IMPL = Path(__file__).resolve().parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

import numpy as np  # noqa: E402

from ascend_multi_connector import AscendMultiConnector, register_connector  # noqa: E402
from multi_connector import MultiConnector  # noqa: E402
from mooncake_layerwise_connector import (  # noqa: E402
    MooncakeLayerwiseConnector,
    MooncakeLayerwiseConnectorScheduler,
    group_concurrent_contiguous,
)
from pool_scheduler import KVPoolScheduler, LoadSpec  # noqa: E402
from load_balance_proxy_server_example import (  # noqa: E402
    ServerRole,
    SharedProxyScheduler,
    build_prefill_request,
    calculate_decode_score,
    calculate_prefill_score,
)
from runtime_stub import (  # noqa: E402
    KVConnectorFactory,
    KVConnectorRole,
    KVCacheBlocks,
    Request,
)


# ---------------------------------------------------------------------------
# 测试替身（fake 子连接器 / 请求 / 块 / lookup client）—— 仅喂控制流。
# ---------------------------------------------------------------------------
class _RecordingChild:
    """子连接器替身：记录被喂了什么 blocks，可配置 matched tokens 返回。"""

    def __init__(self, toks=0, load_async=False, is_layerwise=False):
        self._toks = toks
        self._load_async = load_async
        self.is_layerwise = is_layerwise
        self.alloc_calls = []  # (num_external_tokens, block_ids)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self._toks, self._load_async

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self.alloc_calls.append((num_external_tokens, blocks.get_block_ids()))


def _make_multi(children):
    """绕过真实 __init__（要读 vllm 配置），直接装配 fan-out 状态。"""
    mc = MultiConnector.__new__(MultiConnector)
    mc._connectors = children
    mc._requests_to_connector = {}
    mc._extra_async_saves = {}
    mc._all_support_hma = False
    return mc


# ---------------------------------------------------------------------------
# 第 1 层：连接器分发 —— MultiConnector 选举 + AscendMultiConnector 覆写
# ---------------------------------------------------------------------------
def test_election_picks_first_advertiser():
    """get_num_new_matched_tokens：按 config 顺序，第一个 toks>0 的子连接器赢得 load。"""
    children = [
        _RecordingChild(toks=0),
        _RecordingChild(toks=5, load_async=True),
        _RecordingChild(toks=3),
    ]
    mc = _make_multi(children)
    req = Request("r1", prompt_token_ids=list(range(10)))
    toks, load_async = mc.get_num_new_matched_tokens(req, 0)
    assert (toks, load_async) == (5, True)
    assert mc._requests_to_connector["r1"] == 1  # index 1 owns the load


def test_election_pending_lookup_returns_none():
    """任一子连接器仍在查（toks=None）→ 整体返回 (None, False)，本步不调度。"""
    children = [_RecordingChild(toks=0), _RecordingChild(toks=None)]
    mc = _make_multi(children)
    req = Request("r2", prompt_token_ids=list(range(4)))
    assert mc.get_num_new_matched_tokens(req, 0) == (None, False)


def _make_layerwise_recorder():
    """真·MooncakeLayerwiseConnector 实例（绕过重 __init__），让 isinstance 命中，
    再挂一个 record-only 的 update_state_after_alloc。"""
    lw = MooncakeLayerwiseConnector.__new__(MooncakeLayerwiseConnector)
    lw.alloc_calls = []
    lw.update_state_after_alloc = (
        lambda request, blocks, n: lw.alloc_calls.append((n, blocks.get_block_ids()))
    )
    return lw


def test_ascend_layerwise_always_gets_real_blocks():
    """AscendMultiConnector.update_state_after_alloc：chosen 连接器 + 任何 layerwise
    连接器都拿到真实 blocks；其余拿空 blocks。这是昇腾对基座的关键分歧。"""
    chosen = _RecordingChild(toks=4)          # idx 0, 赢得 load
    middle = _RecordingChild(toks=0)          # idx 1, 既非 chosen 也非 layerwise
    layerwise = _make_layerwise_recorder()    # idx 2, layerwise push（非 chosen）
    amc = AscendMultiConnector.__new__(AscendMultiConnector)
    amc._connectors = [chosen, middle, layerwise]
    amc._requests_to_connector = {"r3": 0}

    req = Request("r3", prompt_token_ids=list(range(8)))
    blocks = KVCacheBlocks([10, 11, 12])
    amc.update_state_after_alloc(req, blocks, num_external_tokens=4)

    assert chosen.alloc_calls == [(4, [10, 11, 12])]       # chosen → real
    assert middle.alloc_calls == [(0, [])]                 # other → empty
    assert layerwise.alloc_calls == [(4, [10, 11, 12])]    # layerwise → real (push needs them)


def test_base_multiconnector_only_chosen_gets_blocks():
    """对照基座：base MultiConnector 只给 chosen 真实 blocks，其余全空（无 layerwise 豁免）。"""
    chosen = _RecordingChild(toks=4)
    other = _RecordingChild(toks=0)
    mc = _make_multi([chosen, other])
    mc._requests_to_connector = {"r4": 0}
    req = Request("r4", prompt_token_ids=list(range(8)))
    mc.update_state_after_alloc(req, KVCacheBlocks([1, 2]), num_external_tokens=4)
    assert chosen.alloc_calls == [(4, [1, 2])]
    assert other.alloc_calls == [(0, [])]


def test_register_connector_overrides_multiconnector():
    """register_connector：pop 掉内置 'MultiConnector' 再指向 AscendMultiConnector，
    并按名注册三个 mooncake 连接器。"""
    KVConnectorFactory._registry.clear()
    KVConnectorFactory._registry["MultiConnector"] = ("vllm.builtin", "MultiConnector")
    register_connector()
    assert KVConnectorFactory._registry["MultiConnector"][1] == "AscendMultiConnector"
    for name in ("MooncakeConnectorV1", "MooncakeHybridConnector", "MooncakeLayerwiseConnector"):
        assert name in KVConnectorFactory._registry


# ---------------------------------------------------------------------------
# 第 1 层（layerwise scheduler）：方向由 kv_transfer_params 标志驱动
# ---------------------------------------------------------------------------
def test_layerwise_scheduler_remote_prefill_pulls_whole_prompt():
    """do_remote_prefill（本节点是 decoder）→ count = 整段 prompt 未算部分，异步拉取。"""
    sched = MooncakeLayerwiseConnectorScheduler.__new__(MooncakeLayerwiseConnectorScheduler)
    sched.block_size = [16]
    req = Request("d1", prompt_token_ids=list(range(64)),
                  kv_transfer_params={"do_remote_prefill": True})
    count, load_async = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert (count, load_async) == (64, True)


def test_layerwise_scheduler_no_remote_prefill():
    sched = MooncakeLayerwiseConnectorScheduler.__new__(MooncakeLayerwiseConnectorScheduler)
    sched.block_size = [16]
    req = Request("d2", prompt_token_ids=list(range(32)), kv_transfer_params=None)
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)


# ---------------------------------------------------------------------------
# 第 2 层：P2P 传输 —— 连续块合并
# ---------------------------------------------------------------------------
def test_group_concurrent_contiguous_coalesces_runs():
    """仅当 local 与 remote 块号都 +1 连续时才并入同一批；任一断开即分批。"""
    src = [1, 2, 3, 7]
    dst = [10, 11, 12, 20]
    src_groups, dst_groups = group_concurrent_contiguous(src, dst)
    assert src_groups == [[1, 2, 3], [7]]
    assert dst_groups == [[10, 11, 12], [20]]


def test_group_concurrent_contiguous_breaks_on_remote_gap():
    """local 连续但 remote 跳号 → 仍要分批。"""
    src_groups, dst_groups = group_concurrent_contiguous([1, 2, 3], [10, 11, 99])
    assert src_groups == [[1, 2], [3]]
    assert dst_groups == [[10, 11], [99]]


def test_group_concurrent_contiguous_empty():
    assert group_concurrent_contiguous([], []) == ([], [])


# ---------------------------------------------------------------------------
# 第 3 层：proxy 负载均衡分发
# ---------------------------------------------------------------------------
def test_prefill_decode_scores():
    assert calculate_prefill_score(4) == 1.0 * 0.0345 + 120.0745
    assert calculate_decode_score(123) == 123


def test_build_prefill_request_stamps_prefiller_role():
    """proxy 给 prefiller 盖章：do_remote_decode=True / do_remote_prefill=False / max_tokens=1。"""
    payload = build_prefill_request({"prompt": "hi", "max_tokens": 100, "stream": True})
    kv = payload["kv_transfer_params"]
    assert kv["do_remote_decode"] is True
    assert kv["do_remote_prefill"] is False
    assert payload["max_tokens"] == 1 and payload["min_tokens"] == 1
    assert payload["stream"] is False


def test_shared_scheduler_picks_least_loaded():
    """两台 prefiller 起步零负载：begin_request 轮流落到当前最轻的一台。"""
    sched = SharedProxyScheduler(
        prefiller_instances=[("0.0.0.0", 8001), ("0.0.0.0", 8002)],
        decoder_instances=[("0.0.0.0", 9001), ("0.0.0.0", 9002)],
    )
    first = sched.begin_request(load=10.0)
    second = sched.begin_request(load=10.0)
    # 第二次应避开刚加了 kv_cache 压力的第一台
    assert first["key"] != second["key"]
    assert sched.request_num == 2


def test_shared_scheduler_decoder_priority_is_active_tokens():
    sched = SharedProxyScheduler(
        prefiller_instances=[("0.0.0.0", 8001)],
        decoder_instances=[("0.0.0.0", 9001), ("0.0.0.0", 9002)],
    )
    d1 = sched.pick_decoder(load=5.0)
    d2 = sched.pick_decoder(load=5.0)
    assert d1["key"] != d2["key"]  # 第二个解码请求落到另一台空闲 decoder


# ---------------------------------------------------------------------------
# ★ 高潮：KV 亲和（cache-hit-aware）路由 —— lookup → need_to_allocate
# ---------------------------------------------------------------------------
class _FakeLookupClient:
    def __init__(self, hit):
        self._hit = hit
        self.calls = []

    def lookup(self, token_len, block_hashes, kv_cache_group_ids=None):
        self.calls.append((token_len, len(block_hashes), kv_cache_group_ids))
        return self._hit


def _make_pool_scheduler(hit, *, discard=False, granularity=16,
                         load_async=True, use_layerwise=False):
    s = KVPoolScheduler.__new__(KVPoolScheduler)
    s.kv_role = "kv_producer"
    s.consumer_is_to_load = False
    s._discard_partial_chunks = discard
    s.cache_transfer_granularity = granularity
    s.kv_cache_group_ids = [0]
    s.client = _FakeLookupClient(hit)
    s.load_specs = {}
    s.load_async = load_async
    s.use_layerwise = use_layerwise
    return s


def test_affinity_need_to_allocate_is_hit_minus_computed():
    """命中 H=300，本地已算 C=100 → 只需拉 H-C=200 个 token，跨节点字节按 200 算。"""
    s = _make_pool_scheduler(hit=300)
    req = Request("a1", prompt_token_ids=list(range(400)), num_tokens=400,
                  block_hashes=[b"h"] * 25)
    need, load_async = s.get_num_new_matched_tokens(req, num_computed_tokens=100)
    assert need == 200
    assert load_async is True  # load_async and not use_layerwise
    spec = s.load_specs["a1"]
    assert spec.vllm_cached_tokens == 100
    assert spec.kvpool_cached_tokens == 300


def test_affinity_full_prompt_hit_is_clamped():
    """全 prompt 命中（H==num_tokens）→ 砍 1，留 1 个 token 必须本地跑 forward。"""
    s = _make_pool_scheduler(hit=400)
    req = Request("a2", prompt_token_ids=list(range(400)), num_tokens=400,
                  block_hashes=[b"h"] * 25)
    need, _ = s.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert need == 399
    assert s.load_specs["a2"].kvpool_cached_tokens == 399


def test_affinity_no_gain_when_hit_below_computed():
    """命中不超过已算 → need_to_allocate<=0 → 不走外部加载。"""
    s = _make_pool_scheduler(hit=50)
    req = Request("a3", prompt_token_ids=list(range(400)), num_tokens=400,
                  block_hashes=[b"h"] * 25)
    assert s.get_num_new_matched_tokens(req, num_computed_tokens=100) == (0, False)
    assert "a3" not in s.load_specs


def test_affinity_consumer_not_to_load_short_circuits():
    s = _make_pool_scheduler(hit=300)
    s.kv_role = "kv_consumer"
    s.consumer_is_to_load = False
    req = Request("a4", prompt_token_ids=list(range(400)), num_tokens=400,
                  block_hashes=[b"h"] * 25)
    assert s.get_num_new_matched_tokens(req, 0) == (0, False)


def test_affinity_layerwise_disables_async_load():
    """use_layerwise=True → 第二返回值（load_async and not use_layerwise）变 False。"""
    s = _make_pool_scheduler(hit=300, load_async=True, use_layerwise=True)
    req = Request("a5", prompt_token_ids=list(range(400)), num_tokens=400,
                  block_hashes=[b"h"] * 25)
    need, load_async = s.get_num_new_matched_tokens(req, 0)
    assert need == 300 and load_async is False


def test_loadspec_is_a_dataclass_with_token_counts():
    spec = LoadSpec(vllm_cached_tokens=10, kvpool_cached_tokens=30, can_load=False)
    assert spec.vllm_cached_tokens == 10 and spec.kvpool_cached_tokens == 30
