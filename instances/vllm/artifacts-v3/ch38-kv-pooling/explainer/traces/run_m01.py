# m1 池的构成与契约再嵌套：facade best-effort 三件 + OffloadingSpecFactory 选引擎 + 生态注册表。
from _driver_common import TESTS, ctx, dump

import sys

sys.path.insert(0, str(TESTS))
from _kv_harness import make_kv_caches, make_kv_config, make_vllm_config  # noqa: E402

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (  # noqa: E402
    OffloadingConnector,
)

kv_config = make_kv_config()

# ── facade 三重身份（property 求值不建实例）──
facade = {
    "requires_kv_delivery_is_false": OffloadingConnector.requires_kv_delivery.fget(None) is False,
    "prefer_cross_layer_blocks_is_true": OffloadingConnector.prefer_cross_layer_blocks.fget(None)
    is True,
    "required_kvcache_layout": OffloadingConnector.get_required_kvcache_layout(None),
    "comment_anchor": "offloading_connector.py:L55-L58（a dropped save is just a future cache miss）",
}

# ── role 分裂：同一 facade 按 role 只建半边 ──
vcfg = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
sched = OffloadingConnector(vcfg, KVConnectorRole.SCHEDULER, kv_config)
work = OffloadingConnector(vcfg, KVConnectorRole.WORKER, kv_config)
role_split = {
    "total_connector_objects": 2,
    "non_none_halves": 2,
    "sched_has_scheduler_half": sched.connector_scheduler is not None,
    "sched_worker_half_is_none": sched.connector_worker is None,
    "work_has_worker_half": work.connector_worker is not None,
    "work_scheduler_half_is_none": work.connector_scheduler is None,
    "scheduler_half_type": type(sched.connector_scheduler).__name__,
    "worker_half_type": type(work.connector_worker).__name__,
}

# ── OffloadingSpecFactory：第二层注册表（spec_name → spec 类）──
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (  # noqa: E402
    build_offloading_config,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory  # noqa: E402

spec_choices = {}
for tag, extra in [
    ("default_no_spec_name", {"cpu_bytes_to_use": 1 << 20}),
    ("cpu_explicit", {"cpu_bytes_to_use": 1 << 20, "spec_name": "CPUOffloadingSpec"}),
    (
        "tiering",
        {
            "cpu_bytes_to_use": 1 << 20,
            "spec_name": "TieringOffloadingSpec",
            "secondary_tiers": [],
        },
    ),
]:
    spec = OffloadingSpecFactory.create_spec(build_offloading_config(make_vllm_config(extra_config=extra), kv_config))
    spec_choices[tag] = type(spec).__name__
try:
    OffloadingSpecFactory.get_spec_cls({"spec_name": "NoSuchSpec"})
    spec_choices["unknown"] = "resolved"
except ValueError as e:
    spec_choices["unknown"] = f"ValueError: {e}"

# ── 契约再嵌套：OffloadingSpec 的双半边工厂（get_manager 调度器进程 / get_worker worker 进程）──
nested = {
    "manager_primitives": [
        "lookup",
        "prepare_load",
        "touch",
        "complete_load",
        "prepare_store",
        "complete_store",
    ],
    "worker_primitives": ["submit_store", "submit_load", "get_finished", "wait"],
    "manager_docstring_anchor": "vllm/v1/kv_offload/base.py:L162-L186",
    "worker_docstring_anchor": "vllm/v1/kv_offload/base.py:L545-L566",
    "outer_contract_anchor": "ch16 KVConnectorBase_V1（v1/base.py:L7-L41）",
    "manager_instance_type": type(sched.connector_scheduler.manager).__name__,
}

# ── 不逐层/不同步填法（ch16 逐层钩子的空实现）──
import torch  # noqa: E402

noop_hooks = {
    "wait_for_layer_load_returns_none": OffloadingConnector.wait_for_layer_load(None, "layer") is None,
    "save_kv_layer_returns_none": OffloadingConnector.save_kv_layer(None, "l", torch.zeros(1), None) is None,
    "wait_for_save_returns_none": OffloadingConnector.wait_for_save(None) is None,
    "anchor": "offloading_connector.py:L103-L134（wait_for_save 注释：Store deferral is handled in get_finished()）",
}

# ── 生态注册表：六条注册行（十条已删的不再注册）──
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory  # noqa: E402

eco = {
    "six_registered": [
        "OffloadingConnector",
        "MultiConnector",
        "MooncakeConnector",
        "MooncakeStoreConnector",
        "LMCacheConnectorV1",
        "NixlConnector",
    ],
    "loadable_in_tree": [
        KVConnectorFactory.get_connector_class_by_name(n).__name__
        for n in ("OffloadingConnector", "MultiConnector", "MooncakeStoreConnector")
    ],
}
try:
    KVConnectorFactory.get_connector_class_by_name("DecodeBenchConnector")
    eco["deleted_example"] = "resolved"
except ValueError as e:
    eco["deleted_example"] = f"ValueError: {e}"

dump(
    "m01.json",
    {
        "facade": facade,
        "role_split": role_split,
        "spec_factory_choices": spec_choices,
        "nested_contract": nested,
        "noop_hooks": noop_hooks,
        "eco_registry": eco,
    },
)
