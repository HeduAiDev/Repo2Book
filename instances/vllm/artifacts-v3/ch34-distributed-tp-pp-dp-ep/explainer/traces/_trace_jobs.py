# ch34 explainer 取证作业表：每个 job = 一个机制的取证现场。
# 全部跑在精简版（implementation/，host gloo 退化形态）上——与 pin（NCCL/CUDA）
# 的差异统一登记在 explainer.json 的 trace_environment，数字旁按需标注。
# 对被测面只做 monkeypatch 记录 / 测试替身注入（逐处标注），不携带目标代码仓
# 没有的行为。
from __future__ import annotations

import socket

JOBS: dict[int, dict[str, object]] = {1: {}, 2: {}, 4: {}, 8: {}}


def _job(world: int, name: str):
    def deco(fn):
        JOBS[world][name] = fn
        return fn

    return deco


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fresh_store() -> dict:
    # 路径带进程号防跨轮复用（tests/_pool.py 同款防毒化约定）
    import os

    from _trace_pool import _TMP_DIR

    _TMP_DIR.mkdir(exist_ok=True)
    seq = getattr(_fresh_store, "_seq", 0) + 1
    _fresh_store._seq = seq
    return {
        "path": (_TMP_DIR / f"s{os.getpid()}_{seq}").as_posix(),
        "port": _free_port(),
    }


def _destroy_model_parallel_quietly():
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass


# ═════════ world=8：5 维 rank 张量四刀（m2/m3） ═════════


@_job(8, "cuts_full")
def _job_cuts(rank, world, payload):
    """真 tensor 数学 + 真调用位（monkeypatch init_model_parallel_group 记录
    group_ranks；world 组 new_group 走真实路径）。is_moe 由 payload 控制。"""
    import torch
    import torch.distributed as dist

    import vllm.distributed.parallel_state as ps
    from _pool import _init_model_parallel

    class _FakeGroup:
        def __init__(self, group_ranks):
            self.group_ranks = group_ranks
            self.world_size = len(group_ranks[0]) if group_ranks else 1
            self.rank_in_group = 0

        def destroy(self):
            pass

    recorded: dict[str, object] = {}
    n_factory_calls = [0]

    def fake_factory(group_ranks, local_rank, backend, **kwargs):
        n_factory_calls[0] += 1
        name = kwargs.get("group_name", "?")
        recorded.setdefault(name, group_ranks)
        if kwargs.get("use_all2all"):
            recorded[f"{name}:use_all2all"] = True
        if kwargs.get("use_message_queue_broadcaster"):
            recorded[f"{name}:mq"] = True
        return _FakeGroup(group_ranks)

    real_factory = ps.init_model_parallel_group
    ps.init_model_parallel_group = fake_factory
    try:
        # sizes = (tp, pp, dp, pcp) = (2, 2, 2, 1)——m2 worked example 参数
        _init_model_parallel(rank, world, (2, 2, 2, 1), payload["is_moe"], payload["store"])
        global_rank = dist.get_rank()
        engine_size = 2 * 2 * 1  # tp*pp*pcp
        out = {
            "is_moe": payload["is_moe"],
            "recorded": recorded,
            "factory_calls": n_factory_calls[0],
            # 站 5 公式：global = dp_rank*engine_size + engine_rank（偏移后真值）
            "global_rank": global_rank,
            "dp_rank": rank // engine_size,
            "engine_rank": rank % engine_size,
        }
        if rank == 0:
            # 机械公式就地复演（与源码 L1812-L1827 完全同式，纯 tensor 数学）
            dp, pp, pcp, tp = 2, 2, 1, 2
            all_ranks = torch.arange(world).reshape(-1, dp, pp, pcp, tp)
            out["all_ranks_shape"] = list(all_ranks.shape)
            out["all_ranks"] = all_ranks.tolist()
            out["derivation"] = {
                "tp_view": [x.tolist() for x in all_ranks.view(-1, tp).unbind(0)],
                "pp_transpose24": [
                    x.tolist()
                    for x in all_ranks.transpose(2, 4).reshape(-1, pp).unbind(0)
                ],
                "dp_transpose14": [
                    x.tolist()
                    for x in all_ranks.transpose(1, 4).reshape(-1, dp).unbind(0)
                ],
                "ep_transpose12": [
                    x.tolist()
                    for x in all_ranks.transpose(1, 2)
                    .reshape(-1, dp * pcp * tp)
                    .unbind(0)
                ],
            }
            # 每维组数 × 双群组（L455-L470：每组 new_group 出 device+cpu 两个 PG）
            groups_per_dim = {
                k: len(v) for k, v in recorded.items() if isinstance(v, list)
            }
            out["groups_per_dim"] = groups_per_dim
            out["groups_total"] = sum(groups_per_dim.values())
            out["new_group_total_estimate"] = 2 * sum(groups_per_dim.values())
            # 四刀的机械式原文（源码 L1829-L1950 同式，记录在案）
            out["cut_formulas"] = [
                "all_ranks.view(-1, 2).unbind(0)",
                "all_ranks.transpose(2, 4).reshape(-1, 2).unbind(0)",
                "all_ranks.transpose(1, 4).reshape(-1, 2).unbind(0)",
                "all_ranks.transpose(1, 2).reshape(-1, 4).unbind(0)",
            ]
        return out
    finally:
        ps.init_model_parallel_group = real_factory
        _destroy_model_parallel_quietly()
        from vllm.distributed.parallel_state import destroy_distributed_environment

        destroy_distributed_environment()
        if dist.is_initialized():
            dist.destroy_process_group()


# ═════════ world=2 / world=1：GroupCoordinator 解剖 + all_reduce 派发（m1/m5） ═════════


@_job(2, "tp_anatomy")
def _job_tp_anatomy(rank, world, payload):
    """真建组（无 fake）：TP=2 的 GroupCoordinator 解剖 + all_reduce 三条路径中的
    两条（直接路径 / 强开 custom-op 路径）。partial=[rank+1]*6 → AR 出 3.0。"""
    import torch
    import torch.distributed as dist

    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group
    from _pool import _init_model_parallel

    try:
        _init_model_parallel(rank, world, (2, 1, 1, 1), False, payload["store"])
        tp = get_tp_group()
        partial = torch.full((2, 3), float(rank + 1))
        out = tensor_model_parallel_all_reduce(partial)
        # m5：use_custom_op_call 强开（host CPU 平台 seam 默认 False——真实 CUDA
        # 平台为 True；强开走 torch.ops.vllm.all_reduce(group_name=...) 真路径）
        use_custom_op_default = tp.use_custom_op_call
        tp.use_custom_op_call = True
        partial2 = torch.full((2,), float(rank + 1))
        out_custom_op = tensor_model_parallel_all_reduce(partial2)
        tp.use_custom_op_call = use_custom_op_default
        # barrier 刻意走 cpu_group（真实调用位）；对象广播走 cpu_group
        tp.barrier()
        obj = {"k": rank, "list": [1, 2]} if rank == 0 else None
        got_obj = tp.broadcast_object(obj, src=0)
        dc = tp.device_communicator
        return {
            "partial": partial.tolist(),
            "out_direct": out.tolist(),
            "out_custom_op": out_custom_op.tolist(),
            "use_custom_op_call_host_default": use_custom_op_default,
            "ranks": tp.ranks,
            "world_size": tp.world_size,
            "rank_in_group": tp.rank_in_group,
            "local_rank": tp.local_rank,
            "device": str(tp.device),
            "cpu_backend": str(dist.get_backend(tp.cpu_group)),
            "device_backend": str(dist.get_backend(tp.device_group)),
            "dc_type": type(dc).__name__,
            "dc_pynccl_seam": type(dc.pynccl_comm).__name__,
            "dc_pynccl_disabled": bool(dc.pynccl_comm.disabled),
            # 七级回退链前六级在 host 的构造面状态（seam：never enabled）
            "dc_qr": dc.qr_comm is not None,
            "dc_fi": dc.fi_ar_comm is not None,
            "dc_aiter": dc.aiter_ar_comm is not None,
            "dc_ca": dc.ca_comm is not None,
            "dc_symm": dc.symm_mem_comm is not None,
            "has_mq": tp.mq_broadcaster is not None,
            "mq_type": type(tp.mq_broadcaster).__name__,
            "unique_name": tp.unique_name,
            "broadcast_object": got_obj,
        }
    finally:
        _destroy_model_parallel_quietly()


@_job(1, "tp1_shortcircuit")
def _job_tp1(rank, world, payload):
    """m5 短路分支：world_size==1 的组 all_reduce 原样返回同一张量（is 同一对象）。"""
    import torch

    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group
    from _pool import _init_model_parallel

    try:
        _init_model_parallel(rank, world, (1, 1, 1, 1), False, payload["store"])
        tp = get_tp_group()
        x = torch.tensor([5.0, 7.0])
        out = tensor_model_parallel_all_reduce(x)
        return {
            "ranks": tp.ranks,
            "world_size": tp.world_size,
            "in": x.tolist(),
            "out": out.tolist(),
            "out_is_input": out is x,
        }
    finally:
        _destroy_model_parallel_quietly()


# ═════════ world=2：PP 段间张量字典（m8） ═════════


@_job(2, "pp_dict")
def _job_pp_dict(rank, world, payload):
    """isend/irecv_tensor_dict 全链：metadata（非张量）走 cpu_group 对象通道、
    张量各自 isend/irecv；首段发次段收。捕获句柄数=张量 isend 数。"""
    import torch

    from vllm.distributed.parallel_state import get_pp_group
    from _pool import _init_model_parallel

    try:
        _init_model_parallel(rank, world, (1, 2, 1, 1), False, payload["store"])
        pp = get_pp_group()
        if pp.is_first_rank:
            d = {
                "hidden": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "residual": torch.ones(3, 4) * 7.0,
                "scalar_meta": {"num": 42},
            }
            handles = pp.isend_tensor_dict(d)
            for h in handles:
                h.wait()
            return {
                "role": "sender",
                "keys_sent": sorted(d.keys()),
                "n_tensor_handles": len(handles),
            }
        else:
            tensor_dict, handles, post = pp.irecv_tensor_dict()
            for h in handles:
                h.wait()
            for fn in post:
                fn()
            return {
                "role": "receiver",
                "hidden": tensor_dict["hidden"].tolist(),
                "residual": tensor_dict["residual"].tolist(),
                "scalar_meta": tensor_dict["scalar_meta"],
                "n_tensor_handles": len(handles),
                "n_postprocess": len(post),
            }
    finally:
        _destroy_model_parallel_quietly()


# ═════════ world=2：PP 懒同步（m9 实证部分） ═════════


@_job(2, "lazy_mp")
def _job_lazy_mp(rank, world, payload):
    """真 irecv/isend 下的懒同步结构证据：sender 故意延迟 0.5s 才 isend；
    receiver 的 irecv_tensor_dict 会在 metadata 通道上等到 0.5s（cpu_group 对象
    通道是同步门），但返回时张量句柄仍未 wait——结构事实『句柄随返回外带、
    wait 被推迟到 .tensors 首触』由 n_handles_unwaited 与时间戳实证；
    数值正确性由 roundtrip 相等实证。"""
    import time

    import torch

    from vllm.distributed.parallel_state import get_pp_group
    from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors
    from _pool import _init_model_parallel

    try:
        _init_model_parallel(rank, world, (1, 2, 1, 1), False, payload["store"])
        pp = get_pp_group()
        if pp.is_first_rank:
            t0 = time.perf_counter()
            time.sleep(payload["sender_delay_s"])
            d = {"hidden": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
            handles = pp.isend_tensor_dict(d)
            for h in handles:
                h.wait()
            t_done = time.perf_counter() - t0
            return {
                "role": "sender",
                "sender_delay_s": payload["sender_delay_s"],
                "t_isend_done_s": round(t_done, 2),
            }
        else:
            t0 = time.perf_counter()
            tensor_dict, handles, post = pp.irecv_tensor_dict()
            t_irecv_return = time.perf_counter() - t0  # metadata 门 + 句柄外带
            unwated_before_touch = sum(1 for h in handles if not h.is_completed())
            a = AsyncIntermediateTensors(
                tensor_dict, comm_handles=handles, comm_postprocess=post
            )
            # 本地准备（模拟 KV 取址 / 注意力元数据构建等与接收无关的本地活）
            time.sleep(payload["local_prep_s"])
            t_touch = time.perf_counter() - t0
            _ = a.tensors  # 首次触碰 → wait 全部句柄
            t_done = time.perf_counter() - t0
            return {
                "role": "receiver",
                "sender_delay_s": payload["sender_delay_s"],
                "local_prep_s": payload["local_prep_s"],
                "t_irecv_return_s": round(t_irecv_return, 2),
                "n_handles": len(handles),
                "n_handles_unwaited_on_return": unwated_before_touch,
                "t_touch_s": round(t_touch, 2),
                "t_done_s": round(t_done, 2),
                "hidden_ok": tensor_dict["hidden"].tolist()
                == torch.arange(12, dtype=torch.float32).reshape(3, 4).tolist(),
            }
    finally:
        _destroy_model_parallel_quietly()


# ═════════ world=4：PP×TP 切片优化（m11） ═════════


@_job(4, "slice_capture")
def _job_slice(rank, world, payload):
    """m11：numel 整除 TP 的张量只发本 rank 1/tp 切片（发送侧就地复演源码
    L1061-L1063 的切片式），接收端 irecv 切片后 all_gather 重建全张量。
    rank0/1 = PP 首段（TP0/TP1），rank2/3 = PP 次段。"""
    import torch

    from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    from _pool import _init_model_parallel

    try:
        _init_model_parallel(rank, world, (2, 2, 1, 1), False, payload["store"])
        pp = get_pp_group()
        tp = get_tp_group()
        if pp.is_first_rank:
            full = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            handles = pp.isend_tensor_dict(
                {"hidden": full}, all_gather_group=tp, all_gather_tensors={}
            )
            for h in handles:
                h.wait()
            # 发送侧切片式（源码 L1061-L1063：tensor.reshape(ws, -1)[rank]）
            all_gather_size = tp.world_size
            sent = full.reshape(all_gather_size, -1)[tp.rank_in_group]
            return {
                "role": "pp0",
                "tp_rank": tp.rank_in_group,
                "full_numel": full.numel(),
                "full_shape": list(full.shape),
                "sent_shape": list(sent.shape),
                "sent_numel": sent.numel(),
                "sent_rows": sent.tolist(),
            }
        else:
            tensor_dict, handles, post = pp.irecv_tensor_dict(
                all_gather_group=tp, all_gather_tensors={}
            )
            for h in handles:
                h.wait()
            for fn in post:
                fn()
            rebuilt = tensor_dict["hidden"]
            expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            return {
                "role": "pp1",
                "tp_rank": tp.rank_in_group,
                "rebuilt_shape": list(rebuilt.shape),
                "rebuilt_equals_original": torch.equal(rebuilt, expected),
                "rebuilt": rebuilt.tolist(),
            }
    finally:
        _destroy_model_parallel_quietly()


# ═════════ world=2：EP dispatch/combine 全链（m22） ═════════


@_job(2, "ep_full")
def _job_ep_full(rank, world, payload):
    """m22 worked example：2 rank 各 3 token、topk 一奇一偶跨 rank 命中。
    裸 dispatch/combine + MoE prepare/finalize 消费现场两段都取。"""
    import torch

    from vllm.distributed.parallel_state import get_ep_group
    from vllm.forward_context import (
        DPMetadata,
        ForwardContext,
        override_forward_context,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceContiguous,
    )
    from _pool import _init_model_parallel

    try:
        _init_model_parallel(rank, world, (1, 1, 2, 1), True, payload["store"])
        ep = get_ep_group()
        torch.manual_seed(7)
        n_local, hidden, topk = 3, 4, 2
        h = torch.randn(n_local, hidden)
        topk_ids = torch.tensor([[0, 1], [2, 0], [1, 2]], dtype=torch.int32)
        topk_w = torch.full((n_local, topk), 0.5)

        sizes = [n_local, n_local]
        ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
        ctx.dp_metadata = DPMetadata(torch.tensor(sizes, dtype=torch.int32))
        with override_forward_context(ctx):
            with ctx.dp_metadata.sp_local_sizes(1):
                gh, gw, gids = ep.dispatch(h, topk_w, topk_ids)
        # 本地专家 = 偶数专家在 rank0、奇数在 rank1（e % world == rank）
        local_experts = [e for e in range(4) if e % world == rank]
        hit_rows = [t for t in range(gh.shape[0]) if any(int(gids[t, k]) % world == rank for k in range(topk))]
        contrib = torch.zeros_like(gh)
        n_contrib_slots = 0
        for t in range(gh.shape[0]):
            for k in range(topk):
                if int(gids[t, k]) % world == rank:
                    contrib[t] += 0.5 * gh[t]
                    n_contrib_slots += 1
        with override_forward_context(ctx):
            with ctx.dp_metadata.sp_local_sizes(1):
                back = ep.combine(contrib)

        # MoE 消费现场（naive_dp_ep prepare/finalize，恒等『专家』）
        torch.manual_seed(11)
        a1 = torch.randn(n_local, hidden)
        pf = MoEPrepareAndFinalizeNaiveDPEPModular(is_sequence_parallel=False)
        with override_forward_context(ctx):
            with ctx.dp_metadata.sp_local_sizes(1):
                a1q, scale, _, gids2, gw2 = pf.prepare(
                    a1, topk_w, topk_ids, num_experts=4, expert_map=None,
                    apply_router_weight_on_input=False, quant_config=None,
                )
            fused = torch.zeros(n_tok := gids2.shape[0], topk, hidden)
            for t in range(n_tok):
                for k in range(topk):
                    if int(gids2[t, k]) % world == rank:
                        fused[t, k] = a1q[t]
            out = torch.zeros(n_local, hidden)
            with ctx.dp_metadata.sp_local_sizes(1):
                pf.finalize(
                    out, fused, gw2, gids2,
                    apply_router_weight_on_input=False,
                    weight_and_reduce_impl=TopKWeightAndReduceContiguous(),
                )
        return {
            "rank": rank,
            "ep_name": ep.unique_name,
            "dc_all2all": type(ep.device_communicator.all2all_manager).__name__,
            "sizes": sizes,
            "h": h.tolist(),
            "topk_ids": topk_ids.tolist(),
            "topk_weights": topk_w.tolist(),
            "gathered_rows": int(gh.shape[0]),
            "gathered_ids": gids.tolist(),
            "local_experts": local_experts,
            "hit_rows": hit_rows,
            "n_hit_rows": len(hit_rows),
            "n_contrib_slots": n_contrib_slots,
            "back": back.tolist(),
            "back_equals_h": bool(
                all(
                    abs(a - b) < 1e-6
                    for ra, rb in zip(back.tolist(), h.tolist())
                    for a, b in zip(ra, rb)
                )
            ),
            "moe_prepare_rows": int(gids2.shape[0]),
            "moe_finalize_rows": int(out.shape[0]),
            "moe_out_equals_a1": bool(
                all(
                    abs(a - b) < 1e-6
                    for ra, rb in zip(out.tolist(), a1.tolist())
                    for a, b in zip(ra, rb)
                )
            ),
            "a1": a1.tolist(),
        }
    finally:
        _destroy_model_parallel_quietly()


# ═════════ world=1：DPCoordinator 三 socket 控制面（m18/m20） ═════════


@_job(1, "coord_timed")
def _job_coord(rank, world, payload):
    """真 XPUB/XSUB/PULL 三 socket + 真 100ms 发布节拍 + wave 状态机。
    相比 tests 的 coordinator_e2e 增测发布时延（stats push → 前端收到）。"""
    import threading
    import time

    import msgspec.msgpack
    import zmq

    from vllm.utils.network_utils import make_zmq_socket
    from vllm.v1.engine import (
        EngineCoreOutputs,
        EngineCoreRequestType,
        SchedulerStats,
    )
    from vllm.v1.engine.coordinator import DPCoordinatorProc

    def _reserve_addr() -> str:
        ctx = zmq.Context()
        s = ctx.socket(zmq.PUSH)
        s.bind("tcp://127.0.0.1:0")
        addr = s.getsockopt(zmq.LAST_ENDPOINT).decode()
        s.close(0)
        ctx.term()
        return addr

    front_addr, back_out_addr, back_pub_addr = (
        _reserve_addr(), _reserve_addr(), _reserve_addr(),
    )
    coord = DPCoordinatorProc(engine_count=2, enable_wave_coordination=True)
    t = threading.Thread(
        target=coord.process_input_socket,
        args=(front_addr, back_out_addr, back_pub_addr),
        daemon=True,
    )
    t.start()

    ctx = zmq.Context()
    events: list = []
    eng = ctx.socket(zmq.XSUB)
    eng.connect(back_pub_addr)
    eng.send(b"\x01")  # 订阅帧（win32 libzmq XSUB 不认 setsockopt SUBSCRIBE）
    eng1 = ctx.socket(zmq.XSUB)
    eng1.connect(back_pub_addr)
    eng1.send(b"\x01")
    deadline = time.time() + 10
    ready = None
    while time.time() < deadline:
        if eng.poll(100):
            ready = eng.recv()
            if ready == b"READY":
                break
    assert ready == b"READY", f"no READY from coordinator: {ready!r}"
    while eng1.poll(100):
        assert eng1.recv() == b"READY"
    events.append(("ready", True))

    front = ctx.socket(zmq.XSUB)
    front.connect(front_addr)
    front.send(b"\x01")

    push = ctx.socket(zmq.PUSH)
    push.connect(back_out_addr)
    stats = SchedulerStats(
        num_waiting_reqs=2, num_running_reqs=1, kv_cache_usage=0.6,
        step_counter=1, current_wave=0,
    )
    t_push = time.perf_counter()
    push.send(msgspec.msgpack.encode(EngineCoreOutputs(scheduler_stats=stats)))
    got_stats = None
    t_publish = None
    deadline = time.time() + 10
    while time.time() < deadline:
        if front.poll(200):
            counts, wave, running = msgspec.msgpack.decode(front.recv())
            if counts is not None:
                got_stats = (counts, wave, running)
                t_publish = time.perf_counter() - t_push
                break
    events.append(("stats", got_stats))
    events.append(("publish_latency_cold_s", round(t_publish, 2)))

    # 暖路径：再上报一份变化后的 stats，量『连接已热』的发布时延
    stats2 = SchedulerStats(
        num_waiting_reqs=3, num_running_reqs=0, kv_cache_usage=0.9,
        step_counter=2, current_wave=0,
    )
    t_push2 = time.perf_counter()
    push.send(msgspec.msgpack.encode(EngineCoreOutputs(scheduler_stats=stats2)))
    t_publish2 = None
    deadline = time.time() + 10
    while time.time() < deadline:
        if front.poll(200):
            counts, wave, running = msgspec.msgpack.decode(front.recv())
            if counts is not None and counts[0][0] == 3:
                t_publish2 = time.perf_counter() - t_push2
                break
    events.append(("stats2_counts00", 3))
    events.append(("publish_latency_warm_s", round(t_publish2, 2)))

    front.send(msgspec.msgpack.encode((1, got_stats[1])))
    got_wave = None
    deadline = time.time() + 10
    while time.time() < deadline:
        if eng.poll(200):
            frame = eng.recv_multipart()
            if frame[0] == EngineCoreRequestType.START_DP_WAVE.value:
                got_wave = msgspec.msgpack.decode(frame[1])
                break
    events.append(("start_wave", got_wave))

    push.send(
        msgspec.msgpack.encode(EngineCoreOutputs(wave_complete=0, engine_index=0))
    )
    got_state = None
    deadline = time.time() + 10
    while time.time() < deadline:
        if front.poll(200):
            counts, wave, running = msgspec.msgpack.decode(front.recv())
            if counts is None and wave == 1 and running is False:
                got_state = (counts, wave, running)
                break
    events.append(("wave_complete", got_state))

    push.send(
        msgspec.msgpack.encode(EngineCoreOutputs(start_wave=3, engine_index=0))
    )
    got_stale = None
    deadline = time.time() + 10
    while time.time() < deadline:
        if eng.poll(200):
            frame = eng.recv_multipart()
            if frame[0] == EngineCoreRequestType.START_DP_WAVE.value:
                got_stale = msgspec.msgpack.decode(frame[1])
                break
    events.append(("stale_wave", got_stale))

    eng.close(0)
    eng1.close(0)
    front.close(0)
    push.close(0)
    ctx.term()
    return {
        "events": events,
        "source_constants": {
            "stats_update_interval_ms": 100,
            "heartbeat_ms": 5000,
            "lockstep_min_timeout_ms": 50,
            "provenance": "coordinator.py:L158/L256-L261/L266-L271（源码常量，非本机实测）",
        },
    }


# ═════════ world=2：wave 共识（m19） ═════════


@_job(2, "sync_truth")
def _job_sync_truth(rank, world, payload):
    """sync_dp_state 2 元素 SUM 双共识真值表 + has_unfinished_dp MAX≡OR。
    同一 stateless gloo 组上连续跑多组输入（sync_dp_state 无状态，每次全新张量）。"""
    from vllm.config import ParallelConfig
    from vllm.distributed.utils import (
        stateless_destroy_torch_distributed_process_group,
    )
    from _pool import _make_config

    cfg = _make_config((1, 1, 2, 1), True)
    pc = cfg.parallel_config
    pc.data_parallel_rank = rank
    pc.data_parallel_master_ip = "127.0.0.1"
    pc.data_parallel_master_port = payload["port"]
    group = pc.stateless_init_dp_group()
    rows = []
    for has_unfinished, pending_pause in payload["combos"]:
        hu, pp = ParallelConfig.sync_dp_state(
            group, has_unfinished=has_unfinished[rank], pending_pause=pending_pause[rank]
        )
        or_r = ParallelConfig.has_unfinished_dp(
            group, has_unfinished=has_unfinished[rank]
        )
        rows.append(
            {
                "has_unfinished_inputs": has_unfinished,
                "pending_pause_inputs": pending_pause,
                "tensor_before": [int(has_unfinished[rank]), int(pending_pause[rank])],
                "has_global": hu,
                "pause_consensus": pp,
                "or_result": or_r,
            }
        )
    stateless_destroy_torch_distributed_process_group(group)
    return {"rows": rows, "dp_size": group.size()}


@_job(2, "wave_lockstep")
def _job_wave_lockstep(rank, world, payload):
    """DPEngineCoreProc 忙循环真跑：全员无活 → dummy batch 维持锁步 → 32 步共识
    → dp_rank0 发 wave_complete(client_index=-1) → current_wave+=1 → 暂停空转。"""
    import queue

    from vllm.distributed.utils import (
        stateless_destroy_torch_distributed_process_group,
    )
    from vllm.v1.engine.core import DPEngineCoreProc
    from _pool import _make_config

    cfg = _make_config((1, 1, 2, 1), True)
    pc = cfg.parallel_config
    pc.data_parallel_rank = rank
    pc.data_parallel_master_ip = "127.0.0.1"
    pc.data_parallel_master_port = payload["port"]
    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    proc.log_stats = False
    proc.dp_rank = rank
    proc.dp_size = 2
    proc.dp_group = pc.stateless_init_dp_group()
    proc.step_counter = 0
    proc.current_wave = 0
    proc.pending_pause = False
    proc.ignore_start_dp_wave = False
    proc.engines_running = True
    proc.has_coordinator = True
    proc.publish_dp_lb_stats = False
    proc.output_queue = queue.Queue()
    proc.shutdown_state = 0
    proc.iter_budget = [payload.get("max_iters", 40)]

    class _Sched:
        def has_unfinished_requests(self):
            return False

        def has_requests(self):
            return False

        def get_request_counts(self):
            return (0, 0)

        def get_kv_cache_usage(self):
            return 0.0

    class _Exec:
        is_sleeping = False

    proc.scheduler = _Sched()
    proc.model_executor = _Exec()
    dummies = []
    steps_at_dummy = []

    def _drain_budget():
        proc.iter_budget[0] -= 1
        if proc.iter_budget[0] <= 0:
            proc.shutdown_state = 1

    proc._process_input_queue = _drain_budget
    proc._process_engine_step = lambda: False
    proc._handle_shutdown = lambda: proc.shutdown_state == 0
    _real_dummy = proc.execute_dummy_batch

    def _counting_dummy():
        steps_at_dummy.append(proc.step_counter + 1)
        dummies.append(1)

    proc.execute_dummy_batch = _counting_dummy

    try:
        proc.run_busy_loop()
    except SystemExit:
        pass
    wave_events = []
    while not proc.output_queue.empty():
        client_index, eco = proc.output_queue.get_nowait()
        if eco.wave_complete is not None:
            wave_events.append((client_index, eco.wave_complete))
    stateless_destroy_torch_distributed_process_group(proc.dp_group)
    return {
        "max_iters": payload.get("max_iters", 40),
        "dummies": len(dummies),
        "first_dummy_at_step": steps_at_dummy[0] if steps_at_dummy else None,
        "last_dummy_at_step": steps_at_dummy[-1] if steps_at_dummy else None,
        "wave_events": wave_events,
        "final_wave": proc.current_wave,
        "dp_rank": rank,
    }
