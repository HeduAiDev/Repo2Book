# ch34 测试基础设施：常驻 gloo 进程池（spawn），跨多个测试复用同一批子进程，
# 避免每个测试重复付 spawn + torch import 的启动成本。
#
# 池协议：worker 经 Pipe 等待 (job, payload) 指令，派发到 JOBS 表（按
# world_size 分组注册），把返回值经 Pipe 回传。纯测试脚手架——不携带任何
# 目标代码仓没有的行为；对被测面只做 monkeypatch 记录 / 测试替身注入，
# 均在 job 内逐处标注。

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

import numpy as np  # 类体（_InputBatch/_Discard 替身）读模块全局——函数局部对类体不可见

_IMPL = Path(__file__).resolve().parents[1] / "implementation"
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))

_TMP_DIR = Path(__file__).resolve().parent / ".tmp"


def _make_config(sizes, is_moe: bool):
    """测试侧 VllmConfig 工厂（ch03 装配线替身）：只填本章建组要读的字段。"""
    from vllm.config import VllmConfig

    tp, pp, dp, pcp = sizes
    vllm_config = VllmConfig()
    pc = vllm_config.parallel_config
    pc.tensor_parallel_size = tp
    pc.pipeline_parallel_size = pp
    pc.data_parallel_size = dp
    pc.prefill_context_parallel_size = pcp
    pc.decode_context_parallel_size = 1
    pc.world_size = pp * tp * pcp
    vllm_config.model_config.is_moe = is_moe
    return vllm_config


def _worker_main(rank: int, world: int, conn):
    try:
        while True:
            msg = conn.recv()
            if msg[0] == "STOP":
                break
            job, payload = msg
            fn = JOBS[world][job]
            result = fn(rank, world, payload)
            conn.send(("OK", result))
    except Exception as e:  # noqa: BLE001
        conn.send(("ERR", f"{e}\n{traceback.format_exc()}"))
    finally:
        import torch.distributed as dist

        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
            )

            destroy_model_parallel()
        except Exception:
            pass
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        conn.close()


class GlooPool:
    """N 个 spawn 子进程；map(job, payload) 把同一 job 广播给全部 rank、收集
    各 rank 返回值（集合通信测试需要全员同时进组）。"""

    def __init__(self, world: int):
        _TMP_DIR.mkdir(exist_ok=True)
        self.world = world
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        for rank in range(world):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(rank, world, child))
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def map(self, job: str, payload=None):
        for c in self.conns:
            c.send((job, payload))
        results = []
        for c in self.conns:
            tag, val = c.recv()
            if tag == "ERR":
                raise RuntimeError(f"worker error:\n{val}")
            results.append(val)
        return results

    def close(self):
        for c in self.conns:
            try:
                c.send(("STOP", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(30)
            if p.exitcode not in (0, None):
                raise RuntimeError(f"worker exited with {p.exitcode}")
        for p in self.procs:
            if p.is_alive():
                p.kill()
        for c in self.conns:
            c.close()


JOBS: dict[int, dict[str, object]] = {1: {}, 2: {}, 4: {}, 8: {}}


def _job(world: int, name: str):
    def deco(fn):
        JOBS[world][name] = fn
        return fn

    return deco


def _destroy_model_parallel_quietly():
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
    except Exception:
        pass


def _init_model_parallel(rank, world, sizes, is_moe, store):
    """每个 job 的标准前奏：gloo world + set_current_vllm_config + 建组。

    dp>1 时走真实『worker 入全局 world』路径：per-engine world_size=tp*pp*pcp，
    init_distributed_environment 内部按 dp_rank 偏移 rank、把 world 扩成
    across_dp（station 5——测试断言 get_rank()==dp_rank 即在验它）。
    每个 job 收尾 destroy_model_parallel，同一池可串多种切法。"""
    import torch.distributed as dist

    import vllm.distributed.parallel_state as ps
    from vllm.config import set_current_vllm_config

    tp, pp, dp, pcp = sizes
    cfg = _make_config(sizes, is_moe)
    with set_current_vllm_config(cfg):
        if dp > 1:
            # 引擎内坐标 vs 引擎序号：worker 全局 rank = dp_rank*engine_size +
            # engine_rank（站 5 公式）。engine_size>1 ∧ dp>1 时两者必须拆开——
            # data_parallel_rank 只在 0..dp-1，init 的 rank 是引擎内 0..engine-1。
            engine_size = tp * pp * pcp
            cfg.parallel_config.data_parallel_rank = rank // engine_size
            cfg.parallel_config.data_parallel_master_ip = "127.0.0.1"
            cfg.parallel_config.data_parallel_master_port = store["port"]
            cfg.parallel_config._data_parallel_master_port_list = [store["port"]]
            call_world, call_rank = engine_size, rank % engine_size
            if dist.is_initialized():
                # 同池上一个 DP 作业只销了 model-parallel 组——world PG 残留会
                # 让新一轮 rank 偏移 rendezvous 错位，这里全量拆干净再建。
                _destroy_model_parallel_quietly()
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                )

                destroy_distributed_environment()  # 含默认 PG 的销毁
            assert not dist.is_initialized()
        else:
            if not dist.is_initialized():
                dist.init_process_group(
                    "gloo",
                    init_method=f"file://{store['path']}",
                    rank=rank,
                    world_size=world,
                )
            call_world, call_rank = world, rank
        ps.init_distributed_environment(
            world_size=call_world,
            rank=call_rank,
            distributed_init_method="env://",
            local_rank=rank,
            backend="gloo",
        )
        ps.initialize_model_parallel(tp, pp, pcp, 1, backend="gloo")
    return cfg


# ═════════════════ world=8：5 维 rank 张量四刀（m2/m3 worked example） ═════


@_job(8, "groups")
def _job_groups(rank, world, payload):
    """monkeypatch init_model_parallel_group 记录四刀的 group_ranks（真 tensor
    数学 + 真调用位；『为每维建几十个 new_group』的代价旁路——world 组的
    new_group 仍是真实路径）。is_moe 由 payload 控制。"""
    import torch.distributed as dist

    import vllm.distributed.parallel_state as ps

    class _FakeGroup:
        def __init__(self, group_ranks):
            self.group_ranks = group_ranks
            self.world_size = len(group_ranks[0]) if group_ranks else 1
            self.rank_in_group = 0

        def destroy(self):
            pass

    recorded: dict[str, object] = {}

    def fake_factory(group_ranks, local_rank, backend, **kwargs):
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
        _init_model_parallel(
            rank, world, (2, 2, 2, 1), payload["is_moe"], payload["store"]
        )
        global_rank_after_offset = None
    finally:
        ps.init_model_parallel_group = real_factory
        _destroy_model_parallel_quietly()
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
        )

        destroy_distributed_environment()
        if dist.is_initialized():
            dist.destroy_process_group()
    return recorded


# ═══════════════════ world=2：TP=2 全链（m1/m5/m6/m7） ═══════════════════


@_job(2, "tp_all_reduce")
def _job_tp_all_reduce(rank, world, payload):
    import torch
    import torch.distributed as dist

    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group

    try:
        _init_model_parallel(rank, world, (2, 1, 1, 1), False, payload["store"])
        tp = get_tp_group()
        partial = torch.full((2, 3), float(rank + 1))
        out = tensor_model_parallel_all_reduce(partial)
        return {
            "out": out.tolist(),
            "ws": tp.world_size,
            "rank_in_group": tp.rank_in_group,
            "unique_name": tp.unique_name,
            "cpu_backend": str(dist.get_backend(tp.cpu_group)),
            "dev_backend": str(dist.get_backend(tp.device_group)),
            "has_dc": tp.device_communicator is not None,
            "dc_type": type(tp.device_communicator).__name__,
            "pynccl_disabled": bool(
                tp.device_communicator.pynccl_comm is None
                or tp.device_communicator.pynccl_comm.disabled
            ),
            "has_mq": tp.mq_broadcaster is not None,
        }
    finally:
        _destroy_model_parallel_quietly()


@_job(2, "tp_custom_op_route")
def _job_tp_custom_op(rank, world, payload):
    """use_custom_op_call=True → torch.ops.vllm.all_reduce(group_name=...) 路径。"""
    import torch

    from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import get_tp_group

    try:
        _init_model_parallel(rank, world, (2, 1, 1, 1), False, payload["store"])
        tp = get_tp_group()
        assert tp.use_custom_op_call is False  # CPU 平台 seam：直接路径为默认
        tp.use_custom_op_call = True
        partial = torch.full((2,), float(rank + 1))
        out = tensor_model_parallel_all_reduce(partial)
        return {"out": out.tolist()}
    finally:
        _destroy_model_parallel_quietly()


@_job(2, "tp_barrier_object")
def _job_tp_barrier_object(rank, world, payload):
    """barrier 刻意走 cpu_group；broadcast_object 对象级广播（mq_broadcaster 为
    None 时走 broadcast_object_list on cpu_group）。"""
    from vllm.distributed.parallel_state import get_tp_group

    try:
        _init_model_parallel(rank, world, (2, 1, 1, 1), False, payload["store"])
        tp = get_tp_group()
        tp.barrier()  # 不挂死即通过（全员到齐）
        obj = {"k": rank, "list": [1, 2]} if rank == 0 else None
        got = tp.broadcast_object(obj, src=0)
        return {"obj": got}
    finally:
        _destroy_model_parallel_quietly()


@_job(2, "rowparallel")
def _job_rowparallel(rank, world, payload):
    """RowParallelLinear TP=2：列切权重部分和 → all_reduce == 全量 Linear+bias
    （bias 只在 rank0 融进 GEMM → AR 后恰好加一次）。A 按输入特征维切块：
    本 rank weight = full.weight[:, rank*(in/tp):...]，输入同切片。"""
    import torch

    from vllm.model_executor.layers.linear import RowParallelLinear

    try:
        _init_model_parallel(rank, world, (2, 1, 1, 1), False, payload["store"])
        torch.manual_seed(0)
        in_f, out_f = 8, 6
        full = torch.nn.Linear(in_f, out_f, bias=True, dtype=torch.float64)
        x = torch.randn(4, in_f, dtype=torch.float64)
        layer = RowParallelLinear(
            input_size=in_f,
            output_size=out_f,
            bias=True,
            params_dtype=torch.float64,
            reduce_results=True,
            input_is_parallel=True,
        )
        with torch.no_grad():
            per_rank_in = in_f // layer.tp_size
            layer.weight.copy_(full.weight[:, rank * per_rank_in : (rank + 1) * per_rank_in])
            if rank == 0:
                layer.bias.copy_(full.bias)
        out, out_bias = layer(x[:, rank * per_rank_in : (rank + 1) * per_rank_in])
        assert out_bias is None  # skip_bias_add=False → bias 已融进 GEMM
        return {
            "out": out.tolist(),
            "expected": full(x).tolist(),
            "w_shape": list(layer.weight.shape),
        }
    finally:
        _destroy_model_parallel_quietly()


# ═══════════════════ world=2：PP=2 段间 P2P（m8/m12） ═══════════════════


@_job(2, "pp_tensor_dict")
def _job_pp_tensor_dict(rank, world, payload):
    """isend/irecv_tensor_dict：metadata 走 cpu_group、张量走 device_group、
    首段发次段收（dst=next / src=prev 的环语义）。"""
    import torch

    from vllm.distributed.parallel_state import get_pp_group

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
            return {"role": "sender"}
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
            }
    finally:
        _destroy_model_parallel_quietly()


@_job(2, "pp_token_broadcast")
def _job_pp_token_broadcast(rank, world, payload):
    """末段 broadcast 回首段（m12 async 版，gloo 上同样可跑）；chunked prefill
    分支跳过广播。"""
    import torch

    from vllm.distributed.parallel_state import get_pp_group
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    try:
        _init_model_parallel(rank, world, (1, 2, 1, 1), False, payload["store"])
        pp = get_pp_group()

        class _InputBatch:
            num_reqs = 3
            req_ids = ["r0", "r1", "r2"]
            num_tokens_no_spec = [0, 0, 0]
            prev_sampled_token_ids = None
            is_token_ids = np.zeros((3, 8), dtype=bool)

        class _Req:
            def __init__(self):
                self.output_token_ids = []

        class _Discard:
            np = np.zeros(3, dtype=bool)

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.input_batch = _InputBatch()
        runner.device = torch.device("cpu")
        runner.discard_request_mask = _Discard()
        runner.requests = {rid: _Req() for rid in _InputBatch.req_ids}
        runner._is_all_reqs_chunked_prefill = lambda: payload["chunked"]
        if pp.is_last_rank:
            sampled = torch.tensor([[11], [22], [33]], dtype=torch.int32)
            runner._pp_broadcast_prev_sampled_token_ids(sampled)
            return {"role": "last", "sent": sampled.tolist()}
        runner._pp_receive_prev_sampled_token_ids_to_input_batch()
        ib = runner.input_batch
        return {
            "role": "first",
            "recv": None
            if ib.prev_sampled_token_ids is None
            else ib.prev_sampled_token_ids.tolist(),
            "placeholder_appended": [
                r.output_token_ids for r in runner.requests.values()
            ],
        }
    finally:
        _destroy_model_parallel_quietly()


# ═══════════════════ world=4：TP=2 × PP=2 切片优化（m11） ═══════════════════


@_job(4, "pp_tp_slice")
def _job_pp_tp_slice(rank, world, payload):
    """m11：numel 整除 TP 的张量只发本 rank 1/tp 切片，接收端 all_gather 重建。
    rank0/1 = PP 首段（TP0/TP1），rank2/3 = PP 次段。"""
    import torch

    from vllm.distributed.parallel_state import get_pp_group, get_tp_group

    try:
        _init_model_parallel(rank, world, (2, 2, 1, 1), False, payload["store"])
        pp = get_pp_group()
        tp = get_tp_group()
        if pp.is_first_rank:
            d = {"hidden": torch.arange(16, dtype=torch.float32).reshape(4, 4)}
            handles = pp.isend_tensor_dict(
                d, all_gather_group=tp, all_gather_tensors={}
            )
            for h in handles:
                h.wait()
            return {"role": "pp0", "tp": tp.rank_in_group}
        else:
            tensor_dict, handles, post = pp.irecv_tensor_dict(
                all_gather_group=tp, all_gather_tensors={}
            )
            for h in handles:
                h.wait()
            for fn in post:
                fn()
            return {
                "role": "pp1",
                "tp": tp.rank_in_group,
                "hidden": tensor_dict["hidden"].tolist(),
            }
    finally:
        _destroy_model_parallel_quietly()


# ═══════════════ world=2：DP=2（MoE）批对齐 + EP（m21/m22） ═══════════════


@_job(2, "dp_align")
def _job_dp_align(rank, world, payload):
    """m21：4×dp all-reduce 互见 (orig,padded,ubatch,cg)；cg 取 min、pad 到 max。"""
    from vllm.v1.worker.dp_utils import coordinate_batch_across_dp

    try:
        cfg = _init_model_parallel(
            rank, world, (1, 1, 2, 1), True, payload["store"]
        )
        mode = payload["mode"]
        if mode == "pad":
            tokens, cg = 100 + 20 * rank, 1  # rank0:100 / rank1:120，双 PIECEWISE
        elif mode == "min_cg":
            tokens, cg = 100, rank  # rank0 NONE(0) / rank1 PIECEWISE(1)
        ubatch, padded, cg_out = coordinate_batch_across_dp(
            num_tokens_unpadded=tokens,
            allow_microbatching=False,
            parallel_config=cfg.parallel_config,
            cudagraph_mode=cg,
        )
        return {
            "ubatch": ubatch,
            "padded": None if padded is None else padded.tolist(),
            "cg_out": int(cg_out),
            "global_rank_offset": __import__("torch").distributed.get_rank(),
        }
    finally:
        _destroy_model_parallel_quietly()


@_job(2, "ep_dispatch_combine")
def _job_ep_dispatch_combine(rank, world, payload):
    """m22 worked example：2 rank 各 3 token；dispatch 全网可见、本地只算命中
    本地专家的 token、combine 按原 sizes 归位求和（0.5+0.5 权重 → 原值）。"""
    import torch

    from vllm.distributed.parallel_state import get_ep_group
    from vllm.forward_context import (
        DPMetadata,
        ForwardContext,
        override_forward_context,
    )

    try:
        _init_model_parallel(rank, world, (1, 1, 2, 1), True, payload["store"])
        ep = get_ep_group()
        torch.manual_seed(7)
        n_local, hidden = 3, 4
        h = torch.randn(n_local, hidden)
        topk_ids = torch.tensor([[0, 1], [2, 0], [1, 2]], dtype=torch.int32)
        topk_w = torch.full((n_local, 2), 0.5)

        sizes = [n_local, n_local]
        ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
        ctx.dp_metadata = DPMetadata(torch.tensor(sizes, dtype=torch.int32))
        with override_forward_context(ctx):
            # 真实消费位（moe_runner.py:L628-L640）：MoE 层进 dispatch 前经
            # sp_local_sizes(sp_size) 声明各 rank chunk 尺寸——纯 DP 即 sp=1
            with ctx.dp_metadata.sp_local_sizes(1):
                gh, gw, gids = ep.dispatch(h, topk_w, topk_ids)
        local_mask = (gids % 2) == rank
        n_hits = int(local_mask.any(dim=1).sum())
        contrib = torch.zeros_like(gh)
        for t in range(gh.shape[0]):
            for k in range(gids.shape[1]):
                if int(gids[t, k]) % 2 == rank:
                    contrib[t] += 0.5 * gh[t]
        with override_forward_context(ctx):
            with ctx.dp_metadata.sp_local_sizes(1):
                back = ep.combine(contrib)
        return {
            "gathered_rows": int(gh.shape[0]),
            "n_hits": n_hits,
            "back": back.tolist(),
            "h": h.tolist(),
            "ep_name": ep.unique_name,
            "dc_all2all": type(ep.device_communicator.all2all_manager).__name__,
            "gathered_ids": gids.tolist(),
        }
    finally:
        _destroy_model_parallel_quietly()


@_job(2, "ep_moe_prepare_finalize")
def _job_ep_moe(rank, world, payload):
    """naive_dp_ep 消费现场（m22 尾）：prepare → dispatch；finalize → 真实
    TopKWeightAndReduceContiguous（Σ_k w_k·expert_out）→ combine 归位。"""
    import torch

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

    try:
        _init_model_parallel(rank, world, (1, 1, 2, 1), True, payload["store"])
        n_local, hidden = 3, 4
        torch.manual_seed(11)
        a1 = torch.randn(n_local, hidden)
        topk_ids = torch.tensor([[0, 1], [2, 0], [1, 2]], dtype=torch.int32)
        topk_w = torch.full((n_local, 2), 0.5)
        pf = MoEPrepareAndFinalizeNaiveDPEPModular(is_sequence_parallel=False)

        sizes = [n_local, n_local]
        ctx = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
        ctx.dp_metadata = DPMetadata(torch.tensor(sizes, dtype=torch.int32))

        with override_forward_context(ctx):
            # 真实消费位同上（moe_runner.py:L628-L640）：sp_local_sizes(1) 声明
            # 纯 DP 的各 rank chunk 尺寸
            with ctx.dp_metadata.sp_local_sizes(1):
                a1q, scale, _, gids, gw = pf.prepare(
                    a1,
                    topk_w,
                    topk_ids,
                    num_experts=4,
                    expert_map=None,
                    apply_router_weight_on_input=False,
                    quant_config=None,
                )
            # fused_expert_output = 本地专家核产出 [m, topk, K]：命中本地专家的
            # (token,k) 槽位产出该行 hidden（恒等『专家』），远端槽位为零——
            # combine 跨 rank 求和后凑齐 Σ_k w_k·y_k。
            n_tok, n_topk = gids.shape
            fused = torch.zeros(n_tok, n_topk, hidden)
            for t in range(n_tok):
                for k in range(n_topk):
                    if int(gids[t, k]) % 2 == rank:
                        fused[t, k] = a1q[t]
            # combine 归位到 owner 视角：reduce_scatterv 按 sizes[rank] 切回
            # 本 rank 的 3 行（output 是 MoE 层本地输出，不是全网 6 行）
            out = torch.zeros(n_local, hidden)
            with ctx.dp_metadata.sp_local_sizes(1):
                pf.finalize(
                    out,
                    fused,
                    gw,
                    gids,
                    apply_router_weight_on_input=False,
                    weight_and_reduce_impl=TopKWeightAndReduceContiguous(),
                )
        return {"back": out.tolist(), "rows": int(out.shape[0])}
    finally:
        _destroy_model_parallel_quietly()


# ═══════════════ world=2：wave 共识（stateless dp_group，m13/m19） ═══════════════


@_job(2, "stateless_sync")
def _job_stateless_sync(rank, world, payload):
    """引擎级 gloo dp_group（stateless_init_dp_group）+ sync_dp_state 双共识 +
    has_unfinished_dp（MAX≡OR）。"""
    from vllm.config import ParallelConfig
    from vllm.distributed.utils import (
        stateless_destroy_torch_distributed_process_group,
    )

    cfg = _make_config((1, 1, 2, 1), True)
    pc = cfg.parallel_config
    pc.data_parallel_rank = rank
    pc.data_parallel_master_ip = "127.0.0.1"
    pc.data_parallel_master_port = payload["port"]
    group = pc.stateless_init_dp_group()
    has_global, pause_consensus = ParallelConfig.sync_dp_state(
        group,
        has_unfinished=payload["has_unfinished"][rank],
        pending_pause=payload["pending_pause"][rank],
    )
    or_result = ParallelConfig.has_unfinished_dp(
        group, has_unfinished=payload["has_unfinished"][rank]
    )
    stateless_destroy_torch_distributed_process_group(group)
    return {
        "has_global": has_global,
        "pause_consensus": pause_consensus,
        "or_result": or_result,
        "group_size": group.size(),
    }


@_job(2, "wave_consensus")
def _job_wave_consensus(rank, world, payload):
    """DPEngineCoreProc 忙循环双打：全员无活 → dummy batch 维持锁步 → 32 步共识
    → dp_rank0 发 wave_complete(client_index=-1) → current_wave+=1。"""
    import queue

    from vllm.distributed.utils import (
        stateless_destroy_torch_distributed_process_group,
    )
    from vllm.v1.engine.core import DPEngineCoreProc

    cfg = _make_config((1, 1, 2, 1), True)
    pc = cfg.parallel_config
    pc.data_parallel_rank = rank
    pc.data_parallel_master_ip = "127.0.0.1"
    pc.data_parallel_master_port = payload["port"]
    proc = DPEngineCoreProc.__new__(DPEngineCoreProc)
    proc.log_stats = False  # super().__init__ 会设的观测开关（__new__ 旁路需自备）
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
    proc.shutdown_state = 0  # EngineShutdownState.RUNNING
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

    def _drain_budget():
        proc.iter_budget[0] -= 1
        if proc.iter_budget[0] <= 0:
            proc.shutdown_state = 1  # REQUESTED → 退出循环

    proc._process_input_queue = _drain_budget
    proc._process_engine_step = lambda: False
    proc._handle_shutdown = lambda: proc.shutdown_state == 0
    proc.execute_dummy_batch = lambda: dummies.append(1)

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
        "dummies": len(dummies),
        "wave_events": wave_events,
        "final_wave": proc.current_wave,
    }


# ═══════════════ world=1：DPCoordinatorProc 真三 socket 循环（m18/m20） ═══════════════


@_job(1, "coordinator_e2e")
def _job_coordinator_e2e(rank, world, payload):
    """真 XPUB/XSUB/PULL 三 socket + 真 100ms 发布节拍 + wave 状态机 + START_DP_WAVE
    广播（m18/m19/m20/m22 的控制面全链）。全部交互发生在 worker 进程内。"""
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
        _reserve_addr(),
        _reserve_addr(),
        _reserve_addr(),
    )
    coord = DPCoordinatorProc(
        engine_count=2, enable_wave_coordination=True
    )
    t = threading.Thread(
        target=coord.process_input_socket,
        args=(front_addr, back_out_addr, back_pub_addr),
        daemon=True,
    )
    t.start()

    ctx = zmq.Context()
    events: list = []

    # 引擎侧：两个 XSUB 订阅控制通道（engine_count=2 → 循环等齐两份订阅）
    eng = ctx.socket(zmq.XSUB)
    # 订阅即发送 b"" 帧；setsockopt(SUBSCRIBE) 在 XSUB 上无效（win32 EINVAL）
    eng.connect(back_pub_addr)
    eng.send(b"\x01")  # 订阅帧
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

    # 前端侧：XSUB 订阅聚合统计
    front = ctx.socket(zmq.XSUB)
    front.connect(front_addr)
    front.send(b"\x01")

    # 引擎 0 上报 stats（PULL → coordinator）
    push = ctx.socket(zmq.PUSH)
    push.connect(back_out_addr)
    stats = SchedulerStats(
        num_waiting_reqs=2,
        num_running_reqs=1,
        kv_cache_usage=0.6,
        step_counter=1,
        current_wave=0,
    )
    push.send(msgspec.msgpack.encode(EngineCoreOutputs(scheduler_stats=stats)))

    # 等一次前端发布（100ms 变化刷）
    got_stats = None
    deadline = time.time() + 10
    while time.time() < deadline:
        if front.poll(200):
            counts, wave, running = msgspec.msgpack.decode(front.recv())
            if counts is not None:
                got_stats = (counts, wave, running)
                break
    assert got_stats is not None, "no stats published to front-end"
    events.append(("stats", got_stats))

    # FIRST_REQ：engines 暂停（running=False）→ 前端经 XSUB 发通知 →
    # coordinator 广播 START_DP_WAVE(exclude=已收请求引擎)
    front.send(msgspec.msgpack.encode((1, got_stats[1])))
    got_wave = None
    deadline = time.time() + 10
    while time.time() < deadline:
        if eng.poll(200):
            frame = eng.recv_multipart()
            if frame[0] == EngineCoreRequestType.START_DP_WAVE.value:
                got_wave = msgspec.msgpack.decode(frame[1])
                break
    assert got_wave is not None, "no START_DP_WAVE broadcast"
    events.append(("start_wave", got_wave))

    # wave_complete（dp rank 0 引擎上报）→ current_wave+1、engines_running=False
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
    assert got_state is not None, "no wave-complete state published"
    events.append(("wave_complete", got_state))

    # stale wave 请求：引擎上报 start_wave=3 → coordinator 补广播（exclude=引擎 0）
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
    return {"events": events}
