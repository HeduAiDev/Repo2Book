"""ch18 持久批次与固定地址 —— 单元+契约测试（不 import vllm）。

测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观测行为**
（锚点 = vllm/... 行号，基线 v0.27.1 现核，非 v2 资产的 v0.21.0 旧行号）。
纯 host 单元/契约测试：不 import vllm、不触 CUDA（worker 侧 CUDA 面以
HOST SEAM 承载——HostEvent/HostCopyStream/前向脚本化 logits，见 impl-notes）。

行为清单（按 dossier.mechanisms 对账）：
- m01 差量协议：scheduled_new_reqs 全量建 CachedRequestState 快照、
  scheduled_cached_reqs 只发 diff、finished 通知清缓存
  （sched/output.py:L193-L205 / gpu_model_runner.py:L1202-L1217）
- m02 _update_states 差量调和：finished 出缓存+出批次；unscheduled 出批次
  留缓存（gpu_model_runner.py:L1202-L1253）
- m03 slot 复用与压实：remove 打洞 → pop_removed 复用最小空 slot →
  condense 尾部滑入（gpu_input_batch.py:L324-L348 / L530-L548 / L708-L838）
- m04 InputBatch 内存布局：token_ids_cpu R×L 行式 + 列式 CPU 镜像
  （gpu_input_batch.py:L127-L172 / L350-L398）
- m05 CpuGpuBuffer 固定双端缓冲：cpu(pinned)+gpu+np 三视图、
  copy_to_gpu(n) 只传活跃前缀（v1/utils.py:L110-L149）；runner 持久缓冲
  一次分配（gpu_model_runner.py:L763-L810）
- m06 token 扁平收集：np.repeat 展开 + cumsum/arange + token_indices =
  pos + req_index·M + torch.index_select（gpu_model_runner.py:L1743-L1767 /
  L1977-L2024）
- m07 query_start_loc CU 偏移 + 尾部 pad 非递减（L2073-L2078）
- m08 block_table 双镜像：append_row CPU 增量 / commit_block_table 活跃行
  （block_table.py:L138-L155 / L213-L214）
- m09 写回闭环：采样 token 写 token_ids_cpu 行 + output_token_ids 增长；
  异步时留 GPU prev_sampled_token_ids + CPU 行写 -1 占位
  （gpu_model_runner.py:L3815-L3846 / L3797-L3813）
- m10 可变性裁决：execute_model 入口 replace() 浅拷贝 + 就地裁剪
  （gpu_model_runner.py:L4180-L4195 / ngram_proposer_gpu.py:L475-L515）
- m11 resumed 语义分叉：new_block_ids append vs 整体替换
  （sched/output.py:L118-L121 / gpu_model_runner.py:L1441-L1452）
- m12 批次重排钩子 + swap_states 只交换活跃前缀（L1115-L1138 /
  gpu_input_batch.py:L586-L653）
- m13 pinned buffer 防踩：synchronize_input_prep 等上拍事件（L3864-L3877）
- m14 固定地址：跨拍 data_ptr 不变（ch19 CUDA graph 回放的前提）
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from implementation._host_seams import (  # noqa: E402
    HostEvent,
    SamplingParams,
    SamplingType,
)
from implementation.block_table import (  # noqa: E402
    PAD_SLOT_ID,
    BlockTable,
    MultiGroupBlockTable,
    SlotMappingMode,
)
from implementation.gpu_input_batch import (  # noqa: E402
    CachedRequestState,
    InputBatch,
)
from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.logits_processor.state import BatchUpdateBuilder  # noqa: E402
from implementation.ngram_proposer_gpu import (  # noqa: E402
    update_scheduler_for_invalid_drafts,
)
from implementation.output import (  # noqa: E402
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from implementation.utils import CpuGpuBuffer  # noqa: E402

VOCAB = 32


# ---------------------------------------------------------------- fixtures
def _greedy():
    return SamplingParams(temperature=0.0)


def _vllm_config(
    max_num_reqs=8,
    max_model_len=64,
    max_num_batched_tokens=64,
    async_scheduling=False,
):
    model_config = SimpleNamespace(
        max_model_len=max_model_len,
        runner_type="generate",
        enable_prompt_embeds=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        is_encoder_decoder=False,
        dtype=torch.float32,
        logits_processors=None,
        get_vocab_size=lambda: VOCAB,
    )
    cache_config = SimpleNamespace(
        block_size=16,
        calculate_kv_scales=False,
        use_replayssm=False,
        mamba_cache_mode="align",
        kv_sharing_fast_prefill=False,
    )
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_reqs,
        async_scheduling=async_scheduling,
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        speculative_config=None,
        compilation_config=None,
        lora_config=None,
        load_config=None,
        offload_config=None,
        observability_config=None,
        reasoning_config=None,
    )


def _runner(async_scheduling=False, **kw):
    cfg = _vllm_config(async_scheduling=async_scheduling, **kw)
    runner = GPUModelRunner(cfg, torch.device("cpu"))
    # 单一 full-attention KV group（initialize_kv_cache 的 ch14 域装配；
    # _may_reorder_batch L1130 读 len(kv_cache_config.kv_cache_groups)）
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    return runner


def _new_req(req_id, prompt, block_ids, num_computed=0, sampling=None):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=sampling if sampling is not None else _greedy(),
        pooling_params=None,
        block_ids=(list(block_ids),),
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def _cached_data(req_ids=(), resumed=(), new_blocks=(), computed=(),
                 outputs=(), all_token_ids=None, new_tokens=()):
    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(resumed),
        new_token_ids=[list(t) for t in new_tokens],
        all_token_ids=all_token_ids or {},
        new_block_ids=[(list(b),) if b is not None else None for b in new_blocks],
        num_computed_tokens=list(computed),
        num_output_tokens=list(outputs),
    )


def _sched_output(new_reqs=(), cached=None, num_scheduled=None, total=None,
                  finished=(), spec_tokens=None):
    num_scheduled = num_scheduled or {}
    if total is None:
        total = sum(num_scheduled.values())
    return SchedulerOutput(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached or _cached_data(),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=total,
        scheduled_spec_decode_tokens=spec_tokens or {},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )


def _input_batch(max_num_reqs=8, max_model_len=64):
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=64,
        device=torch.device("cpu"),
        vocab_size=VOCAB,
        block_sizes=[16],
        kernel_block_sizes=[16],
        max_num_blocks_per_req=[4],
    )


def _req_state(req_id, prompt, block_ids, output=(), num_computed=0):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=_greedy(),
        generator=None,
        block_ids=(list(block_ids),),
        num_computed_tokens=num_computed,
        output_token_ids=list(output),
    )


def _logits_row(pick):
    """one-hot 行：argmax == pick（贪心采样的可预测脚本）。"""
    row = [0.0] * VOCAB
    row[pick % VOCAB] = 1.0
    return row


# ================================================================ m03/m04
# BatchUpdateBuilder（vllm/v1/sample/logits_processor/state.py:L18-L145）


def test_batch_update_builder_removed_descending_and_pop_lowest():
    b = BatchUpdateBuilder()
    b.removed_append(3)
    b.removed_append(1)
    b.removed_append(5)
    # docstring 保证：self.removed 恒降序
    assert b.removed == [5, 3, 1]
    # peek/pop 返回**最小**空 slot
    assert b.peek_removed() == 1
    assert b.pop_removed() == 1
    assert b.pop_removed() == 3
    assert b.pop_removed() == 5
    assert b.pop_removed() is None
    assert b.has_removed() is False


def test_batch_update_builder_append_after_read_raises():
    b = BatchUpdateBuilder()
    b.removed_append(2)
    _ = b.removed  # 首次读取后封账
    with pytest.raises(RuntimeError):
        b.removed_append(0)


def test_batch_update_builder_get_and_reset():
    b = BatchUpdateBuilder()
    b.removed_append(4)
    b.added.append((0, None, [1], []))
    b.moved.append((3, 0, None))
    upd = b.get_and_reset(3)
    assert upd is not None
    assert upd.batch_size == 3
    assert list(upd.removed) == [4]
    assert list(upd.added) == [(0, None, [1], [])]
    assert list(upd.moved) == [(3, 0, None)]
    # 空步返回 None（refresh_metadata 不重建采样元数据）
    assert b.get_and_reset(3) is None


# InputBatch 布局与增删（gpu_input_batch.py:L324-L348 / L350-L501 / L530-L584）


def test_add_request_writes_row_and_columns():
    ib = _input_batch()
    ib.add_request(_req_state("a", [5, 6, 7], [11], output=[9], num_computed=4))
    # 一行 = 一个请求的全长 token 缓冲：prompt 前缀 + output 紧随
    assert ib.token_ids_cpu[0, :4].tolist() == [5, 6, 7, 9]
    assert ib.is_token_ids[0, :4].tolist() == [True] * 4
    assert ib.num_prompt_tokens[0] == 3
    assert ib.num_tokens_no_spec[0] == 4
    assert ib.num_computed_tokens_cpu[0] == 4
    assert ib.req_id_to_index == {"a": 0}
    assert ib.block_table[0].num_blocks_per_row[0] == 1
    assert ib.block_table[0].block_table.np[0, 0] == 11
    # CachedRequestState.num_tokens = prompt + output
    assert ib.req_ids == ["a"]


def test_remove_punches_hole_without_moving_data():
    ib = _input_batch()
    for rid, prompt in [("a", [1, 2]), ("b", [3, 4]), ("c", [5, 6])]:
        ib.add_request(_req_state(rid, prompt, [1]))
    ib.remove_request("b")
    # 打洞：行标记 None、映射解绑、块表行清零——但数据不搬
    assert ib._req_ids[1] is None
    assert "b" not in ib.req_id_to_index
    assert ib.block_table[0].num_blocks_per_row[1] == 0
    assert ib.token_ids_cpu[1, :2].tolist() == [3, 4]  # 数据仍在
    assert ib.num_reqs == 2


def test_add_reuses_smallest_popped_slot():
    ib = _input_batch()
    for rid in ["a", "b", "c"]:
        ib.add_request(_req_state(rid, [1, 2], [1]))
    ib.remove_request("a")
    ib.remove_request("c")
    # 批末统一压实前的 add：pop_removed 复用最小空 slot（= 0）
    ib.add_request(_req_state("d", [7, 8], [2]))
    assert ib.req_id_to_index["d"] == 0
    # 复用后 removed 已被填平 → condense 零成本早退（docstring 行为）
    ib.condense()
    assert ib.req_ids == ["d", "b"]
    assert ib.token_ids_cpu[0, :2].tolist() == [7, 8]


def test_condense_slides_tail_into_holes_active_prefix_only():
    ib = _input_batch()
    # a=10 tokens @0, b=3 tokens @1, c/d 占 2/3
    ib.add_request(_req_state("a", list(range(10)), [1]))
    ib.add_request(_req_state("b", [20, 21, 22], [2]))
    ib.add_request(_req_state("c", [30], [3]))
    ib.add_request(_req_state("d", [40], [4]))
    ib.remove_request("b")  # hole @1
    ib.remove_request("c")  # hole @2
    ib.condense()
    # [0, num_reqs) 连续：a@0, d 滑入 1（removed 降序保证双指针正确）
    assert ib.req_ids == ["a", "d"]
    assert ib.req_id_to_index == {"a": 0, "d": 1}
    # 只拷活跃前缀：d 的 1 个 token 进 row1；row1 之外的陈旧 b 数据不动
    assert ib.token_ids_cpu[1, 0] == 40
    assert ib.token_ids_cpu[1, 1] == 21  # b 的陈旧尾巴仍可见（真实行为）
    assert ib.num_tokens_no_spec[1] == 1
    assert ib.block_table[0].num_blocks_per_row[1] == 1
    assert ib.block_table[0].block_table.np[1, 0] == 4


def test_condense_early_return_when_no_removals():
    ib = _input_batch()
    ib.add_request(_req_state("a", [1], [1]))
    ib.condense()  # 无 removal → 早退
    assert ib.req_ids == ["a"]


def test_condense_empty_batch_clears_lists():
    ib = _input_batch()
    ib.add_request(_req_state("a", [1], [1]))
    ib.remove_request("a")
    ib.condense()
    assert ib.req_ids == []
    assert ib.num_reqs == 0


def test_swap_states_swaps_active_prefix_window():
    ib = _input_batch()
    ib.add_request(_req_state("a", [1, 2, 3, 4], [1]))
    ib.add_request(_req_state("b", [9], [2]))
    ib.swap_states(0, 1)
    assert ib.req_ids == ["b", "a"]
    assert ib.req_id_to_index == {"b": 0, "a": 1}
    # 交换窗口 = max(活跃长度)：a 的 4 个 token 整窗搬去 row1
    assert ib.token_ids_cpu[1, :4].tolist() == [1, 2, 3, 4]
    assert ib.token_ids_cpu[0, :4].tolist() == [9, 0, 0, 0]
    assert ib.num_tokens_no_spec.tolist()[:2] == [1, 4]
    assert ib.block_table[0].block_table.np[0, 0] == 2
    assert ib.block_table[0].block_table.np[1, 0] == 1


def test_refresh_metadata_rebuilds_and_feeds_logitsprocs():
    from implementation.logits_processor.state import LogitsProcessors

    class _Spy:
        def __init__(self):
            self.updates = []
            self.argmax_invariant = True

        def is_argmax_invariant(self):
            return True

        def update_state(self, batch_update):
            self.updates.append(batch_update)

    spy = _Spy()
    ib = _input_batch()
    ib.logitsprocs = LogitsProcessors([spy])
    ib.add_request(_req_state("a", [1], [1]))
    ib.refresh_metadata()
    # 非 pooling 路径：get_and_reset 产出 BatchUpdate 喂给每个 processor
    assert len(spy.updates) == 1
    assert spy.updates[0].batch_size == 1
    assert list(spy.updates[0].added)[0][0] == 0  # (index, params, prompt, out)
    assert ib.sampling_metadata.all_greedy is True
    # 无变更步 → update_state(None)
    ib.refresh_metadata()
    assert spy.updates[-1] is None


def test_update_req_spec_token_ids_places_draft_tokens():
    ib = _input_batch()
    st = _req_state("a", [1, 2, 3], [1])
    ib.add_request(st)
    ib.update_req_spec_token_ids(st, {"a": [7, 8]})
    # spec 占位写行：从 num_tokens_no_spec 起写 draft token
    assert ib.token_ids_cpu[0, 3:5].tolist() == [7, 8]
    assert ib.spec_token_ids[0] == [7, 8]
    assert st.prev_num_draft_len == 2
    # 空步清空
    ib.update_req_spec_token_ids(st, {})
    assert ib.spec_token_ids[0] == []


# ================================================================ m05
# CpuGpuBuffer（vllm/v1/utils.py:L110-L149）


def test_cpu_gpu_buffer_prefix_copy_only():
    buf = CpuGpuBuffer(8, dtype=torch.int32, device=torch.device("cpu"))
    assert buf.np.shape == (8,)
    buf.np[:8] = np.arange(1, 9, dtype=np.int32)
    buf.copy_to_gpu(5)
    assert buf.gpu[:5].tolist() == [1, 2, 3, 4, 5]
    assert buf.gpu[5:].tolist() == [0, 0, 0]  # 尾部不拷（活跃前缀语义）
    buf.np[0] = 99
    buf.copy_to_gpu()
    assert buf.gpu[0].item() == 99  # n=None 全量


def test_cpu_gpu_buffer_copy_to_cpu_roundtrip():
    buf = CpuGpuBuffer(4, dtype=torch.int64, device=torch.device("cpu"))
    buf.gpu[:] = torch.arange(10, 14)
    buf.copy_to_cpu(2)
    assert buf.cpu[:2].tolist() == [10, 11]
    assert buf.cpu[2:].tolist() == [0, 0]


def test_cpu_gpu_buffer_addresses_never_change():
    buf = CpuGpuBuffer(8, dtype=torch.int32, device=torch.device("cpu"))
    p_cpu, p_gpu = buf.cpu.data_ptr(), buf.gpu.data_ptr()
    for i in range(3):
        buf.np[i] = i
        buf.copy_to_gpu(4)
    assert buf.cpu.data_ptr() == p_cpu
    assert buf.gpu.data_ptr() == p_gpu


def test_cpu_gpu_buffer_bfloat16_rejects_numpy_view():
    with pytest.raises(ValueError):
        CpuGpuBuffer(4, dtype=torch.bfloat16, device=torch.device("cpu"))


# ================================================================ m08
# BlockTable 双镜像（block_table.py:L48-L267 / L270-L376）


def _block_table(max_reqs=4, kernel_block_size=16):
    return BlockTable(
        block_size=16,
        max_num_reqs=max_reqs,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=kernel_block_size,
        cp_kv_cache_interleave_size=1,
    )


def test_block_table_append_add_clear_move():
    bt = _block_table()
    bt.append_row([5, 6], 0)
    assert bt.num_blocks_per_row[0] == 2
    bt.append_row([7], 0)  # 差量追加：行内偏移由 num_blocks_per_row 记账
    assert bt.block_table.np[0, :3].tolist() == [5, 6, 7]
    bt.add_row([9], 0)  # 整行重置重写
    assert bt.block_table.np[0, 0] == 9
    assert bt.num_blocks_per_row[0] == 1
    bt.append_row([1, 2], 1)
    bt.move_row(1, 2)  # 压实搬移：搬活跃前缀 + 清源行
    assert bt.block_table.np[2, :2].tolist() == [1, 2]
    assert bt.num_blocks_per_row[2] == 2
    assert bt.num_blocks_per_row[1] == 0
    bt.clear_row(2)
    assert bt.num_blocks_per_row[2] == 0
    assert int(bt.block_table.np[2, 0]) == 0


def test_commit_block_table_copies_only_active_rows():
    bt = _block_table()
    bt.append_row([1, 2, 3], 0)
    bt.append_row([4, 5, 6], 1)
    bt.append_row([7, 8, 9], 2)
    bt.commit_block_table(2)
    assert bt.block_table.gpu[:2].tolist() == [[1, 2, 3, 0], [4, 5, 6, 0]]
    assert bt.block_table.gpu[2].tolist() == [0, 0, 0, 0]  # 非活跃行不拷


def test_multi_group_block_table_fans_out():
    mgb = MultiGroupBlockTable(
        max_num_reqs=4,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 16],
        kernel_block_sizes=[16, 16],
        max_num_blocks=[4, 4],
        cp_kv_cache_interleave_size=1,
        slot_mapping_modes=[SlotMappingMode.TOKEN_TO_KV_SLOT] * 2,
    )
    mgb.add_row(([1], [8]), 0)
    assert mgb[0].block_table.np[0, 0] == 1
    assert mgb[1].block_table.np[0, 0] == 8
    mgb.append_row(([2], [9]), 0)
    assert mgb[0].block_table.np[0, 1] == 2
    assert mgb[1].block_table.np[0, 1] == 9
    mgb.clear_row(0)
    assert mgb[0].num_blocks_per_row[0] == 0
    assert mgb[1].num_blocks_per_row[0] == 0


def test_compute_slot_mapping_identity_and_pad_tail():
    bt = _block_table()
    bt.append_row([5, 3], 0)  # 请求行：块 5（pos0-15）、块 3（pos16-31）
    qsl = torch.tensor([0, 3, 3], dtype=torch.int32)  # 1 请求 3 token + 空请求
    positions = torch.tensor([0, 14, 16], dtype=torch.int64)
    bt.compute_slot_mapping(1, qsl, positions)
    # 恒等式：slot = 块号 × block_size + 块内偏移（kernel 本体的 host 镜像）
    assert bt.slot_mapping.np[:3].tolist() == [5 * 16, 5 * 16 + 14, 3 * 16]
    # PAD 尾：[num_tokens, max_num_batched_tokens) 每拍重填 PAD_SLOT_ID
    assert (bt.slot_mapping.np[3:] == PAD_SLOT_ID).all()


# ================================================================ m01/m02/m11
# _update_states 差量调和（gpu_model_runner.py:L1192-L1566）


def test_finished_requests_leave_cache_and_batch():
    r = _runner()
    r._update_states(_sched_output(new_reqs=[_new_req("a", [1, 2], [10])]))
    r._update_states(_sched_output(new_reqs=[_new_req("b", [3], [11])]))
    so = _sched_output(
        cached=_cached_data(req_ids=["b"], new_blocks=[[12]], computed=[1],
                            outputs=[1]),
        num_scheduled={"b": 1},
        finished={"a"},
    )
    r._update_states(so)
    # finished：出 requests 缓存 + 出持久批次
    assert "a" not in r.requests
    assert r.input_batch.req_ids == ["b"]


def test_unscheduled_removed_from_batch_but_kept_in_cache():
    r = _runner()
    r._update_states(_sched_output(
        new_reqs=[_new_req("a", [1, 2], [10]), _new_req("b", [3], [11])],
        num_scheduled={"a": 2, "b": 1}, total=3))
    # 下一拍只调度 a：b 出批次、留缓存（未来还会回来）
    r._update_states(_sched_output(
        cached=_cached_data(req_ids=["a"], new_blocks=[[13]], computed=[2],
                            outputs=[0]),
        num_scheduled={"a": 1}))
    assert "b" in r.requests
    assert r.input_batch.req_ids == ["a"]
    assert r.requests["b"].prompt_token_ids == [3]


def test_new_request_full_snapshot_then_only_diff():
    r = _runner()
    r._update_states(_sched_output(new_reqs=[_new_req("a", [5, 6, 7], [10, 11])],
                                   num_scheduled={"a": 3}))
    # worker 缓存请求全量：此后该请求只收差量
    st = r.requests["a"]
    assert st.prompt_token_ids == [5, 6, 7]
    assert st.block_ids == ([10, 11],)
    assert st.output_token_ids == []
    assert r.input_batch.token_ids_cpu[0, :3].tolist() == [5, 6, 7]
    assert r.input_batch.block_table[0].block_table.np[0, :2].tolist() == [10, 11]
    # 差量拍：new_block_ids append、num_computed 覆盖
    r._update_states(_sched_output(
        cached=_cached_data(req_ids=["a"], new_blocks=[[12]], computed=[3],
                            outputs=[1]),
        num_scheduled={"a": 1}))
    assert r.requests["a"].block_ids == ([10, 11, 12],)
    assert r.input_batch.block_table[0].block_table.np[0, 2] == 12
    assert r.input_batch.num_computed_tokens_cpu[0] == 3


def test_resumed_request_replaces_block_ids_entirely():
    r = _runner()
    r._update_states(_sched_output(new_reqs=[_new_req("a", [5, 6], [10, 11])],
                                   num_scheduled={"a": 2}))
    # 抢占：出批次留缓存
    r._update_states(_sched_output(num_scheduled={}))
    assert r.input_batch.num_reqs == 0
    assert "a" in r.requests
    # 恢复：resumed 语义——new_block_ids 整体替换而非 append
    r._update_states(_sched_output(
        cached=_cached_data(req_ids=["a"], resumed=["a"], new_blocks=[[20]],
                            computed=[0], outputs=[0]),
        num_scheduled={"a": 2}))
    assert r.requests["a"].block_ids == ([20],)
    assert r.input_batch.block_table[0].block_table.np[0, 0] == 20
    assert r.input_batch.block_table[0].num_blocks_per_row[0] == 1


def test_resumed_in_batch_set_arithmetic_forced_preemption_case():
    # reset_prefix_cache 强制抢占：请求同时在 batch 与 resumed 集合——
    # 集合差 unscheduled = cached − (scheduled − resumed) 先清出批次再重加
    r = _runner()
    r._update_states(_sched_output(new_reqs=[_new_req("a", [5, 6], [10])],
                                   num_scheduled={"a": 2}))
    r._update_states(_sched_output(
        cached=_cached_data(req_ids=["a"], resumed=["a"], new_blocks=[[30]],
                            computed=[0], outputs=[0]),
        num_scheduled={"a": 2}))
    assert r.requests["a"].block_ids == ([30],)
    assert r.input_batch.req_ids == ["a"]


def test_async_resumed_recovers_output_tokens_from_all_token_ids():
    r = _runner(async_scheduling=True)
    r._update_states(_sched_output(new_reqs=[_new_req("a", [5, 6], [10])],
                                   num_scheduled={"a": 2}))
    r._update_states(_sched_output(
        cached=_cached_data(
            req_ids=["a"], resumed=["a"], new_blocks=[[20]], computed=[2],
            outputs=[2],
            all_token_ids={"a": [5, 6, 8, 9]}),
        num_scheduled={"a": 1}))
    # 异步调度：恢复请求的 output_token_ids 从 all_token_ids 尾部重建
    assert r.requests["a"].output_token_ids == [8, 9]
    assert r.input_batch.token_ids_cpu[0].tolist()[:4] == [5, 6, 8, 9]


def test_slot_reuse_prefers_smallest_hole_across_steps():
    r = _runner()
    r._update_states(_sched_output(new_reqs=[
        _new_req("a", [1], [10]), _new_req("b", [2], [11]),
        _new_req("c", [3], [12])], num_scheduled={"a": 1, "b": 1, "c": 1}))
    assert r.input_batch.req_ids == ["a", "b", "c"]
    # a 完成 + d 新进：d 复用 slot 0（最小空洞优先）
    r._update_states(_sched_output(
        new_reqs=[_new_req("d", [4], [13])],
        num_scheduled={"b": 1, "c": 1, "d": 1},
        finished={"a"}))
    assert r.input_batch.req_id_to_index == {"d": 0, "b": 1, "c": 2}
    assert r.input_batch.token_ids_cpu[0, 0] == 4


def test_streaming_same_id_reuses_cached_state():
    r = _runner()
    r._update_states(_sched_output(new_reqs=[_new_req("a", [5], [10])],
                                   num_scheduled={"a": 1}))
    # streaming：同 req_id 再入 scheduled_new_reqs → 复用缓存快照重铺行
    r._update_states(_sched_output(
        new_reqs=[_new_req("a", [5, 6, 7], [10, 11])],
        num_scheduled={"a": 3}))
    assert r.input_batch.req_ids == ["a"]
    assert r.input_batch.token_ids_cpu[0, :3].tolist() == [5, 6, 7]
    assert r.requests["a"].output_token_ids == []


def test_update_states_returns_none_without_async_spec():
    # deferred 纠偏闭包随 async spec decode 链删除：普通拍隐式返回 None
    r = _runner()
    r._update_states(_sched_output(new_reqs=[_new_req("a", [1], [10])],
                                   num_scheduled={"a": 1}))
    assert r._update_states(_sched_output(
        cached=_cached_data(req_ids=["a"], new_blocks=[[11]], computed=[1],
                            outputs=[1]),
        num_scheduled={"a": 1})) is None


# ================================================================ m06/m07
# _prepare_inputs 收集与装配（gpu_model_runner.py:L1960-L2282）


def _flattening_setup():
    """3 请求 [2,5,3] 调度窗——源码注释的基准场景（chunked prefill 第二拍）。"""
    r = _runner()
    r._update_states(_sched_output(new_reqs=[
        _new_req("a", [100, 101, 102, 103, 104, 105], [5, 7], num_computed=0),
        _new_req("b", list(range(200, 215)), [9], num_computed=0),
        _new_req("c", [300, 301, 302], [3], num_computed=0),
    ], num_scheduled={"a": 6, "b": 15, "c": 3}))
    # 手工推进 computed（正常由调度器拍间下发；首拍 chunk 已算部分）
    r.input_batch.num_computed_tokens_cpu[:] = [4, 10, 0] + [0] * 5
    so = _sched_output(
        cached=_cached_data(
            req_ids=["a", "b", "c"],
            new_blocks=[[8], [], []],
            computed=[4, 10, 0],
            outputs=[0, 6, 0]),
        num_scheduled={"a": 2, "b": 5, "c": 3},
    )
    return r, so


def test_get_cumsum_and_arange_source_comment_example():
    r = _runner()
    out = np.empty(16, dtype=np.int64)
    cu = r._get_cumsum_and_arange(np.array([2, 5, 3], dtype=np.int32), out)
    assert cu.tolist() == [2, 7, 10]
    assert out[:10].tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]


def test_prepare_inputs_flattening_and_index_select():
    r, so = _flattening_setup()
    num_sched = np.array([2, 5, 3], dtype=np.int32)
    logits_indices, spec_meta = r._prepare_inputs(so, num_sched)
    # [2,5,3] → req_indices [0,0,1,1,1,1,1,2,2,2]（np.repeat）
    req_indices = np.repeat(r.arange_np[:3], num_sched)
    assert req_indices.tolist() == [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    # positions = computed[req] + 请求内偏移
    np.testing.assert_array_equal(
        r.query_pos.np[:10],
        np.array([0, 1, 0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64),
    )
    # token_indices = pos + req_index·M（二维坐标编一维）→ index_select 收齐
    # a: pos 4,5 → 104,105；b: pos 10..14 → 210..214；c: pos 0..2 → 300..302
    expected = [104, 105, 210, 211, 212, 213, 214, 300, 301, 302]
    assert r.input_ids.cpu[:10].tolist() == expected
    assert r.input_ids.gpu[:10].tolist() == expected  # 前缀上载（同步拍）
    # logits_indices = 每请求窗尾（非 spec：query_start_loc[1:]−1）
    assert logits_indices.tolist() == [1, 6, 9]
    assert spec_meta is None


def test_prepare_inputs_query_start_loc_pad_and_seq_lens():
    r, so = _flattening_setup()
    r._prepare_inputs(so, np.array([2, 5, 3], dtype=np.int32))
    qsl = r.query_start_loc.np
    # CU 偏移 + 尾部 pad 非递减（FlashAttention 要求）
    assert qsl[0] == 0
    assert qsl[1:4].tolist() == [2, 7, 10]
    assert (qsl[4:] == 10).all()
    assert (np.diff(qsl) >= 0).all()
    assert r.query_start_loc.gpu[:4].tolist() == [0, 2, 7, 10]
    # GPU 端最终 positions = computed[req] + 请求内偏移
    assert r.positions[:10].tolist() == [4, 5, 10, 11, 12, 13, 14, 0, 1, 2]
    # seq_lens = computed + scheduled；尾部清零
    assert r.seq_lens[:3].tolist() == [6, 15, 3]
    assert (r.seq_lens[3:] == 0).all()
    # 乐观 seq_lens 同值（discard_request_mask 的数据源）
    assert r.optimistic_seq_lens_cpu[:3].tolist() == [6, 15, 3]
    assert (r.optimistic_seq_lens_cpu[3:] == 0).all()
    # num_accepted_tokens 无 spec：默认全 1（事件门控删除后的无条件路径）
    assert (r.num_accepted_tokens.np == 1).all()
    # req_indices/query_pos/num_scheduled_tokens 持久缓冲前缀上载
    assert r.req_indices.gpu[:10].tolist() == [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    assert r.num_scheduled_tokens.gpu[:3].tolist() == [2, 5, 3]


def test_prepare_inputs_slot_mapping_identity():
    r, so = _flattening_setup()
    r._prepare_inputs(so, np.array([2, 5, 3], dtype=np.int32))
    # pos 4,5 → 块 5 → slot 84,85；pos 10..14 → 块 9 → 154..158；
    # pos 0..2 → 块 3 → 48,49,50（slot = 块号×16 + 块内偏移）
    slots = r.input_batch.block_table[0].slot_mapping.np[:10]
    assert slots.tolist() == [84, 85, 154, 155, 156, 157, 158, 48, 49, 50]
    assert (r.input_batch.block_table[0].slot_mapping.np[10:] == PAD_SLOT_ID).all()
    # 块表先行拷贝：commit 只拷活跃行 → GPU 镜像前 3 行就绪
    bt = r.input_batch.block_table[0]
    assert bt.block_table.gpu[0, 0] == 5
    assert bt.block_table.gpu[1, 0] == 9


def test_discard_request_mask_masks_partial_chunked_prefill():
    r = _runner()
    # prompt 8 token，首拍只排 5：optimistic(5) < num_tokens(8) → 不采样
    r._update_states(_sched_output(new_reqs=[_new_req("a", list(range(8)), [1])],
                                   num_scheduled={"a": 5}))
    so = _sched_output(
        cached=_cached_data(req_ids=["a"], new_blocks=[[]], computed=[0],
                            outputs=[0]),
        num_scheduled={"a": 5})
    r._prepare_inputs(so, np.array([5], dtype=np.int32))
    assert r.discard_request_mask.np[0] == True  # noqa: E712
    # 第二 chunk：computed=5 再排 3 → optimistic 8 == num_tokens 8 → 采样
    r.input_batch.num_computed_tokens_cpu[0] = 5
    so2 = _sched_output(
        cached=_cached_data(req_ids=["a"], new_blocks=[[]], computed=[5],
                            outputs=[0]),
        num_scheduled={"a": 3})
    r._prepare_inputs(so2, np.array([3], dtype=np.int32))
    assert r.discard_request_mask.np[0] == False  # noqa: E712


# ================================================================ m05/m14
# 固定地址：持久缓冲跨拍不换（CUDA graph 回放前提）


def test_fixed_addresses_across_steps():
    r = _runner()
    ptrs = {
        "input_ids.cpu": r.input_ids.cpu.data_ptr(),
        "input_ids.gpu": r.input_ids.gpu.data_ptr(),
        "positions": r.positions.data_ptr(),
        "query_start_loc.gpu": r.query_start_loc.gpu.data_ptr(),
        "seq_lens": r.seq_lens.data_ptr(),
        "token_ids_cpu": r.input_batch.token_ids_cpu_tensor.data_ptr(),
    }
    for step in range(3):
        if step == 0:
            r._update_states(_sched_output(
                new_reqs=[_new_req("a", [1, 2], [10])],
                num_scheduled={"a": 2}))
        else:
            r._update_states(_sched_output(
                cached=_cached_data(req_ids=["a"], new_blocks=[[11 + step]],
                                    computed=[2], outputs=[step]),
                num_scheduled={"a": 1}))
        r._prepare_inputs(
            _sched_output(num_scheduled={"a": 1}, total=1),
            np.array([1], dtype=np.int32))
    for k, v in ptrs.items():
        assert v, k
    assert r.input_ids.cpu.data_ptr() == ptrs["input_ids.cpu"]
    assert r.input_ids.gpu.data_ptr() == ptrs["input_ids.gpu"]
    assert r.positions.data_ptr() == ptrs["positions"]
    assert r.query_start_loc.gpu.data_ptr() == ptrs["query_start_loc.gpu"]
    assert r.seq_lens.data_ptr() == ptrs["seq_lens"]
    assert r.input_batch.token_ids_cpu_tensor.data_ptr() == ptrs["token_ids_cpu"]


# ================================================================ 两段式 + m09
# execute_model / sample_tokens / 写回闭环


def _step_decode(r, req_id, pick, num_computed, new_blocks=(), output=0):
    r.enqueue_logits([{req_id: _logits_row(pick)}])
    so = _sched_output(
        cached=_cached_data(req_ids=[req_id], new_blocks=[list(new_blocks)],
                            computed=[num_computed], outputs=[output]),
        num_scheduled={req_id: 1},
    )
    assert r.execute_model(so) is None
    out = r.sample_tokens(None)
    return out


def test_two_phase_state_machine_and_empty_batch():
    r = _runner()
    so = _sched_output(new_reqs=[_new_req("a", [1], [10])],
                       num_scheduled={"a": 1})
    with pytest.raises(RuntimeError):
        # execute_model() 后必须先 sample_tokens()（两段式契约）
        r.execute_model(so)
        r.execute_model(so)


def test_empty_batch_returns_singleton_empty_output():
    from implementation.outputs import EMPTY_MODEL_RUNNER_OUTPUT

    r = _runner()
    out = r.execute_model(_sched_output())
    assert out is EMPTY_MODEL_RUNNER_OUTPUT


def test_bookkeeping_sync_writeback_closes_the_loop():
    r = _runner()
    r.enqueue_logits([{"a": _logits_row(9)}])
    r.execute_model(_sched_output(new_reqs=[_new_req("a", [5, 6, 7], [10])],
                                  num_scheduled={"a": 3}))
    out = r.sample_tokens(None)
    # 写回：采样 token 写 token_ids_cpu 行、列前移、req_state 增长
    assert out.sampled_token_ids == [[9]]
    assert r.input_batch.token_ids_cpu[0, :4].tolist() == [5, 6, 7, 9]
    assert r.input_batch.num_tokens_no_spec[0] == 4
    assert r.requests["a"].output_token_ids == [9]
    assert r.requests["a"].num_tokens == 4
    # 下一拍收集到的 input_ids 含自产 token（持久批次自产自销）
    out2 = _step_decode(r, "a", pick=12, num_computed=3, new_blocks=[11],
                        output=1)
    assert out2.sampled_token_ids == [[12]]
    assert r.input_ids.cpu[0] == 9
    assert r.requests["a"].output_token_ids == [9, 12]


def test_bookkeeping_discard_clears_partial_request_tokens():
    r = _runner()
    # chunked prefill：prompt 8、首拍排 5 → discard → 采样 token 清空
    r.enqueue_logits([{"a": _logits_row(9)}])
    r.execute_model(_sched_output(new_reqs=[_new_req("a", list(range(8)), [1])],
                                  num_scheduled={"a": 5}))
    out = r.sample_tokens(None)
    assert out.sampled_token_ids == [[]]
    # 行不推进、req_state 不增长（token 被丢弃）
    assert r.input_batch.num_tokens_no_spec[0] == 8
    assert r.requests["a"].output_token_ids == []


def test_async_bookkeeping_leaves_tokens_on_gpu_with_minus1_placeholder():
    r = _runner(async_scheduling=True)
    r.enqueue_logits([{"a": _logits_row(9)}])
    assert r.execute_model(_sched_output(
        new_reqs=[_new_req("a", [5, 6], [10])],
        num_scheduled={"a": 2})) is None
    out = r.sample_tokens(None)
    # 异步：真 token 留 GPU（prev_sampled_token_ids），CPU 行只写 -1 占位
    assert out.sampled_token_ids == []
    assert r.input_batch.prev_sampled_token_ids is not None
    assert r.input_batch.prev_sampled_token_ids.tolist() == [[9]]
    assert r.input_batch.prev_req_id_to_index == {"a": 0}
    assert r.input_batch.token_ids_cpu[0, 2] == -1
    assert r.requests["a"].output_token_ids == [-1]
    # 下一拍 _prepare_input_ids：GPU 回填——common-case 单 slice 直拷
    so = _sched_output(
        cached=_cached_data(req_ids=["a"], new_blocks=[[11]], computed=[2],
                            outputs=[1]),
        num_scheduled={"a": 1})
    r.enqueue_logits([{"a": _logits_row(8)}])
    assert r.execute_model(so) is None
    assert r.input_ids.cpu[0] == -1  # CPU 侧仍是占位
    assert r.input_ids.gpu[0] == 9  # GPU 侧已是真 token
    out2 = r.sample_tokens(None)
    assert r.requests["a"].output_token_ids == [-1, -1]
    assert r.input_batch.prev_sampled_token_ids.tolist() == [[8]]


def test_async_prepare_input_ids_scatter_after_reorder():
    # 批次变化/重排后 common-case 不命中 → 按 index scatter 回填
    r = _runner(async_scheduling=True)
    r.enqueue_logits([{"a": _logits_row(9), "b": _logits_row(8)}])
    r.execute_model(_sched_output(
        new_reqs=[_new_req("a", [5], [10]), _new_req("b", [6], [11])],
        num_scheduled={"a": 1, "b": 1}))
    r.sample_tokens(None)
    assert r.input_batch.prev_sampled_token_ids.tolist() == [[9], [8]]
    # b 先完成出批 → prev_req_id_to_index 快照仍含两行；a 的 prev_index=0
    so = _sched_output(
        cached=_cached_data(req_ids=["a"], new_blocks=[[12]], computed=[1],
                            outputs=[1]),
        num_scheduled={"a": 1},
        finished={"b"})
    r.enqueue_logits([{"a": _logits_row(7)}])
    assert r.execute_model(so) is None
    # a 单请求 decode：先全量前缀上载（含 -1 占位），再 scatter 真 token
    assert r.input_ids.gpu[0] == 9


def test_synchronize_input_prep_waits_then_records():
    r = _runner(async_scheduling=True)
    ev = r.prepare_inputs_event
    assert isinstance(ev, HostEvent)
    with r.synchronize_input_prep():
        pass
    # 防踩协议：进入等上拍 record、退出 record 本拍（HOST SEAM 计数观测）
    assert ev.syncs >= 1 and ev.records >= 1
    # 同步调度：event 为 None → 直通
    r2 = _runner()
    assert r2.prepare_inputs_event is None
    with r2.synchronize_input_prep():
        pass


# ================================================================ m10
# 可变性裁决（ngram_proposer_gpu.py:L475-L515）


def test_update_scheduler_for_invalid_drafts_trims_token_accounts():
    so = _sched_output(
        cached=_cached_data(req_ids=["a", "b"], computed=[10, 20],
                            outputs=[5, 5]),
        num_scheduled={"a": 4, "b": 2},
        total=6,
        spec_tokens={"a": [7, 8, 9], "b": [4]},
    )
    ev = HostEvent()
    valid = torch.tensor([1, 0], dtype=torch.int32)  # a:1/3 有效，b:0/1
    update_scheduler_for_invalid_drafts(ev, valid, so, {"a": 0, "b": 1})
    # a 回退 2：total 6→4、num_scheduled 4→2、spec 截 [7]
    assert so.total_num_scheduled_tokens == 4 - 1  # b 再回退 1 → 3
    assert so.total_num_scheduled_tokens == 3
    assert so.num_scheduled_tokens == {"a": 2, "b": 1}
    assert so.scheduled_spec_decode_tokens == {"a": [7]}
    # clamp 语义：valid_k 负值/超界钳到 [0, scheduled_k]
    so2 = _sched_output(
        num_scheduled={"c": 3}, total=3, spec_tokens={"c": [1, 2]})
    update_scheduler_for_invalid_drafts(
        ev, torch.tensor([99], dtype=torch.int32), so2, {"c": 0})
    assert so2.scheduled_spec_decode_tokens == {"c": [1, 2]}
    assert so2.total_num_scheduled_tokens == 3


def test_update_scheduler_skips_unknown_requests():
    so = _sched_output(
        cached=_cached_data(req_ids=["a"], computed=[1], outputs=[1]),
        num_scheduled={"a": 2}, total=2, spec_tokens={"a": [5]})
    ev = HostEvent()
    # req_id_to_index 缺 a → continue；spec_token_ids None 的请求也跳过
    update_scheduler_for_invalid_drafts(
        ev, torch.tensor([0], dtype=torch.int32), so, {})
    assert so.total_num_scheduled_tokens == 2
