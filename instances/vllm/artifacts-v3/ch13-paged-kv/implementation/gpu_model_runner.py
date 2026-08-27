# SOURCE: vllm/v1/worker/gpu_model_runner.py
# GPUModelRunner 的**KV 镜像切面**（m7 第 7/8 站 + m9/m10/m14/m15 的 worker 半边）：
#   _update_states——清零新块 + block_ids 差量 extend/整表替换 + 块表行写；
#   _prepare_inputs——第一句先拷块表（"overlap the copy"）+ positions GPU 组装
#   + compute_slot_mapping 派发；
#   _get_block_table——块表张量交 attention metadata builder（读腿出口）；
#   _allocate/_reshape_kv_cache_tensors——GPU 物理池：每层一块 int8 原始缓冲
#   → num_blocks = numel // page_size_bytes → 视图。
# 模型执行/采样/cudagraph 全景归 ch17/18；attention 后端消费归 ch21。
# SUBTRACTED（dossier.delete 批准项的落点）：第 9 条 CoW 拷贝
#   （copy_kv_cache_blocks_inplace，L1223-L1228）；spec/ngram/async 影子账
#   （L1263/L1290/L1337-L1404/L1459-L1469——ch12/33）；PP token 回传
#   （L1408-L1439/L1476-L1494——ch17）；encoder/mamba/LORA/mm（第 4 条）；
#   packed 别名分配（L7326-L7336——ch14）；layer_packing（L7383-L7404
#   ——ch14）；attn_backend.get_kv_cache_shape 的形状仲裁（L7421-L7453
#   ——ch21，主流后端为 [num_blocks, kv_heads, block_size, 2*head_dim]、
#   K/V 打进内容维；切面换用 2(K,V) 显式的说明性布局，页字节数不变）；
#   MambaSpec 支（L7455-L7468——ch14）。
# HOST SEAM：device=CPU 时 KV 池/块表/清零全走 CPU 镜像（BlockTable/
#   KVBlockZeroer 的 HOST SEAM 承载）；容器内为真 GPU。
import numpy as np
import torch

from .block_table import (
    NULL_BLOCK_ID,
    BlockTable,
    SlotMappingMode,
)
from .gpu_input_batch import CachedRequestState, InputBatch
from .kv_cache_interface import AttentionSpec, KVCacheConfig
from .worker_utils import AttentionGroup, KVBlockZeroer


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2250 _StandardLayoutBackend 的
#   契约位（HOST SEAM 装配切面：真实 backend 由注意力后端注册表装配 → ch21；
#   说明性布局 = 块最外层（block_dim=0），每层一块
#   [num_blocks, 2, block_size, kv_heads, head_dim] 张量——m10 视图；
#   真实主流后端 get_kv_cache_shape 是 (num_blocks, kv_heads, block_size,
#   2*head_size)——K/V 打进内容维 → ch21，页字节数不变）
class _StandardLayoutBackend:
    # HOST SEAM 装配切面：真实 backend 由注意力后端注册表装配（→ ch21）；
    #   说明性布局取值 = 块最外层（1 段/缓冲）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2250 get_kv_cache_block_dim
    def get_kv_cache_block_dim(
        self, kernel_block_size: int, num_kv_heads: int, head_size: int,
        cache_dtype_str: str,
    ) -> int:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2250（block_dim=0）
        return 0


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2490 GPUModelRunner（KV 镜像
#   切面——真实类为千行执行器，ch17/18 全文）
class GPUModelRunner:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2492 __init__（切面装配）
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        block_size: int,
        max_num_reqs: int,
        max_blocks_per_req: int,
        max_num_batched_tokens: int,
        device: torch.device,
    ) -> None:
        # SUBTRACTED: 模型/采样/cudagraph/spec 装配面（L2492-L2670——ch17）；
        #   cache_config.device 装配（HOST SEAM：device 直供）。
        self.kv_cache_config = kv_cache_config
        self.device = device
        self.block_size = block_size
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7303-L7310 块尺寸一致性
        #   断言（InputBatch 与 kv_cache 同源）
        assert len(kv_cache_config.kv_cache_groups) >= 1

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2870 InputBatch 装配
        #   （真实经 initialize_kv_cache_tensors 统一建；切面直构单组表）
        self.input_batch = InputBatch(
            block_table=BlockTable(
                block_size=block_size,
                max_num_reqs=max_num_reqs,
                max_num_blocks_per_req=max_blocks_per_req,
                max_num_batched_tokens=max_num_batched_tokens,
                pin_memory=False,  # HOST SEAM：CPU host 无 pinned memory
                device=device,
                kernel_block_size=block_size,
                slot_mapping_mode=SlotMappingMode.TOKEN_TO_KV_SLOT,
            ),
            max_num_reqs=max_num_reqs,
        )
        self.requests: dict[str, CachedRequestState] = {}
        self.runner_only_attn_layers: set[str] = set()

        # GPU 物理池（m10）：每层一块原始缓冲 → 视图
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7541 initialize_kv_cache_
        #   tensors 的装配序列（切面直调两步）
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        self.kv_caches = self._reshape_kv_cache_tensors(kv_cache_raw_tensors)
        spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        # HOST SEAM 观测位：一块页的 int32 元素数（测试对账清零范围用）
        self.page_size_el = spec.page_size_bytes // 4

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1140 _init_kv_zero_meta 的
        #   装配位（真实由 gpu_worker 在 CuMem 池外调；切面构造期直建）
        self._init_kv_zero_meta()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1140 _init_kv_zero_meta
    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Called from gpu_worker.py outside the CuMem pool context.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1145-L1152（static_
        #   forward_context 真实来自 compilation_config；切面以 kv_caches
        #   承载同一 layer→tensor 契约——HOST SEAM 装配位）
        class _Ctx:
            # HOST SEAM 装配账位：static_forward_context[layer].kv_cache
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1151（装配面）
            def __init__(self, kv_cache: torch.Tensor):
                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1151
                self.kv_cache = kv_cache

        self._kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=[
                g.kv_cache_spec.block_size
                for g in self.kv_cache_config.kv_cache_groups
            ],
            cache_dtype="auto",
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context={
                name: _Ctx(kv) for name, kv in self.kv_caches.items()
            },
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1154 _zero_block_ids
    def _zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1156-L1157
        if hasattr(self, "_kv_block_zeroer"):
            self._kv_block_zeroer.zero_block_ids(block_ids)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1190 _update_states（KV 镜像
    #   切面：清零新块 + 块账镜像 + 页表行写）
    def _update_states(self, scheduler_output) -> None:
        """Update the cached states of the requests and the persistent batch
        with the scheduler output for this step（KV 切面）.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.
        """
        # Remove finished requests from the cached states.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1202-L1205
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # SUBTRACTED: num_prompt_logprobs/late_interaction_runner 清缓存
        #   （L1206-L1209——ch08/ch17）。
        # Remove the finished requests from the persistent batch.
        # NOTE(Woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1216-L1217
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1219-L1222 清零新块
        #   （m8：块从自由队列回收，上一任主人留下的字节还在）
        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
        # SUBTRACTED: kv_cache_block_copies 的 CoW 原地拷贝（L1223-L1228
        #   ——第 9 条 → ch15）；encoder 缓存清理与未调度请求摘批
        #   （L1230-L1253——ch10/12/14 的批维护面）。

        reqs_to_add: list[CachedRequestState] = []

        # Add new requests to the cached states.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1266-L1309 新请求建档
        #   （block_ids 全量随首帧过线——m7 全量块表半边）
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            # SUBTRACTED: 重提同 ID 的流式边界（L1268-L1294——streaming）与
            #   mm/logprobs建档（L1296-L1317）。
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )
            self.requests[req_id] = req_state
            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1334-L1335
        req_data = scheduler_output.scheduled_cached_reqs
        # SUBTRACTED: spec/ngram 影子账（L1335/L1337-L1353——ch33/12）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1355-L1361 在跑/恢复请求
        #   循环头
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # SUBTRACTED: prev_num_draft_len 乐观纠错（L1363-L1404——ch12/33）。
            # Update the cached states.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1406
            req_state.num_computed_tokens = num_computed_tokens
            # SUBTRACTED: PP 非末 rank 的 token 回传（L1408-L1439——ch17）。

            # Update the block IDs.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1441-L1446 差量 extend
            #   （m7 增量半边）
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1447-L1452 整表替换
                #   （被抢占恢复——恢复者块全换新）
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1454-L1465
                # SUBTRACTED: async 恢复 output_token_ids（L1459-L1463——ch12）
                #   与 ngram_gpu 全量拷贝（L1466-L1468——ch33）
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1471-L1474 num_computed
            #   落行 + 块表行写（append_row 差量追加进 CPU 页表行）
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                # SUBTRACTED: 真实 L1474 单调用 MultiGroupBlockTable.append_row(
                #   new_block_ids, req_index) 扇出各组；MultiGroup 删（第 4 条）
                #   后单组表逐组直写——同语义。
                for group_ids in new_block_ids:
                    self.input_batch.block_table.append_row(group_ids, req_index)

        # SUBTRACTED: PP 回传（L1476-L1494——ch17）。
        # Add the new requests to the persistent batch.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~1495-L1560 的建档尾段
        #   （InputBatch.add_request 内景 → ch18）
        for req_state in reqs_to_add:
            self.input_batch.add_request(req_state)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1960 _prepare_inputs（块表
    #   先行拷贝 + positions/槽位切面）
    def _prepare_inputs(
        self,
        scheduler_output,
        num_scheduled_tokens: np.ndarray,
    ) -> None:
        """Prepare the block table copy + GPU positions + slot mapping
        （KV 切面；token ids/采样元数据组装归 ch12/17）."""
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1972-L1975
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1977-L1979 m15：_prepare_
        #   inputs 第一句就是 commit（拷贝与后续 CPU 活重叠）
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1983
        req_indices = np.repeat(self._arange_np(num_reqs), num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1985-L1989（cumsum + 组内
        #   arange——query_start_loc 的来源）
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
        query_start_loc_np[1:] = cu_num_tokens
        query_pos_np = np.concatenate(
            [np.arange(n, dtype=np.int64) for n in num_scheduled_tokens]
        )

        # SUBTRACTED: M-RoPE/XD-RoPE/token_indices 的 CPU 组装（L1997-L2014
        #   ——多模态章）；input_ids 组装（L2016-L2174 的主体——ch12/17）。
        # Update num_computed_tokens on GPU（非 async spec 的直接拷贝支）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2175-L2178
        num_computed_tokens_gpu = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu[:num_reqs].astype(np.int64)
        ).to(self.device)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2180-L2182 req_indices
        req_indices_gpu = torch.from_numpy(req_indices.astype(np.int64)).to(self.device)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2184 query_pos 上载
        query_pos_gpu = torch.from_numpy(query_pos_np).to(self.device)
        query_start_loc = torch.from_numpy(query_start_loc_np).to(self.device)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2188-L2191 positions = GPU
        #   num_computed_tokens[req_indices_gpu] + query_pos.gpu（GPU 张量进、
        #   换算全程不落 CPU——WC3 的痛点源头）
        positions = num_computed_tokens_gpu[req_indices_gpu] + query_pos_gpu

        # SUBTRACTED: seq_lens 组装（L2192-L2195——attention metadata 的消费
        #   面 → ch21）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2197-L2201 compute_slot_
        #   mapping 派发（GPU 张量直进 kernel）
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            query_start_loc,
            positions,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~1850 _arange_np 的装配位
    #   （HOST SEAM 账位：真实预构 np.arange 缓冲——ch18）
    def _arange_np(self, n: int) -> np.ndarray:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~1850
        return np.arange(n)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2284 _build_attention_metadata
    #   的块表出口段（切面抽出：读腿交棒）
    def _get_block_table(self, num_reqs: int, num_reqs_padded: int):
        """块表张量交给 attention metadata builder（读走 block_table——
        写侧 slot_mapping / 读侧块表张量的两条腿，m14/F7 伏笔埋点）。"""
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2326-L2336
        # SUBTRACTED: EncoderOnlyAttentionSpec 零块表分支（L2328-L2333
        #   ——第 4 条；本章全组都持真块表）
        blk_table = self.input_batch.block_table
        blk_table_tensor = blk_table.get_device_tensor(num_reqs_padded)

        # Fill unused block table entries with NULL_BLOCK_ID (null block)
        # for CUDAGraph padding. Block 0 is reserved for padding.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2338-L2341（pad 行填
        #   NULL_BLOCK_ID=0——与 block_id=0 被 null_block 占用呼应）
        blk_table_tensor[num_reqs:num_reqs_padded].fill_(NULL_BLOCK_ID)
        return blk_table_tensor

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7312 _allocate_kv_cache_
    #   tensors —— GPU 物理池的出生
    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7325-L7342（packed 别名
        #   分支删——L7326-L7336，ch14：每 tensor 独立一块 int8 缓冲）
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # SUBTRACTED: block_stride>0 的 packed_backing 别名（L7328-L7336
            #   ——跨层重叠布局 → ch14）
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7344-L7352
        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7355 _attn_group_iterator /
    #   L7358 _kv_cache_spec_attn_group_iterator
    def _attn_group_iterator(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7356（真实 attn_groups
        #   由模型注册装配——ch17/21；切面从 kv_cache_config 逐组现构）
        for i, group in enumerate(self.kv_cache_config.kv_cache_groups):
            yield AttentionGroup(
                backend=_StandardLayoutBackend(),
                layer_names=list(group.layer_names),
                kv_cache_spec=group.kv_cache_spec,
                kv_cache_group_id=i,
            )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7358 _kv_cache_spec_attn_
    #   group_iterator
    def _kv_cache_spec_attn_group_iterator(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7359-L7362
        if not self.kv_cache_config.kv_cache_groups:
            return
        yield from self._attn_group_iterator()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7364 _reshape_kv_cache_
    #   tensors（切面：num_blocks 换算 + 标准视图）
    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # SUBTRACTED: kernel_block_sizes 参数与 per-group kernel 块细分
        #   （L7367、L7393-L7396——第 4 条）、layer_packing（L7383-L7404
        #   ——ch14）、attn_backend.get_kv_cache_shape/stride_order 的形状
        #   仲裁（L7421-L7453——ch21：主流后端 [num_blocks, kv_heads,
        #   block_size, 2*head_dim]，K/V 打进内容维）、MambaSpec 支
        #   （L7455-L7468——ch14）。
        kv_caches: dict[str, torch.Tensor] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7400-L7407
                #   num_blocks = numel // page_size_bytes（两侧同源同值的
                #   worker 半边——单一事实源 = KVCacheConfig）
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7408-L7413（isinstance
                #   AttentionSpec 判定与 blocks_per_kv_block=1 的细分乘子删）
                if isinstance(kv_cache_spec, AttentionSpec):
                    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7433-L7439
                    #   （get_kv_cache_shape 形状仲裁）的 host 说明性视图：
                    #   [num_blocks, 2(K,V), block_size, kv_heads, head_dim]——
                    #   real_page_size_bytes 的 2× 取成显式两半；真实主流后端
                    #   为 (num_blocks, kv_heads, block_size, 2*head_size)、
                    #   K/V 打进内容维（flash_attn.py:L143），页字节数不变
                    kv_caches[layer_name] = raw_tensor.view(kv_cache_spec.dtype).view(
                        num_blocks,
                        2,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                    )
                else:
                    raise NotImplementedError
        return kv_caches
