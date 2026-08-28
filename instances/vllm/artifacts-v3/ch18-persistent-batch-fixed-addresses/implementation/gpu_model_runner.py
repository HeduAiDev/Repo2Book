# SOURCE: vllm/v1/worker/gpu_model_runner.py
# Subtract-only companion for v3 ch18 — 持久批次与固定地址（pin v0.27.1 /
# 6e448d0ea）。Same names, same structure, same control flow; only dossier-
# approved deletions (each marked `# SUBTRACTED:`), plus 本章切面之外的域段
# 以「SUBTRACTED + 归属章」注记（ch09-ch17 立下的切面惯例）。
#
# 本章切面（dossier.code_spine / stations）：
#   * execute_model 入口（L4166-L4535）：可变性裁决（ngram replace 浅拷贝
#     L4180-L4195）→ synchronize_input_prep 内 _update_states（L4203-L4208）
#     → 空批早退 → _prepare_inputs（L4250-L4253）→ 前向段（ENGINE SEAM，
#     ch17/ch19 边界）→ ExecuteModelState 打包（L4516-L4535）；
#   * _update_states（L1192-L1566）：差量调和——移除 finished/unscheduled →
#     新请求建 CachedRequestState 快照 → 老请求 append/替换块号 → 落位 →
#     condense 压实 → 重排钩子 → 刷元数据；
#   * _prepare_inputs（L1960-L2282）：commit_block_table 先行 → np.repeat/
#     cumsum/arange/index_select 扁平收集 → query_start_loc 非递减 pad →
#     乐观 seq_lens + discard mask → GPU 端 positions/seq_lens →
#     compute_slot_mapping → _prepare_input_ids 前缀上载；
#   * _prepare_input_ids（L1784-L1913）/ _bookkeeping_sync（L3723-L3862，
#     写回闭环）/ synchronize_input_prep（L3865-L3877，pinned 防踩）/
#     sample_tokens（L4553-L4840 的 bookkeeping 切面）；
#   * __init__ 持久缓冲块（L763-L810 起，"Persistent buffers for CUDA
#     graphs"）——地址从此永不变（m05/m14）。
#
# SUBTRACTED（dossier.subtraction_plan.delete 批准项，九条 → 落点见各处
# 就地注释；第 0/1/2/3/4/5/6/7/8 条分别标注「delete[N]」）。
# 前向本体（L4255-L4514）/attention metadata/cudagraph/模型装配归 ch19/
# ch21/ch22/ch17——以 ENGINE SEAM 脚本化 logits 承载两段式契约（ch12 同款）。
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

from ._host_seams import (
    CacheConfig,
    HostCopyStream,
    HostEvent,
    KVBlockZeroer,
    LateInteractionRunner,
    Sampler,
    SamplingType,
    envs,
    get_dcp_group,
    get_pp_group,
    init_logger,
    length_from_prompt_token_ids_or_embeds,
    reorder_batch_to_split_decodes_and_prefills,
)
from .block_table import SlotMappingMode
from .gpu_input_batch import CachedRequestState, InputBatch
from .logits_processor import build_logitsprocs
from .math_utils import cdiv
from .ngram_proposer_gpu import update_scheduler_for_invalid_drafts
from .outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput, SamplerOutput
from .torch_utils import PIN_MEMORY
from .utils import CpuGpuBuffer, record_function_or_nullcontext

if TYPE_CHECKING:
    from .output import NewRequestData, SchedulerOutput

logger = init_logger(__name__)


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L437-L451 ExecuteModelState ——
# 两段式契约的暂存协议本体（两段之间传递的短命缓存态）
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L441-L450（SpecDecodeMetadata/
    #   CommonAttentionMetadata/ECConnectorOutput/CUDAGraphStat 是章界外类型
    #   ——future 注解下永不求值，逐字保留真实名）
    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None


# SUBTRACTED: 三支 mixin 基类（LoRAModelRunnerMixin / KVConnectorModelRunnerMixin /
#   ECConnectorModelRunnerMixin——LoRA 面与 KV/EC 连接器面，ch16/ch33 域；
#   本章 runner 无混合基类）。
# SOURCE: vllm/v1/worker/gpu_model_runner.py:L453 GPUModelRunner —— 本章舞台类
class GPUModelRunner:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L456-L458 __init__ 签名
    def __init__(
        self,
        vllm_config: Any,
        device: torch.device,
    ):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L461-L471 配置解包
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L473-L479
        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.dtype = self.model_config.dtype

        # SUBTRACTED: check_ep_fault / kv_cache_dtype 装配（L480-L486——
        #   MoE 容错与 KV dtype，ch34/ch14 域）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L488 is_pooling_model 派生
        #   （delete[2] 未列 runner 侧此行；InputBatch 构造 L727 与
        #   build_logitsprocs 的实参消费位保留）
        self.is_pooling_model = model_config.runner_type == "pooling"
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L489 enable_prompt_embeds
        #   旗标（delete[1] 只删缓冲与守卫支；旗标保留——精简配置恒 False）
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        # SUBTRACTED: mm raw-input/pruning/routed_experts 旗标（L490-L498
        #   ——mm/MoE 域）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L499-L506
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L501-L502
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L503-L504 dcp_world_size
        #   属性及其赋值（delete[8] 明示保留——单卡 decode_context_parallel_
        #   size=1 短路，不触 get_dcp_group）
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L505-L506
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # SUBTRACTED: broadcast_pp_output（L508-L515——PP 广播装配，delete[7]
        #   PP 面）。
        # SUBTRACTED: num_query_heads/inputs_embeds_size/use_alibi/cascade/
        #   is_mm_prefix_lm（L517-L525——ch21/22 注意力域；inputs_embeds_size
        #   的唯一消费点 inputs_embeds 缓冲随 delete[1] 删）。
        # SUBTRACTED: mm_registry 装配与 supports_mm_inputs 派生（L527/L530-
        #   L532——mm 域）；HOST SEAM 替代：sample_tokens L4775 消费位恒 False
        #   （真实语义 = 纯文本模型 supports_multimodal_inputs(model_config)
        #   返回 False）。
        self.supports_mm_inputs = False  # HOST SEAM — mm 域（L530-L532）
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L528-L529 uses_mrope/
        #   uses_xdrope_dim 派生（delete[0] 明示保留——守卫分支已删净）
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L534-L539 max_encoder_len
        #   （L697-L698 占位块数推导的输入；seam 配置 is_encoder_decoder=False
        #   → 0）
        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Async scheduling
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L541-L542
        self.use_async_scheduling = self.scheduler_config.async_scheduling

        # SUBTRACTED: Sampler 真装配（L544-L548——logprobs_mode/fp64 gumbel 面，
        #   ch08/ch29 域）；HOST SEAM greedy 支承载 _sample 调用位。
        self.sampler = Sampler()  # HOST SEAM — ch08 采样域（greedy 支）
        # SUBTRACTED: eplb/_moe 装配（L550-L558——MoE 域）。

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L563-L576 惰性装配位
        self.kv_caches: list[torch.Tensor] = []
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: Any | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[Any]] = []
        # self.kv_cache_config: KVCacheConfig  # set in initialize_kv_cache (ch14 域)

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L573 late_interaction_
        #   runner（delete[6] 明示 5 处调用位整组保留）
        self.late_interaction_runner = LateInteractionRunner()  # HOST SEAM

        # Encoder CUDA graph manager (initialized after model load if enabled)
        self.encoder_cudagraph_manager: Any | None = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L578
        self.use_aux_hidden_state_outputs = False
        # SUBTRACTED: speculative drafter 装配 elif 链（L579-L655——ch33 域；
        #   其中 ngram_gpu elif 支 L610-L626 是 delete[5] 批准删除项：镜像
        #   张量 num_tokens_no_spec_gpu/token_ids_gpu_tensor/pinned 双 buf 与
        #   NgramProposerGPU 构造整支删）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L660-L673 spec 计数面
        #   （prev_num_spec_tokens 是 _prepare_input_ids L1840 的乘数）
        self.num_spec_tokens = 0
        self.prev_num_spec_tokens = 0
        self.valid_sampled_token_count_gpu: torch.Tensor | None = None
        # SUBTRACTED: speculative_config 分支（L663-L670——精简配置 None）。
        # SOURCE: vllm/v1/worker/gpu_input_batch 用途的 use_async_spec_decode
        #   （L671-L673——delete[4] 删除其全部消费点后为死存；定义保留）
        self.use_async_spec_decode = (
            self.use_async_scheduling and self.num_spec_tokens > 0
        )

        # Request states.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L675-L679 requests 全量缓存
        #   （req_id→CachedRequestState——差量协议的 worker 侧根本载体）
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L690-L731 InputBatch 装配
        #   （占位块尺寸推导 + logitsprocs 装填）
        logits_processors = model_config.logits_processors
        custom_logitsprocs = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        placeholder_block_size = (
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
        )
        placeholder_max_num_blocks = cdiv(
            max(self.max_model_len, self.max_encoder_len), placeholder_block_size
        )
        self._init_block_sizes = [placeholder_block_size]
        self._init_kernel_block_sizes = [placeholder_block_size]
        self._init_max_num_blocks = [placeholder_max_num_blocks]
        self._init_slot_mapping_modes = [SlotMappingMode.TOKEN_TO_KV_SLOT]
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoder
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[placeholder_block_size],
            kernel_block_sizes=[placeholder_block_size],
            max_num_blocks_per_req=[placeholder_max_num_blocks],
            num_spec_tokens=self.num_spec_tokens,
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                PIN_MEMORY,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively. Thinking-budget
            # tracking is requested dynamically when a budgeted request is in the batch.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
            reasoning_config=self.vllm_config.reasoning_config,
            use_replayssm=self.cache_config.use_replayssm,
        )

        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L733-L743 async 流/事件对
        #   （HOST SEAM：CPU host 无 CUDA stream/event——HostCopyStream/HostEvent
        #   站同一契约位，ch12 同款）
        self.async_output_copy_stream: Any | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: Any | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = HostCopyStream()  # HOST SEAM
            # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
            # under TP contention that spin can balloon and make the rank a straggler.
            self.prepare_inputs_event = HostEvent()  # HOST SEAM

        # SUBTRACTED: cudagraph_batch_sizes 装配（L745-L754——ch19 捕获域）。
        # SUBTRACTED: 设备属性缓存与 encoder timing registry（L756-L761
        #   ——ch17/观测域）。

        # Persistent buffers for CUDA graphs.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L763-L810 持久缓冲块
        #   （一次分配、地址永不变——m05/m14 本章标题之二）
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.optimistic_seq_lens_cpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.prev_num_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.req_indices = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        # Maps current batch position -> previous batch position (-1 for new reqs)
        self.prev_positions = self._make_buffer(self.max_num_reqs, dtype=torch.int64)
        self.num_scheduled_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L790 encoder_seq_lens
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # SUBTRACTED: dcp_local_seq_lens 守卫分配（L791-L794——delete[8]：单卡
        #   dcp_world_size=1 恒 False；守卫消费点 L2451-L2462 在 _build_
        #   attention_metadata（ch21/22 切面外），两侧成对删）。
        # SUBTRACTED: inputs_embeds 缓冲分配（L795-L800——delete[1]：bf16 免
        #   numpy 注释与分配整段；is_token_ids L801【不删】——全链跨
        #   InputBatch/写回多处，删缓冲不删链即残留死引用）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L801-L810
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_mask = self._make_buffer(
            self.max_num_reqs, dtype=torch.bool
        )
        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )

        # SUBTRACTED: M-RoPE / XD-RoPE 位置缓冲（L812-L833——delete[0]：纯
        #   文本模型 uses_mrope/uses_xdrope_dim 恒 False，守卫删净无悬空；
        #   『+1 dummy 保非连续以兼容 torch.compile』的 PR 12128 讨论归 ch19）。

        # None in the first PP rank. The rest are set after load_model.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L835（死存保留——PP 面
        #   delete[7] 未列此行）
        self.intermediate_tensors: Any | None = None

        # OPTIMIZATION: Cache the arange tensors rather than creating them
        # every step. Keep in int64 to avoid overflow with long context.
        # - arange_np: immutable [0, 1, 2, ...] used as source for batched computation
        # - query_pos: CpuGpuBuffer for the computed batched arange result
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L838-L845 arange 缓存
        arange_size = max(self.max_num_reqs + 1, self.max_num_tokens)
        self.arange_np = np.arange(arange_size, dtype=np.int64)
        self.query_pos = self._make_buffer(arange_size, dtype=torch.int64)
        self._arange_scratch = np.empty(arange_size, dtype=np.int64)

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L847-L851
        self.shared_kv_cache_layers: dict[str, str] = {}
        # SUBTRACTED: kv_sharing_fast_prefill 簿记+守卫分配（L852-L858
        #   ——delete[6]⑤：eligible_layers 集与 logits_indices 缓冲）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L860
        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # SUBTRACTED: cudagraph_dispatcher / mm_budget 装配（L862-L869——
        #   ch19 捕获域 / mm 域）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L871 重排阈值（_may_
        #   reorder_batch L1133 消费）
        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L873-L876（_init_kv_zero_
        #   meta 的构造实参之一，保留）
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        # SOURCE: vllm/v1/worker/gpu_input_batch 写回/spec 面的缓存输出族
        #   （L878-L895 逐字——_draft_token_ids 是 _prepare_input_ids L1894
        #   的守卫读点；_num_valid_draft_tokens* 是 _update_states L1346-L1348
        #   ngram 守卫块的实参，delete[5] 明示保留）
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self._draft_probs: torch.Tensor | None = None
        self._draft_prob_req_ids: list[str] | None = None
        # N-gram GPU path: async D2H buffer/event for per-request valid draft counts.
        self._num_valid_draft_tokens: torch.Tensor | None = None
        self._num_valid_draft_tokens_cpu: torch.Tensor | None = None
        self._num_valid_draft_tokens_event: Any | None = None
        self._num_valid_draft_tokens_copy_stream: Any | None = None
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            self._num_valid_draft_tokens_cpu = torch.empty(
                self.max_num_reqs, dtype=torch.int32, pin_memory=PIN_MEMORY
            )
            self._num_valid_draft_tokens_event = HostEvent()  # HOST SEAM
            self._num_valid_draft_tokens_copy_stream = HostCopyStream()  # HOST SEAM

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L897-L904
        self._draft_token_req_ids: list[str] | None = None
        self.transfer_event = HostEvent()  # HOST SEAM — L898 torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L906-L935（num_spec_
        #   tokens=0 → 恒不创建；字段定义逐字保留——num_accepted_tokens_event
        #   是 delete[4] 门控语义的说明锚，事件本身永不诞生）
        self.valid_sampled_token_count_event: Any | None = None
        self.valid_sampled_token_count_copy_stream: Any | None = None
        # We also copy the drafted tokens to the CPU asynchronously,
        # in case we need them for structured outputs.
        self.draft_token_ids_event: Any | None = None
        self.draft_token_ids_copy_stream: Any | None = None
        self.valid_sampled_token_count_cpu: torch.Tensor | None = None
        self.draft_token_ids_cpu: torch.Tensor | None = None
        self.num_accepted_tokens_event: Any | None = None
        if self.num_spec_tokens:
            self.draft_token_ids_event = HostEvent()  # HOST SEAM
            self.num_accepted_tokens_event = HostEvent()  # HOST SEAM
            self.draft_token_ids_copy_stream = HostCopyStream()  # HOST SEAM
            self.draft_token_ids_cpu = torch.empty(
                (self.max_num_reqs, self.num_spec_tokens),
                dtype=torch.int64,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            if self.use_async_scheduling:
                self.valid_sampled_token_count_event = HostEvent()  # HOST SEAM
                self.valid_sampled_token_count_copy_stream = HostCopyStream()  # SEAM
                self.valid_sampled_token_count_cpu = torch.empty(
                    self.max_num_reqs,
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )

        # SUBTRACTED: set_offloader/create_offloader（L937-L939——权重卸载，
        #   ch17/ch34 域）。

        # Ephemeral state transferred between execute_model() and sample_tokens().
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L941-L951 单槽字段族
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: Any | None = None
        self.mamba_state_idx: dict[str, int] = {}
        self._mamba_bufs: Any | None = None
        self.mamba_prev_last_scheduled_idx: CpuGpuBuffer | None = None
        if self.cache_config.mamba_cache_mode == "all" and self.num_spec_tokens > 0:
            self.mamba_prev_last_scheduled_idx = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        self.layerwise_nvtx_hooks_registered = False

        # ENGINE SEAM（ch17 边界）：脚本化前向的脚本队列（每步一个
        # {req_id: logits 行} 字典；真实前向在 GPU 上算 compute_logits(
        # hidden_states[logits_indices])，见 L4484-L4485）。
        self._scripted_logits: deque = deque()

    # ------------------------------------------------------------------ #
    # 位置缓冲出口（delete[0] 后的纯文本主支）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1032-L1044 _get_positions
    #   （mrope/xdrope 两臂 L1034-L1037/L1040-L1043 随 delete[0] 整删——
    #   独立 if 语句，删净后纯文本主支完整）
    def _get_positions(self, num_tokens: Any):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1033-L1038
        if isinstance(num_tokens, int):
            return self.positions[:num_tokens]
        else:
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1039-L1044
            return self.positions[num_tokens]

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1046-L1054 _make_buffer ——
    #   持久缓冲的分配入口
    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1049-L1054
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            with_numpy=numpy,
        )

    # SUBTRACTED: _get_mamba_bufs / _init_model_kwargs（L1056-L1113——
    #   mamba align 簿记与 pooling token_type_ids 装配，ch14/ch29 域）。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1115-L1138 _may_reorder_batch
    #   —— attention backend 重排钩子（m12）
    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_groups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1130-L1131
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1133-L1138
        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1140-L1152 _init_kv_zero_meta
    #   ——【不删（delete[6]①：gpu_worker 启动期外部调用，仅变无人使用）】
    def _init_kv_zero_meta(self) -> None:
        """One-time precomputation for _zero_block_ids.

        Called from gpu_worker.py outside the CuMem pool context.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1145-L1152
        self._kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=self.compilation_config.static_forward_context,
        )

    # SUBTRACTED: _zero_block_ids（L1154-L1157——delete[6]①：方法本体随调用点
    #   L1219-L1222 同删；_init_kv_zero_meta 与 _kv_block_zeroer 保留——
    #   gpu_worker 启动期外部调用，仅变无人使用）。
    # SUBTRACTED: _init_device_properties / _sync_device / _get_or_create_
    #   async_output_copy_stream（L1160-L1174——ch17/ch12 域）。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1176-L1182 _on_request_state_
    #   removed —— 平台 runner 的请求级清理钩子（默认 no-op）
    def _on_request_state_removed(
        self,
        req_id: str,
        req_state: CachedRequestState | None,
    ) -> None:
        """Hook for platform runners to clean request-scoped side caches."""
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1181-L1182
        del req_id, req_state

    # SUBTRACTED: _process_encoder_cache_scheduler_output（L1184-L1190——
    #   delete[6]②：encoder cache 释放调用 L1230-L1231 与方法本体一并删）。

    # ------------------------------------------------------------------ #
    # 本章主方法①：差量调和（m02/m11）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1192-L1566 _update_states
    def _update_states(self, scheduler_output: "SchedulerOutput") -> Callable | None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1202-L1209
        for req_id in scheduler_output.finished_req_ids:
            req_state = self.requests.pop(req_id, None)
            self._on_request_state_removed(req_id, req_state)
            self.num_prompt_logprobs.pop(req_id, None)
        self.late_interaction_runner.on_requests_finished(
            scheduler_output.finished_req_ids
        )
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1216-L1217
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # SUBTRACTED: 新块 GPU 置零与块级 CoW 拷贝（L1219-L1228——delete[6]①：
        #   new_block_ids_to_zero→_zero_block_ids 与 kv_cache_block_copies→
        #   copy_kv_cache_blocks_inplace——缓存卫生命令，精简版无 prefix cache
        #   复用，不影响 token 账与块表正确性）。
        # SUBTRACTED: encoder cache 释放调用（L1230-L1231——delete[6]②）。

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1238-L1247 集合差算术
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1252-L1253
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1255-L1260 ngram 局部量
        #   （delete[5] 明示保留——可变性裁决协议的守卫面；speculative_config
        #   None 时全为不触发的守卫，删了反而留 NameError）
        is_ngram_gpu = (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        )
        if is_ngram_gpu:
            ngram_gpu_new_reqs: list[CachedRequestState] = []

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1262-L1263
        reqs_to_add: list[CachedRequestState] = []
        deferred_spec_decode_corrections = []

        # Add new requests to the cached states.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1266-L1272 streaming 支
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1274-L1284 采样参数
            #   解包与 RANDOM_SEED generator 构造
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            # SUBTRACTED: pooling pooler 更新（L1286-L1293——delete[2]：
            #   is_pooling_model 主路径）。
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1295-L1308 新请求全量
            #   快照（worker 从此缓存请求全量，此后只收差量）
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                prompt_is_token_ids=new_req_data.prompt_is_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1309-L1310
            self.requests[req_id] = req_state
            self.late_interaction_runner.register_request(req_id, pooling_params)

            # SOURCE: vllm/v1/worker/gpu_input_batch prompt logprobs 登记
            #   （L1312-L1317）
            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # SUBTRACTED: M-RoPE/XD-RoPE 初始化（L1319-L1325——delete[0]）。
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1327-L1330
            reqs_to_add.append(req_state)
            # Track new requests for ngram_gpu full tensor copy
            if is_ngram_gpu:
                ngram_gpu_new_reqs.append(req_state)

        # Update the states of the running/resumed requests.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1332-L1335
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Save scheduler-allocated spec lengths before trimming so
        # prev_num_draft_len keeps the optimistic count for rejection correction.
        # SOURCE: vllm/v1/worker/gpu_input_batch original_num_spec_per_req 声明
        #   （L1339——delete[4] 明示【不删】：保留的 ngram 守卫块 L1340-L1351
        #   在 L1344-L1345 填充该 dict，删声明即悬空名；删恢复段后填充退化为
        #   无害死存储）
        original_num_spec_per_req: dict[str, int] = {}
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1340-L1351 ngram 守卫块
        #   （delete[5] 明示保留——可变性裁决协议：就地裁剪在此发生）
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            for req_id, toks in scheduled_spec_tokens.items():
                original_num_spec_per_req[req_id] = len(toks)
            update_scheduler_for_invalid_drafts(
                self._num_valid_draft_tokens_event,
                self._num_valid_draft_tokens_cpu,
                scheduler_output,
                self.input_batch.req_id_to_index,
            )
        # SUBTRACTED: prev_num_draft_tokens.np.fill(0)（L1352-L1353——
        #   delete[4]：async spec decode 乐观记账链）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1355-L1361 老请求循环头
        #   与字段解包（delete[4] ⚠️ 明示保留——非连续区间）
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # SUBTRACTED: prev_num_draft_len 乐观记账（L1363-L1403——delete[4]：
            #   async spec decode 链，无 spec 时 prev_num_draft_len 恒 0、
            #   条件恒 False）。
            # Update the cached states.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1405-L1406 num_
            #   computed_tokens 就地覆盖
            req_state.num_computed_tokens = num_computed_tokens

            # SUBTRACTED: PP 非末 rank 回填与 KV-load/乐观回滚对齐（L1408-
            #   L1439——delete[7]：if not is_last_rank 整支连同 elif 对齐臂
            #   一并删，删 if 留 elif 是语法错误；单机单卡 is_last_rank=True
            #   恒不进）。
            # Update the block IDs.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1441-L1452 块号差量：
            #   常规 append vs resumed 整体替换（协议注释 L118-L121 的消费者）
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1454-L1469 不在批次内
            #   的入 reqs_to_add 等落位
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1459-L1463 异步
                #   调度恢复请求的 output_token_ids 重建（ch12 已立、ch18 消费）
                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                # Track resumed requests for ngram_gpu full tensor copy
                if is_ngram_gpu:
                    ngram_gpu_new_reqs.append(req_state)
                continue

            # Update the persistent batch.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1471-L1474 批次列更新
            #   与块表行追加
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # SUBTRACTED: 非末 PP rank 补写 token_ids_cpu 增量（L1476-L1499
            #   ——delete[7]：自足 if 块；末 rank 由采样直接写——即 m09 写回）。
            # Add spec_token_ids to token_ids_cpu.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1501-L1502
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
            # SUBTRACTED: ngram 裁剪后的 draft 计数恢复段（L1503-L1507——
            #   delete[4]：恢复段的读端已删，删后无人读）。

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1509-L1513 收尾落位
        #   （add_request 内 pop_removed 复用最小空 slot）
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1515-L1520 四连收尾
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        # SUBTRACTED: update_ngram_gpu_tensors_incremental（L1522-L1532——
        #   delete[5]：drafter 侧 GPU 张量镜像增量维护）。
        # SUBTRACTED: deferred_spec_decode_corrections 纠偏闭包（L1534-L1566
        #   ——delete[4]：if 块、else 行与 return None 一并删——_update_states
        #   隐式返回 None，行为等价；L1263 的空列表初始化保留为死存）。

    # _update_states_after_model_execute —— 签名保留（sample_tokens L4591
    #   调用位）；方法体（mamba align GPU 后处理 + num_accepted 记账）归
    #   ch14/ch33 域
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1568-L1571 _update_states_after_model_execute
    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Update the cached states after model execution.
        # SUBTRACTED: 方法体（L1572-L1624——mamba 对齐 GPU 后处理与
        #   num_accepted_tokens 事件记账，ch14/ch33 域；精简配置下
        #   spec None → 真实首行守卫 L1579-L1580 即返回）。
        """
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1626-L1659 _update_streaming_
    #   request —— streaming 同 req_id 复用分支
    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`.

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.

        NOTE: prompt_token_ids includes intermediate output tokens - tokens
        previously generated but now are input context (part of the prompt).
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1637-L1638
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1640-L1650 全量快照刷新
        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        self.late_interaction_runner.register_request(req_id, req_state.pooling_params)
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1652-L1654
        req_state.output_token_ids.clear()

        # SUBTRACTED: M-RoPE 位置重算守卫（L1656-L1657——delete[0]）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1659
        return req_state

    # SUBTRACTED: _init_mrope_positions/_init_xdrope_positions（L1661-L1704）
    #   与 _calc_mrope_positions/_calc_xdrope_positions（L2755-L2850）——
    #   delete[0]：四个方法定义整删（Qwen2-VL/HunYuan-VL 多模态，纯文本恒
    #   False）；_extract_mm_kwargs/_dummy_mm_kwargs（L1706-L1741——mm 域）。

    # ------------------------------------------------------------------ #
    # 扁平收集的算术件（m06）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1743-L1767 _get_cumsum_and_
    #   arange —— CU 偏移与请求内 arange 的原地展开（预分配缓冲上写）
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        arange_out: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> np.ndarray:
        """Get the cumulative sum and batched arange of the given array.
        E.g., [2, 5, 3] -> [2, 7, 10], arange written to
        arange_out[:10] as [0, 1, 0, 1, 2, 3, 4, 0, 1, 2].
        Equivalent to but faster than:
        np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1755-L1757
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1758-L1759
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1760-L1765
        np.subtract(
            self.arange_np[:total_num_tokens],
            cumsums_offsets,
            out=arange_out[:total_num_tokens],
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1767
        return cu_num_tokens

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1769-L1782 _compute_prev_
    #   positions —— cur → prev 槽位映射（-1=新请求）
    def _compute_prev_positions(self, num_reqs: int) -> None:
        """Build prev_positions mapping: current pos -> prev pos (-1 if new).

        Populates self.prev_positions.np[:num_reqs] with the mapping.
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1774-L1782
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        prev_positions = self.prev_positions.np[:num_reqs]

        if not prev_req_id_to_index:
            prev_positions.fill(-1)
            return

        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            prev_positions[i] = prev_req_id_to_index.get(req_id, -1)

    # ------------------------------------------------------------------ #
    # 本章主方法②：固定地址前缀上载与异步 prev_sampled_token_ids 消费
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1784-L1913 _prepare_input_ids
    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        num_reqs: int,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids.

        Uses self.prev_positions[:num_reqs] which maps current pos -> prev pos
        (-1 for new requests).
        """

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1801-L1803 正常拍
        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            # SUBTRACTED: enable_prompt_embeds 的 inputs_embeds/is_token_ids
            #   补拷（L1804-L1806——delete[1]）。
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1809-L1820 索引细算
        prev_positions = self.prev_positions.np[:num_reqs]
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        prev_indices: list[int] = []
        common_indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1822-L1849 逐请求映射
        for cur_index in range(num_reqs):
            prev_index = prev_positions[cur_index]
            if prev_index < 0:
                continue
            prev_indices.append(prev_index)
            req_id = self.input_batch.req_ids[cur_index]
            # We need to compute the flattened input_ids index of the
            # last token in each common request.
            draft_len = len(scheduled_spec_tokens.get(req_id, ()))
            total_num_spec_tokens += draft_len
            flattened_index = cu_num_tokens[cur_index].item() - 1
            # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
            # sample_flattened_indices = [0, 2, 5]
            # spec_flattened_indices = [1,   3, 4,    6, 7]
            sample_flattened_indices.append(flattened_index - draft_len)
            spec_flattened_indices.extend(
                range(flattened_index - draft_len + 1, flattened_index + 1)
            )
            start = prev_index * self.prev_num_spec_tokens
            # prev_draft_token_indices is used to find which draft_tokens_id
            # should be copied to input_ids
            # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
            # flatten draft_tokens_id [1,2,3,4,5,6]
            # draft_len of each request [1, 2, 1]
            # then prev_draft_token_indices is [0,   2, 3,   4]
            prev_draft_token_indices.extend(range(start, start + draft_len))
            common_indices_match &= prev_index == flattened_index
            max_flattened_index = max(max_flattened_index, flattened_index)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1851-L1852
        num_common_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        # SUBTRACTED: enable_prompt_embeds 的 is_token_ids 补拷守卫（L1853-
        #   L1857——delete[1]）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1858-L1863
        if num_common_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # we need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            # SUBTRACTED: enable_prompt_embeds 的 inputs_embeds 补拷（L1862-
            #   L1863——delete[1]）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1864-L1867
        if num_common_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1868-L1877 common-case
        #   优化：批次未变未重排 → 单 slice 直拷
        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1878-L1891 scatter 兜底
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1893-L1913 spec draft 的
        #   二段 scatter（守卫恒早退：_draft_token_ids 在精简配置恒 None——
        #   drafter 装配归 ch33；守卫块本身未列删除，原样保留）
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=PIN_MEMORY
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    # SUBTRACTED: _get_encoder_seq_lens（L1915-L1958——encoder-decoder 交叉
    #   注意力域，消费点 _build_attention_metadata 在 ch21/22 切面外）。

    # ------------------------------------------------------------------ #
    # 本章主方法③：收集与装配（m06/m07/m08）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1960-L2282 _prepare_inputs
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
    ]:
        """
        Returns:
            tuple[logits_indices, spec_decode_metadata]
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1972-L1975
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1977-L1979 先行拷块表
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1981-L1983 np.repeat 展开
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1985-L1989
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )

        # Get positions.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1991-L1995 positions =
        #   computed[req] + 请求内偏移
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )

        # SUBTRACTED: M-RoPE/XD-RoPE position 计算（L1997-L2005——delete[0]：
        #   独立 if 语句整删）。
        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2007-L2014 二维坐标编一维
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2016-L2024 一次收齐
        #   input_ids
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        # SUBTRACTED: prompt_embeds 的 is_token_ids 收集与逐请求 embeds 填充
        #   （L2025-L2070——delete[1]）。

        # Prepare the attention metadata.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2072-L2079 query_start_loc
        #   CU 偏移 + 尾部 pad 非递减
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        # Compute optimistic seq_lens (assumes all draft tokens from previous
        # iteration accepted). Store in optimistic_seq_lens_cpu for use by
        # _build_attention_metadata (max_seq_len) and discard_request_mask.
        # seq_lens (GPU) will be computed later using the same optimistic values.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2085-L2090 乐观序列长
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # Build prev_positions mapping: current pos -> prev pos (-1 if new).
        # Used for gathering from previous iteration's GPU tensors.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2092-L2095
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        self._compute_prev_positions(num_reqs)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2097-L2098
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returning
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2100-L2105 discard mask
        #   （【不删】delete[4] 明示无条件默认路径）
        self.discard_request_mask.np[:num_reqs] = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        )
        self.discard_request_mask.copy_to_gpu(num_reqs)

        # SUBTRACTED: num_accepted_tokens 事件门控同步（L2107-L2137——
        #   delete[4]：num_accepted_tokens_event 仅 num_spec_tokens>0 时诞生，
        #   精简配置恒走默认填充——保留体去缩进降为无条件）：
        # Default to 1; [update_num_computed_tokens_for_batch_change below
        # corrects rows that had drafts from valid_sampled_token_count.]
        #   （指向被删方法的两行悬空注释随删）
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2140-L2141
        self.num_accepted_tokens.np.fill(1)
        self.num_accepted_tokens.gpu.fill_(1)

        # SUBTRACTED: mamba/GDN spec 预处理（L2143-L2150——delete[6]③：
        #   mamba_prev_last_scheduled_idx 门控恒 None）。
        # SUBTRACTED: update_num_computed_tokens_for_batch_change GPU 修正
        #   （L2152-L2174——delete[4]：async spec 乐观纠偏的 GPU 侧；
        #   else 体默认拷贝去缩进降为无条件）：
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2175-L2178 num_computed_
        #   tokens 的 CPU→GPU 镜像（positions/seq_lens 的数据源，删了全错）
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2180-L2187 req_indices/
        #   query_pos/num_scheduled_tokens 持久缓冲前缀上载
        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2188-L2195 GPU 端算最终
        #   positions/seq_lens
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2197-L2201 slot 数学入口
        #   （position→物理 slot 的 Triton 数学 → ch22）
        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # Copy the tensors to the GPU.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2203-L2209 前缀上载
        self._prepare_input_ids(
            scheduler_output,
            num_reqs,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        # SUBTRACTED: mrope/xdrope GPU 拷贝与异步漂移修正（L2211-L2230——
        #   delete[0]：if/elif 两臂无 else，整删）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2232-L2241 非 spec 主支
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        # SUBTRACTED: spec else 支（L2242-L2267——delete[6]④：num_draft/
        #   num_decode_draft_tokens 填充与 _calc_spec_decode_metadata，
        #   ch33 域；两个返回值都在主支定义）。
        # SUBTRACTED: LoRA 热切换块（L2269-L2277——delete[3]：set_active_loras
        #   调用点；make_lora_inputs 读端随 gpu_input_batch 侧同条删净）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2279-L2282
        return (
            logits_indices,
            spec_decode_metadata,
        )

    # SUBTRACTED: _build_attention_metadata（L2284-L2750——ch21/22 注意力
    #   metadata 构建域，含 dcp 消费 L2451-L2462 与 cascade 判定——后者由
    #   注意力后端 builder 自带能力决定，不在 _prepare_inputs 主干）；
    #   _calc_spec_decode_metadata（L2851-L2900——ch33 域）；
    #   _prepare_kv_sharing_fast_prefill（L2926 起——delete[6]⑤）。

    # SUBTRACTED: _preprocess（L3545-L3690——mm/embeds 前处理与 PP 中间张量
    #   对齐，ch17/ch34 域；mrope/xdrope 两臂 L3657-L3660 随 delete[0] 删）。

    # ------------------------------------------------------------------ #
    # 采样与写回（m09）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3692-L3721 _sample —— 采样
    #   调用位（采样栈归 ch08；本章测试全 greedy，走 sampler 的 greedy 支）
    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3698
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3701
        self.input_batch.update_async_output_token_ids()
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3702-L3706 非 spec 直采
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        # SUBTRACTED: spec 的 rejection sampler 支（L3708-L3721——ch33 域：
        #   drafter 装配删除后 self.rejection_sampler 不存在，该支在精简
        #   配置不可达）。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3723-L3862 _bookkeeping_sync
    #   —— 写回闭环（采样 token 写 token_ids_cpu 行、output_token_ids 增长；
    #   异步时留 GPU + CPU 行写 -1 占位）
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3739-L3741 NaN 观测面
        #   （envs 开关 HOST SEAM 恒 False）
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3743-L3750 discard 行定位
        #   与 generator 回退
        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3752-L3755
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3757-L3760
        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        logprobs_lists = None
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3762 同步分支
        if not self.use_async_scheduling:
            # SUBTRACTED: routed experts 的先行 D2H（L3763-L3776——MoE 域）。
            # Get the valid generated tokens.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3778-L3788
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()

                if logprobs_tensors is not None:
                    logprobs_lists = logprobs_tensors.tolists()
            # SUBTRACTED: spec 的 RejectionSampler.parse_output else 支
            #   （L3789-L3796——ch33 域；精简配置 max_gen_len 恒 1）。
        else:
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3797-L3800 invalid 行
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3802-L3808 影子①：
            #   真 token 的 GPU 住所
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3809-L3813 影子②：
            #   上拍槽位表快照
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        # SUBTRACTED: use_pp 的回传分支（L3817-L3819 注释中 PP 例外段——
        #   delete[7] PP 面；写回循环主体【不删】——持久批次自闭环的本体）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3820-L3846 写回循环
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3822-L3825 async 分支：
            #   token_ids_cpu 行只写占位 -1
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3832-L3838
            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3840-L3842 账本照走
            #   （token_ids_cpu 行写 + 列前移）
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3844-L3846 req_state
            #   输出面增长
            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # SUBTRACTED: prompt logprobs 取用（L3848-L3852——ch8 域：_get_prompt_
        #   logprobs_dict 依赖 hidden_states（前向深水），本章恒 None）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3854-L3862 返回七元组
        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            None,  # prompt_logprobs_dict（ch8 域——取用支已删）
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3864-L3877 synchronize_input_
    #   prep —— pinned buffer 防踩协议（m13：等上拍 CPU 张量用完再复用）
    @contextmanager
    def synchronize_input_prep(self):
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3866-L3868
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3870-L3877
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    # SUBTRACTED: _model_forward（L3879-L3909——前向本体，ch17 域；ENGINE
    #   SEAM 脚本化 logits 承载同一位形）。

    # SUBTRACTED: _is_uniform_decode/_determine_batch_execution_and_padding/
    #   _get_slot_mappings 等（L3911-L4165——ch19 捕获/padding 域）。

    # ------------------------------------------------------------------ #
    # 两段式契约·上半段（ch9/ch17 立过；本章保可变性裁决+差量调和编排）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4166 execute_model
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4171-L4175 双向断言·入口
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # SUBTRACTED: routed capturer 清理（L4177-L4178——MoE 域）。

        # If ngram_gpu is used, we need to copy the scheduler_output to avoid
        # the modification has influence on the scheduler_output in engine core process.
        # The replace is much faster than deepcopy.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4180-L4195 可变性裁决：
        #   ngram-GPU 开启时 replace() 浅拷贝两个 dict（m10——delete[5] 明示
        #   保留；精简配置 speculative_config=None 恒不触发，守卫删了反而
        #   丢协议）
        if (
            self.speculative_config is not None
            and self.speculative_config.use_ngram_gpu()
        ):
            num_scheduled_tokens_copy = scheduler_output.num_scheduled_tokens.copy()
            spec_decode_tokens_copy = (
                scheduler_output.scheduled_spec_decode_tokens.copy()
            )
            scheduler_output = replace(
                scheduler_output,
                num_scheduled_tokens=num_scheduled_tokens_copy,
                scheduled_spec_decode_tokens=spec_decode_tokens_copy,
            )

        # SUBTRACTED: kv connector 抢占处理（L4197-L4200——ch16 域）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4202-L4206
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("gpu_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            # Update persistent batch states.
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4207-L4208 差量调和
            deferred_state_corrections_fn = self._update_states(scheduler_output)

            # SUBTRACTED: ec transfer 早退（L4210-L4216——mm connector 域）。

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4218-L4233 空批早退
            if not num_scheduled_tokens:
                # SUBTRACTED: DP+external_launcher 的 dummy run 与 kv connector
                #   分支（L4219-L4231/L4234——DP/ch16 域）。
                # Return empty ModelRunnerOutput if no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4236-L4241 kv_sharing_
            #   fast_prefill 断言（delete[6]⑤ 未列此行——守卫保留；精简配置
            #   开关恒 False，断言空转）
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4243-L4248
            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4250-L4253 本章主方法
            #   调用位：收集与装配
            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            # SUBTRACTED: cascade/DBO/cudagraph 派发/ubatch/slot_mappings/
            #   _build_attention_metadata/_preprocess/set_forward_context 前向
            #   段（L4255-L4456——ch19 编译捕获 + ch21/22 attention metadata +
            #   ch34 connector + ch17 前向深水；回放前 DEBUG 断言 data_ptr 一致
            #   在 vllm/compilation/cuda_graph.py:L346-L355——m14 叙事锚，
            #   编译与捕获全章归 ch19）。
            # ENGINE SEAM（ch17 边界，ch12 同款）：脚本化 logits 行——每请求
            #   一行（真实 = sample_hidden_states = hidden_states[logits_
            #   indices]; logits = model.compute_logits(...)，L4484-L4485）。
            logits = self._seam_model_forward(logits_indices)
            hidden_states = None  # ENGINE SEAM（前向深水 ch17）
            sample_hidden_states = None  # ENGINE SEAM
            aux_hidden_states = None  # ENGINE SEAM
            ec_connector_output = None  # ENGINE SEAM（mm connector 域）
            cudagraph_stats = None  # ENGINE SEAM（ch19 观测面）
            slot_mappings = None  # ENGINE SEAM（ch22）
            kv_connector_output = None  # ENGINE SEAM（ch16）

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4516-L4527 打包暂存
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            None,  # spec_decode_common_attn_metadata（ch33 域）
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4528
        self.kv_connector_output = kv_connector_output

        # Now the batch has been launched we can wait for corrections from the
        # previous model forward without breaking async scheduling.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4530-L4533 纠偏回调位
        #   （_update_states 删纠偏闭包后恒 None——行为等价，调用位保留存证）
        if deferred_state_corrections_fn:
            deferred_state_corrections_fn()

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4535
        return None

    # ------------------------------------------------------------------ #
    # 两段式契约·下半段：sample_tokens 的 bookkeeping 切面（m09 发生地）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4552-L4555 sample_tokens
    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        # SUBTRACTED: 空槽早退（L4556-L4564——kv connector 传递面 + PP 回传，
        #   ch16/ch34 域；本章 execute_model 后必有暂存态）。

        # Unpack ephemeral state.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4566-L4578 解包
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4579-L4580 解包即清
        self.execute_model_state = None

        # SUBTRACTED: 结构化输出 bitmask 应用（L4582-L4586——ch30 域；
        #   本章 grammar_output 恒 None）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4588-L4589 采样调用位
        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4591-L4593 模型执行后的
        #   批记账调用位（方法体归 ch14/ch33 域，保留调用面）
        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        # SUBTRACTED: PP 广播 sampled token ids（L4594-L4602——delete[7] PP 面；
        #   _pp_broadcast/_pp_receive 两方法 L4842-L4887 随删，ch34 域）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4604-L4609 prev 缓存清点
        #   （spec 的 _draft_* 清理与 GPU token 计数随 drafter 域删；保
        #   prev_sampled_token_ids=None——上一拍影子不跨拍泄漏，
        #   _bookkeeping_sync 随后缓存新采样）
        self._draft_token_ids = None
        self._draft_probs = None
        self._draft_prob_req_ids = None
        self._draft_token_req_ids = None
        self.valid_sampled_token_count_gpu = None
        self.input_batch.prev_sampled_token_ids = None

        # SUBTRACTED: propose_draft_token_ids 闭包与 spec drafter 决策区
        #   （L4611-L4751——ch33 域：speculative_config=None 恒不进）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4716-L4731 bookkeeping
        #   调用位（m09 写回闭环的入口）
        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
            )

        # SUBTRACTED: draft_after_bookkeeping（L4733-L4751——ch33 域）。
        # SUBTRACTED: kv connector finalize / eplb_step（L4753-L4760——
        #   ch16/MoE 域）。

        # self.kv_connector_output may be modified during drafting
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4762-L4764
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4766-L4780 ModelRunnerOutput
        #   组装
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
                cudagraph_stats=cudagraph_stats,
                routed_experts=None,
            )

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4782-L4792 同步版直返
        # SUBTRACTED: routed experts 的同步回填（L4783-L4791——MoE 域）。
        if not self.use_async_scheduling:
            return output

        # SUBTRACTED: AsyncGPUModelRunnerOutput 异步包裹与 set_async_sampled_
        #   token_ids（L4794-L4840——ch12 已全文立，逐字在 ch12 精简版
        #   gpu_model_runner.py；本章保 _bookkeeping_sync 的 async 分支本体
        #   ——真 token 留 GPU/快照 prev_req_id_to_index 都在那里发生）。
        # ENGINE SEAM（ch12 边界）：异步路径的 D2H 重叠包裹不在本章面内，
        #   直接返回同步形态（sampled_token_ids=[]，与 _bookkeeping_sync
        #   async 分支的语义一致）。
        return output

    # ------------------------------------------------------------------ #
    # ENGINE SEAM test hooks（ch17 边界：前向本体不在本章切面）
    # ------------------------------------------------------------------ #
    # ENGINE SEAM test hook：脚本化 logits 行注入（每步一个 {req_id: 行}
    # 字典；真实前向在 GPU 上算 compute_logits(hidden_states[logits_indices])）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4458-L4514 前向/后处理段 — SEAM
    def enqueue_logits(self, steps: list) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4458-L4514 — SEAM
        for step in steps:
            self._scripted_logits.append(
                {rid: list(row) for rid, row in dict(step).items()}
            )

    # ENGINE SEAM：按 input_batch.req_ids 序取脚本行，拼 [num_reqs, vocab]
    # 的 logits 张量（真实 = hidden_states[logits_indices] 过 lm_head）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4484-L4485 — SEAM
    def _seam_model_forward(self, logits_indices) -> torch.Tensor:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4484-L4485 — SEAM
        rows = self._pop_scripted_rows()
        ordered = []
        for rid in self.input_batch.req_ids:
            if rid not in rows:
                raise RuntimeError(
                    f"no scripted logits row for request {rid!r} "
                    "(ch17 boundary: script each request's row)"
                )
            ordered.append(rows[rid])
        return torch.tensor(ordered, dtype=torch.float32)

    # ENGINE SEAM：弹出一步的脚本行（真实引擎在此处等 GPU）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4432-L4456 模型前向 — SEAM
    def _pop_scripted_rows(self) -> dict:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4432-L4456 — SEAM
        if not self._scripted_logits:
            raise RuntimeError(
                "scripted forward ran dry (ch17 boundary: the real "
                "engine waits on the GPU here)"
            )
        return self._scripted_logits.popleft()

    # ------------------------------------------------------------------ #
    # D2H 落地（同步分支 _to_list）
    # ------------------------------------------------------------------ #
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7839-L7851 _to_list —— 同步
    #   分支的 D2H（事件同步避免 cudaStreamSynchronize 阻塞其他拷贝流；
    #   HOST SEAM：HostEvent 即刻完成）
    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7847-L7851
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()

    # SUBTRACTED: 其余执行/装配域（load_model/get_kv_cache_spec/
    #   initialize_kv_cache/profile_run/capture_model/_dummy_run/take_draft_
    #   token_ids/gpu_worker 启动序列——ch14/ch17/ch19/ch33 域；_kv_cache_
    #   spec_attn_group_iterator（L7358 起）供 _init_kv_zero_meta 引用，
    #   该方法仅 gpu_worker 启动期调用，精简版不提供实现）。
