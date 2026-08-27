# SOURCE: vllm/v1/worker/gpu_model_runner.py
# GPUModelRunner 的**账本切面**（m2/m3/m9/m12 站 4/12 的 runner 半边）：
#   profile_run——max_num_tokens 假数据跑真前向 + 采样器，测激活峰值；
#   _init_minimal_kv_cache_for_profiling——CUDA graph 估计前先建最小 KV
#   池（临时 num_gpu_blocks_override = min_blocks——账本机器的复用）；
#   profile_cudagraph_memory——图池估计（first_capture + per-graph×(n−1)、
#   跨 mode 取 max 防重叠计账）；
#   get_kv_cache_spec——遍历 attention 层收 KVCacheSpec（每层自报形状）；
#   initialize_kv_cache(_tensors)——按 config 分配张量绑到层 + kernel 块
#   细分（256→4×64 例）。
# ENGINE SEAM（站点抽块纪律——抽出而非改写，控制流逐字；真实类为千行
#   执行器，ch09/17 全文）：_dummy_run/_dummy_sampler_run/_sync_device/
#   _warmup_and_capture/cudagraph_dispatcher/attn_layers（前向与捕获机器）；
#   测试以 FakeRunner/FakeAttn 注入同契约位。
# HOST SEAM：设备读数（torch.accelerator）host 上经 monkeypatch 注入
#   （见 tests）；容器内真跑。_reshape 的张量形状仲裁（attn_backend.
#   get_kv_cache_shape）→ ch21（ch13 同款切面：块最外层说明性布局）。
# SUBTRACTED（dossier.delete 批准项的落点 + 邻章边界）：
#   第 1 条 mm：profile_run 的 encoder/encoder-cache 段（L6434-L6490）与
#     supports_mm_inputs 装配；
#   第 2 条 startup_plan：maybe_apply/save_startup_plan 调用位；
#   第 3 条 reserve_mm_ipc_gpu_memory 包装；
#   第 4 条 packed/layer_packing 布局（L7312-L7336 的 packed_backing、
#     L7383-L7404 的 layer_packing——重叠别名布局 → 正文 why 注点名）；
#   第 5 条 encoder-only/mamba ssu 装配（L7638/L7641-L7643）；
#   第 9 条 metrics/编译计数观测；
#   捕获机器内景（_warmup_and_capture 的真身、graph_pool 调换、ROCm 流
#     选择 L6683-L6777 → ch19——采样循环骨架保留：mem_before/after 差值）；
#   encoder_cudagraph_manager（mm 图池，随第 1 条）；
#   bind_kv_cache 前向上下文绑定位（L7588-L7593 → ch21）、shared_kv_cache_
#   layers 跨层共享（ch13 边界）、uniform_kv_caches 优化路径（ch21 后端）；
#   update_max_model_len 的 runner 缓存面（L645-L646 由 worker 侧直供）。
import gc
from dataclasses import replace

import torch

from .kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
)
from .kv_cache_spec_registry import KVCacheSpecRegistry
from .kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
)
from .worker_utils import prepare_kernel_block_sizes


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2490 GPUModelRunner（账本切面
#   ——真实类为千行执行器，ch09/17 全文）
class GPUModelRunner:
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L~2492 __init__（切面装配）
    def __init__(
        self,
        vllm_config,
        device: torch.device,
    ) -> None:
        # SUBTRACTED: 模型/采样/cudagraph/spec 装配面（L2492-L2670——ch17）。
        self.vllm_config = vllm_config
        self.device = device
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6493-L6494（profile_run
        #   的假数据规模 = max_num_tokens）
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        # SUBTRACTED: supports_mm_inputs/is_pooling_model 的真实装配
        #   （L6435/L6497——第 1 条 mm / pooling → ch02；账位 False）
        self.is_pooling_model = False
        self.supports_mm_inputs = False
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6518-L6521（min_blocks
        #   的两个输入：max_num_reqs 与 max_cudagraph_capture_size——真实
        #   各自的装配链（scheduler/compilation config）→ ch03/17；切面
        #   getattr 账位，测试直供）
        self.max_num_reqs = getattr(vllm_config.scheduler_config, "max_num_seqs", 1)
        self.compilation_config = getattr(vllm_config, "compilation_config", None)
        # ENGINE SEAM：attn_layers——真实经 get_layers_from_vllm_config 从
        #   static forward context 收（L7812，config/vllm.py:L2454）；切面
        #   由装配方注入 {layer_name: attn_module}。attn_module 契约位：
        #   get_kv_cache_spec(vllm_config) -> KVCacheSpec、get_attn_backend()
        #   .indexes_kv_by_block_stride() -> bool。
        self.attn_layers: dict = {}
        # ENGINE SEAM：cudagraph_dispatcher（捕获形状分派，→ ch19）——契约位
        #   get_capture_descs() -> [(CUDAGraphMode, [desc, ...]), ...]，
        #   desc.num_tokens 可读。
        self.cudagraph_dispatcher = None
        # ENGINE SEAM：attn_groups（注意力后端分组，→ ch21）——契约位
        #   list[list[group]]，group.backend.get_supported_kernel_block_sizes()。
        self.attn_groups: list[list] = []
        self.runner_only_attn_layers: set[str] = set()
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_cache_config = None

    # ------------------------------------------------------------------ #
    # m2 激活峰测量
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6433 profile_run
    def profile_run(self) -> None:
        # SUBTRACTED: multimodal encoder & encoder cache 段（L6434-L6490
        #   ——dossier.delete 第 1 条：纯文本主线不触发多模态；删后
        #   profile_run = dummy 前向 + 采样器，控制流不变）。
        # Add `is_profile` here to pre-allocate communication buffers
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6492-L6495（max_num_
        #   tokens 假数据跑真前向）
        hidden_states, last_hidden_states = self._dummy_run(
            self.max_num_tokens, is_profile=True
        )
        # SUBTRACTED: PP 非末段分支（L6496/L6501-L6502——单卡恒末段）与
        #   pooling 分支（L6497-L6498——ch02）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6500（末段跑采样器——
        #   logits/采样缓冲也进峰值账）
        output = self._dummy_sampler_run(last_hidden_states)
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6503-L6506
        self._sync_device()
        del hidden_states, output
        gc.collect()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5830 _dummy_run（ENGINE
    #   SEAM——真实为 dummy 数据组批 + 模型 forward + 各后端预分配，ch09/17
    #   全文；本章只消费它的调用契约：max_num_tokens、is_profile=True）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5830
    def _dummy_run(self, num_tokens: int, is_profile: bool = False):
        # SUBTRACTED: dummy 输入组批/模型执行/输出收集（L5830-L6230——
        #   ch09/17 的执行机器）。HOST SEAM：切面 no-op（返回 None 账位）。
        return None, None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6242 _dummy_sampler_run
    #   （ENGINE SEAM——真实预分配 logits/采样缓冲并跑采样器）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6242
    def _dummy_sampler_run(self, last_hidden_states):
        # SUBTRACTED: logits 张量分配与采样 kernel（L6242-L6300——ch17）。
        #   HOST SEAM：切面 no-op。
        return None

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1166 _sync_device
    def _sync_device(self) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1167（HOST SEAM：host
        #   上无 CUDA 流可同步）
        pass

    # ------------------------------------------------------------------ #
    # m3 CUDA graph 内存估计
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6508 _init_minimal_kv_
    #   cache_for_profiling（dossier m3 锚点：估计前先建最小 KV 池——临时
    #   num_gpu_blocks_override = min_blocks，账本机器的复用）
    def _init_minimal_kv_cache_for_profiling(self) -> None:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6514-L6516
        kv_cache_spec = self.get_kv_cache_spec()
        KVCacheSpecRegistry.check_kv_cache_spec_registry(kv_cache_spec)
        kv_cache_groups = get_kv_cache_groups(self.vllm_config, kv_cache_spec)
        # the minimum number of blocks required is 1 block *per sequence*
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6517-L6521
        min_blocks = (
            min(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
            or 1
        )

        # Temporarily change num_gpu_blocks_override to allocate a minimal KV cache
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6523-L6529
        saved_override = self.vllm_config.cache_config.num_gpu_blocks_override
        self.vllm_config.cache_config.num_gpu_blocks_override = min_blocks
        minimal_config = get_kv_cache_config_from_groups(
            self.vllm_config, kv_cache_groups, available_memory=0
        )
        self.vllm_config.cache_config.num_gpu_blocks_override = saved_override

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6531-L6534
        self.initialize_kv_cache(minimal_config, is_profiling=True)
        self.vllm_config.cache_config.num_gpu_blocks = minimal_config.num_blocks

    # SUBTRACTED: debug 日志（L6536-L6537——观测面）。

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6645 profile_cudagraph_memory
    def profile_cudagraph_memory(self) -> int:
        # SUBTRACTED: set_current_vllm_config 上下文（L6646——ch03 装配域）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6647
        self._init_minimal_kv_cache_for_profiling()

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6649（编译计数——观测
        #   面，第 9 条删；saved/restore 语义保留于 finally 的捕获清理）
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6651（capture_descs
        #   ——ENGINE SEAM：cudagraph_dispatcher.get_capture_descs()，捕获
        #   形状分派 → ch19）
        capture_descs = self.cudagraph_dispatcher.get_capture_descs()
        # SUBTRACTED: encoder_cudagraph_manager（L6652-L6654、L6657-L6661、
        #   L6675-L6679、L6766-L6777——mm 图池，dossier.delete 第 1 条）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6656-L6666（图数清点；
        #   0 图早退）
        decoder_graphs = sum(len(descs) for _, descs in capture_descs)
        total_graphs = decoder_graphs
        if total_graphs == 0:
            return 0

        # SUBTRACTED: graph_groups 日志（L6668-L6681——观测面）；临时图池
        #   调换与 ROCm 流选择（L6683-L6713——捕获机器内景 → ch19）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6694-L6695（估计账本）
        shared_memory_estimate = {}
        per_graph_estimate = {}
        # SUBTRACTED: encoder_memory_estimate（L6696——第 1 条）。

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6717-L6718（捕获开关
        #   与 GC 冻结——ch19 的捕获机器；切面保留 try/finally 清理骨架）
        try:
            # SUBTRACTED: graph_capture 上下文与流编排（L6719-L6724——ch19）。
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6726-L6728（每 mode
            #   取前 2 个形状采样）
            for mode, descs in capture_descs:
                profile_descs = descs[:2]
                mem_samples: list[int] = []

                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6730-L6746 采样
                #   循环（mem_before/after 差值 = 一张图的增量；真实捕获在
                #   _warmup_and_capture——ENGINE SEAM → ch19）
                for i, desc in enumerate(profile_descs):
                    mem_before = torch.accelerator.get_memory_info()[0]
                    self._warmup_and_capture(
                        desc,
                        cudagraph_runtime_mode=mode,
                        profile_seq_lens=(
                            min(
                                self.max_model_len,
                                self.max_num_tokens // desc.num_tokens,
                            )
                            if mode.name == "FULL" and i == 0
                            else None
                        ),
                    )
                    torch.accelerator.synchronize()
                    free_after = torch.accelerator.get_memory_info()[0]
                    mem_samples.append(mem_before - free_after)

                # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6748-L6755 估计
                #   算术：first_capture（共享池）+ per-graph×(n−1)；每图至少
                #   1 MiB driver 开销
                first_capture = mem_samples[0]
                per_graph = max(
                    mem_samples[1] if len(mem_samples) > 1 else 0, 1 << 20
                )

                shared_memory_estimate[mode] = first_capture
                per_graph_estimate[mode] = per_graph * (len(descs) - 1)

                # SUBTRACTED: debug 日志（L6757-L6764——观测面）。
        finally:
            # SUBTRACTED: 捕获清理机器（L6778-L6795——clear_all_graphs/
            #   池还原/键清理/LORA/编译计数还原，ch19）。
            pass

        # FULL and PIECEWISE graphs share the global pool at runtime and are
        # never replayed concurrently, so the pool overlays their memory.
        # Take the max to avoid double-counting the overlap.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6797-L6805（跨 mode 取
        #   max 防重复计账 + 求和）
        decoder_estimate = max(shared_memory_estimate.values(), default=0) + sum(
            per_graph_estimate.values()
        )
        # SUBTRACTED: encoder 叠加（L6803-L6805——第 1 条）。
        total_estimate = decoder_estimate

        # SUBTRACTED: 估计量日志（L6806-L6809——观测面）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6811
        return int(total_estimate)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6730 调用位的 _warmup_and_
    #   capture（ENGINE SEAM——真实为 warmup + cudagraph 捕获，L6813 起的
    #   捕获编排 → ch19；切面 no-op，host 测试经 fake desc 注入内存差值）
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L6732
    def _warmup_and_capture(self, desc, cudagraph_runtime_mode, profile_seq_lens=None):
        # SUBTRACTED: warmup/捕获机器（→ ch19）。HOST SEAM：切面 no-op——
        #   采样差值由测试经 torch.accelerator.get_memory_info 注入。
        return None

    # ------------------------------------------------------------------ #
    # 站 4 每层自报形状
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7800 get_kv_cache_spec
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        # SUBTRACTED: ec_transfer 消费端早退（L7808-L7809——ch16）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7810-L7812
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = self.attn_layers  # ENGINE SEAM: 真实 get_layers_from_vllm_config
        for layer_name, attn_module in attn_layers.items():
            # SUBTRACTED: kv_sharing_target_layer_name 跨层共享层跳过
            #   （L7814-L7825——ch13 边界：本章不做 KV 共享）。
            # Skip modules that don't need KV cache (eg encoder-only attention)
            # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7826-L7835
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                if isinstance(spec, AttentionSpec):
                    backend = attn_module.get_attn_backend()
                    # indexes_kv_by_block_stride() -> get_kv_cache_stride_order()
                    # -> get_kv_cache_layout() needs the current vLLM config.
                    # SUBTRACTED: set_current_vllm_config 上下文包装
                    #   （L7832——ch03 装配域）
                    indexes = backend.indexes_kv_by_block_stride()
                    spec = replace(spec, indexes_kv_by_block_stride=indexes)
                kv_cache_spec[layer_name] = spec

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7837
        return kv_cache_spec

    # ------------------------------------------------------------------ #
    # 站 12 worker 侧落地
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7541 initialize_kv_cache_tensors
    def initialize_kv_cache_tensors(
        self, kv_cache_config, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # SUBTRACTED: use_uniform_kv_caches 优化路径（L7556-L7569——ch21
        #   后端族的 uniform 分配）；真实走 general fallback。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7570-L7578
        # Fallback to the general case
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(
            kv_cache_raw_tensors, kernel_block_sizes
        )

        # SUBTRACTED: shared_kv_cache_layers 跨层共享映射（L7580-L7583——
        #   ch13 边界）；num_attn_module longcat 特判（L7585-L7587——单模型
        #   家族特路）；bind_kv_cache 前向上下文绑定（L7588-L7593——ch21）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7594
        return kv_caches

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7312 _allocate_kv_cache_
    #   tensors（GPU 物理池：每层一块原始 int8 缓冲）
    def _allocate_kv_cache_tensors(
        self, kv_cache_config
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
        # SUBTRACTED: packed_backing 别名分配（L7316、L7322-L7336——dossier.
        #   delete 第 4 条 packed 重叠布局）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7337-L7344
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(
                kv_cache_tensor.size, dtype=torch.int8, device=self.device
            )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7346-L7352（层集对账）
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

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7364 _reshape_kv_cache_tensors
    def _reshape_kv_cache_tensors(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # SUBTRACTED: layer_packing 重叠视图（L7383-L7389、L7401-L7404——
        #   第 4 条 packed）；MLA 压缩 shape_block_size（L7415-L7419——第 4
        #   条）；kv_cache_dtype skip-layers（L7421-L7432——ch27）；
        #   attn_backend.get_kv_cache_shape/get_kv_cache_stride_order 的形状
        #   仲裁（L7433-L7453——ch21；切面换用块最外层的说明性布局
        #   [kernel_num_blocks, 每块字节]——页字节数不变，ch13 同款切面）；
        #   MambaSpec reshape（L7455-L7468——mamba 状态形状 → 邻章）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7380-L7390（逐组逐层：
        #   组 id 越界 = 无 KV 的尾组，跳过）
        kv_caches: dict[str, torch.Tensor] = {}
        for attn_groups_per_kv_group, kernel_block_size in zip(
            self.attn_groups, kernel_block_sizes
        ):
            for group in attn_groups_per_kv_group:
                kv_cache_spec = group.kv_cache_spec
                kv_cache_group_id = group.kv_cache_group_id
                if kv_cache_group_id == len(kernel_block_sizes):
                    # There may be a last group for layers without kv cache.
                    continue
                kernel_block_size = kernel_block_sizes[kv_cache_group_id]
                for layer_name in group.layer_names:
                    if layer_name in self.runner_only_attn_layers:
                        continue
                    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7400、L7405-L7407
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7408-L7413
                    #   （kernel 细分乘子：256-token 块 × 后端只认 64 → 4×64）
                    if isinstance(kv_cache_spec, AttentionSpec):
                        num_blocks_per_kv_block = (
                            kv_cache_spec.block_size // kernel_block_size
                        )
                        kernel_num_blocks = num_blocks * num_blocks_per_kv_block
                        # HOST SEAM 说明性布局：块最外层（block_dim=0）、
                        # 每块一页（页字节数不变；真实主流后端为内容维
                        # 打包 [num_blocks, kv_heads, block_size, 2*head_size]
                        # ——形状仲裁 → ch21）
                        kv_caches[layer_name] = raw_tensor.view(
                            kernel_num_blocks, -1
                        )
        return kv_caches

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7624 initialize_kv_cache
    def initialize_kv_cache(
        self,
        kv_cache_config,
        is_profiling: bool = False,
    ) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7635-L7637
        from copy import deepcopy

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        # SUBTRACTED: _mamba_bufs/may_add_encoder_only_layers/maybe_add_kv_
        #   sharing_layers（L7637-L7639——第 5 条/ch13 边界）、
        #   initialize_attn_backend（L7640——ch21：attn_groups 为 ENGINE
        #   SEAM 装配）、initialize_mamba_ssu_backend（L7641-L7643——邻章）。
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7644-L7652 kernel 块细分
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        self._kernel_block_sizes = kernel_block_sizes

        # SUBTRACTED: initialize_metadata_builders（L7654-L7655——ch21）与
        #   may_reinitialize_input_batch（L7657-L7658——ch18）。
        # SOURCE: vllm/v1/worker/gpu_model_runner.py:L7659-L7661
        self.initialize_kv_cache_tensors(
            kv_cache_config, kernel_block_sizes
        )

        # SUBTRACTED: spec-decode/mamba buffer/事件收尾（L7663 起——各邻章）。
