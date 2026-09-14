# ch26《DeepSeek 索引器 NSA→DSA》impl-notes —— 只做减法精简版

对应真实源码 pin **vLLM v0.27.1（6e448d0ea）**，行号全部现核（2026-09-14，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。

运行：`cd instances/vllm/artifacts-v3/ch26-deepseek-indexer-nsa-dsa && python -m pytest tests/ -q`
→ **36 passed**（~4s）。纯 host 单元测试：真 torch/transformers（真
DeepseekV2Config、真 `torch.library` 注册的 sparse_attn_indexer 统一算子与
torch.ops._C 的 cooperative/persistent_topk 双核）；无 vllm 包安装、无 CUDA
上下文——CUDA/Triton kernel 面（DeepGEMM mqa_logits 族 / FlashMLA 稀疏核 /
top_k_per_row 族 / 融合量化核）以 HOST SEAM 镜像承载**精确数学**（等价性
测试直接数值对照测试自有的独立参考实现——Eq.(1) 打分 / 因果 top-k / 双源
softmax / 压缩窗 softmax 门控）；装配按真实注入位（`attention_config.backend`
显式名 → selector → registry——MLAAttention 的真实显式后端路径）。

**验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST
SEAM 例外见 §Seam 清单——每个 seam 行内标注真实源锚）。SUBTRACTED 标记逐条
挂 dossier.subtraction_plan.delete[0..5] 批准项编号（章界外域段另以
「→ chN」注记——ch18 起的切面惯例）。

**lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无
BLOCKING）**；must_keep 52 符号经 linter `over_subtraction` 项全数核在。

**锚点双置惯例**（ch22/ch25 同款）：每个 def/class 的 `# SOURCE:` 锚点在声明
上方（供读者）与 def/class 体内/紧邻上方复置一行——后者是
`lint_fidelity._spans_missing_source` 的跨度判据。

## 本章四幕 ↔ 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 同名 | **主文件 1**（站 8-11/m03/m05/m06/m09/m16）：op 本体（L295-L688 减法——delete[0] DCP、delete[1] XPU/ROCm、delete[2] FP4、delete[3] PCP、delete[4] dense-MHA 跳过、delete[5] 假跑）+ sparse_attn_indexer_fake + direct_register_custom_op（mutates_args）+ SparseAttnIndexer CustomOp + fused_indexer_q_rope_quant（HOST SEAM）|
| `vllm/v1/attention/backends/mla/indexer.py` | 同名 | **主文件 2**（站 5/m04）：split_indexer_prefill_chunks（L77-L123 逐字）+ 双后端族（V32 块 64/V4 块 256）+ get_max_prefill_buffer_size（魔数 40 注释原文）+ DeepseekV32IndexerMetadataBuilder（delete[0]①-⑤ 五小块）+ build_prefill_chunk_metadata（DCP 段删）+ _prepare_decode_tensors 逐字（flatten 均匀/变长 + native 2D）|
| `vllm/model_executor/models/deepseek_v2.py` | 同名 | **主文件 3**（站 1-4/7）：DeepseekV32IndexerCache（L616-L642 逐字）+ Indexer（L645-L819——__init__ 逐字 + forward else 分支逐字）+ _try_load_fp8_indexer_wk（L822-L871 逐字）+ DeepseekV2MLAAttention 装配段（skip_topk 三旋钮块 L1091-L1141 逐字）+ DeepseekV2Model 的 buffer 段（站 1）+ load_weights 的 indexer 切面 |
| `vllm/model_executor/layers/mla.py` | 同名 | **主文件 4**（站 6）：全文减 delete[4]（dense_mha 绑定）——**L205-L206 的 indexer 接线位逐字**（本章命脉：纯副作用、返回值无人接收） |
| `vllm/models/deepseek_v4/attention.py` | 同名 | **主文件 5**（站 10/13/m10-m12）：compress_ratios 逐层表 L207-L213 逐字 + compress_ratio==4 才建 indexer L276-L297 逐字 + DeepseekV4IndexerCache 逐字 + DeepseekV4Indexer（delete[2] fp4 分支/缓冲；//4 压缩坐标 + 短上下文全选 + 双流 join 逐字）+ _fill_short_context_topk_indices（kernel 垫片）|
| `vllm/models/deepseek_v4/compressor.py` | 同名 | **主文件 6**（站 13/m11）：CompressorStateCache（L155-L208 逐字——块共享注释原文）+ DeepseekCompressor（fused_wkv_wgate L284-L292 逐字 + save_partial_states 前置 + 融合尾步分派的 indexer triton 臂；head=512 cutedsl/two-stage → ch25/ch28）|
| `vllm/models/deepseek_v4/nvidia/flashmla.py` | 同名 | **主文件 7**（站 14/m13）：forward_mqa 切刀 + _forward_decode 逐字（C4A 换算/C128A 直算表/tile_sched 三型/一核双源调用面）+ _forward_prefill 逐字（双 gather + combine_topk_swa_indices 并集 + flash_mla_sparse_fwd）|
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | 同名 | **主文件 8**（站 12/m14）：656B/584B 布局 docstring 逐字 + FlashMLASparseBackend/Metadata 逐字 + get_prefill_workspace_size（魔数 5）+ FlashMLASparseImpl 三条 KV 路径（builder → ch21/ch25）|
| `vllm/v1/attention/backends/mla/sparse_utils.py` | 同名 | m07：triton_convert_req_index_to_global_index（HOST SEAM 精确数学：块表换算/-1 直通/越界守卫/prefill workspace 偏移/valid 计数；DCP 过滤器删——delete[0] 族）|
| `vllm/v1/attention/backends/mla/sparse_swa.py` | 同名 | m10/m13：DeepseekV4SWACache（L56-L104 逐字）+ 三类层分型常量 + DeepseekSparseSWAMetadata 消费字段 + get_prefill_chunk_plan 逐字（builder → ch25/ch21）|
| `vllm/v1/attention/backends/mla/compressor_utils.py` | 同名 | m11：get_compressed_slot_mapping（HOST SEAM：(pos+1)%ratio==0 → block_table 落压缩 slot）|
| `vllm/model_executor/layers/attention/sparse_mla_attention.py` | 同名 | m14：SparseMLACommonImpl——buffer 接管（indexer 自带 or 显式传参兜底，L483-L490 逐字）+ _slice/_remap/_project 逐字 + _run_masked_mha（位掩码 varlen；chunked-context 分支 → ch25 m08）|
| `vllm/model_executor/layers/attention/mla_attention.py` | 同名 | 站 6 插座侧：MLAAttention sparse 布线（extra_impl_args['topk_indices_buffer'] L482-L484 逐字 + 显式 attn_backend 注入位 + _canonicalize L331-L344 逐字 + get_kv_cache_spec L1140-L1152 逐字）；MQA 吸收腿/混批切刀 → ch25 |
| `vllm/models/deepseek_v4/common/ops/*.py` | 同名 | m11/m13/m15：save_partial_states（HOST SEAM 逐式）/ compress_norm_rope_store_triton（HOST SEAM——压缩→RMSNorm→RoPE→FP8 全式）/ fused_indexer_q（MXFP4_BLOCK_SIZE 逐字 + V4 fused q 镜像）/ cache_utils（compute_global + combine_topk + dequant gather 三镜像）|
| `vllm/models/deepseek_v4/{sparse_mla,common/rope}.py`、`vllm/model_executor/layers/attention_layer_base.py`、`vllm/v1/kv_cache_interface.py`、`vllm/v1/attention/backend.py`、`vllm/v1/attention/backends/utils.py` | 各自真实路径 | 协议/装配积木消费面（MLAAttentionSpec L388-L468 逐字含 compress_ratio/storage_block_size/584B/656B 特账与 merge 四字段断言；split_decodes_and_prefills L564-L635 逐字；spec/reshape 家族）|
| `vllm/{config,platforms,distributed,forward_context,envs,logger,compilation}`、`vllm/utils/*`、`vllm/triton_utils.py`、`vllm/v1/worker/workspace.py`、`vllm/vllm_flash_attn/` | 各自真实路径 | HOST SEAM 载体（见 §Seam 清单）|

## 1:1 Source Map（核心行；改动=减法或 seam，原因=批准条/章节边界）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| 模型入口 buffer 分配 | deepseek_v2.py:L1379-L1389 | **逐字** | must_keep（topk_indices_buffer/is_v32）；站 1——全模型共享裸 buffer |
| IndexCache 跨层复用装配段 | deepseek_v2.py:L1091-L1141 | **逐字** | must_keep（skip_topk/index_topk_freq/indexer_rope_emb）；m08——arXiv:2603.12201 注释原文 + MTP 恒建护栏 |
| Indexer.__init__ | deepseek_v2.py:L646-L726 | **逐字** | must_keep（wq_b/wk_weights_proj/k_norm/softmax_scale/n_head_scale/indexer_op）；站 3——独立小头字面证据 |
| Indexer.forward（else 分支+共享尾） | deepseek_v2.py:L780-L819 | 逐字（ROCm L734-L750/CUDA fused L751-L779 平台死支省略——dossier elide 项） | must_keep（per_token_group_quant_fp8 消费位）；站 7——scale 全折 weights |
| _try_load_fp8_indexer_wk | deepseek_v2.py:L822-L871 | **逐字** | must_keep；m01 可信度——FP8 wk 权重的融合加载工程痕迹 |
| MLA 外层接线 | mla.py:L150-L226 | 逐字减 delete[4]（L128-L147 绑定块） | must_keep；**站 6 命脉 L205-L206**——indexer 先于 mla_attn、纯副作用 |
| sparse_attn_indexer op | sparse_attn_indexer.py:L295-L688 | 减法（delete[0]-[5] 六批准项；⚠ L321-L325 取值保留/L886 主路径不删的警示条目全部遵守） | must_keep（cu_seqlen_ks/ke/indexer_k_quant_and_cache/三核分派）；站 8-11 |
| top-k 三核分派 | sparse_attn_indexer.py:L616-L665 | **逐字**（cooperative/persistent 经 torch.ops._C 注册使分派可跑） | must_keep（cooperative_topk/persistent_topk/top_k_per_row_decode）；m06 |
| split_indexer_prefill_chunks | indexer.py:L77-L123 | **逐字** | must_keep；m04——双预算（N≤max_model_len×40、M·N·4≤512MB） |
| _prepare_decode_tensors | indexer.py:L616-L734 | **逐字**（uniform kernel 走垫片） | 站 5——flatten 变长注释原例 [8,9,10,7,9,10,11,12] 与 native 2D |
| builder 的压缩坐标换算 | indexer.py:L793-L811/L932-L941 | **逐字** | m11——compress_ratio>1 时 slot/seq 全部 //ratio |
| DeepseekV4Indexer.__init__/forward | deepseek_v4/attention.py:L719-L892 | 逐字减 delete[2]（L782-L785 fp4 分支、L870-L871 fused 缓冲） | must_keep（compress_ratio//4、skip_k_cache_insert、_fill_short_context、maybe_execute_in_parallel）；站 13 |
| DeepseekCompressor.forward | compressor.py:L329-L478 | 逐字减章界（cutedsl L422-L439/two-stage L440-L448 → ch25/ch28） | must_keep（fused_wkv_wgate/save_partial_states）；m11 |
| V4 _forward_decode | nvidia/flashmla.py:L151-L244 | **逐字** | must_keep（compute_global_topk_indices_and_lens/extra_k_cache/flash_mla_with_kvcache）；站 14——NSA 三支路回归字面载体 |
| V4 _forward_prefill 并集 | nvidia/flashmla.py:L246-L363 | **逐字** | must_keep（combine_topk_swa_indices）；m13——topk ∪ SWA |
| FlashMLASparseImpl.forward_mqa | flashmla_sparse.py:L838-L875 | **逐字** | must_keep（forward_mqa/取 buffer 前 num_actual_toks 行）；站 12 |
| SparseMLACommonImpl.__init__ buffer 接管 | sparse_mla_attention.py:L483-L490 | **逐字** | must_keep（skip 层没 indexer 也能读——显式传参兜底）；m14 |
| triton_convert_req_index_to_global_index | sparse_utils.py:L120-L236 | HOST SEAM（kernel L11-L117 逐式） | must_keep（m07——F7 间接寻址在稀疏路径的再现） |
| MLAAttentionSpec | kv_cache_interface.py:L388-L468 | **逐字**（ch25 同源切面复用） | 站 4/m02/m10——storage÷compress、584B/656B 特账 |

## §Seam 清单（真实代码之外唯一允许的承载面；每处行内另有源锚）

- **B1 CUDA/Triton kernel 镜像（精确数学）**：
  `vllm/utils/deep_gemm.py`（fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits /
  get_paged_mqa_logits_metadata——DeepGEMM 打分核：I=Σ w·ReLU(q·k)，k 侧
  逐 token 反量化、q scale 折 weights 的契约）；`vllm/_custom_ops.py`
  （indexer_k_quant_and_cache 132B 布局 + ue8m0 幂次 scale /
  cp_gather_indexer_k_quant_cache 分页 gather / top_k_per_row_prefill·decode
  【降序、tie 取小、prefill 输出相对 cu_ks、decode 1D 自算
  rowEnd=L−next_n+j+1——对齐 csrc/libtorch_stable/sampler.cu 的插入排序
  tie-break 与 −rowStart 相对化】/ torch.ops._C 的 cooperative·persistent_topk
  注册 / concat_mla_q / gather·upconvert）；`vllm/third_party/flashmla/
  flash_mla_interface.py`（flash_mla_sparse_fwd 的 O(Lk) 稀疏 softmax /
  flash_mla_with_kvcache 的一核双源 + attn_sink / FlashMLASchedMeta 容器）；
  `vllm/triton_utils.py` 的 TritonKernelShim（三处 kernel[grid](...) 直调点：
  _fill_short_context_topk_indices、_prepare_uniform_decode_kernel、
  _BUILD_PREFILL_CHUNK_METADATA_KERNEL——调用面逐字）；
  `vllm/models/deepseek_v4/common/ops/`（save_partial_states /
  compress_norm_rope_store_triton【压缩窗 softmax 门控→RMSNorm→GPT-J RoPE
  打末 rope_dim、压缩位→单块 FP8】/ fused_indexer_q / compute_global_topk_
  indices_and_lens / combine_topk_swa_indices / dequantize_and_gather_k_cache）；
  `vllm/v1/attention/backends/mla/{sparse_utils,compressor_utils}.py`（块表
  换算/压缩 slot 换算）；`vllm/vllm_flash_attn/`（FA4 masked varlen 的位掩码
  softmax——_build_topk_mask 的 word=idx>>5、bit=1<<(idx&31) 打包契约）；
  `sparse_attn_indexer.py` 的 fused_indexer_q_rope_quant 镜像（一核三合一）；
  `vllm/v1/attention/ops/triton_merge_attn_states.py`（ch25 同款 LSE 恒等式）。
- **B2 平台面**：`platforms/__init__.py`（is_cuda 等恒 False→topk 三核分派
  走 per_row 兜底核、Indexer.forward 走 else 分支——真实非 CUDA 平台同型
  分派；get_attn_backend_cls 只实现显式名支——自动优先级面 → ch21）。
- **C 配置面**：`config/__init__.py`（六 namespace 消费字段子集 +
  get/set_current_vllm_config 家族 + CUDAGraphMode 枚举）、
  `config/attention.py`（use_fp4_indexer_cache/indexer_kv_dtype 逐字）、
  `config/cache.py`（CacheDType Literal 面）。
- **D 装配积木**：`model_executor/layers/linear.py`（TP=1 退化，[out,in]
  朝向断言于测试——防转置手滑）、`rotary_embedding/`（GPT-J/NeoX 双形态
  旋转数学 + cos_sin_cache 布局契约；yarn → ch24）、
  `quantization/utils/{fp8,quant}_utils.py`（per_token_group_quant_fp8 的
  ue8m0 幂次 scale / scaled_dequantize 组广播）、`models/utils.py`
  （extract_layer_index 逐字 + make_layers 单进程退化——PP → ch17）。
- **E 顶层设施**：`envs.py`（VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512 真实默认）、
  `logger.py`（no-op）、`distributed/`（单进程组）、`forward_context.py`
  （ch23/ch25 同款骨架）、`compilation/breakable_cudagraph.py`（直通）、
  `utils/{torch,math,import,platform,multi_stream,flashinfer}_utils.py`
  （direct_register_custom_op 真注册 / maybe_execute_in_parallel 逐字 /
  resolve_obj_by_qualname / has_cutedsl·has_flashinfer 恒 False）、
  `v1/worker/workspace.py`（get_simultaneous 独立分配面）、
  `v1/attention/{selector,backends/registry}.py`（显式后端支——
  attention_config.backend 名经真实 registry 解析）、
  `v1/attention/backends/mla/prefill/`（get_mla_prefill_backend 的
  『无可用后端 → ValueError』真实失败面——MLAAttention 捕获后落
  top-k MQA-only）。

## SUBTRACTED 台账（挂 dossier.subtraction_plan.delete 编号）

- **delete[0] DCP**：sparse_attn_indexer.py 的 _assert_cutedsl_dcp_merge_
  supported（L48-L71）+ _merge_dcp_topk_global（L74-L124）+ 两处调用
  （L520-L528、L667-L675）+ dcp 三形参（L315-L317，默认 0/1/1）+ __init__
  的 dcp_rank/interleave 两标量（L770-L771）；indexer.py 五小块——①
  _dcp_localize_decode_seq_lens（L596-L614）②build 调用的 dcp 透传
  （L855-L857）③global_seq_lens 守卫+下游（L883-L885/L893-L901/L963 +
  dataclass 字段 L420 + 方法本体 L736-L762 + global_decode_seq_lens_buffer
  L545-L549）④_dcp_localize 调用块（L922-L928 含其上局部化长注释）⑤
  build_prefill_chunk_metadata 的 DCP 段（L1013-L1027）；sparse_utils.py 的
  triton_filter_and_convert_dcp_index（L239-L343）。**⚠ L886-L921 decode 主
  路径逐字保留**（⚠ 警示条目遵守）。
- **delete[1] XPU/ROCm**：sparse_attn_indexer.py 的 XPU 分支（L488-L498、
  L587-L601）与 forward_hip/rocm_aiter 分支（L842-L873）；forward_native
  平台分派保留 def 分支。
- **delete[2] FP4**：_gather_workspace_shapes fp4 形状（L264-L268）、
  kv_cache_as_quant_view fp4 视图（L282-L291）、q_scale 断言（L372-L377）、
  prefill/decode fp4 cast（L480-L487、L552-L558、L582-L586）、
  DeepseekV4Indexer 的 use_fp4_kv 分支（L782-L785）与 fused 输出缓冲
  （L870-L871）。MXFP4_BLOCK_SIZE 常量本体保留（must_keep 布局锚）。
- **delete[3] PCP**：sparse_attn_indexer.py 的 use_pcp 消费（L384-L385 +
  maybe_gather_indexer_k 的 gather 分支 L391-L396）——prefill_context_
  parallel_size=1 恒不触发。
- **delete[4] dense-MHA 跳过**：sparse_attn_indexer.py L408-L424（提前返回）
  + mla.py L128-L147（dense_mha_metadata_layer_name 绑定）——纯优化性跳过，
  不跳只多算一次打分、结果不变。
- **delete[5] 假跑分支**：sparse_attn_indexer.py L327-L364（profiling 预留）；
  L321-L325 的取值保留（真实路径必需）。
- **章界外（→ chN，非 delete 项）**：mla_attention.py 的 MQA 吸收腿/混批切刀
  /输出量化/吸收重排（→ ch25）；deepseek_v2.py 的 min-latency GEMM
  （→ ch25 站 2）、DecoderLayer 选型三岔与 MoE/MLP、Model forward/load 的
  MoE 段（→ ch25/ch28）、旧 MHA/GQA 层（→ ch24/ch25）；deepseek_v4 的
  融合 GEMM/norm/RoPE/kv 插入链与三路重叠（→ ch25/ch28/ch19）、fp8 o_proj
  （→ ch27）、compressor 的 head=512 cutedsl/two-stage（→ ch25/ch28）；
  sparse_mla_attention 的 builder 与 chunked-context 机器（→ ch25）；
  flashmla_sparse 的 MetadataBuilder（→ ch21/ch25）；sparse_swa 的 builder/
  tile scheduler（→ ch25/ch21）；sparse_mla.py 的 V4 MetadataBuilder（正文
  按真实源码 excerpt 讲 _build_c128a_metadata）；rotary 的 yarn（→ ch24）；
  make_layers 的 PP 分段（→ ch17）。
