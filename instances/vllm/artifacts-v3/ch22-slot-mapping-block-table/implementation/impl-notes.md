# ch22《slot_mapping 与 block_table》impl-notes —— 只做减法精简版

对应真实源码 pin **vLLM v0.27.1（6e448d0ea）**，行号全部现核（2026-08-29，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。
运行：`cd instances/vllm/artifacts-v3/ch22-slot-mapping-block-table && python -m pytest tests/ -q`
→ **27 passed**（~1.7s）。纯 host 单元测试：真 torch/numpy/triton（kernel 定义
逐字保留、host 上不 launch）；无 vllm 包、无 CUDA 上下文——worker 侧 CUDA 面
（Triton launch / vllm_flash_attn CUDA ops）以 HOST SEAM 承载（见 §Seam 清单）。

**验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST SEAM
例外见 §Seam 清单——每个 seam 行内标注并在此登记）。SUBTRACTED 标记逐条挂
dossier.subtraction_plan.delete[0..14] 批准项编号（章界外域段另以「→ chN」
注记——ch18 立下的切面惯例）。

**lint**：`python scripts/lint_fidelity.py <本章目录>` → **全部通过（无
BLOCKING）**；must_keep 49 符号经 linter `over_subtraction` 项全数核在。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `block_table.py` | `vllm/v1/worker/block_table.py` | **主角文件近乎全文**：get_block_table_width（128-token 对齐 m11）/ SlotMappingMode 两值（m10）/ BlockTable 全方法（五原语 + hybrid 细分 + PCP/DCP 探测）/ MultiGroupBlockTable 全扇出（m12）/ `_compute_slot_mapping_kernel` **逐字**（do_not_specialize 装饰器 + PAD 程序 + 恒等式 + CP 分片三件）。仅 import 面归一；CPU host 镜像 seam 见清单 |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | 块表线切面：ExecuteModelState（L438-L451 逐字）/ GPUModelRunner 切面 __init__（持久缓冲 L763-L845 子集镜像）/ _update_states 块号差量落行（L1355-L1474 切面）/ _get_cumsum_and_arange（L1743-L1767 逐字）/ _prepare_inputs（commit 先行 L1977-L1979 + query_start_loc 非递减 L2073-L2078 + GPU positions/seq_lens L2180-L2201）/ _build_attention_metadata（_get_block_table L2325-L2341 + cm_base L2430-L2449）/ _preprocess（positions 尾清零 L3662-L3664）/ _get_slot_mappings（L4082-L4154）/ execute_model（L4165-L4535 切面：has_separate_kv_update L4307-L4318 + 三元选择 L4367-L4376 + set_forward_context L4432-L4444 + 打包 L4516-L4527）/ may_reinitialize_input_batch（L7240-L7301） |
| `attention.py` | `vllm/model_executor/layers/attention/attention.py` | 写腿通道三件：get_attention_context（L734-L772）/ unified_kv_cache_update（L775-L798 逐字）/ unified_attention_with_output（L817-L846 逐字）+ 两个 fake 实现 |
| `flash_attn.py` | `vllm/v1/attention/backends/flash_attn.py` | 消费端：FlashAttentionMetadata 全 dataclass（L242-L299 逐字）/ FlashAttentionBackend 两语义位（get_supported_kernel_block_sizes=[MultipleOf(16)] L82-L84、forward_includes_kv_cache_update=False L86）/ FlashAttentionImpl（__new__ 组探测 + __init__ L746-L782 逐字 + forward 读腿切面 L838-L1067 + do_kv_cache_update L1098-L1132 逐字）+ 两个 CUDA op 的 HOST SEAM 镜像（见清单） |
| `backend.py` | `vllm/v1/attention/backend.py` | MultipleOf（L49-L53）/ AttentionType 枚举 / AttentionBackend 两语义位（L66-L71）/ AttentionMetadata（L404）/ **CommonAttentionMetadata**（L411-L533：字段面 + batch_size/naive_query_lens/replace + 两个 deprecated 属性 L505-L533 逐字——D2H 禁忌的成文纪律 WC2） |
| `forward_context.py` | `vllm/forward_context.py` | BatchDescriptor（L30-L58）/ ForwardContext 切片（slot_mapping 字段 **L136** + docstring L137-L143）/ get_forward_context / create_forward_context（L212-L241）/ override_forward_context / set_forward_context（L259-L344 切面：slot_mapping 参数 L268 与 create/yield 主干） |
| `worker_utils.py` | `vllm/v1/worker/utils.py` | AttentionGroup（L217-L227）/ select_common_block_size（L266-L332 逐字）/ prepare_kernel_block_sizes（L335-L376 逐字） |
| `kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | 消费面载体：KVCacheSpecKind / KVCacheSpec.max_num_blocks_per_req（L139-L145）/ FullAttentionSpec / MambaSpec（page_size_bytes + max_num_blocks_per_req 逐字）/ EncoderOnlyAttentionSpec / UniformTypeKVCacheSpecs / get_kv_cache_spec_kind（full/mamba/encoder 三支）/ KVCacheGroupSpec / KVCacheConfig.has_mamba_layers |
| `v1_utils.py` | `vllm/v1/utils.py` | CpuGpuBuffer 全类（L110-L149 逐字——双镜像基座 m4） |
| `backends_utils.py` | `vllm/v1/attention/backends/utils.py` | PAD_SLOT_ID/NULL_BLOCK_ID 两常量（L42-L46——m3 的两块界碑） |
| `math_utils.py` | `vllm/utils/math_utils.py` | cdiv |
| `torch_utils.py` | `vllm/utils/torch_utils.py` | PIN_MEMORY（HOST SEAM False）/ _resolve_layer_name（L882-L885） |
| `_host_seams.py` / `_dist_seams.py` | （跨域缝合，见 §Seam 清单） | HOST SEAM 登记处 |

## 1:1 Source Map（核心行；改动=减法或 seam，原因=批准条/章节边界）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `_compute_slot_mapping_kernel` 全体 | block_table.py:L379-L442 | **逐字**（含 `@triton.jit(do_not_specialize=...)` 装饰器行） | must_keep（m1/m2/m3/m8 主角）；WC1 行号核验注：卡内 L379 含装饰器、锚点规范口径 L380 起 def——本包 SOURCE 标签两者都标 |
| `BlockTable.__init__` | block_table.py:L49-L136 | 逐字（hybrid 支 L82-L101、`_kernel_block_arange` L114-L119、PCP/DCP 探测 try/except L121-L134 全保留） | must_keep ×7（use_hybrid_blocks/blocks_per_kv_block/kv_cache_block_size…）；delete 无本类删除项 |
| `append_row`/`add_row`/`clear_row`/`move_row`/`swap_row` | block_table.py:L138-L180 | 逐字（move_row 的 #49757 清源注释原文保留） | must_keep ×5；m13 |
| `compute_slot_mapping` + CPU 镜像 | block_table.py:L182-L211 / kernel L397-L442 | 派发逐字 + HOST SEAM 镜像（NONE 早退 L189-L192 原样） | must_keep；m2/m10 |
| `commit_block_table` | block_table.py:L213-L214 | **逐字**（只拷活跃行） | must_keep；m4/站3 |
| `get_block_table_width` | block_table.py:L20-L40 | 逐字（token_alignment=128 对齐乘子全保留） | must_keep（token_alignment）；m11 |
| `map_to_kernel_blocks` | block_table.py:L220-L248 | **逐字**（docstring 的 [0,1,2]→[0..5] 例保留） | must_keep；m9 |
| `MultiGroupBlockTable` 全类 | block_table.py:L270-L376 | 逐字 | must_keep；m12 |
| `_update_states` 块表线 | gpu_model_runner.py:L1355-L1474 | 差量 extend/恢复替换（L1441-L1452）+ 落行（L1471-L1474）逐字；reqs_to_add 落位/condense/重排 → ch18 | 站1；delete[2][10][11] 各支就地标注 |
| `_prepare_inputs` | gpu_model_runner.py:L1960-L2282 | commit 三行（L1977-L1979）+ qsl 非递减（L2073-L2078）+ GPU 组装（L2180-L2201）+ 非 spec 支（L2232-L2241）逐字；token 收集/mrope/spec 纠偏删除 | 站3-5；delete[0][1][2][3] |
| `_get_slot_mappings` + 闭包 | gpu_model_runner.py:L4082-L4154 | docstring + 切片 + 尾段 fill_(-1)（L4128-L4130）+ by-layer dict 逐字；encoder 零向量/ubatch 分支删 | 站9；delete[5][7] |
| `_build_attention_metadata` | gpu_model_runner.py:L2284-L2449 | _get_block_table（尾行 NULL_BLOCK_ID L2338-L2341）+ cm_base（L2430-L2449）逐字；dcp/kv_sharing/builder 循环删除（seam 组装 FA metadata） | 站10；delete[6][8][9][13] + ch21 边界 |
| `execute_model` | gpu_model_runner.py:L4165-L4535 | 两段式契约（L4171-L4535 主干）+ has_separate_kv_update（L4307-L4318）+ 三元选择（L4367-L4376）+ set_forward_context（L4432-L4444）+ 打包（L4516-L4527）逐字；logits/采样尾 → ch18 | 站8/11；delete[4][5][9][11][12] + ch19 seam |
| `unified_kv_cache_update` | attention.py:L775-L798 | **逐字**（do_kv_cache_update 派发 + `key.new_empty(0)` dummy） | must_keep；m5/站12 |
| `do_kv_cache_update` | flash_attn.py:L1098-L1132 | **逐字**（woosuk NOTE 原文：slot_mapping 的形状定 token 数） | must_keep；m5/站13 |
| `forward` 读腿 | flash_attn.py:L838-L1067 | K/V 视图（L904-L905）+ block_table 解包（L929-L935）+ varlen 调用（L1041-L1067）逐字；cascade/dcp/mm/rswa/encoder 支删除 | m6/站14/F7；delete[7] 同域 + 分布式/多模态/ch21 边界 |
| `reshape_and_cache_flash` | csrc/libtorch_stable/cache_kernels.cu:L315-L344 | HOST SEAM 镜像：slot<0 return（L329-L331）+ slot//bs、slot%bs 逆分解（L332-L333）+ 行拷贝 | m1/m3/m5；kernel 本体在 .cu（正文内嵌） |
| `CommonAttentionMetadata` | backend.py:L411-L533 | 字段面 + 两 deprecated 属性**逐字**（'avoid implicit H<>D sync … breaks full async scheduling' 措辞原文） | must_keep；WC2 官印 |
| `set_forward_context` / `ForwardContext.slot_mapping` | forward_context.py:L259-L344 / L136 | 签名（slot_mapping 参数）+ create/override/yield 主干逐字；DP/MoE/观测面删除 | must_keep；站11 |
| `select_common_block_size` / `prepare_kernel_block_sizes` | worker/utils.py:L266-L332 / L335-L376 | **逐字**（Case 1/Case 2 注释证明原文） | must_keep ×2；m9 |
| `may_reinitialize_input_batch` | gpu_model_runner.py:L7240-L7301 | 模式装配（L7254-L7272）+ 重建判据逐字；encoder continue 删 | must_keep；m10；delete[7] |

## 删除账本（dossier.subtraction_plan.delete[0..14] 落点）

| delete | 内容 | 落点（# SUBTRACTED 标注） |
|---|---|---|
| [0] | M-RoPE/XD-RoPE 守卫与拷贝 | gpu_model_runner.py：__init__ 缓冲、_prepare_inputs L1997-L2005/L2211-L2230、_preprocess L3657-L3660 |
| [1] | prompt_embeds 路径 | gpu_model_runner.py：_prepare_inputs L2007-L2070、_preprocess L3566-L3647 |
| [2] | async spec 乐观纠偏全家 | gpu_model_runner.py：_update_states L1363-L1403/L1408-L1439、_prepare_inputs L2092-L2141/L2152-L2173（else 支 L2174-L2178 去缩进无条件——ch18 同款；optimistic_seq_lens_cpu 的**无条件装配** L2085-L2090 保留：_build_attention_metadata 消费它，无 spec 时 optimistic==精确） |
| [3] | mamba 预处理管线 | gpu_model_runner.py：L2143-L2150、execute_model L4320-L4362 |
| [4] | cascade prefix 预计算 | gpu_model_runner.py L4255-L4263（cascade_attn_prefix_lens=None 占位） |
| [5] | ubatching/DBO | gpu_model_runner.py L4293-L4299/_get_slot_mappings L4145-L4152/ubatch_slices 恒 None；AttentionGroup 的 metadata_builders |
| [6] | routed_experts 私有快照 | gpu_model_runner.py L2347-L2358 |
| [7] | Encoder-only 三分支 | _get_block_table 零表 L2328-L2333、_get_slot_mapping 零向量 L4115-L4123、may_reinitialize continue L7261-L7262（判别位 prepare_kernel_block_sizes L359-L360 原样保留——真实代码） |
| [8] | hybrid metadata 缓存复用 | _build_attention_metadata L2472-L2600 的 builder 循环整体（cache 命中支在其内）→ ch21 边界 + seam 组装 |
| [9] | KV/EC connector 与 kv_sharing_fast_prefill | execute_model L4197-L4200/L4210-L4216/L4231-L4234/L4236-L4241/L4445-L4448、_build_attention_metadata L2466-L2470 |
| [10] | PP 相关 | _update_states L1333/L1408-L1439/L1476-L1494、execute_model 后段 PP 两臂（L4467-L4514 内） |
| [11] | ngram-GPU replace 复制 | execute_model L4180-L4195、_update_states ngram 账（L1337-L1351/L1466-L1468） |
| [12] | external_launcher+DP dummy-run | execute_model L4219-L4230（空拍早退保留） |
| [13] | dcp_local_seq_lens | _build_attention_metadata L2451-L2464 |
| [14] | InputBatch 与块表无关的装填 | InputBatchSeam 只承载块表线字段面（构造镜像 gpu_input_batch.py:L186-L195 的 MultiGroupBlockTable 装配逐字）；采样列/thinking/replayssm/reasoning 不进 |

## Seam 清单（HOST/ENGINE SEAM——真实代码之外唯一允许的承载，每个行内标注）

1. **`BlockTable._compute_slot_mapping_host`**（CPU 镜像，ch13/ch18 同款）：
   kernel L397-L442 的逐行 numpy 镜像——同一 PAD 尾、同一 CP 变量名与单卡退化、
   同一恒等式；收尾一次 `gpu.copy_(cpu)` 承载「kernel 写设备缓冲」（device=cpu
   时两端同机）。CUDA 设备分支 kernel 派发逐字保留（容器内真跑）。CP 三件按
   `self.dcp_*`/`cp_kv_cache_interleave_size` 取值（kernel 侧为烘干 constexpr）。
2. **`reshape_and_cache_flash`**（写腿 op 承载）：vllm_flash_attn 的 CUDA op 在
   host 无库——以 cache_kernels.cu:L315-L344 kernel 本体的逐 token torch 镜像
   承载（slot<0 跳过 + slot//bs、slot%bs 逆分解 + 行拷贝；"auto" dtype 无缩放、
   向量化拷贝退化为行赋值）。调用面 do_kv_cache_update 逐字。
3. **`flash_attn_varlen_func`**（读腿 op 承载）：精确 attention 数学镜像——每
   请求**穿 block_table 逐逻辑块 gather K/V**（间接寻址——F7 的读侧内景）、
   GQA 广播、causal、逐 Q 头 softmax(QK^T·scale)V（ch20 已立数学；fp64 softmax
   稳定数值）。只承载本章用面（causal=True、(-1,-1) 无窗口、无 alibi/softcap）
   ——其余参数面断言拒绝。
4. **`GPUModelRunner` 切面 __init__ + `_model_forward`**（ENGINE SEAM，ch17
   域/ch18 同款）：真实 __init__（L456-L760）的模型/采样/cudagraph 装配面以
   装配参数直供（ch13 同款切面构造）；持久缓冲块 L763-L845 的块表线子集逐字。
   `_model_forward` 以脚本化单 Attention 层前向承载：每层先
   `unified_kv_cache_update`（写腿）再 `unified_attention_with_output`（吃同一
   dummy 数据依赖）——真实 Attention.forward 的两算子调用序（attention.py 真身）；
   观测位记录前向内按 layer_name 取到的 slot_mapping。
5. **`_build_attention_metadata` 的 FA metadata 组装**（ENGINE SEAM）：
   FlashAttentionMetadataBuilder.build 的非级联 pass-through 半边（common →
   FlashAttentionMetadata 字段直拷，锚 flash_attn.py:L458-L476 解包 +
   L672-L697 组装）；AOT scheduler/cascade/量化分支归 ch21。
6. **`execute_model` 的 padded 口径装配**（ch19 域 seam）：
   `_determine_batch_execution_and_padding`（BatchDescriptor 查表命中/实际
   batch padding——ch19 全文）以观测位 `seam_cudagraph_mode`/`seam_batch_desc`
   直供（测试置 FULL/128×8 验证 padded 口径）；后续 num_tokens_padded/
   num_reqs_padded 消费链逐字。
7. **组探测/logger/envs**（_host_seams.py/_dist_seams.py）：get_pcp_group/
   get_dcp_group 与真实「测试环境未初始化组」同型抛 AssertionError，由
   BlockTable/FlashAttentionImpl 的 try/except 捕获退化单卡（源码原生路径）；
   init_logger/record_function_or_nullcontext/deprecated 为 no-op/等价镜像
   （deprecated 保 DeprecationWarning 原措辞可见——测试断言它）；
   CUDAGraphMode 枚举逐字（ch19 域）；get_flash_attn_version 恒 FA2；
   is_quantized_kv_cache 等价装配（"auto"→False）。
8. **协议载体**（SchedulerOutputSeam/InputBatchSeam）：delete[14] 明示
   「InputBatch 直接复用 ch18 精简版产物」——本章以同接口最小承载
   （block_table + num_reqs/req_ids/req_id_to_index + 四个列式 CPU 镜像，np 与
   torch 共享存储同真实 InputBatch）；SchedulerOutput 只带块表线消费字段。
9. **配置/环境占位**：PIN_MEMORY=False（host 无 pinned——只影响拷贝速度）；
   VllmConfigSeam 各 config namespace 的块表线消费子集（mm/rswa/replayssm/
   spec/lora 等开关全取默认关的真实值——被删支的守卫位保留、永不触发）。

## 已知偏差（非 seam 的显式记录）

- `_update_states` 返回 None（真实可返回 deferred corrections callable——其
  生产支是 delete[2] 的 async spec 纠偏，删后无生产者；execute_model 的
  `if deferred_state_corrections_fn:` 消费位原样保留）。
- `execute_model` 的 logits/采样尾（L4458-L4514）→ ch18 采样域：logits 恒
  None、hidden_states 为 seam 前向占位（两段式契约的后半 sample_tokens 归
  ch18）；空拍早退的 EMPTY_MODEL_RUNNER_OUTPUT 以 None 占位（真身
  vllm/v1/outputs.py，ch12/ch18 域）。
- `unified_kv_cache_update`/`unified_attention_with_output` 的
  `direct_register_custom_op` 注册块与两装饰器（L809-L814、L817-L818）归
  ch19/ch16——直调 Python 函数体（同一控制流）；两个 *_fake 实现保留（编译
  fake 模式语义位）。
- kv_cache_interface 的 spec 类为**消费面裁剪**：字段只留 block_size 与
  max_num_blocks_per_req/page_size_bytes（MLA/SWA/Sink 等判别支与
  page_size_bytes/storage_block_size 抽象 → ch13/ch14 全文）。
