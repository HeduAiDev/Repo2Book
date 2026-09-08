# ch23《模型定义层拼装术》impl-notes —— 只做减法精简版

对应真实源码 pin **vLLM v0.27.1（6e448d0ea）**，行号全部现核（2026-09-06，
`instances/vllm/source`），**不是** v2 资产的 v0.21.0 旧行号。
**对齐修订版 dossier**（2026-09-06 21:00 dossier-fix2：delete 重钉为 14 项
[0..13]、must_keep 扩至 71、delete[7]/[10]/[11] 增豁免/保留指令——本版已逐条
落实：linear.py 恢复 fused-on-disk 豁免骨架、gpu_model_runner 恢复 L5362
调用与 EPLB 块、aux 解包段与 broadcast_pp_output 头）。
运行：`cd instances/vllm/artifacts-v3/ch23-model-layer-assembly && python -m pytest tests/ -q`
→ **54 passed**（~5s）。纯 host 单元测试：真 torch + 真 transformers（LlamaConfig）；
无 vllm 包安装、无 CUDA 上下文——分布式/平台/编译面以 HOST SEAM 单进程退化承载
（见 §Seam 清单），Attention 的 impl/attn_backend 按 ch21 域的真实注入位
（`__init__` 的 `attn_backend` 参数）以参考后端注入（数学 = ch20 已立的精确注意力）。

**验收判据**：把真实源码删掉所有 `# SUBTRACTED:` 分支 ≈ 得到本包（HOST SEAM
例外见 §Seam 清单——每个 seam 行内标注并在此登记）。SUBTRACTED 标记逐条挂
dossier.subtraction_plan.delete[0..13] 批准项编号（章界外域段另以「→ chN」注记
——ch18 立下的切面惯例）。

**lint**：`python scripts/lint_fidelity.py <本章目录>` → 全部通过（无 BLOCKING）；
must_keep 71 符号经 linter `over_subtraction` 项全数核在。

**锚点双置惯例**（ch22 同款）：每个 def/class 的 `# SOURCE:` 锚点既放在
声明上方（供读者），也在 def/class 体内（docstring 后）复置一行——后者是
`lint_fidelity._spans_missing_source` 的跨度判据（装饰器/多行注释会切断
「上方一行」窗口）。两处锚点同文。

## 包结构（与真实树同名同构）

| 精简版文件 | 真实文件 | 本章切面 |
|---|---|---|
| `vllm/model_executor/models/llama.py` | 同名 | **主角文件**：Llama 五类拼装全文减法——LlamaMLP（L79-L119 逐字）/ LlamaAttention（L122-L245：TP 切头数学 L140-L160 逐字 + QKV/o_proj 构造 + forward 五行 L221-L231 逐字）/ LlamaDecoderLayer（L248-L331：四件套装配 + 残差穿针 L310-L327 逐字）/ LlamaModel（@support_torch_compile + hf_to_vllm_mapper L345-L354 + __init__ L356-L395 + forward L400-L439 + load_weights）/ LlamaForCausalLM（bases 标记族 + 三方法契约 L516-L540 逐字） |
| `vllm/model_executor/layers/linear.py` | 同名 | 三积木承重墙：LinearBase/ColumnParallelLinear/MergedColumnParallelLinear/QKVParallelLinear/RowParallelLinear + 各自 weight_loader 分片算术（QKV 段定位 L1317-L1333 逐字 + 秩算术 L1375-L1383 逐字 + **fused-on-disk 骨架 L1236-L1265/L1309-L1313 豁免保留**；Merged L829-L834 段定位逐字 + **fused 骨架 L788-L793/L823-L826 豁免保留** + validate_shard_id tuple 支；Row input_dim 行切 L1717-L1737 逐字；adjust_scalar_to_fused_array L116-L140 逐字；forward 三型逐字） |
| `vllm/model_executor/layers/logits_processor.py` | 同名 | 出口三步：__init__（org_vocab_size/use_all_gather/head_dtype 位）/ _gather_logits（L84-L96 逐字——woosuk NOTE 原话）/ _apply_head 主路径 / _get_logits（L137-L153 逐字——GEMM→gather→裁 padding） |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | 同名 | 词表分片：UnquantizedEmbeddingMethod + pad_vocab_size + VocabParallelEmbedding（__init__ 区间账 L240-L334 逐字 + weight_loader + forward）+ ParallelLMHead（子类 + tie_weights + forward 即 raise） |
| `vllm/model_executor/layers/attention/attention.py` | 同名 | Attention 插座：__init__（sliding_window 三态/属性面/自注册 static_forward_context L443-L446 逐字）+ forward（L488-L582 减法：docstring 官方自述 + view 重排 + 双通道）+ get_attention_context（L732-L772 逐字）/ unified_kv_cache_update（L775-L798 逐字）/ unified_attention_with_output（L817-L846 减法）+ 两个 fake |
| `vllm/model_executor/layers/layernorm.py` | 同名 | RMSNorm：__init__（pass_weight 位）+ forward_native（L74-L95 逐字——融合 add-norm 二元组语义本体，经 ir.ops）+ 平台派发位 |
| `vllm/model_executor/layers/activation.py` | 同名 | SiluAndMul（register L117 + __init__ L130 + forward_native L139 逐字 + 平台派发位） |
| `vllm/model_executor/layers/rotary_embedding/` | 同名三文件 | get_rope 工厂（default-scaling 主路径 + _ROPE_DICT 缓存）+ RotaryEmbeddingBase/RotaryEmbedding + ApplyRotaryEmb.forward_static（neox/GPT-J 旋转数学逐字）——正文当黑盒工厂 |
| `vllm/model_executor/models/registry.py` | 同名 | 五条目单表（新旧布局对照）+ _LazyRegisteredModel（load_model_cls L1017-L1019 逐字）+ _resolve_module_name（L1439-L1445 逐字）+ ModelRegistry 构造（L1448-L1458 逐字） |
| `vllm/model_executor/models/utils.py` | 同名 | WeightsMapper（_map_name_with_shard——stacked 映射本体 L116-L120 逐字 + apply 盖章）/ AutoWeightsLoader 全递归分发 / PPMissingLayer / make_layers（L798-L826 减法：offloader 包装删）/ make_empty_intermediate_tensors_factory / maybe_prefix / extract_layer_index |
| `vllm/model_executor/models/interfaces.py` | 同名 | llama.py bases 消费面：SupportsLoRA/SupportsPP/SupportsQuant（_find_quant_config 逐字）/ LocalArgmaxMixin/EagleModelMixin（标记类保留声明——delete[0] 明示） |
| `vllm/model_executor/model_loader/__init__.py` | 同名 | get_model_loader 查表（default/dummy 两项）+ get_model 统一入口（L122-L142 逐字） |
| `vllm/model_executor/model_loader/base_loader.py` | 同名 | BaseModelLoader.load_model 四段编排（L43-L82 减法：dtype/device 上下文→建空壳→load_weights→process→eval 逐字主干） |
| `vllm/model_executor/model_loader/default_loader.py` | 同名 | DefaultModelLoader：load_weights 核心（L415-L427 减法——`model.load_weights(self.get_all_weights(...))` 主行逐字；checkpoint IO 族 SUBTRACTED，host 以 SeededLoader 覆写 get_all_weights 注入） |
| `vllm/model_executor/model_loader/dummy_loader.py` | 同名 | DummyModelLoader（L22-L42 减法：随机权重初始化删除——只消费「免 checkpoint 建模型」语义） |
| `vllm/model_executor/model_loader/utils.py` | 同名 | initialize_model（新式签名校验 + set_current_vllm_config 构造）/ process_weights_after_loading 两轮 / device_loading_context / get_model_architecture（hash 键缓存逐字） |
| `vllm/model_executor/model_loader/weight_utils.py` | 同名 | default_weight_loader（L1231 逐字——_load_param 的兜底） |
| `vllm/model_executor/parameter.py` | 同名 | BasevLLMParameter（weight_loader 属性/tp 戳记）+ _ColumnvLLMParameter/RowvLLMParameter/ModelWeightParameter（量化参数族 SUBTRACTED——delete[7]） |
| `vllm/model_executor/custom_op.py` | 同名 | op_registry/PluggableLayer（register/register_oot——m13 OOT 面）/ CustomOp（dispatch_forward 派发主干 + enabled/default_on） |
| `vllm/v1/worker/gpu_model_runner.py` | 同名 | 两薄层（delete[13]）：load_model 线（L5303-L5390 减法——三段删除 + **delete[10] 明示保留的 L5362 调用与 L5376-L5390 EPLB 块**）+ execute_model 尾段（L4432-L4456 with 块 + **delete[11] 明示保留的 L4459-L4465 aux 解包与 L4467 主路径头** + L4484-L4485 采样位切片两行逐字）+ _model_forward（L3879-L3907 逐字） |
| `vllm/models/deepseek_v4/__init__.py` | 同名 | m9 布局证据：平台分发骨架（L14-L32 的 if/elif/else 逐字；子包本体归 ch28） |
| `vllm/sequence.py` | 同名 | IntermediateTensors 全类（L12-L53 逐字） |
| `vllm/forward_context.py` | 同名 | ForwardContext（no_compile_layers/attn_metadata/slot_mapping 三字段）+ get/create/override/set_forward_context 主干 |
| `vllm/ir/ops/layernorm.py` + `vllm/ir/` | 同名 | rms_norm/fused_add_rms_norm native 数学（逐字）+ maybe_inplace 通道的 HOST SEAM 载体 |
| `vllm/distributed/__init__.py` | parallel_state/utils/communication_op 门面 | HOST SEAM：divide/split/get_pp_indices（逐字）+ GroupCoordinator rank 面/集合通信的 world_size==1 旁路 |
| 其余（config/envs/logger/platforms/utils/torch_utils/…） | 各自真实路径 | HOST SEAM 载体（见 §Seam 清单） |

## 1:1 Source Map（核心行；改动=减法或 seam，原因=批准条/章节边界）

| 精简版符号 | 真实源码锚点（v0.27.1 现核） | 改动 | 原因 |
|---|---|---|---|
| `LlamaForCausalLM` 全类 | llama.py:L446-L540 | bases 逐字；__init__ L466-L503 逐字；forward L516-L526 逐字；compute_logits L528-L533 逐字；load_weights L535-L540 逐字 | must_keep ×6（契约三方法持有者） |
| `LlamaModel.forward` | llama.py:L400-L439 | aux 采集线删（L419/L426-L428/L437-L438）；embed/islice/PP 打包/norm 收尾逐字 | must_keep；delete[0] |
| `LlamaDecoderLayer.forward` | llama.py:L310-L327 | **逐字**（首层特判 + 融合 add-norm 二元组） | must_keep ×3；m1 |
| `LlamaAttention.__init__` 切头段 | llama.py:L138-L160 | **逐字**（total%tp 断言、KV partition/replicate、q_size/kv_size/scaling） | must_keep；m2；测试 tp=1/2/断言三视角 |
| `LlamaAttention.forward` | llama.py:L221-L231 | **逐字**（五行——签名无 metadata/kv_cache，Part VI hook 实证） | must_keep；站9 |
| `QKVParallelLinear.weight_loader` | linear.py:L1222-L1400 | 量化块 L1266-L1307/L1329-L1374 删；**fused-on-disk 骨架 L1236-L1265 + 递归尾 L1309-L1313 + needs/警告尾 L1385-L1397 豁免保留**；q/k/v 段定位 L1317-L1333 与秩算术 L1375-L1383 **逐字** | must_keep ×5；delete[7]；m7 worked example；测试断言 fused-on-disk 装载段位 |
| `MergedColumnParallelLinear.weight_loader` | linear.py:L746-L889 | 量化块 L779-L787/L794-L821/L836-L868 删；**fused-on-disk 骨架 L788-L793 构造+循环头 + L823-L826 narrow/递归尾 + L873-L886 尾部豁免保留**；段定位 L829-L834 **逐字**；validate_shard_id tuple 支 L729-L744 **逐字** | delete[7]；m7 |
| `adjust_scalar_to_fused_array` | linear.py:L116-L140 | **逐字**（恢复——豁免段 L1240-L1243/L763-L766/L874-L877 的调用位仍在） | delete[7] 豁免面 |
| `RowParallelLinear.weight_loader`+forward | linear.py:L1717-L1737 / L1748-L1774 | bnb 判整删；input_dim 行切**逐字**；forward（bias 只 rank0 加/all_reduce）**逐字** | delete[7]；m7 |
| `hf_to_vllm_mapper` + `WeightsMapper` | llama.py:L345-L354 / models/utils.py:L47-L139 | 五条 stacked 映射**逐字**；renaming/regex/kv_scale 面 | must_keep ×3；delete 面见 SUBTRACTED 注 |
| `AutoWeightsLoader.load_weights`→`_load_module` | models/utils.py:L398-L424 / L317-L398 | quant_config cache mapper 合并段删；PPMissingLayer 早退 L322-L323 / 子模块委派 / 递归分组 / 无主错误逐字 | must_keep；m6 |
| `LogitsProcessor._get_logits` | logits_processor.py:L137-L153 | **逐字**（三步） | must_keep ×3；m5/站12 |
| `_gather_logits` | logits_processor.py:L84-L96 | **逐字**（woosuk NOTE 原话） | must_keep ×3 |
| `Attention.__init__` 尾部自注册 | attention.py:L443-L446 | **逐字**（Duplicate layer name raise）；get_attn_backend 选择面→ch21（真实注入位 L251 参数保留） | must_keep；m11；测试断言 raise |
| `get_attention_context` / `unified_kv_cache_update` / `unified_attention_with_output` | attention.py:L732-L772 / L775-L798 / L817-L846 | 前两者**逐字**；第三者函数体逐字、两装饰器删（ch19/ch16）；custom op 注册块删（ch19，直调同控制流） | must_keep ×3；m3；测试断言 dummy 依赖与 spec decode list[dict] |
| `make_layers` + `PPMissingLayer` | models/utils.py:L798-L826 / L785-L797 | get_pp_indices 均分 + 三段 ModuleList **逐字**；offloader 包装删 | must_keep ×2；m14；测试 pp_rank=1/size=2 断言哨兵占位 |
| registry 五条目 + `_resolve_module_name` + `load_model_cls` | registry.py:L92-L95/L147/L617/L643 选段 / L1439-L1445 / L1017-L1019 | 条目表收缩为 delete[8] 批准五条目；两函数**逐字**；十表合并缩单表 | must_keep ×2；m8/m9；测试断言新旧两布局条目解析 |
| `initialize_model` | model_loader/utils.py:L41-L64 | 新式签名校验三行**逐字**；老式 kwargs 段删（deprecated） | must_keep；站3；测试断言 (vllm_config, prefix) 契约 |
| `BaseModelLoader.load_model` | base_loader.py:L43-L82 | 四段主干**逐字**；观测/在线量化段删 | must_keep；m10；测试断言 eval() 与编排次序 |
| `DefaultModelLoader.load_weights` | default_loader.py:L415-L427 | 核心行 `model.load_weights(self.get_all_weights(...))` **逐字**；torchao/EP 前置删 | m6/站5；测试 SeededLoader 覆写 get_all_weights 走全链 |
| `process_weights_after_loading` | model_loader/utils.py:L100-L135 | 两轮主干**逐字**；UMA/HPC/torchao 尾段删 | must_keep；m10 |
| `GPUModelRunner.load_model` + `execute_model` 尾段 | gpu_model_runner.py:L5303-L5390 / L4432-L4456 / L4459-L4485 | 启动线 L5323-L5326 **逐字**；**L5362 `_setup_eagle3_aux_hidden_state_outputs()` 调用与 L5376-L5390 EPLB enable 块整块含头逐字保留**（delete[10] 明示——_moe_model 恒 None 块体不进；no-op 守卫 def L5483-L5484 逐字）；尾段 **aux 解包 L4459-L4465 + `if not self.broadcast_pp_output:` 头 L4467 + 主路径切片 L4484-L4485 逐字保留**（delete[11] 明示）；set_forward_context with 块签名减参 | delete[10]/[11]/[13]；m5/站6/站11；测试断言 L5362 调用发生、EPLB 块恒假、aux 二元组解包与主路径数值对照 |
| `logits_indices = query_start_loc[1:] - 1` | gpu_model_runner.py:L2232-L2240 | **逐字**（含 woosuk NOTE 五行原话） | must_keep ×2；测试两请求批断言 [4,7] |

## 删除账本（dossier.subtraction_plan.delete[0..13] 落点——修订版 dossier）

| delete | 内容 | 落点（# SUBTRACTED 标注） |
|---|---|---|
| [0] | EAGLE3 aux hidden states 线（llama.py 侧钩子） | llama.py：L419/L426-L428 采集、L437-L438 返回分支；bases 标记类保留声明。**runner 侧 L4459-L4465 解包段不删**（delete[11] 明示保留——见下） |
| [1] | layer_types/sliding_window 分支 | llama.py：L182-L201 + per_layer_sliding_window 传参（sliding_window 变量保留于 Attention.__init__ 的三态解析） |
| [2] | bias 兼容变体 | llama.py：L264-L271（固定 False 数值等价） |
| [3] | ENCODER_ONLY 分流与适配器 | llama.py：L203-L207 attn_cls 三元 / L273-L280 / EncoderOnlyAttention import / L543-L552 两适配器类 |
| [4] | logits_as_input/soft_cap/scale≠1 | logits_processor.py forward：L69-L81 对应分支 |
| [5] | _apply_head 的 head_dtype 分支 | logits_processor.py：L110-L135（head_dtype 配置位 __init__ 保留——m13 叙事锚） |
| [6] | get_top_tokens | logits_processor.py：L155-L205（LocalArgmaxMixin.get_top_tokens 保留声明——调用位归 ch33 回填） |
| [7] | weight_loader 量化/低比特特例 + weight_loader_v2 族 | linear.py：QKV 循环内量化块 L1266-L1307 + q/k/v 路径 L1329-L1374；Merged bnb raise L779-L787 + 循环内 L794-L810/L812-L821 + 单段 L836-L868；Row bnb 判整 L1719-L1723；is_sharded_weight 分支头（narrow 落成无条件）；parameter.py 量化参数族；vocab packed_dim 段；_maybe_allow_fp8_block_shape_mismatch **未删**（原样保留——未量化路径早退无操作）。**豁免保留**（checkpoint 格式面非量化面，不删）：QKV fused-on-disk 骨架 L1236-L1265 + 递归装载尾 L1309-L1313 + needs/警告尾 L1385-L1397；Merged L788-L793 构造+循环头 + L823-L826 narrow/递归尾 + L873-L886 尾部；adjust_scalar_to_fused_array（L116-L140 逐字）与 validate_shard_id 的 tuple 支（L729-L744 逐字）随之恢复 |
| [8] | registry 收缩 | registry.py：条目表→五条目单表；_RegisteredModel/探测族/transformers 后端/_PREVIOUSLY_SUPPORTED/OOT 名单删 |
| [9] | loader 族收缩 | model_loader/__init__.py：表只留 default/dummy；其余 loader 文件不进 |
| [10] | load_model 内三段（LoRA L5327-L5330 / drafter L5331-L5360 / MoE 解析 L5364-L5374，均连头删） | gpu_model_runner.py。**明示原样保留**：启动线 L5323-L5326、**L5362 的 `_setup_eagle3_aux_hidden_state_outputs()` 调用**（no-op 守卫 def L5482-L5484 恢复——use_aux 恒 False 首行即 return）、**L5376-L5390 EPLB enable 条件块整块含 if 头**（_moe_model 维持 __init__ L551 的 None → 条件第一项恒假、块体不进；支撑属性 L550-L551/L468/L512/L578 在切面 __init__ 恢复） |
| [11] | execute_model 尾段三处（①PP 中间 rank return 段 L4469-L4473 ②pooling 分支 L4475-L4482 ③PP 广播 else 块 L4486-L4514，均连头删不留悬空 if） | gpu_model_runner.py。**明示原样保留**：L4459-L4465 aux 解包段（use_aux 恒 False 恒走 else 常路）、L4467 `if not self.broadcast_pp_output:` 头 + L4484-L4485 主路径切片两行 |
| [12] | base_loader 观测/在线量化 | base_loader.py：L61/L66-L73/L77-L78；log_model_inspection/_has_online_quant 两函数删；L64 的 self.load_weights 编排调用保留 |
| [13] | gpu_model_runner 复用 ch18 产物 | 本章只新增两薄层（load_model 线 + execute_model 尾段）；_prepare_inputs 等不重刻；批元数据三件（query_start_loc/attn_metadata/slot_mapping）直供——SchedulerOutput 切面最小承载 |

## Seam 清单（HOST/ENGINE SEAM——真实代码之外唯一允许的承载，行内标注）

1. **`vllm/config/__init__.py`**（CONFIG SEAM）：VllmConfig 六 config namespace 的
   本章消费字段子集（ch03 域的数千行装配链不重建）；set/get_current_vllm_config
   语义逐字（compilation_counter 计数→ch19）。
2. **`vllm/distributed/__init__.py`**（DIST SEAM，ch34 领地）：divide/
   split_tensor_along_last_dim/get_pp_indices **逐字**；GroupCoordinator 以
   `_GroupCoordinatorSeam`（rank_in_group/world_size/first/last/is_first/
   is_last_rank）承载；集合通信保真实 world_size==1 旁路语义（parallel_state.py
   L677-L787 各 bypass 的同型），多进程面 raise 指引进容器。
3. **`vllm/platforms/__init__.py`**（PLATFORM SEAM）：Platform 基类默认行为位
   （interface.py 的 is_* 全 False / opaque_attention_op False→直调分支 /
   use_all_gather False→gather 通道）——host 无加速器即基类态。
4. **`vllm/envs.py` / `vllm/logger.py`**：VLLM_BATCH_INVARIANT=False /
   VLLM_PP_LAYER_PARTITION=None（真实默认值）；no-op logger。
5. **`vllm/forward_context.py`**（FORWARD CONTEXT 切面）：CUDAGraphMode/
   UBatchSlices 以 None 常量位承载（ch19/ch34 域）；create/set_forward_context
   装配主干逐字。
6. **`vllm/compilation/decorators.py`**：@support_torch_compile 签名与 docstring
   逐字；包装本体→ch19（原类直返——标记语义保留）。
7. **`vllm/ir/ops/layernorm.py`**：rms_norm/_fused_add_rms_norm native 数学逐字；
   IrOpInplaceOverload 的 torch.library 注册面→ch19，maybe_inplace 以直调 native
   承载（op.py:L529-L533 未启 torch wrap 的同款路径）。
8. **`torch.ops.*` 算子注册面**：attention.py 三个算子的 direct_register_custom_op
   注册块删（ch19）——直调 Python 函数体（同控制流）；`if self.calculate_kv_scales:`
   的守卫位保留（默认 False 不达）。
9. **`GPUModelRunner` 切面**（ENGINE SEAM，ch17/ch18 同款）：真实 __init__ 的
   模型/采样/cudagraph 装配面以装配参数直供；scheduler_output 以
   `_EmptySchedulerOutputSeam`（scheduled_spec_decode_tokens=[] 默认位）承载。
10. **`Attention.impl` 的 None 占位**：真实代码经 ch21 选择面必得 impl；本章
    选择面删除（→ch21），attn_backend=None 时 impl=None 延后绑定，测试按真实
    注入位（L251 参数）注入参考后端——与 kv_cache 的 placeholder-then-bind 同型。
11. **checkpoint IO**（default_loader）：Source/_prepare_weights/get_all_weights
    的 safetensors 流式族 SUBTRACTED（仓库 IO 面）；host 测试以 SeededLoader
    覆写 get_all_weights 注入 in-memory 权重流（同签名同语义）。

## 已知偏差（非 seam 的显式记录）

- `GPUModelRunner.load_model` 的观测/EPLB 预构面：logger.info_once 启动日志
  （L5308-L5312）、EplbState 预构（L5314-L5316，`eplb_models = 0` 计数位所在）、
  try/DeviceMemoryProfiler/time_before_load（L5318-L5320）与计时汇报
  （L5392-L5393）、except OutOfMemoryError 收尾（L5394 起）——**不在新
  delete[10] 的三段批准内**，按章界外惯例删（tracing 观测域 + EPLB 域 ch34）。
  后果显式记录：保留的 EPLB enable 块体内 `eplb_models += 1`（L5390 逐字）在
  块体恒不进的削减语义下为死代码位（其初始化位随 EplbState 预构删——同一
  恒假条件族，正常路径不可达）。
- `UnquantizedLinearMethod.apply` / `UnquantizedEmbeddingMethod.apply`：batch
  invariant 分支整支删除（ch20 域——envs 默认 False 不进）；`process_weights_
  after_loading` 的 CPU dispatch 分支整支删除（onednn/zentorch 平台 GEMM 族，
  layers/utils.py 域——删除后真实源码剩余体即本包）。
- `initialize_model`：老式模型类走不到新式分支时显式 raise（真实落入 L66-L97
  的 deprecated kwargs 猜参段；本章六条目全是新式，该段删除后以 raise 封底）。
- `_ModelRegistry._try_load_model_cls`：模块级 `_try_load_model_cls`（L1023-L1033，
  current_platform.verify_model_arch 包裹 + 异常吞噬）随探测族删除，方法体直调
  `self.models[arch].load_model_cls()`——异常直抛（平台校验位由 HOST SEAM no-op 承载）。
- `registry` 条目表：delete[8] 批准五条目（LlamaForCausalLM/DeepseekV32/V4/
  DSparkDraftModel/DeepSeekV4MTPModel）；真实 L92-L95 四行新旧布局对照由正文
  embed 摘录直引真源码（精简表只保留可解析的最小面）。
- `CustomOp.dispatch_forward`/`default_on`：compilation_config.custom_ops 的
  HOST SEAM 默认 ["none"]（Inductor 后端真实默认基模式——count_none==1，
  dispatch 走 native）。
- `tests` 侧的 RefAttentionImpl/RefAttentionBackend 为测试注入件（真实注入位 =
  attention.py L251 的 attn_backend 参数）；参考数学 = ch20 已立的精确注意力
  （KV 按 slot 写 cache、按请求窗读、GQA 广播、causal，双精度对照）。
