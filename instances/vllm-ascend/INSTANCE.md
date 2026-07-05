# 实例：vllm-ascend（vLLM-Ascend 源码解读：昇腾 NPU 后端如何接入 vLLM）

> 本文件 = 本实例的「源码版本 + 当前状态 + 专属规则」。通用方法论/工厂运转见仓库根 `CLAUDE.md`；实例配置见 `instances/vllm-ascend/repo2book.json`。
> 中文、高级读者。解读对象 = vllm-ascend（昇腾 NPU 的 vLLM 后端插件）。

## 源码版本与「配对依赖」（本书的关键前提）
- **锁定 vllm-ascend `v0.21.0rc1`**（commit `80610e44`，`instances/vllm-ascend/source/` 工作树即此版）。规范路径前缀 `vllm_ascend/…`。
- **配套并依赖 vLLM `v0.21.0`**（README 明示："CI commitment for vLLM main branch and vLLM v0.21.0 tag"）。这个基座**已经在本仓库** `instances/vllm/source/`（vllm 实例已锁 v0.21.0）——解读 vllm-ascend 时凡涉及它接入/改写的 vLLM 接口，直接对照该目录，无需另克隆。
- 因此本书是 vLLM 书的**姊妹篇**：vLLM 书讲引擎本体（CUDA 线），本书讲「同一个 v0.21.0 引擎如何被搬到昇腾 NPU 上」。

## 它是什么（一句话 解读）
vllm-ascend 是 vLLM 的 **out-of-tree 平台插件**：不改 vLLM 源码，而是经 **setuptools entry points** 把自己注册进去，再用 **monkey-patch** 替换昇腾上跑不动/跑不快的实现。
- 入口（`setup.py` entry points → `vllm_ascend/__init__.py`）：
  - `vllm.platform_plugins`: `ascend = vllm_ascend:register` → 返回 `vllm_ascend.platform.NPUPlatform`（`PlatformEnum.OOT`）。
  - `vllm.general_plugins`: `register_connector` / `register_model_loader` / `register_service_profiling` / `register_model`（在 engine-core 子进程里生效）。
- **两段式 patch**（`vllm_ascend/patch/`）：`platform/`（25 个，平台初始化期打，改 distributed / kv_cache_coordinator / multiproc_executor / mla_prefill_backend / mamba 等 vLLM 内部）+ `worker/`（22 个，worker 期打）。

## 子系统地形（~108k LoC Python + csrc/ AscendC 算子）
按体量与解读价值排序（详见 `book/cartography/ARCHITECTURE.md` 种子）：
`ops`(19k，昇腾自定义/融合算子) · `distributed`(14k，NPU 通信/并行) · `attention`(12.8k) · `worker`(12.6k，NPUWorker/ModelRunner) · `quantization`(6k) · `core`(3k，调度/KV) · `spec_decode`(2.7k) · `compilation`(2.5k，torchair/图模式) · `models`(2.2k) · `sample`(1.9k)；外加 `platform.py`（NPUPlatform 总入口）与 `patch/`（接入机制）。

## 实例专属硬规则
- 解读以 **vllm_ascend/** 为主线；对照基座写 `vllm/...`（指 `instances/vllm/source` @ v0.21.0），二者都用规范路径，**绝不**出现 `instances/*/source/`。
- 昇腾相关代码 host 无法跑（无 NPU/CANN）：精简版只验证可读控制流，行为以源码为准，不强求在本机运行 NPU 算子。
- 章节用 `ch`-前缀 slug，置于 `instances/vllm-ascend/artifacts/`。

## 当前状态（本 fork 完成）
- ✅ 锁定 v0.21.0rc1（80610e44）、blobless clone 进 source/。
- ✅ 摸清「配对依赖 vLLM v0.21.0 + OOT 插件 + 两段 patch」的接入骨架，写入本文件 + `cartography/ARCHITECTURE.md` 种子。
- ⏭ 下一步：补完整大纲（按子系统 + 接入机制分 Part），把顶层 `repo2book.json` 的 `active_instance` 切到 `vllm-ascend`，再逐章发车。

## Part VIII — 算法原理篇（primer，v4 新增）

- **规划**：`book/cartography/outline-final.json` 新增 Part VIII「算法原理篇：论文里的 DeepSeek」，6 章 `ch21`–`ch26`，全部 `mode: "primer"`：
  - `ch21-primer-mla`（MLA：低秩 KV 压缩/解耦 RoPE/权重吸收，deps `ch22`）
  - `ch23-primer-sparse-attention`（NSA→DSA/Lightning Indexer 谱系，deps `ch24`+`ch21`）
  - `ch34-primer-speculative-sampling`（拒绝采样定理+MTP+DSpark 前瞻，deps `ch35`）
  - `ch09-primer-eplb`（EPLB 均衡算法本体，deps `ch10`）
  - `ch31-primer-quantization`（量化数学：GPTQ/AWQ/SmoothQuant，deps `ch32`）
  - `ch26-primer-v4-csa-hca`（DeepSeek-V4 CSA/HCA 两级压缩混合注意力，deps `ch21`+`ch23`）
  - 发车顺序（见 RUNBOOK 发车阶段）：串行线 `ch21→ch23→ch26`（记号/概念递进），并行线 `ch34`/`ch09`/`ch31` 互不依赖。
- **硬规则 2 豁免范围**：CLAUDE.md HARD RULE 2「只做减法不做加法」的豁免**仅限 `kind=primer`** 的章——这 6 章的落地代码段仍是忠实参考实现（非杜撰），但正文主线是论文推导而非源码逐段精简，成对启用 `lint_paper_grounding` 门禁（`# PAPER` 全覆盖 + 正文出处可溯源）；其余 30 章（`ch01`–`ch36`，`mode: "code"`）不受影响，`lint_fidelity` 照常跑。
- **论文包位置**：`instances/vllm-ascend/book/papers/<slug>/`（`paper.md` 为主，部分章有辅助论文如 `ch23` 的 `paper-dsa.md`、`ch34` 的 `paper-mtp.md`、`ch31` 的 `paper-awq.md`/`paper-smoothquant.md`），`meta.json` 记来源；总索引 `book/cartography/papers-map.json`。
- **2026-07-04 gap 盘点**（`book-gap-audit` workflow 首跑）：全书 30 章码章体检出 6 处「悬崖」——正文引用了论文级机制但未展开推导，对应 6 章 primer 消解：
  | 悬崖 | 首现处 | 消解章 |
  |---|---|---|
  | 解耦 RoPE（为何不能吸收进 W_UK） | ch22 MLA on NPU | ch21 |
  | DSA/Lightning Indexer 谱系（NSA→V3.2 演进） | ch24 attention backend | ch23 |
  | 拒绝采样定理 + MTP 草稿机制 | ch35 | ch34 |
  | EPLB 重排算法本体（只讲工程接入未讲均衡算法） | ch10 | ch09 |
  | 量化数学（GPTQ/AWQ/SmoothQuant 推导） | ch32 昇腾量化框架 | ch31 |
  | DeepSeek-V4 CSA/HCA 两级压缩注意力 | ch24/ch25/ch36 一带而过 | ch26 |
  - 指路框补丁（6 章全部 APPROVED 后）：在 ch22/ch24/ch35/ch10/ch32 各定点插入一句「本章默认你已了解 X；其数学推导见第 NN 章」回指对应 primer 章（ch26 的 V4 CSA/HCA 暂无需单独补丁，随 ch24/ch25/ch36 的既有指路一并覆盖）；随后重跑 `book-gap-audit` 验证 6 处悬崖降级为「已建立/有指路」，报告存 `book/audits/`。

## 实例专属坑
1. 别写脱离代码的抽象——正文以真实 vllm_ascend 源码为主线、自包含内嵌，对照基座写 `vllm/...`。
2. implementer 别过度删减/误删——只删 `delete` 批准项，`must_keep` 必保留。
3. 标记完成前跑全部 linter（含 `--all` 锚点/半角/图几何）。
4. 昇腾相关代码 host 跑不了（无 NPU/CANN）；行号以 v0.21.0rc1（`80610e44`）为准。
5. 别赌自己的上下文——决策/状态写进 trace、Bible、本文件。

## 源码事实备忘（原 knowledge/ 归并，2026-07-04）
ch07（sleep-mode-camem，`vllm_ascend/device_allocator/camem.py`）：
- 是 `vllm/device_allocator/cumem.py` 的逐行同构移植：`cudart→acl.rt`、`vllm.cumem_allocator→vllm_ascend.vllm_ascend_C`、`torch.cuda.*→torch.npu.*`、`PYTORCH_CUDA_ALLOC_CONF→PYTORCH_NPU_ALLOC_CONF`；连过时注释与死变量 `libcudart=None` 都照搬。
- `HandleType=tuple(设备号,对齐字节数,设备VA=handle[2],物理句柄)`；`pointer_to_data` 账本 key=VA。`sleep()` 里 `unmap_and_release` 在 if 之外——所有 tag 物理页都释放，仅命中 `offload_tags` 的额外留 CPU pin 备份（D2H）；`wake_up()` 用原 handle 把物理页重映射回原 VA（指针不变）。
- 两档 sleep 全在 `worker.py:L207` 一行 `offload_tags`：`level1=('weights',)` 留权重丢 KV，`level2=tuple()` 全丢。`weights`/`kv_cache` tag 由 worker `load_model`（L548）/`initialize_from_config`（L767）的 `use_memory_pool(tag=...)` 贴上。
- `patch_camem_allocator.py` 在锁定的 vLLM v0.21.0 上是 no-op（base 无 `is_cumem_allocator_available`，全仓 grep 零命中）；真正放行 sleep mode 的是 `NPUPlatform.is_sleep_mode_available()=True`（`platform.py:L153`）被 `vllm/config/model.py:L507` 校验命中。文件名易误导，须查证真实生效路径。
- `acl.rt.memcpy(dst,destMax,src,count,kind)` 比 `cudaMemcpy(dst,src,n)` 多 `destMax` 上界（给 `size*2`）与显式方向枚举（`ACL_MEMCPY_DEVICE_TO_HOST=2`/`HOST_TO_DEVICE=1`），是 GPU→NPU 少数必须改而非改名处。
- `CaMemAllocator` 必须单例：C 扩展用一个全局变量存 malloc/free 回调，多实例会互相覆盖导致 free 回调弹错账本。

ch11（kv-transfer-pd，`vllm_ascend/distributed/kv_transfer/`）：
- vllm-ascend 从 `KVConnectorFactory._registry` 弹出 vLLM 内置 `MultiConnector` 再用同名重注册为 `AscendMultiConnector`；另按名注册 `MooncakeConnectorV1`/`MooncakeHybridConnector`/`MooncakeLayerwiseConnector`/`AscendStoreConnector`，用户配 `kv_connector='MultiConnector'` 透明拿到 HMA 感知的昇腾子类（`__init__.py:L21-55`）。
- `AscendMultiConnector(MultiConnector, SupportsHMA)` 只在 3 处偏离基类：`__init__` 的 HMA 断言、`update_state_after_alloc`（`MooncakeLayerwiseConnector` 即使非选中 loader 也**总是**拿到真实分配块，因为 layerwise PUSH/save 需要）、`request_finished_all_groups`（HMA 逐组释放）（`ascend_multi_connector.py:L36,L43-67`）。
- `MooncakeLayerwiseConnector` 是角色门控 facade：`KVConnectorRole.SCHEDULER` 持有 `MooncakeLayerwiseConnectorScheduler`，`WORKER` 持有 `MooncakeLayerwiseConnectorWorker`，基类钩子全转发；方向由 `request.kv_transfer_params` 驱动：`do_remote_prefill` ⇒ 本节点是 decoder/pull，`do_remote_decode` ⇒ 本节点是 prefiller/push（`mooncake_layerwise_connector.py:L690-704,L850-929`）。
- 逐层推送：`MooncakeLayerwiseConnectorWorker.save_kv_layer` 一算完某 transformer 层就把其 KV 塞进后台 `KVCacheSendingLayerThread`，`wait_for_save` 是 no-op；早层传输与晚层计算重叠（L1746-1763,L777-779）。
- mooncake P2P 用进程单例 `TransferEngine`（`GlobalTE`，双检锁），初始化 mode `'P2PHANDSHAKE'`、backend `'ascend'`；`get_transfer_meta` 算平坦 src/dst 绝对地址 `base_addr + block_id*block_len`，`group_concurrent_contiguous` 把连续 (local,remote) 块合并批量传输（`mooncake_transfer_engine.py:L11-29`；`mooncake_layerwise_connector.py:L285-440,L1922-1954`）。
- ★ KV 亲和（命中感知）路由在 `KVPoolScheduler.get_num_new_matched_tokens`：经 zmq REQ socket 调 `LookupKeyClient.lookup(token_len, request.block_hashes, kv_cache_group_ids)`，返回 `num_external_hit_tokens`；`need_to_allocate = max(0, hit - num_computed_tokens)`；prompt 全命中（`hit==num_tokens`）钳到 `hit-1` 保证至少 1 token 跑前向（`pool_scheduler.py:L224-293,L643-658`）。
- disagg proxy（`examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`）用按角色懒删除的最小堆负载均衡：prefill 优先级 = `active_tokens + 0.3*active_kv_cache`，decode 优先级 = `active_tokens`；`assign_instances` 先选 prefiller 发只 prefill 请求（`max_tokens=1, do_remote_decode=True`），从响应读 `kv_transfer_params`（remote_block_ids/engine_id/host/port），再选 decoder 转发 `do_remote_prefill`（L276-300,L790-946）。

ch14/ch15（model-runner，`vllm_ascend/worker/model_runner_v1.py`）：
- `NPUModelRunner(GPUModelRunner)` 继承 244KB/7179 行父类，只 override 4 处设备接缝：`_init_device_properties(num_sms=None)`/`_sync_device(torch.npu.synchronize)`/`capture_model`/`profile_cudagraph_memory`（L255,L580-584,L4798-4824）。
- ch14/ch15 对照判据是父类怎么表达设备差异：GPU Worker 用 else-raise 钉死 cuda（无接缝只能重写全文件）；GPU ModelRunner 用 override 钩子 + 散落 `torch.cuda.*` + 模块级 `graph_capture`/`CUDAGraphWrapper`（有接缝可继承+猴补）（`vllm/v1/worker/gpu_model_runner.py:L1056-1064,L6075-6106`）。
- 两个成对上下文管理器做临时猴补（L4890-4953）：`_torch_cuda_wrapper`（进程级 `torch.cuda.*→torch.npu.*`，try/except/finally，退出落稳态缺省非原样还原，失败 placeholder 兜底）+ `_replace_gpu_model_runner_function_wrapper`（setattr 父模块 `graph_capture`/`CUDAGraphWrapper`→NPU/ACL，`original_attrs` 备份 + finally 还原）。
- `_get_gpu_model_runner_module_name`（L4876-4887）沿 MRO 取 `GPUModelRunner.__module__`——必须 setattr 到父类所在模块，因父方法按其 `__globals__` 解析自由变量，改本模块父方法看不见。
- `capture_model`/`profile` 用未绑定方法 `GPUModelRunner.<m>(self)` 显式调父类（L620-625）；`_use_aclgraph` 三条件 `cudagraph_mode!=NONE ∧ mode==VLLM_COMPILE ∧ not enforce_eager`；`ACLGraphWrapper` 与 `CUDAGraphWrapper` 同形（前 4 位置参数 + `_all_instances`/`clear_all_graphs`）才可热替换（`compilation/acl_graph.py:L96-105`）。

ch18（310P specialization，`vllm_ascend/_310p/`、`vllm_ascend/utils.py`）：
- 310P 分流总开关 `is_310p()`（`utils.py:L122`）== `AscendDeviceType._310P`；真正判定在 `_init_ascend_device_type()`（L778），靠 `_build_info.__soc_version__` 含子串 `'310P'` → `'_310P'`，构建期烧死。`AscendDeviceType` 枚举 `A2=0/A3=1/_310P=2/A5=3`（A2 是 0 不是 1）。`SOC_VERSION_INFERENCE_SERIES=['Ascend310P3']`（L51）只是常量名，不参与分流。
- 310P 按组件挑继承深度（不是一刀切再子类化一层）：①主执行体三层 `NPUModelRunner310(NPUModelRunner)`/`NPUInputBatch310(NPUInputBatch)`；②`BlockTable` 特例——昇腾 `worker/block_table.py` 是独立类 `class BlockTable:`（不继承 vLLM `BlockTable`，只 import 复用 vLLM Triton `_compute_slot_mapping_kernel`），`_310p/block_table.py` 的 `BlockTable(AscendBlockTable)` 建其上；③KV 清零/权重加载与昇腾主栈无关，直接两层继承 vLLM 基类跳过昇腾中间层：`AscendKVBlockZeroer310(KVBlockZeroer)`、`ShardedStateLoader310(ShardedStateLoader)`。
- 310P 无 Triton 的连锁：`slot_mapping` 退到 CPU NumPy（`np.add(block_numbers*block_size, offsets, out=slot_mapping.np)` → copy_to_gpu），`_to_numpy` 强制 CPU 输入（device 张量直接 raise）。副作用：基座靠步末 Triton kernel（读 `block_table.gpu`）隐式形成 NPU 流序；310 改 CPU 丢了它，于是 `_update_states` 在 condense（`finished_req_ids` 非空、`move_row` 改写 `block_table.np`）步骤手动 `torch.npu.current_stream().synchronize()` 补回——只在布局变更步补（`_310p/block_table.py:L14-88`, `model_runner_310p.py:L106-117,L241-288`）。
- 310P 受限 KV cache：`initialize_kv_cache_tensors`（`model_runner_310p.py:L670`）对 KV-transfer/DeepSeek-Sparse/MLA 直接 `raise ValueError`（与 `platform.py:L752` `backend_map_310` 把 MLA/SFA 注释掉=不支持 闭环）；`_allocate` 用 `torch_npu.empty_with_format(acl_format=ACL_FORMAT_FRACTAL_NZ=29)` 分配，k_cache/v_cache 分开两块（非单 Tensor）；页注意力硬约束 `_ATTENTION_BLOCK_SIZE_LIMIT=128*128`，超了用 `block_size_chunk` 拆虚拟块。
- 310P 横切补丁 `patch_distributed.py:L33` `communication_adaptation_310p()` 把 `broadcast` 改成 `all_gather` 后取 src 项、`int64` `all_reduce` 改成 `all_gather`+本地 sum/max（CPU 张量/非 int64 走原生 fn）；只在 `get_ascend_device_type()==_310P` 才安装（L88）；代价 O(P*N) vs 原生 O(N)，只用于 int64 小张量故可接受。
- 310P 权重加载 `ShardedStateLoader310.save_model` 永远单 part（`part_idx=0`，无视 max_size，删了基座按 max_size 切多 part 的循环）；额外 `generate_quant_description` 逐参数判 dtype 产出 `parameters_type_map.json` 供 CANN/310P 量化加载。

ch22（MLA on NPU，`vllm_ascend/attention/mla_v1.py`）：
- `mla_v1.py`（1804 行）= `AscendMLABackend` 的 impl/builder。`ACL_FORMAT_FRACTAL_ND=2`/`ACL_FORMAT_FRACTAL_NZ=29`；`process_weights_after_loading` 对 `kv_b_proj` 用 FRACTAL_ND(2) 不是 29，`W_UK_T` 才经 `maybe_trans_nz` 转 NZ(29)（L928,L988；`utils.py:L54-55`）。
- 本版 `AscendMLAImpl.forward_mqa`/`forward_mha` 仅 `raise NotImplementedError`；真实 decode/prefill 分流在 `forward()->_mla_preprocess` 按 `has_decode`/`has_prefill` 分两路（decode 吸收/prefill 解压）（L1696-1716,L1640-1691）。
- 昇腾 MLA 融合算子：`npu_kv_rmsnorm_rope_cache`（RMSNorm+RoPE+写 KV cache；decode 取前 2 返回值，prefill `is_output_kv=True` 取后 2）、`npu_fused_infer_attention_score`（prefill TND）、`npu_fused_infer_attention_score_v2`（decode，K=V=k_nope 隐向量；图捕获先 `get_max_workspace` 预取）、`npu_attention_update`（chunked-context LSE 在线合并）、`npu_transpose_batchmatmul`（`_v_up_proj`）。
- absorb 形状代数：`W_UK_T=W_UK.permute(1,2,0)=(N,P,Lkv)`；`ql_nope=bmm(q_nope.transpose(0,1),W_UK_T)`；decode 对 latent 做 MQA 免解压 K，输出经 `W_UV` 投回 V；`decode_threshold=1+num_speculative_tokens<=16`（FIA TND 限制）（L910-922,L924-957,L261-271）。
- `mla_v1.py` 主路被旁支淹没（mlapo/fa_quant/A5 量化、Context-Parallel、ACL graph 捕获、MTP/spec padding、Flash-Comm allgather/layer_sharding）；减法只留 bf16+标准 PA+`head_padding=0`+非捕获 else 主线。

ch32（Ascend quantization framework，`vllm_ascend/quantization/`）：
- 三入口量化 `Config` 经 `@register_quantization_config` 注进 vLLM：`modelslim_config.py` `AscendModelSlimConfig('ascend')`；`compressed_tensors_config.py`/`fp8_config.py` 对 `'compressed-tensors'`/`'fp8'`（+`'deepseek_v4_fp8'`）先 `QUANTIZATION_METHODS.remove` 原版再同名注册顶替。
- scheme 注册表 `registry.py`：`_SCHEME_REGISTRY dict[(quant_type,layer_type)->cls]`，`@register_scheme` 登记，`get_scheme_class` 查表；`methods/__init__.py` import 各 scheme 模块触发装饰器填表。
- 三 wrapper（`method_adapters.py`）：`AscendLinearMethod(LinearMethodBase)`/`AscendKVCacheMethod(BaseKVCacheMethod)`/`AscendFusedMoEMethod(FusedMoEMethodBase)`，`self.quant_method=scheme`，`create_weights`/`apply` 全转交 scheme；`AscendEmbeddingMethod` 继承 `AscendLinearMethod`。
- `get_quant_method`（`modelslim_config.py:L512`）按 `isinstance(layer,...)` 四岔分发；FLOAT 层经 `is_layer_skipped_ascend` 走 `AscendUnquantized*` 跳过；逐层 `quant_type` 来自 ModelSlim `quant_model_description.json` 解析进 `quant_description`，`get_linear_quant_type` 按 `prefix+'.weight'` 查、融合层校验各 shard 同类型。
- `W8A8_DYNAMIC` linear scheme（`w8a8_dynamic.py`）：`get_weight` int8[out,in] + `get_perchannel_param` weight_scale/offset[out,1]；`apply` 走 `torch_npu.npu_dynamic_quant`（per-token）+ `npu_quant_matmul`；`process_weights_after_loading` 转置 + `maybe_trans_nz`（NZ）。NPU/CANN 硬特化 host 不可真跑。
- MXFP microscaling：`quant_type.py` `QuantTypeMapping` 给 `W8A8_MXFP8`/`W4A4_MXFP4`/`W4A8_MXFP` 指定 act/weight/scale dtype（`FLOAT8_E8M0FNU` per-group 共享指数）；`methods/__init__.py` `is_mx_quant_type` 标记，`create_weights` 据此设 `scale.input_dim=1`；W4A8 用 per-group `weight_scale_second[out,in//group_size]`。

ch33（sampling NPU adaptation，`vllm_ascend/sample/`）：
- `AscendTopKTopPSampler.self.q` 与 `self.async_event` 不在 `__init__` 初始化——由 `set_q_event` 在模型前向期间的 `do_async_exponential` 写入；`AscendSampler.__init__` 只建 `self.async_exponential_event = torch.npu.Event()`。
- `rejection_sampler.py` 的 `zero_threshold` 是函数内现造的 0.0 常量张量 `torch.tensor([0.0], pin_memory=True).to(device, non_blocking=True)`，acceptance 判据用它要求 draft 概率严格 >0 防除零。
- vLLM v1 主采样路径（PyTorch 原生/CUDA）不用 `torch.multinomial`，但 ROCm 的 aiter 旁路 `aiter_sample` 仍调它——写"全程没用"是过宽的。
- vLLM 上游 `random_sample`（`vllm/v1/sample/ops/topk_topp_sampler.py:L385`）早已用 Gumbel-max 等价式 `probs.div_(q).argmax(q~Exp(1))` 规避 `torch.multinomial` 的 CPU-GPU 同步；昇腾 `sampler.py:L19` 逐字继承，仅把指数随机用 `with npu_stream_switch(global_stream()):...wait_stream()` 两行包成异步。归属务必写准：数学来自上游，异步 stream 编排才是昇腾 delta。
- `AscendSampler`/`AscendTopKTopPSampler`/`AscendRejectionSampler` 都是薄壳：基类 `Sampler.sample`（`vllm/v1/sample/sampler.py:L232`）的温度派发/topk-topp/where 全继承不动，子类仅覆写 `apply_penalties`/`greedy_sample`/`forward_native`/`__init__` 装配等少数热点；`greedy_sample` 的 else 分支与基类逐字相同（`logits.argmax`），覆写只为加 `enable_reduce_sample` 多卡旁路。
- penalties/top-k-top-p/拒绝采样每个热点统一骨架 `if not HAS_TRITON:` 回退基类/`_pytorch`；`apply_all_penalties`（`penalties.py:L25`）与基座同签名，仅换内核为 `apply_penalties_triton`；`apply_top_k_top_p` 在模块加载期按芯片型号定派发（A2/A3→AscendC `npu_apply_top_k_top_p`，否则 `_apply_top_k_top_p_pytorch`）。
- `AscendTopKTopPSampler.forward_native`（`sampler.py:L145`）三分支优先级：`VLLM_BATCH_INVARIANT`→回退基类（与 ch30 确定性同源）；`enable_async_exponential`→复用预算 `self.q` 直接 `div_`/`argmax`；默认→`random_sample`；default off 的 `reduce_sample`/`async-exponential` 是次要优化，精简版按 subtraction_plan 删除。
- 投机解码拒绝采样判据：random 走 `target_token_probs/draft_token_probs >= uniform`（`rejection_sampler.py:L1036`）；被拒从残差 `max(0,target-draft)/q` argmax 重采（`sample_recovered_tokens_pytorch L1238`），残差重采同样用 `q~Exp(1)` 的 Gumbel-max；贪心走逐位比对 `draft==target_argmax`，全接受补 bonus。


## 章序交错(2026-07-06 生效)
- 36 章新序:六章原理章已归位(ch09 EPLB 原理/ch21 MLA/ch23 稀疏谱系/ch26 V4 收束/ch31 量化数学/ch34 投机采样),P8 解散并入 P3/P5/P7。映射存档 book/cartography/renumber-2026-07-05.json;补章走 RUNBOOK「补章发车 SOP」。
- 跨章链接一律 ../../ 两层+文字号=目录号(lint_anchors 三规);全书节号=目录号。
- 终验:gap-audit 2026-07-06 cliffs=0(唯一 cliff 已补指路),bumps 68 条 advisory 待日后 triage。
