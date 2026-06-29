# ch17 精简版实现笔记 — 310P 推理芯片全栈特化（subtract-only）

本章精简版按 dossier `subtraction_plan` 对 vllm-ascend 真实源码**只做减法**：与真实代码
同名/同结构/同控制流，每处删除标 `# SUBTRACTED:`，每 def/class 标 `# SOURCE:`。立意：
310P 按组件挑继承深度——主执行体三层（vLLM→昇腾主栈→310）、BlockTable 建在昇腾独立类之上、
KV 清零/权重加载直接两层继承 vLLM 基类跳过昇腾中间层；差异集中在「无 Triton / 无 MLA /
受限 dtype·格式」，由 `is_310p()` 横切分流。

## 文件清单
| 精简版文件 | 真实源码 | 角色 |
|---|---|---|
| `utils.py` | vllm_ascend/utils.py | is_310p 全栈分流总开关 + ACL_FORMAT_FRACTAL_NZ |
| `block_table.py` | vllm_ascend/_310p/block_table.py | 310 BlockTable：CPU NumPy slot_mapping（替换 Triton） |
| `npu_input_batch.py` | vllm_ascend/_310p/npu_input_batch.py | NPUInputBatch310：唯一改动=换 _310p MultiGroupBlockTable |
| `model_runner_310p.py` | vllm_ascend/_310p/model_runner_310p.py | NPUModelRunner310：设备路径特化（4 主线落点） |
| `kv_block_zeroer.py` | vllm_ascend/_310p/kv_block_zeroer.py | 去 Triton 的 KV block 清零 |
| `sharded_state_loader_310p.py` | vllm_ascend/_310p/sharded_state_loader_310p.py | 单 part + parameters_type_map.json |
| `worker_310p.py` | vllm_ascend/_310p/worker_310p.py | 310 栈入口 worker |
| `patch_distributed.py` | vllm_ascend/patch/platform/patch_distributed.py | broadcast/all_reduce all_gather 模拟（ch06 伏笔收口） |
| `platform.py` | vllm_ascend/platform.py | worker_cls→NPUWorker310 + backend_map_310 |

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）
| 精简版符号 | 真实源码:行 | 改动 | 原因 |
|---|---|---|---|
| `BlockTable._compute_slot_mapping_numpy` | _310p/block_table.py:L19-L51 | 删 `dcp*pcp>1` CP 交错分支(L25-L43)，留 else 主路径逐字 | CP 正交，单卡 dcp=pcp=1 已演示 NumPy `block_numbers*block_size+offset` |
| `BlockTable._to_numpy` | _310p/block_table.py:L77-L88 | 逐字保留 | "must be CPU" 守卫点明 310P 规避 device 算术/D2H |
| `MultiGroupBlockTable.__init__` | _310p/block_table.py:L92-L158 | 删 `kv_cache_groups is None` 备用构造分支(L142-L158) | 主分支已说明「多 group 转发到 310 BlockTable」 |
| `MultiGroupBlockTable.compute_slot_mapping_draft` | _310p/block_table.py:L187-L202 | 整体删除 | draft 与 spec-decode 绑定（另章），转发逻辑与 compute_slot_mapping 同构 |
| `NPUInputBatch310.__init__` | _310p/npu_input_batch.py:L10-L59 | 逐字保留 | 「再子类化」最干净样本：唯一改动=末尾换 MultiGroupBlockTable |
| `NPUModelRunner310._prepare_inputs` | _310p/model_runner_310p.py:L241-L579 | 留主干(setup→positions np.add→compute_slot_mapping→各 buffer copy_to_gpu→非 spec logits_indices)，删 CP/PCP、prompt_embeds、async-scheduling、spec-decode、lmhead/lora 支路 | docstring 点题「无 Triton slot-mapping/无通用 NPU Add」；正交支路按 delete 折叠 |
| `NPUModelRunner310._update_states` | _310p/model_runner_310p.py:L106-L117 | 逐字保留 | 「CPU 替换 Triton 的连锁后果」：condense 步手动 stream.synchronize() 补流序 |
| `NPUModelRunner310._allocate_kv_cache_tensors` | _310p/model_runner_310p.py:L703-L791 | 删 linear_attn(Mamba)分支(L727-L745)，留 `attn` 分支逐字 | Mamba KV 正交；保留即说明 FRACTAL_NZ + 128*128 约束 |
| `NPUModelRunner310.initialize_kv_cache_tensors` | _310p/model_runner_310p.py:L670-L701 | 逐字保留 | 三处 raise = 受限硬件能力边界（MLA/Sparse/KV-transfer） |
| `NPUModelRunner310._prepare_input_ids` | _310p/model_runner_310p.py:L795-L887 | 留 `prev_sampled_token_ids is None` 正常调度主路径，删 async scatter 分支(L816-L886) | 异步调度正交 |
| `AscendKVBlockZeroer310` | _310p/kv_block_zeroer.py:L25-L82 | 逐字保留（全 82 行） | 去 Triton：收集 (k,v) 列表 + 切片 `.zero_()`，对照基类 Triton+绝对地址 |
| `ShardedStateLoader310` | _310p/sharded_state_loader_310p.py:L27-L80 | 逐字保留 | 单 part(去 max_size 分片) + parameters_type_map.json |
| `NPUWorker310` | _310p/worker_310p.py:L48-L77 | 留 init_device + save_sharded_state，删 determine_available_memory/_init_device/_warm_up_atb/_is_rc_device | RC 设备显存/启动细节正交 |
| `communication_adaptation_310p` | patch/platform/patch_distributed.py:L33-L89 | 逐字保留 | 横切回收(ch06)：all_gather 模拟 broadcast / int64 all_reduce |
| `get_attn_backend_cls` (backend_map_310) | platform.py:L738-L765 | 删 FA3 早返回分支(L743-L744) | backend_map_310 只有 (False,False)，MLA/SFA 注释掉=不支持 |
| `select_worker_cls_and_custom_ops` | platform.py:L602-L618 | 删 xlite 分支(L608-L610)、refresh_block_size(L614) | 入口选择点：is_310p→NPUWorker310 + 不开 custom_ops |
| `utils.is_310p / AscendDeviceType / _init_ascend_device_type` | utils.py:L122,L768-L816 | 删 check_ascend_device_type 运行期校验(L789-L809) | 「Ascend310P3 子串→_310P 枚举」分流主线；数值区间二次校验正交 |

## host 可跑性
昇腾 NPU/CANN/Triton 不在 host 真跑。精简版保留的纯 Python/NumPy/torch(CPU) 控制流——CPU
NumPy slot_mapping、`_to_numpy` CPU 守卫、KV 切片 `.zero_()`、`parameters_type_map.json`
生成、all_gather 模拟 all_reduce、is_310p 分流、backend_map_310 选择——在 `../tests` 用
sys.modules 桩注入后直接验证（33 passed）。重型 runner 子类（model_runner_310p/
npu_input_batch/worker_310p）触运行时无法 host 实例化，改以源码级结构断言验证子类化覆写点 +
must_keep 符号。
