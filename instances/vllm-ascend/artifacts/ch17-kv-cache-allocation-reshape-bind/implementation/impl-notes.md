# ch16 implementation notes — KV cache 在昇腾上的分配 / reshape / 绑定

本章解读的真实源码（规范路径）：
- `vllm_ascend/worker/model_runner_v1.py`（主角：initialize_kv_cache_tensors / _allocate_kv_cache_tensors / _reshape_kv_cache_tensors / _adjust_kv_layout / _align_memory / _align_up / bind）
- `vllm_ascend/utils.py`（calc_split_factor / extract_dsv4_layer_index / kv_cache_spec_uses_sparse_c8 等几何算术）
- `vllm/v1/worker/gpu_model_runner.py`（对照基座：同名 initialize/_allocate/_reshape KV cache 方法——昇腾覆写它们改内存几何）

只做减法（subtract-only）精简版，镜像 `vllm_ascend/worker/model_runner_v1.py` 的 KV 显存几何路径。
host 无 NPU/CANN：真实 torch_npu 显存分配/物理布局不真跑；但本章核心（对齐算术、int8 裸分配、
K/V 字节拆分、as_strided 跨步重排、bind 三分支派发、kernel_block_sizes 装配）都是纯 Python/CPU
torch，可在 host 验证与真实仓一致的可观察控制流（`tests/test_kv_cache_geometry.py`，16 用例全过）。

## 1:1 Source Map

| 精简版符号 | 真实源码 (规范路径:行) | 改动 | 原因 |
|---|---|---|---|
| `NPUModelRunner.initialize_kv_cache` | `vllm_ascend/worker/model_runner_v1.py:L3700` | 删投机解码 drafter / kv_transfer register / routed_experts capturer 三段旁路注册 + `_bind_routed_experts_capturer`（L3729-L3756） | KV 几何主线之外的注册，删去不影响 allocate→reshape→bind 控制流（subtraction_plan.delete[0]） |
| `initialize_kv_cache_tensors` | `:L3764` | 全保留（含 bind 三分支：deepseek_v4 / longcat / 普通 + hamming） | must_keep：三步骨架与 bind 派发是本章主干 |
| `_allocate_kv_cache_tensors` | `:L3929` | 删单张量分支（mamba/linear/cache_only L3965-3980）与 use_compress 分支（L3981-3992）的分配体（保留分支条件注释）；删 sparse-c8 A5 设备分叉，保留 A3/通用一支 | 单张量分支与标准 attn 同构、只不拆 K/V；A5/A3 控制流同构（delete[1][2]）。must_keep 的 int8 裸分配/K-V 拆分/2MB 对齐全保留 |
| `_reshape_kv_cache_tensors` | `:L4144` | 删 hybrid attn+mamba conv_block_padding 切分（L4301-4331）；删 sparse-c8 A5 设备分叉（shape/dtype/装配），保留 A3 | conv padding 是 mamba block 专题（delete[3]）；A5/A3 同构（delete[2]）。保留标准 attn + MLA(nope/rope) + sparse-c8 A3 + mamba 骨架 |
| `_align_memory` | `:L3758` | 全保留 | must_keep：2MB 对齐怎么实现的原语 |
| `_align_up` | `:L3847` | 全保留 | must_keep：对齐算术原语 |
| `_allocate_int8_cache_tensor` | `:L3851` | 全保留 | must_keep：「KV 一律 int8 裸分配」统一物化点（含 kv_transfer 对齐分支） |
| `_allocate_sparse_c8_indexer_tensors` | `:L3874` | docstring 的 ASCII 图折叠 | must_keep：dsa_k/dsa_k_scale 共享一块对齐 int8 的两个视图 |
| `_adjust_kv_layout` | `:L4111` | 全保留 | must_keep：as_strided 按 page_size_bytes 跨步重排 NPU 物理布局 |
| `_get_attention_kv_cache_dims` | `:L3826` | 全保留 | must_keep：MLA 的 (k_dim,v_dim)=(kv_lora_rank,qk_rope_head_dim) 推导 |
| `_get_layer_kv_cache_specs` | `:L3815` | 全保留（辅助） | _allocate/_reshape 共用的 spec 解包，保 control flow 完整 |
| `may_reinitialize_input_batch` | `:L4464` | 删 pcp_manager.slot_mapping、CPU offload assert、MambaSpec 容量上调 | CP/offload/容量是正交细节（delete[4]）。保留 kernel_block_sizes 装配 + 重建判定 |
| `get_kv_cache_spec` | `:L4657` | 删 kv_sharing/普通 Attention/CacheOnly/MambaBase 分支 + mamba page_size 对齐尾段，保留 MLA→AscendMLAAttentionSpec | 本章对 spec 只需展示 MLA 这一回指 ch04 的落点（delete[5]） |
| `calc_split_factor`（utils.py） | `vllm_ascend/utils.py:L1441` | 全保留 | must_keep：K/V 字节拆分比例算术来源 |
| `extract_dsv4_layer_index`（utils.py） | `vllm_ascend/utils.py:L82` | 全保留 | must_keep：deepseek_v4 自定层序排序键 |
| `kv_cache_spec_uses_sparse_c8`（utils.py） | `vllm_ascend/utils.py:L1546` | 全保留 | must_keep：是否走 sparse-c8 几何的开关 |

## 验收判据自检
把真实源码删掉所有标 `# SUBTRACTED:` 的分支（单张量分支体 / use_compress 分配体 / A5 设备分叉 /
conv_block_padding / CP·offload·容量细节 / 非 MLA 的 spec 分支），≈ 得到本精简版。must_keep 的
18 个符号（含 `torch.as_strided` / `alignment` / `torch.int8` 等用法 token）全部在场，
`lint_fidelity.py` 通过、无 BLOCKING。

## A5 vs A3 的处理说明
真实仓按昇腾代际（`get_ascend_device_type()` == A5 / A3）对 sparse-c8 的 ratio 排布、indexer 三段
视图、ckv k_shape/dtype 各有分叉。本章 subtraction_plan.delete[2] 批准「保留 A3/通用一支、删 A5
专属重排」（控制流同构），故精简版不引入 `get_ascend_device_type`/`AscendDeviceType`，A5 分支均以
`# SUBTRACTED:` 注释点名其存在与所在行号。
