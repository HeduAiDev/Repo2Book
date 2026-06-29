# ch18 精简版实现笔记（subtract-only）

本章只讲「一个 OOT 后端要接进 vLLM 注意力框架，需要实现/伪装哪些契约点」——**后端选择(路由) +
注册占位 + get_name 伪装 + 静态契约 + 运行期分流**。4 个昇腾后端实体（AscendMLABackend /
AscendAttentionBackend / AscendSFABackend / AscendDSABackend）的算子留 ch19-22 展开，本章只点名其
在 backend_map 中的 key。

精简版纯 Python，可在 host 跑（真实 NPU 注意力算子不真跑）；`../tests/` 以 sys.modules 桩掉
NPU/CANN 重依赖，按规范模块名加载这 6 个文件，验证与真实仓一致的可观察控制流。

## 1:1 Source Map

| 精简版文件:符号 | 真实源码 (规范路径:行) | 改动 | 原因 |
|---|---|---|---|
| `backend.py` : `AttentionBackend`（4 个 @abstractmethod + get_supported_kernel_block_sizes + MultipleOf） | `vllm/v1/attention/backend.py:L48-L96` | 删 AttentionType/quant import 与 L98+ 一众带默认实现的 @classmethod | 本章只锁「契约的 4 个 @abstractmethod + 可覆写的 get_supported_kernel_block_sizes」；其余默认方法与契约对账正交 |
| `registry.py` : `AttentionBackendEnum`(FLASH_ATTN/CUSTOM) + `register_backend` + `_ATTN_OVERRIDES` | `vllm/v1/attention/backends/registry.py:L18-L90, L194-L255` | 删几十个内建后端枚举成员、get_path/get_class/is_overridden、mamba override 分支 | 本章只需 FLASH_ATTN（伪装/FA3 判定）与 CUSTOM（OOT 占位槽）；真正解析走 get_attn_backend_cls 点分路径，不依赖 CUSTOM.get_class() |
| `flash_attn.py` : `FlashAttentionBackend.get_name` | `vllm/v1/attention/backends/flash_attn.py:L69, L105-L107` | 删全部 CUDA FA 算子实现（约上千行），只留 get_name | get_name 是昇腾伪装冒充的「样板对象」，算子非本章主线 |
| `selector.py` : `_cached_get_attn_backend` + `resolve_obj_by_qualname` | `vllm/v1/attention/selector.py:L105-L136` + `vllm/utils/import_utils.py:L104-L110` | 删 required_layout / set_kv_cache_layout 旁支（dossier elide 批准「可一句带过」） | 主路是「点分路径字符串 → resolve_obj_by_qualname → 后端类」，KV layout 调整非本章主线 |
| `attention_v1.py` : `AscendAttentionBackend`（get_name 伪装 / get_impl·builder_cls / get_kv_cache_shape / swap·copy_blocks / get_supported_kernel_block_sizes） | `vllm_ascend/attention/attention_v1.py:L73-L140` | 删顶部 torch_npu/acl_graph/... 数十行 import 与 AscendAttentionBackendImpl(L357+)/Metadata 算子实体（→ ch19）；CP 实现与 swap/copy 的 `.to(device)` 用占位替身/CPU 张量 | 本章只要契约骨架；CP 内核（ch15 延伸）与 MHA 算子（ch19）非本章；host 无 NPU 不可真搬运 |
| `platform.py` : `NPUPlatform.get_attn_backend_cls` | `vllm_ascend/platform.py:L738-L765` | 删 FA3 早返回 + `_validate_fa3_backend`（L743-L744, L767-L792）与 310p 旁支（L752-L763），保留注释占位 | FA3/310p 是两条旁支特例，主路由走三元 key backend_map；详见 ch17（310p） |

## must_keep 落点核对

三元 key（`use_mla`/`use_sparse`/`use_compress`，含 getattr 默认 False 前向兼容写法）、`backend_map`、
4 后端类名（`AscendMLABackend`/`AscendAttentionBackend`/`AscendSFABackend`/`AscendDSABackend`）、
`get_attn_backend_cls`、`register_backend`/`AttentionBackendEnum`/`CUSTOM`、`get_name`/`FLASH_ATTN`/
`VLLM_USE_V2_MODEL_RUNNER`、`get_impl_cls`/`get_builder_cls`/`enable_cp`、`get_kv_cache_shape`、
`swap_blocks`/`copy_blocks`、`get_supported_kernel_block_sizes`、`AttentionBackend`、
`resolve_obj_by_qualname` —— 全部原样保留，`lint_fidelity` 校验通过。

## 与真实行为对照（tests）

- 三元 key → 4 后端路由（含 use_compress getattr 默认 False、DSA 仅在带该字段时可达）。
- `@register_backend(CUSTOM,"ASCEND")` → `_ATTN_OVERRIDES[CUSTOM]=="ASCEND"`、装饰器 no-op（类不被改写）。
- `get_name` 在 V2 model-runner 下伪装成 "FLASH_ATTN"，与被冒充的 `FlashAttentionBackend.get_name()` 相等。
- 4 个 @abstractmethod 强制：缺一即 `TypeError` 无法实例化；昇腾后端全实现可实例化。
- `swap_blocks`/`copy_blocks` 不在基座 v1 契约（基座 `AttentionBackend` 无此二属性），按 (2,...) 布局做块级索引搬运。
- `get_impl_cls`/`get_builder_cls` 按 `enable_cp()` 二选一。
- selector → `get_attn_backend_cls`(点分路径) → `resolve_obj_by_qualname` → 后端类 端到端解析。
