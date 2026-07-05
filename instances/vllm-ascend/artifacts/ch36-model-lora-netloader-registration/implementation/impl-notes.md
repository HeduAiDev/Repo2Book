# ch30 impl-notes —— 只做减法精简版 Source Map

全书最后一个代码章：把昇腾接进 vLLM 三个扩展点（模型注册 / LoRA 全局类替换 + 算子 / loader 注册）。
精简版与真实源码**同名、同结构、同控制流，只删不增**，删除处均标 `# SUBTRACTED:`，每 def/class 标 `# SOURCE:`。
host 无 NPU/CANN，真 bgmv/sgmv kernel、网络弹性传输、整模型前向不真跑——精简版只承载可在 host 验证的纯 Python 控制流（注册/分发/类替换/回退）。

## 1:1 Source Map

| 精简版文件 | 对位真实源码 | 改动（只删不增） | 原因 |
|---|---|---|---|
| `models_register.py` | `vllm_ascend/models/__init__.py:L1-L7` | 零删除（整文件 7 行）；仅改文件名（lint 跳过 `__init__.py`） | `register_model()` 调 `ModelRegistry.register_model` 把两个架构名映射到「`<module>:<class>`」懒加载字符串 |
| `deepseek_v4.py` | `vllm_ascend/models/deepseek_v4.py`（1521 行） | 删 ~1500 行 layer/forward/load_weights + 全部辅助类（`DeepseekV4Model`/`DeepseekV2MixtureOfExperts`/`Ascend*Cache`/…）；保留 imports 改动面 + `AscendDeepseekV4ForCausalLM` 类签名 | 本章只点「同一个 DeepSeek-V4 在 NPU 上改了哪几类东西 + 整模型被注册」，不逐行前向（subtraction_plan 批准；cartography 标「中」挑骨架） |
| `lora_utils.py` | `vllm_ascend/lora/utils.py:L1-L82` | 零删除 | 4 个 `Ascend*LinearWithLoRA` 薄壳（重写 `can_replace_layer` 改匹配目标为 `AscendQKVParallelLinear`）+ `refresh_all_lora_classes`（全局类替换 trick 本体） |
| `lora_ops.py` | `vllm_ascend/lora/lora_ops.py:L19-L122` | 内嵌 `bgmv_shrink`/`bgmv_expand` 两代表全保留；`bgmv_expand_slice`/`sgmv_shrink`/`sgmv_expand`/`sgmv_expand_slice` 4 个同构薄壳保留签名、删同构函数体 | 6 个薄壳同构（参数 reorder 后转调 `torch.ops._C_ascend.*`），留 2 代表 + 列其余签名（subtraction_plan 批准） |
| `punica_npu.py` | `vllm_ascend/lora/punica_npu.py:L14-L363` | 删各 `add_*/_apply_*/_shrink_*/_expand_*` 的 docstring 与 Semantics 伪代码块；控制流逐字保留 | LoRA 两招都在 `__init__`（① `refresh_all_lora_classes` ② device/rank 二选一绑 6 个 op）；其余方法是 shrink→expand 分发，语义由正文讲清（subtraction_plan 批准纯注释） |
| `netloader.py` | `vllm_ascend/model_loader/netloader/netloader.py:L1-L369` | 删 `__init__` 配置解析循环(L60-L126)、`load_model` elastic server 启动块(L241-L313)、draft 端口重写(L192-L206) + quant/model_config 深拷贝备份 + 失败释放(del/gc/empty_cache) | 与「注册到 loader 扩展点 + 弹性加载/失败回退」主题无关的样板/副作用/边角分支（subtraction_plan 批准）；保留装饰器+类签名+`load_model` 主干+`revert_to_default` |
| `vllm_registry.py` | `vllm/model_executor/models/registry.py:L931-L982,L1319` | 删 `register_model` 的 docstring/TypeError 校验分支、`_BaseRegisteredModel`/`_RegisteredModel`/`_LazyRegisteredModel` 三 dataclass + `_ModelRegistry` 其余方法、内置模型表初始化字典 | vLLM 侧扩展点(1)：只读「字符串懒加载 + 重名覆盖」注册主干（embed_excerpt 已 elide） |
| `vllm_model_loader.py` | `vllm/model_executor/model_loader/__init__.py:L48-L125` | 删 `register_model_loader` docstring 的 `>>>` 示例、`_LOAD_FORMAT_TO_MODEL_LOADER` 内置条目（留 auto/hf 代表） | vLLM 侧扩展点(3)：`register_model_loader` 装饰器 + 注册表 + `get_model_loader` 分发主干 |
| `vllm_lora_utils.py` | `vllm/lora/utils.py:L78-L124` | 删 `_all_lora_classes` 中间 12 个内置类（留 `VocabParallelEmbeddingWithLoRA` + `QKVParallelLinearWithLoRA` 代表）；`from_layer` 逐字保留 | vLLM 侧扩展点(2)：被昇腾全局追加的候选元组 + `from_layer` 顺序遍历选层（embed_excerpt 已 elide） |

## 验收
- `python3 scripts/lint_fidelity.py <chapter_dir>` → 保真度全过（must_keep 全在、每 def/class 有 SOURCE）。
- `python3 -m pytest tests/ -q` → 20 passed（host 纯控制流：注册映射 / 懒加载格式校验 / 重名覆盖 / 全局元组确定顺序追加 / `can_replace_layer` 严格 type 匹配 / `from_layer` 命中昇腾类 / device·rank 二选一绑 op / 薄壳参数 reorder / loader 注册+分发+非子类拒绝 / source 无效·弹性失败回退 / 量化分支）。

## 关键保真点（reviewer 注意）
- `bgmv_expand` 薄壳**丢弃 `add_inputs` 形参**、固定传 `offset=0 / size=output_tensor.size(1)`——与 C++ 签名对齐，非笔误（已测）。
- `refresh_all_lora_classes` 用**元组 splat 确定顺序**追加（vLLM #35077 把 `_all_lora_classes` 从 set 改成有序 tuple）。
- `register_model` 对已注册同名架构**覆盖**——正是昇腾顶替同名模型的机制（已测）。
- netloader **失败优雅回退** `DefaultModelLoader`：source 无效 / 本 rank 不在 source / `elastic_load` 返回 None 三条路径都回退（已测前两条 + None）。
