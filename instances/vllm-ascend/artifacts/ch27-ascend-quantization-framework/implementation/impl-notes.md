# ch27 实现笔记 —— 昇腾量化框架（只做减法的精简版）

活动实例：`vllm-ascend`。规范路径前缀 `vllm_ascend/...`（正文/SOURCE 绝不带 `instances/.../source/`）。
本精简版与真实 `vllm_ascend/quantization/` 同名、同结构、同控制流，**只删不增**；每处删除标 `# SUBTRACTED:`，
每个 def/class 标 `# SOURCE: vllm_ascend/...:Lxxx`。真实量化 matmul / MXFP kernel 走 torch_npu（NPU/CANN），
host 不真跑——测试以「记录调用」替身验证入参/分流。

## 目录结构（镜像真实源码 `vllm_ascend/quantization/`）

> `implementation/` 直接代表真实包 `vllm_ascend/quantization/`（不再多套一层 `quantization/`）。
> 各文件对应真源：`vllm_ascend/quantization/modelslim_config.py`、`vllm_ascend/quantization/method_adapters.py`、
> `vllm_ascend/quantization/methods/registry.py`、`vllm_ascend/quantization/methods/w8a8_dynamic.py`、
> `vllm_ascend/quantization/compressed_tensors_config.py`、`vllm_ascend/quantization/fp8_config.py`。

```
implementation/
  quant_type.py              # QuantType 枚举（轻量、side-effect free）
  quant_parser.py            # MXFP dtype 映射表（QuantTypeMapping）
  method_adapters.py         # 三个适配器 wrapper
  modelslim_config.py        # 入口1 + 逐层解析 + get_quant_method 分发
  compressed_tensors_config.py # 入口2（先删后替换）
  fp8_config.py              # 入口3（先删后替换 + 别名复用）
  methods/
    __init__.py              # 全表装载点（import 即注册）+ is_mx_quant_type
    registry.py              # _SCHEME_REGISTRY + register_scheme + get_scheme_class
    base.py                  # 三类 scheme 的 ABC 契约
    w8a8_dynamic.py          # 走通全链的样本 scheme（linear + moe）
```

## 1:1 Source Map（精简版 ↔ 真实源码 ↔ 改动 ↔ 原因）

| 精简版符号 | 真实源码 | 改动 | 原因 |
|---|---|---|---|
| `methods/registry.py` 全文 | `vllm_ascend/quantization/methods/registry.py:L20-L62` | 原样保留 | 整章范式核心数据结构，零删减 |
| `methods/base.py` 三个 ABC | `methods/base.py:L42,L131,L186` | 原样保留 | wrapper 盲转交所依赖的固定契约 |
| `w8a8_dynamic.AscendW8A8DynamicLinearMethod.apply` | `methods/w8a8_dynamic.py:L73-L123` | 删 `_chunk_size>0` 大权重切两半分支，保留 else 单次 `npu_quant_matmul` | 针对 >=65536 维权重的 workaround，非主路径 |
| `w8a8_dynamic.*.process_weights_after_loading` | `methods/w8a8_dynamic.py:L125-L149,L350-L391` | 删 `wq_b`/`enable_dsa_cp` 与 `enable_fused_mc2`/`dynamic_eplb` 分支，保留 transpose+NZ+scale view 主干 | 高级特性，正交于「权重转算子友好排布」主干 |
| `w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod.apply` | `methods/w8a8_dynamic.py:L211-L348` | 删 zero_expert / multistream_overlap_gate / force_load_balance / fused_mc2 / dynamic_eplb 分支与 scale 列表准备，保留 `select_experts → build_fused_experts_input → moe_comm_method.fused_experts` 主干 | MoE 量化高级特性，与「量化 MoE 经 wrapper→scheme→fused_experts」主线正交 |
| `method_adapters.AscendLinearMethod.apply` | `method_adapters.py:L140-L162` | 删 RowParallelLinear 的 5 分支 `tp_rank` 选路，保留 `tp_rank=0` + `scheme.apply` 转交 | 并行通信场景细节，与「wrapper 转交 scheme」主线无关 |
| `method_adapters.AscendLinearMethod.create_weights` | `method_adapters.py:L51-L122` | 原样保留 | 「适配器只搬运、scheme 决定形状」的关键证据，不删 |
| `modelslim_config.get_quant_method` | `modelslim_config.py:L512-L581` | 删 minimax/bailing prefix 改写、C8 专属分支、logger.debug；保留 linear/attention(fa,indexer)/moe/embedding 四岔 + FLOAT 跳过 | 模型专属命名 / KV cache 深水区，不伤四岔分发骨架 |
| `modelslim_config.get_linear_quant_type` | `modelslim_config.py:L297-L331` | 原样保留 | 逐层 quant_type 解析 + 融合层一致性校验，主线 |
| `modelslim_config.create_scheme_for_layer` | `modelslim_config.py:L366-L398` | 原样保留 | 解析→查注册表→实例化的合流点，主线 |
| `modelslim_config.AscendModelSlimConfig` 其余 | `modelslim_config.py:L401-L839` | 删 packed_modules 大表（留 2 代表）、prefix/substr 映射、is_c8/enabling_fa/get_kv_quant_* 、maybe_update_config 长 ValueError 文案、_apply_extra hc_head_fn 块 | 静态数据表 / 模型专属字符串映射 / 错误文案，与主线无关 |
| `compressed_tensors_config.py` | `compressed_tensors_config.py:L41-L85` | 保留「先删后替换」+ Config 骨架；删 target_scheme_map 解析 / get_quant_method 等适配体 | 入口2 仅演示注册手法，LLM-Compressor 格式适配超出本章范围 |
| `fp8_config.py` 全文 | `fp8_config.py:L18-L125` | 原样保留（短文件） | 入口3：先删后替换 + deepseek_v4_fp8 别名复用同一 Config |
| `quant_parser.py` | `quant_parser.py:L11-L35` | 留 `QuantTypeMapping` + W8A8_MXFP8 一行示例；删 W4A4/W4A8 表项与 `parse_*`/`get_rollback_*` 函数 | MXFP 是「点出 NPU 硬特化」的旁证而非主线 scheme |

## 防过度删减自检

`subtraction_plan.must_keep` 的 26 个符号全部保留（`lint_fidelity` 通过：`register_quantization_config /
AscendModelSlimConfig / AscendCompressedTensorsConfig / AscendFp8Config / QUANTIZATION_METHODS /
_SCHEME_REGISTRY / register_scheme / get_scheme_class / get_quant_method / get_linear_quant_type /
create_scheme_for_layer / AscendLinearMethod / AscendKVCacheMethod / AscendFusedMoEMethod / create_weights /
apply / AscendLinearScheme / AscendMoEScheme / AscendW8A8DynamicLinearMethod / AscendW8A8DynamicFusedMoEMethod /
get_weight / get_perchannel_param / npu_dynamic_quant / npu_quant_matmul / QuantType / is_mx_quant_type`）。

## 验证

- `python3 scripts/lint_fidelity.py instances/vllm-ascend/artifacts/ch27-ascend-quantization-framework` → 全通过。
- `python3 -m pytest tests/test_quantization_framework.py -q` → 18 passed（host，无 NPU）。
