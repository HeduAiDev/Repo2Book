"""ch28-m10 驱动脚本 —— MegaMoE 专家的 FP4 字节布局: 打包账 + UE8M0 尺度语义 + copy_ 破坏现场。

跑法(host, 纯 CPU torch): python run_ch28_m10_megamoe_fp4_bytes.py
输出: ch28_m10_megamoe_fp4_bytes.json(与本脚本同目录)

素材对应 dossier 机制 ch28-m10(needs_worked_example + needs_figure)。

真源:
- 参数形状 vllm/models/deepseek_v4/nvidia/model.py:L202-L246 verbatim 算术:
  w13_weight=[E_local, 2I, H//2] uint8 / w13_weight_scale=[E_local, 2I, H//32] uint8
  (quant_method='block') / w2 同构。两个 fp4 打包一字节(H//2), 每 32 值一个
  UE8M0 尺度字节(H//32)——OCP MXFP4 组尺度。
- E_local=2, H=128, I=128 取自 tests/models/test_deepseek_v4_mega_moe.py:L58-L67
  的真实测试构造(num_experts=4, num_local_experts=2, hidden 128, intermediate 128;
  该测试还构造了 128x64 的 uint8 专家权重量)。
- UE8M0→fp32 还原 (sf.to(int32)<<23).view(float32) 逐字拷贝自
  model.py:L310-L312(_ue8m0_uint8_to_float); 树内测试
  tests/models/test_deepseek_v4_mega_moe.py:L43-L54 断言 [0,126,127,128]→
  [0.0, 0.5, 1.0, 2.0], 本脚本复跑同值对拍 + 扩展 byte 120(=2^-7)。
- e8m0fnu 数值 copy_ 破坏: model.py:L1268-L1276 注释原文
  "copy_() would do a numeric conversion (e.g. 2^-7 → 0), destroying the raw
  exponent bytes" —— 本脚本用 torch.float8_e8m0fnu 真实 dtype 复现该破坏,
  对照 .view(torch.uint8) 按原始字节装载(model.py:L1277-L1279)。
- finalize 变换后丢弃原参数(model.py:L324-L362)与对称缓冲 7 元组键
  (model.py:L364-L391)为代码引用, 不在本脚本运行范围(deep_gemm/SM100 限定)。

三件:
① 形状→字节账: 四个 uint8 参数张量的字节数, 与 bf16 同逻辑权重对照(压缩比)。
② UE8M0 解码表: byte→2^(byte-127)(byte=0 特判 0.0), 与树内测试对拍。
③ copy_ 破坏现场: 2^-7 的 e8m0fnu 张量数值 copy 到 uint8 → 0;
   .view(uint8) → 120(原始指数字节)。
"""
import json
from pathlib import Path

import torch

torch.manual_seed(0)

E_LOCAL, E_TOTAL, H, I = 2, 4, 128, 128


def r(v, nd=6):
    return round(float(v), nd)


# ---- ① 形状→字节账 (model.py:L202-L246 verbatim 算术) ----
shapes = {
    "w13_weight [E_local, 2I, H//2] uint8": [E_LOCAL, 2 * I, H // 2],
    "w13_weight_scale [E_local, 2I, H//32] uint8": [E_LOCAL, 2 * I, H // 32],
    "w2_weight [E_local, H, I//2] uint8": [E_LOCAL, H, I // 2],
    "w2_weight_scale [E_local, H, I//32] uint8": [E_LOCAL, H, I // 32],
}
bytes_ledger = {}
for name, shp in shapes.items():
    numel = shp[0] * shp[1] * shp[2]
    bytes_ledger[name] = {"shape": shp, "numel": numel, "bytes": numel}

per_expert = {
    "w13_packed_bytes_per_expert": (2 * I) * (H // 2),
    "w13_scale_bytes_per_expert": (2 * I) * (H // 32),
    "w2_packed_bytes_per_expert": H * (I // 2),
    "w2_scale_bytes_per_expert": H * (I // 32),
}
total_fp4 = sum(bytes_ledger[n]["bytes"] for n in bytes_ledger)
bf16_w13 = E_LOCAL * (2 * I) * H * 2  # w1+w3 各 I*H 元素, 2B/元素
bf16_w2 = E_LOCAL * H * I * 2
total_bf16 = bf16_w13 + bf16_w2
per_value = {
    "fp4_bytes_per_value": 0.5 + 1 / 32,
    "bf16_bytes_per_value": 2.0,
    "compression_ratio_vs_bf16": round(total_bf16 / total_fp4, 4),
    "scale_overhead_fraction": round((1 / 32) / (0.5 + 1 / 32), 4),
}

# ---- ② UE8M0 解码表 (model.py:L310-L312 逐字) ----


def _ue8m0_uint8_to_float(sf: torch.Tensor) -> torch.Tensor:
    return (sf.to(torch.int32) << 23).view(torch.float32)


test_bytes = torch.tensor([0, 126, 127, 128], dtype=torch.uint8)  # 树内测试同值
ext_bytes = torch.tensor([0, 120, 121, 126, 127, 128, 129, 200], dtype=torch.uint8)
decoded_test = _ue8m0_uint8_to_float(test_bytes)
decoded_ext = _ue8m0_uint8_to_float(ext_bytes)
tree_test_expected = [0.0, 0.5, 1.0, 2.0]  # tests/models/test_deepseek_v4_mega_moe.py:L54-L57
cross_check_ok = decoded_test.tolist() == tree_test_expected

# ---- ③ e8m0fnu copy_ 破坏现场 (model.py:L1268-L1279 注释的实机复现) ----
sf_fp8 = torch.tensor([2.0**-7, 1.0, 2.0**3], dtype=torch.float8_e8m0fnu)
raw_bytes_view = sf_fp8.view(torch.uint8)  # 按原始字节: .view(torch.uint8)
numeric_dest = torch.zeros(3, dtype=torch.uint8)
numeric_dest.copy_(sf_fp8)  # 数值 copy_: uint8 装不下 0.0078125 → 0

out = {
    "env": "host Miniconda python 3.11.11, torch 2.11.0+cu128 (纯 CPU), pin=vLLM v0.27.1 (6e448d0ea)",
    "config_provenance": "E_local=2/H=128/I=128 取自 tests/models/test_deepseek_v4_mega_moe.py:L58-L67 (E_total=4)",
    "byte_ledger": bytes_ledger,
    "per_expert": per_expert,
    "totals": {
        "fp4_total_bytes": total_fp4,
        "bf16_equivalent_bytes": total_bf16,
        "bf16_w13_bytes": bf16_w13,
        "bf16_w2_bytes": bf16_w2,
        **per_value,
    },
    "ue8m0_decode": {
        "formula": "(sf.to(int32) << 23).view(float32)  # model.py:L310-L312 逐字",
        "tree_test_bytes": test_bytes.tolist(),
        "tree_test_decoded": decoded_test.tolist(),
        "tree_test_expected": tree_test_expected,
        "cross_check_ok": bool(cross_check_ok),
        "extended_bytes": ext_bytes.tolist(),
        "extended_decoded": decoded_ext.tolist(),
        "semantic": "byte b>0 → 2^(b-127); b=0 → 0.0(特判); 纯指数位模式=只表示 2 的幂",
        "byte_120_meaning": "2^-7 = 0.0078125 —— load_weights 注释里被数值 copy_ 杀死的那个值",
    },
    "copy_destruction": {
        "values_chosen": [r(2.0**-7), 1.0, r(2.0**3)],
        "view_uint8_raw_bytes": raw_bytes_view.tolist(),
        "numeric_copy_to_uint8": numeric_dest.tolist(),
        "claim": "数值 copy_ 把 2^-7(=0.0078125) 转成 0; .view(uint8) 保住原始指数字节 120",
    },
    "finalize_note": "finalize_weights(model.py:L324-L362): deep_gemm.transform_sf_into_required_layout + transform_weights_for_mega_moe 变换后 w13_weight/w13_weight_scale/w2_weight/w2_weight_scale 全部置 None(原参数丢弃); symm buffer 按 7 元组键 (group,device,E,max_tokens,topk,H,I) 跨层复用(model.py:L364-L391) —— 代码引用, 需 SM100/deep_gemm, 不在 host 复现范围",
}

dst = Path(__file__).parent / "ch28_m10_megamoe_fp4_bytes.json"
with open(dst, "w", newline="\n", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("wrote", dst)
print(json.dumps(out["ue8m0_decode"], ensure_ascii=True, indent=1))
print(json.dumps(out["copy_destruction"], ensure_ascii=True, indent=1))
