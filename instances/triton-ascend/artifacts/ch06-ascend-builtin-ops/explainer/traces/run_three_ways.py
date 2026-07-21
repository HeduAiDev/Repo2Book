"""ch06 素材驱动 ③:同一个 index_select 的三种写法对照(m13)。

三个 kernel 来自 pin 内官方用例
`third_party/ascend/unittest/pytest_ut/test_index_select.py`:
  A 手写基线      index_select_manual_kernel    (L45-L78)
  B 内建算子      index_select_extension_kernel (L82-L111)
  C 交给编译器    index_select_auto_kernel      (L117-L142)
它们都需要 torch_npu + 真机(宿主没有,见 INSTANCE.md),所以本驱动**按三个 kernel 的
真实循环结构逐句复刻成 numpy 版**(把 tl.load/tl.store/extension.* 换成对同一块假 GM
的读写记录器),跑同一组小输入:验证三条路结果一致,并数出三者对 GM 的**访存形状**差别
——数值语义与访存计数出自这次运行,`ttadapter` 阶段的下降落点出自静态源码锚点(见
lowering_evidence,已逐行核对)。

输出:three_ways.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "three_ways.json"


class GM:
    """假 GM:一块扁平 float32 缓冲区,记录每次访存请求的地址集合。"""

    def __init__(self, arr):
        self.buf = np.asarray(arr, dtype=np.float32).copy()
        self.reads = []      # 每次 load 请求:touched 是这次请求覆盖的扁平地址
        self.writes = []

    def load(self, flat_idx):
        flat_idx = np.asarray(flat_idx, dtype=np.int64)
        self.reads.append({"request": len(self.reads),
                           "flat_addrs": flat_idx.reshape(-1).tolist(),
                           "n_elems": int(flat_idx.size),
                           "contiguous": bool(np.all(np.diff(flat_idx.reshape(-1)) == 1))
                           if flat_idx.size > 1 else True})
        return self.buf[flat_idx]

    def store(self, flat_idx, value):
        flat_idx = np.asarray(flat_idx, dtype=np.int64)
        self.writes.append({"request": len(self.writes),
                            "flat_addrs": flat_idx.reshape(-1).tolist(),
                            "n_elems": int(flat_idx.size)})
        self.buf[flat_idx] = value


# ---- 输入:GM 里一张 4x4 的 fp32 表,取第 2、0 行 ----
OTHER_NUMEL, G_STRIDE = 4, 4      # src 是 (4, 4)
INDICE_LENGTH = 2
G_BLOCK, G_BLOCK_SUB, OTHER_BLOCK = 2, 2, 4
src = np.arange(16, dtype=np.float32).reshape(OTHER_NUMEL, G_STRIDE)
indices = np.array([2, 0], dtype=np.int64)
expected = src[indices]

results = {}

# ================= A:手写基线(get_element + 逐行 tl.load + insert_slice) =================
gm = GM(src.reshape(-1))
out_a = np.zeros((INDICE_LENGTH, G_STRIDE), dtype=np.float32)
ops_a = []
g_begin = 0
for goffs in range(0, G_BLOCK, G_BLOCK_SUB):
    g_idx = np.arange(G_BLOCK_SUB) + g_begin + goffs
    for other_offset in range(0, G_STRIDE, OTHER_BLOCK):
        tmp_buf = np.zeros((G_BLOCK_SUB, OTHER_BLOCK), dtype=np.float32)
        other_idx = np.arange(OTHER_BLOCK) + other_offset
        for i in range(G_BLOCK_SUB):
            gather_offset = int(indices[i]) * G_STRIDE      # extension.get_element(indices,(i,))
            val = gm.load(gather_offset + other_idx)        # tl.load:一段连续
            ops_a.append({"op": "tl.load", "row_i": i, "gather_offset": gather_offset,
                          "n_elems": OTHER_BLOCK})
            tmp_buf[i, :] = val                             # extension.insert_slice
            ops_a.append({"op": "extension.insert_slice", "offsets": [i, 0],
                          "sizes": [1, OTHER_BLOCK]})
        gm.store((g_idx[:, None] * G_STRIDE + other_idx[None, :]).reshape(-1),
                 tmp_buf.reshape(-1))
        ops_a.append({"op": "tl.store", "n_elems": G_BLOCK_SUB * OTHER_BLOCK})
        out_a[g_idx, :] = tmp_buf
results["A_manual"] = {
    "kernel": "index_select_manual_kernel (third_party/ascend/unittest/pytest_ut/"
              "test_index_select.py:L45-L78)",
    "uses": ["extension.get_element", "tl.load", "extension.insert_slice", "tl.store"],
    "gm_read_requests": len(gm.reads),
    "gm_read_shapes": [r["n_elems"] for r in gm.reads],
    "all_reads_contiguous": all(r["contiguous"] for r in gm.reads),
    "ops": ops_a, "reads": gm.reads,
    "ir_ops_emitted": len(ops_a),
    "ir_op_kinds": sorted({o["op"] for o in ops_a}),
    "result": out_a.tolist(), "matches_expected": bool(np.array_equal(out_a, expected)),
}

# ================= B:内建算子(index_select_simd 一条) =================
gm = GM(src.reshape(-1))
out_b = np.zeros((INDICE_LENGTH, G_STRIDE), dtype=np.float32)
ops_b = []
for goffs in range(0, G_BLOCK, G_BLOCK_SUB):
    g_idx = np.arange(G_BLOCK_SUB) + g_begin + goffs
    for other_offset in range(0, G_STRIDE, OTHER_BLOCK):
        other_idx = np.arange(OTHER_BLOCK) + other_offset
        # extension.index_select_simd(src_shape=(other_numel,g_stride), src_offset=(-1,0),
        #                             read_shape=(-1, other_block)) —— 每个 index 取一整条 tile
        tiles = []
        for in_idx in indices:
            tiles.append(gm.load(int(in_idx) * G_STRIDE + other_idx))
        tmp_buf = np.stack(tiles)
        ops_b.append({"op": "extension.index_select_simd", "n_index": len(indices),
                      "tile_shape": [1, OTHER_BLOCK], "n_elems": len(indices) * OTHER_BLOCK})
        gm.store((g_idx[:, None] * G_STRIDE + other_idx[None, :]).reshape(-1),
                 tmp_buf.reshape(-1))
        ops_b.append({"op": "tl.store", "n_elems": G_BLOCK_SUB * OTHER_BLOCK})
        out_b[g_idx, :] = tmp_buf
results["B_builtin"] = {
    "kernel": "index_select_extension_kernel (third_party/ascend/unittest/pytest_ut/"
              "test_index_select.py:L82-L111)",
    "uses": ["extension.index_select_simd", "tl.store"],
    "builtin_calls": 1,
    "gm_read_requests": len(gm.reads),
    "gm_read_shapes": [r["n_elems"] for r in gm.reads],
    "all_reads_contiguous": all(r["contiguous"] for r in gm.reads),
    "ops": ops_b, "reads": gm.reads,
    "ir_ops_emitted": len(ops_b),
    "ir_op_kinds": sorted({o["op"] for o in ops_b}),
    "result": out_b.tolist(), "matches_expected": bool(np.array_equal(out_b, expected)),
    "note": "index_select_simd 在 Python 层是一条 builtin;本驱动为了记录它实际覆盖哪些"
            "地址,按 interpreter 参考实现的语义(每个 index 一条连续 tile)展开成 2 次"
            "tile 读——真机上这是一条算子内部的事。",
}

# ================= C:交给编译器(算好偏移的普通 tl.load) =================
gm = GM(src.reshape(-1))
out_c = np.zeros((INDICE_LENGTH, G_STRIDE), dtype=np.float32)
ops_c = []
for goffs in range(0, G_BLOCK, G_BLOCK_SUB):
    g_idx = np.arange(G_BLOCK_SUB) + g_begin + goffs
    for other_offset in range(0, G_STRIDE, OTHER_BLOCK):
        other_idx = np.arange(OTHER_BLOCK) + other_offset
        src_offsets = indices[:, None] * G_STRIDE + other_idx[None, :]
        tmp_buf = gm.load(src_offsets)                  # 普通 tl.load,偏移是个张量
        ops_c.append({"op": "tl.load", "offset_shape": list(src_offsets.shape),
                      "n_elems": int(src_offsets.size)})
        gm.store((g_idx[:, None] * G_STRIDE + other_idx[None, :]).reshape(-1),
                 tmp_buf.reshape(-1))
        ops_c.append({"op": "tl.store", "n_elems": G_BLOCK_SUB * OTHER_BLOCK})
        out_c[g_idx, :] = tmp_buf
results["C_compiler"] = {
    "kernel": "index_select_auto_kernel (third_party/ascend/unittest/pytest_ut/"
              "test_index_select.py:L117-L142)",
    "uses": ["tl.load(逐元素偏移张量)", "tl.store"],
    "extension_calls": 0,
    "gm_read_requests": len(gm.reads),
    "gm_read_shapes": [r["n_elems"] for r in gm.reads],
    "ops": ops_c, "reads": gm.reads,
    "ir_ops_emitted": len(ops_c),
    "ir_op_kinds": sorted({o["op"] for o in ops_c}),
    "read_addr_order": gm.reads[0]["flat_addrs"],
    "read_is_one_contiguous_run": gm.reads[0]["contiguous"],
    "result": out_c.tolist(), "matches_expected": bool(np.array_equal(out_c, expected)),
}

report = {
    "driver": "explainer/traces/run_three_ways.py",
    "pin": "2badfc89e70a9b7a5e88463a116c2feddce4b101 (v3.2.1)",
    "environment": "host, 无昇腾 NPU/CANN;三个 kernel 按源码循环结构逐句复刻成 numpy 版跑,"
                   "数值与访存计数出自本次运行,不是真机;下降落点出自静态源码锚点",
    "params": {"src_shape": [OTHER_NUMEL, G_STRIDE], "src_values": src.tolist(),
               "indices": indices.tolist(), "expected": expected.tolist(),
               "g_block": G_BLOCK, "g_block_sub": G_BLOCK_SUB, "other_block": OTHER_BLOCK},
    "three_ways": results,
    "all_three_agree": bool(np.array_equal(out_a, out_b) and np.array_equal(out_b, out_c)),
    "lowering_evidence": {
        "C_compiler": [
            "third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp:L367 "
            "—— rewriter.create<triton::ascend::IndirectLoadOp>(...) 把间接寻址的 tt.load "
            "改写成 ascend.indirect_load(阶段=ttadapter)",
            "third_party/ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.td:L299 "
            "—— def IndirectLoadOp : TT_Ascend_Op<\"indirect_load\", ...>",
            "third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp:L2661 "
            "—— IndirectLoadConverter::matchAndRewrite(triton::ascend::IndirectLoadOp ...)",
        ],
        "B_builtin": [
            "third_party/ascend/triton_ascend.cc:L126-L184 —— create_index_select_simd 发出 "
            "ascend.index_select_simd(阶段=ttir)",
            "third_party/ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.td:L249 "
            "—— def IndexSelectSimdOp",
        ],
        "A_manual": [
            "third_party/ascend/triton_ascend.cc:L36-L51 —— create_extract_scalar 落到上游 "
            "tensor 方言(tensor::ExtractOp)",
            "third_party/ascend/triton_ascend.cc:L52-L116 —— create_insert_slice/"
            "create_extract_slice 落到上游 tensor 方言",
        ],
    },
}

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
for k, v in results.items():
    print(k, "ir_ops=", v["ir_ops_emitted"], v["ir_op_kinds"],
          "reads=", v["gm_read_requests"], v["gm_read_shapes"],
          "match=", v["matches_expected"])
print("all agree:", report["all_three_agree"])
print("written:", OUT)
