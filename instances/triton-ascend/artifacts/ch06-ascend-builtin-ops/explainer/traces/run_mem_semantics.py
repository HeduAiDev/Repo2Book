"""ch06 素材驱动 ①:四个 mem_ops 的数值语义(m2/m3/m4/m5)。

**跑的是 pin 内真实源码**:用 ast 从
`python/triton/runtime/ascend_interpreter.py`(pin 2badfc89, v3.2.1)里原样抽出
  - 模块级 `_compute_strides`
  - `AscendInterpreterBuilder.to_int_val`
  - `AscendInterpreterBuilder.create_gather_out_to_ub`
  - `AscendInterpreterBuilder.create_scatter_ub_to_out`
  - `AscendInterpreterBuilder.create_index_put`
  - `AscendInterpreterBuilder.create_index_select_simd`
函数体逐字未改,只把它们依赖的外部名字换成本文件的最小替身:
  - `TensorHandle` / `_get_np_dtype`:numpy 数组 + dtype 映射
  - `_interpreter.load/store`:一块扁平的假 GM 缓冲区(base 地址 0x1000)

宿主无昇腾 NPU/CANN(见 INSTANCE.md):**这些数字来自 pin 内的 interpreter 参考
实现,不是真机**——正文引用时必须如实标注。

输出:mem_semantics.json
"""
import ast
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
INSTANCE = HERE.parents[3]   # instances/triton-ascend
SRC = INSTANCE / "source/python/triton/runtime/ascend_interpreter.py"
OUT = HERE / "mem_semantics.json"

BASE = 0x1000          # 假 GM 基地址
ITEMSIZE = 4           # float32


class FakeDtype:
    def __init__(self, name):
        self.name = name
        self.scalar = self

    def __repr__(self):
        return "<%s>" % self.name


FP32 = FakeDtype("fp32")


class TensorHandle:
    def __init__(self, data, dtype):
        self.data = np.asarray(data)
        self.dtype = dtype


class PtrHandle(TensorHandle):
    """GM 裸指针替身:data 只装一个基地址,元素类型由 get_element_ty() 给出。"""

    def __init__(self, base, elem_ty=FP32):
        super().__init__(np.array([base], dtype=np.int64), elem_ty)
        self._elem_ty = elem_ty

    def get_element_ty(self):
        return self._elem_ty


def _get_np_dtype(_tt):
    return np.float32


class _Memory:
    """一块扁平的假 GM(base=0x1000, float32),记录每次 load/store 的明细。"""

    def __init__(self, values):
        self.buf = np.asarray(values, dtype=np.float32).copy()
        self.log = []
        self.notes = []

    def load(self, addr_array, mask_array, other_array, dtype_np):
        out = np.array(other_array, dtype=dtype_np).copy()
        for i, (a, m) in enumerate(zip(addr_array, mask_array)):
            if not m:
                self.log.append({"op": "load", "slot": i, "addr": None, "valid": False,
                                 "value": float(out[i])})
                continue
            flat = (int(a) - BASE) // ITEMSIZE
            out[i] = self.buf[flat]
            self.log.append({"op": "load", "slot": i, "addr": int(a), "flat_index": flat,
                             "valid": True, "value": float(out[i])})
        return out

    def store(self, addr_array, values, mask_array):
        if len(values) != len(addr_array):
            self.notes.append(
                "len(values)=%d != len(addresses)=%d —— create_index_put 只在合法分支里"
                " append values_to_store(ascend_interpreter.py 的 create_index_put),"
                "越界元素没有对应的 value 占位。本驱动按位置 zip 消费(合法槽位在前,"
                "两种解读一致);真实 C++ _interpreter.store 对长度不等的处理未验证。"
                % (len(values), len(addr_array)))
        for i, (a, v, m) in enumerate(zip(addr_array, values, mask_array)):
            if not m:
                self.log.append({"op": "store", "slot": i, "addr": None, "valid": False,
                                 "value": float(v), "note": "dropped"})
                continue
            flat = (int(a) - BASE) // ITEMSIZE
            self.buf[flat] = v
            self.log.append({"op": "store", "slot": i, "addr": int(a), "flat_index": flat,
                             "valid": True, "value": float(v)})
        for i in range(len(values), len(addr_array)):
            self.log.append({"op": "store", "slot": i, "addr": None, "valid": False,
                             "value": None, "note": "dropped (no value appended)"})


MEM = _Memory(np.zeros(1))          # 每个场景前重新绑定


class _InterpShim:
    def load(self, addr, mask, other, dtype_np):
        return MEM.load(addr, mask, other, dtype_np)

    def store(self, addr, values, mask):
        return MEM.store(addr, values, mask)


WANT = ["to_int_val", "create_gather_out_to_ub", "create_scatter_ub_to_out",
        "create_index_put", "create_index_select_simd"]

tree = ast.parse(SRC.read_text(encoding="utf-8"))
ns = {"np": np, "TensorHandle": TensorHandle, "_get_np_dtype": _get_np_dtype,
      "_interpreter": _InterpShim()}
picked = {}

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_compute_strides":
        mod = ast.Module(body=[node], type_ignores=[])
        exec(compile(mod, str(SRC), "exec"), ns)
        picked["_compute_strides"] = [node.lineno, node.end_lineno]
    if isinstance(node, ast.ClassDef) and node.name == "AscendInterpreterBuilder":
        for m in node.body:
            if isinstance(m, ast.FunctionDef) and m.name in WANT:
                mod = ast.Module(body=[m], type_ignores=[])
                exec(compile(mod, str(SRC), "exec"), ns)
                picked[m.name] = [m.lineno, m.end_lineno]

missing = [w for w in WANT if w not in ns]
assert not missing, missing


class Ref:
    def to_int_val(self, val):
        return ns["to_int_val"](self, val)


for name in WANT[1:]:
    setattr(Ref, name, ns[name])

ref = Ref()

report = {
    "driver": "explainer/traces/run_mem_semantics.py",
    "pin": "2badfc89e70a9b7a5e88463a116c2feddce4b101 (v3.2.1)",
    "semantics_source": "python/triton/runtime/ascend_interpreter.py(函数体逐字执行)",
    "extracted_from_lines": picked,
    "environment": "host, 无昇腾 NPU/CANN;数值来自 interpreter 参考实现,不是真机",
    "scenarios": {},
}

# ---------- 场景 A(m2):gather_out_to_ub ----------
src_2d = np.arange(12, dtype=np.float32).reshape(4, 3)   # 值 0..11, row-major
MEM = _Memory(src_2d.reshape(-1))
index = np.array([[0, 3], [5, 1]], dtype=np.int32)        # 5 越界(boundary=4)
out = ref.create_gather_out_to_ub(
    PtrHandle(BASE), TensorHandle(index, FP32), 4, 0,
    [3, 1], [2, 2], [0, 0], other=-1.0)

rows_a = []
for slot, entry in enumerate(MEM.log):
    coord = (slot // 2, slot % 2)
    idx_val = int(index[coord])
    rows_a.append({
        "slot": slot, "index_coord": list(coord), "index_value": idx_val,
        "in_bounds": bool(entry["valid"]),
        "src_coord": [idx_val, coord[1]] if entry["valid"] else None,
        "byte_offset": (entry["addr"] - BASE) if entry["valid"] else None,
        "loaded_value": entry["value"],
    })
report["scenarios"]["m2_gather_out_to_ub"] = {
    "params": {"src_shape": [4, 3], "src_values": src_2d.tolist(), "dtype": "float32",
               "src_stride": [3, 1], "index": index.tolist(), "index_boundary": 4,
               "dim": 0, "start_offset": [0, 0], "end_offset": [2, 2], "other": -1.0,
               "base_addr": BASE, "elem_bytes": ITEMSIZE},
    "per_element": rows_a,
    "result_tile": out.data.tolist(),
    "load_count": len([e for e in MEM.log if e["op"] == "load"]),
    "raw_log": MEM.log,
}

# ---------- 场景 B(m3):scatter_ub_to_out ----------
MEM = _Memory(np.zeros(12, dtype=np.float32))
value = np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
ref.create_scatter_ub_to_out(
    PtrHandle(BASE), TensorHandle(value, FP32), TensorHandle(index, FP32), 4, 0,
    [3, 1], [2, 2], [0, 0])

rows_b = []
for slot, entry in enumerate(MEM.log):
    coord = (slot // 2, slot % 2)
    rows_b.append({
        "slot": slot, "index_coord": list(coord), "index_value": int(index[coord]),
        "value": float(value[coord]), "in_bounds": bool(entry["valid"]),
        "dst_coord": [int(index[coord]), coord[1]] if entry["valid"] else None,
        "byte_offset": (entry["addr"] - BASE) if entry["valid"] else None,
    })
report["scenarios"]["m3_scatter_ub_to_out"] = {
    "params": {"dst_shape": [4, 3], "dst_stride": [3, 1], "value": value.tolist(),
               "index": index.tolist(), "index_boundary": 4, "dim": 0,
               "start_offset": [0, 0], "end_offset": [2, 2]},
    "per_element": rows_b,
    "dst_after": MEM.buf.reshape(4, 3).tolist(),
    "notes": MEM.notes,
    "raw_log": MEM.log,
}

# ---------- 场景 C(m4):index_put ----------
MEM = _Memory(np.zeros(12, dtype=np.float32))
value_c = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
index_c = np.array([2, 5], dtype=np.int32)               # 5 越界(boundary=4)
ref.create_index_put(
    PtrHandle(BASE), TensorHandle(index_c, FP32), TensorHandle(value_c, FP32),
    0, 4, [2, 2], [0, 1], [3, 1])

rows_c = []
for slot, entry in enumerate(MEM.log):
    coord = (slot // 2, slot % 2)
    iv = int(index_c[coord[0]])
    rows_c.append({
        "slot": slot, "value_coord": list(coord), "index_pos": coord[0], "index_value": iv,
        "value": float(value_c[coord]), "in_bounds": bool(entry["valid"]),
        "dst_coord": [iv, 1 + coord[1]] if entry["valid"] else None,
        "byte_offset": (entry["addr"] - BASE) if entry["valid"] else None,
    })
report["scenarios"]["m4_index_put"] = {
    "params": {"dst_shape": [4, 3], "dst_stride": [3, 1], "value": value_c.tolist(),
               "index": index_c.tolist(), "index_boundary": 4, "dim": 0,
               "start_offset": [0, 1], "end_offset": [2, 2]},
    "per_element": rows_c,
    "dst_after": MEM.buf.reshape(4, 3).tolist(),
    "notes": MEM.notes,
    "raw_log": MEM.log,
}

# ---------- 场景 D(m5):index_select_simd ----------
neighbour = np.arange(900, 908, dtype=np.float32)
MEM = _Memory(np.concatenate([src_2d.reshape(-1), neighbour]))
sel_in = ref.create_index_select_simd(
    PtrHandle(BASE), TensorHandle(np.array([2, 0], dtype=np.int32), FP32), 0,
    [4, 3], [-1, 0], [-1, 2], [2, 2])
log_in = list(MEM.log)

MEM = _Memory(np.concatenate([src_2d.reshape(-1), neighbour]))
sel_oob = ref.create_index_select_simd(
    PtrHandle(BASE), TensorHandle(np.array([2, 5], dtype=np.int32), FP32), 0,
    [4, 3], [-1, 0], [-1, 2], [2, 2])
log_oob = list(MEM.log)

report["scenarios"]["m5_index_select_simd"] = {
    "params": {"src_shape": [4, 3], "src_values": src_2d.tolist(), "dim": 0,
               "src_offset": [-1, 0], "read_shape": [-1, 2], "return_shape": [2, 2],
               "neighbour_values_after_src": neighbour.tolist()},
    "in_bounds": {
        "index": [2, 0], "result_tile": sel_in.data.tolist(),
        "tiles": [{"slot": e["slot"], "byte_offset": e["addr"] - BASE, "value": e["value"]}
                  for e in log_in]},
    "out_of_bounds": {
        "index": [2, 5], "result_tile": sel_oob.data.tolist(),
        "note": "index=5 超出 src_shape[0]=4;该算子没有 index_boundary 参数、不做任何检查,"
                "interpreter 参考实现按公式照算地址,读到了 src 之后的隔壁数据(第 15/16 个 float32,值 903/904)。"
                "真机上这是一次越界访问,行为未在本章验证。",
        "tiles": [{"slot": e["slot"], "byte_offset": e["addr"] - BASE, "value": e["value"]}
                  for e in log_oob]},
    "loads_per_index": 2,
    "gather_loads_for_same_tile": 4,
}

OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print("scenarios:", list(report["scenarios"]))
print("m2 result_tile:", report["scenarios"]["m2_gather_out_to_ub"]["result_tile"])
print("m3 dst_after:", report["scenarios"]["m3_scatter_ub_to_out"]["dst_after"])
print("m4 dst_after:", report["scenarios"]["m4_index_put"]["dst_after"])
print("m5 in :", report["scenarios"]["m5_index_select_simd"]["in_bounds"]["result_tile"])
print("m5 oob:", report["scenarios"]["m5_index_select_simd"]["out_of_bounds"]["result_tile"])
print("written:", OUT)
