#!/usr/bin/env python3
"""ch10 explainer 驱动脚本 —— 发射缓存键相关机制的真值取证。

不 import 已安装的 triton 3.6.0（与本书 pin v3.2.0 不符）；改为把 pin
v3.2.0 源码 python/triton/runtime/jit.py 里相关的纯 Python 函数逐字复制进来
（每处标 # SOURCE: jit.py:Lxxx）并真跑，得真实 trace。只用 stdlib + torch
（torch 仅用来造真实张量以取 data_ptr()%16 的真对齐值）。

覆盖机制：spec-key-alignment / mangle-dtype / kernelparam-annotations /
jitfunction-init（signature→params/切装饰器留 src）/ binder-codegen /
launch-cache-key。
"""
import inspect
import json
import re

import torch

# ============================================================================
# 以下四个函数/字典逐字复制自 pin v3.2.0：python/triton/runtime/jit.py
# ============================================================================

# SOURCE: jit.py:L278-L288
def compute_spec_key(v, align):

    if align and hasattr(v, "data_ptr") and (v.data_ptr() % 16 == 0):
        return "D"
    elif isinstance(v, int):
        # bool is a subclass of int, so we don't check explicitly above.
        if align and (v % 16 == 0):
            return "D"
        elif v == 1:
            return "1"
    return "N"


dtype2str = {}  # SOURCE: jit.py:L291


# SOURCE: jit.py:L294-L318
def mangle_type(arg, is_const=False):

    if arg is None:
        return "none"
    elif isinstance(arg, bool):
        return "i1"
    elif isinstance(arg, int):
        if -(2**31) <= arg and arg <= 2**31 - 1:
            return "i32"
        elif 2**63 <= arg and arg <= 2**64 - 1:
            return "u64"
        else:
            return "i64"
    elif isinstance(arg, float):
        return "fp32"
    elif hasattr(arg, "tma_desc_cpu_ptr"):
        return "nvTmaDesc"
    else:
        # dtypes are hashable so we can memoize this mapping:
        dsk = (arg.dtype, is_const)
        res = dtype2str.get(dsk, None)
        if res is None:
            res = ("*k" if dsk[1] else "*") + type_canonicalisation_dict[str(dsk[0]).split('.')[-1]]
            dtype2str[dsk] = res
        return res


# SOURCE: jit.py:L413-L436 (仅保留本例会命中的条目 + 恒等回填循环 L438-439)
type_canonicalisation_dict = {
    "bool": "i1",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
    "float64": "fp64",
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
}
for _v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[_v] = _v


# SOURCE: jit.py:L222-L227
def _normalize_ty(ty) -> str:
    if isinstance(ty, type):
        return ty.__name__
    elif isinstance(ty, str):
        return ty
    return repr(ty)


# SOURCE: jit.py:L230-L275 (KernelParam，逐字；cached_property 用普通 property 代替，语义同)
class KernelParam:
    """Represents a parameter (name plus metadata) to a @jit'ed function."""

    def __init__(self, num, param, do_not_specialize, do_not_specialize_on_alignment):
        self.num = num
        self._param = param
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment

    @property
    def name(self):
        return self._param.name

    @property
    def annotation(self):
        if not self._param.annotation or self._param.annotation == inspect.Parameter.empty:
            return ""
        return _normalize_ty(self._param.annotation)

    @property
    def annotation_type(self):
        annotation = self.annotation
        for ty1, ty2 in [("uint", 'u'), ("int", 'i')]:
            width = annotation[annotation.find(ty1) + len(ty1):]
            if width and ty1 in annotation:
                return f"{ty2}{width}"
        if annotation == "bool":
            return "u1"
        return ""

    @property
    def is_constexpr(self):
        return "constexpr" in self.annotation

    @property
    def is_const(self):
        return "const" in self.annotation and not self.is_constexpr

    @property
    def default(self):
        return self._param.default

    @property
    def has_default(self):
        return self._param.default != inspect.Parameter.empty


# SOURCE: jit.py:L345-L410 (create_function_from_signature，逐字)
def create_function_from_signature(sig, kparams, backend):
    assert len(sig.parameters) == len(kparams)

    func_args = []
    dict_entries = []
    constexpr_vals = []
    non_constexpr_vals = []
    signature_types = []
    specialisations = []

    for ((name, sp), kp) in zip(sig.parameters.items(), kparams):
        if sp.default is inspect.Parameter.empty:
            func_args.append(name)
            dict_entries.append(f"'{name}': {name}")
        else:
            func_args.append(f"{name}=default_{name}")
            dict_entries.append(f"'{name}': {name}")
        if kp.is_constexpr:
            constexpr_vals.append(name)
        else:
            non_constexpr_vals.append(name)
            if not kp.do_not_specialize:
                if not kp.do_not_specialize_on_alignment:
                    specialisations.append('compute_spec_key(%s, align=True)' % name)
                else:
                    specialisations.append('compute_spec_key(%s, align=False)' % name)
            if kp.annotation_type:
                signature_types.append('"%s"' % kp.annotation_type)
            else:
                signature_types.append('mangle_type(%s, %s)' % (name, 'True' if kp.is_const else 'False'))

    cache_key = ''.join([x + ', ' for x in signature_types + specialisations])
    constexpr_vals = ''.join([x + ', ' for x in constexpr_vals])
    non_constexpr_vals = ''.join([x + ', ' for x in non_constexpr_vals])

    func_args.append('**excess_kwargs')

    args_str = ', '.join(func_args)
    dict_str = ', '.join(dict_entries)
    func_body = "def dynamic_func(%s):\n    return {%s}, (%s), (%s), (%s), excess_kwargs" % (
        args_str, dict_str, cache_key, constexpr_vals, non_constexpr_vals)

    func_namespace = {
        f"default_{name}": param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    func_namespace['mangle_type'] = mangle_type
    func_namespace['compute_spec_key'] = backend.compute_spec_key

    exec(func_body, func_namespace)
    return func_namespace['dynamic_func'], func_body


# 默认后端接缝：backend.compute_spec_key 默认委托 AttrsDescriptor.get_property_key，
# 其 D/1/N 逻辑与 compute_spec_key 逐字相同（backends/compiler.py:L206-L211）。
class _DefaultBackend:
    compute_spec_key = staticmethod(compute_spec_key)


# constexpr 注解替身：triton 里 tl.constexpr 是一个类，_normalize_ty 取其 __name__。
# 用一个同名类精确复现 _normalize_ty(constexpr) == "constexpr"。
class constexpr:  # noqa: N801  (故意小写以匹配 tl.constexpr.__name__)
    pass


out = {"note": "trace from pin v3.2.0 verbatim funcs; torch only for real tensor data_ptr alignment"}

# ---------------------------------------------------------------------------
# 运行示例内核（Triton 经典向量加）——注解仅用于取 signature，不真编译
# ---------------------------------------------------------------------------
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: constexpr):
    pass


sig = inspect.signature(add_kernel)
params = []
for i, p in enumerate(sig.parameters.values()):
    params.append(KernelParam(i, p, do_not_specialize=False, do_not_specialize_on_alignment=False))

# === 机制 jitfunction-init：signature→KernelParam 列表 / arg_names / constexprs ===
# SOURCE: jit.py:L705-L706
arg_names = [pp.name for pp in params]
constexprs = [pp.num for pp in params if pp.is_constexpr]
# SOURCE: jit.py:L681-L682  切掉装饰器行、只从第一个 def 起
raw_src = "@triton.jit\n" + inspect.getsource(add_kernel)  # 模拟带装饰器的源码
sliced_src = raw_src[re.search(r"^def\s+\w+\s*\(", raw_src, re.MULTILINE).start():]
out["jitfunction_init"] = {
    "arg_names": arg_names,
    "num_params": len(params),
    "params_enum": [{"index": pp.num, "name": pp.name, "is_constexpr": pp.is_constexpr} for pp in params],
    "constexprs_indices": constexprs,
    "constexpr_names": [params[i].name for i in constexprs],
    "raw_src_first_line": raw_src.splitlines()[0],
    "sliced_src_first_line": sliced_src.splitlines()[0],
    "sliced_src_starts_with_def": sliced_src.startswith("def "),
}

# === 机制 kernelparam-annotations：is_constexpr/is_const/annotation_type ===
# 用一组 str 注解（_normalize_ty 对 str 原样返回，忠实模拟归一后的注解字符串）
def _mk_param(annotation):
    def _f(x):
        pass
    prm = inspect.Parameter("x", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
    return KernelParam(0, prm, False, False)

anno_cases = ["constexpr", "int32", "uint32", "const int32", "bool", "fp16", ""]
anno_rows = []
for a in anno_cases:
    kp = _mk_param(a if a != "" else inspect.Parameter.empty)
    anno_rows.append({
        "annotation": a if a != "" else "(none)",
        "is_constexpr": kp.is_constexpr,
        "is_const": kp.is_const,
        "annotation_type": kp.annotation_type,
    })
out["kernelparam_annotations"] = anno_rows

# === 机制 spec-key-alignment：compute_spec_key D/1/N（真张量取 data_ptr%16）===
base = torch.empty(64, dtype=torch.float32)          # 分配器对齐，data_ptr%16==0
misaligned = base[1:]                                 # float32 偏移 4 字节
spec_rows = []
for label, v, dp_mod in [
    ("aligned tensor (base)", base, base.data_ptr() % 16),
    ("misaligned tensor base[1:]", misaligned, misaligned.data_ptr() % 16),
    ("int 1024", 1024, None),
    ("int 16", 16, None),
    ("int 1", 1, None),
    ("int 7", 7, None),
]:
    spec_rows.append({
        "input": label,
        "data_ptr_mod16": dp_mod,
        "spec_key": compute_spec_key(v, align=True),
    })
out["spec_key_alignment"] = spec_rows

# === 机制 mangle-dtype：mangle_type 编码 ===
mangle_rows = []
for label, v, is_const in [
    ("None", None, False),
    ("bool True", True, False),
    ("int 1024", 1024, False),
    ("int 2**40", 2**40, False),
    ("float 3.14", 3.14, False),
    ("float32 tensor", torch.empty(4, dtype=torch.float32), False),
    ("int8 tensor (const)", torch.empty(4, dtype=torch.int8), True),
    ("bf16 tensor", torch.empty(4, dtype=torch.bfloat16), False),
]:
    mangle_rows.append({"input": label, "is_const": is_const, "mangle": mangle_type(v, is_const)})
out["mangle_type"] = mangle_rows

# === 机制 binder-codegen：exec 生成 dynamic_func ===
backend = _DefaultBackend()
binder, func_body = create_function_from_signature(sig, params, backend)
out["binder_codegen"] = {
    "func_body": func_body,
    "num_signature_entries": 4,   # 4 个非 constexpr 参数 → 4 个 dtype 签名项
    "num_spec_entries": 4,        # 同 4 个参数 → 4 个 D/1/N 特化项
    "num_tuple_slots": 5,         # dynamic_func 返回 5 元组
}

# === 机制 launch-cache-key：拼发射键，展示同源码不同特化→不同键 ===
x = torch.empty(1024, dtype=torch.float32)
y = torch.empty(1024, dtype=torch.float32)
o = torch.empty(1024, dtype=torch.float32)

def emit_key(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE):
    bound_args, sig_and_spec, cev, ncv, excess = binder(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE)
    # SOURCE: jit.py:L583
    key = ''.join(sig_and_spec) + str((cev, excess))
    return sig_and_spec, cev, key

launches = []
# L1: n=1024(16 对齐→D), BLOCK=1024
s1, c1, k1 = emit_key(x, y, o, 1024, 1024)
launches.append({"desc": "n_elements=1024, BLOCK_SIZE=1024", "sig_and_spec": list(s1), "constexpr_vals": list(c1), "key": k1})
# L2: 只改 BLOCK=512（constexpr 值变）
s2, c2, k2 = emit_key(x, y, o, 1024, 512)
launches.append({"desc": "n_elements=1024, BLOCK_SIZE=512", "sig_and_spec": list(s2), "constexpr_vals": list(c2), "key": k2})
# L3: n=1000（非 16 对齐→N），BLOCK=1024
s3, c3, k3 = emit_key(x, y, o, 1000, 1024)
launches.append({"desc": "n_elements=1000, BLOCK_SIZE=1024", "sig_and_spec": list(s3), "constexpr_vals": list(c3), "key": k3})
# L4: 与 L1 完全同参（命中同键，不重编）
s4, c4, k4 = emit_key(x, y, o, 1024, 1024)
launches.append({"desc": "n_elements=1024, BLOCK_SIZE=1024 (repeat)", "sig_and_spec": list(s4), "constexpr_vals": list(c4), "key": k4})

distinct_keys = sorted(set(l["key"] for l in launches))
out["launch_cache_key"] = {
    "launches": launches,
    "num_launches": len(launches),
    "num_distinct_keys": len(distinct_keys),
    "distinct_keys": distinct_keys,
    "L1_eq_L4": k1 == k4,
}

print(json.dumps(out, indent=2, default=str))
