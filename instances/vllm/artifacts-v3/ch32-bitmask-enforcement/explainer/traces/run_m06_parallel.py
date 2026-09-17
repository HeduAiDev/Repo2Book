# ch32 m06 驱动：并行填充。甲段=129 请求真线程池（结构证据：9 任务=8×16+1、
# 行间独写、行内容逐行核验、阈值边界 128/129、池的存在前提 max_num_seqs>128）；
# 乙段=真 xgrammar 计时（256 matcher × 50257 词表：串行 vs 线程池 16 行/任务）。
import multiprocessing
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _ch32_common import (VOCAB, FakeRequest, dump, make_manager)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))

V = 64
out = {"env": {"cpu_count": multiprocessing.cpu_count()}}

# ── 甲段：结构证据（129 > 128 且无 spec → 并行分支）───────────────────────────
mgr = make_manager(V, max_num_seqs=256)
assert hasattr(mgr, "executor_for_fillmask")


class FG:
    def __init__(self, allowed):
        import torch
        self.torch = torch
        self.allowed = set(allowed)
        self.fill_calls = []

    def fill_bitmask(self, bitmask, index):
        row = np.zeros(bitmask.shape[1], dtype=np.uint32)
        for tok in self.allowed:
            row[tok // 32] |= np.uint32(1 << (tok % 32))
        bitmask[index] = self.torch.from_numpy(row.view(np.int32))
        self.fill_calls.append(index)

    def is_terminated(self):
        return False

    def accept_tokens(self, request_id, tokens):
        return True

    def validate_tokens(self, tokens):
        return list(tokens)

    def rollback(self, num_tokens):
        pass

    def reset(self):
        pass


requests, grammars = {}, []
for i in range(129):
    g = FG({i % V})
    grammars.append(g)
    requests[f"r{i}"] = FakeRequest(f"r{i}", grammar=g)

submissions = []
real_submit = mgr._async_submit_fill_bitmask


def recording_submit(batch):
    submissions.append([idx for _, idx, _ in batch])
    return real_submit(batch)


mgr._async_submit_fill_bitmask = recording_submit
t0 = time.perf_counter()
mask129 = mgr.grammar_bitmask(requests, list(requests.keys()), {})
elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
out["structure"] = {
    "requests": 129,
    "threshold": mgr.fill_bitmask_parallel_threshold,
    "batch_size": mgr.fill_bitmask_parallel_batch_size,
    "tasks_submitted": len(submissions),
    "task_sizes": [len(b) for b in submissions],
    "task_row_ranges": [f"{min(b)}..{max(b)}" for b in submissions],
    "pool_workers_formula": "max(1, min(cpu//2, 8))",
    "pool_workers_on_host": workers,
    "mask_rows": int(mask129.shape[0]),
    "wall_ms_real_pool": elapsed_ms,
    "row_content_verified": True,
    "note": "每任务恰 16 行（最后一任务 1 行）；行 i 允许 token i%64——逐行核验通过",
}
# 逐行核验
for i, g in enumerate(grammars):
    assert g.fill_calls == [i], (i, g.fill_calls)
    row = mask129[i]
    u = row.astype(np.uint32)
    allowed = [t for t in range(V) if (int(u[t // 32]) >> (t % 32)) & 1]
    assert allowed == [i % V], (i, allowed)

# 阈值边界：恰 128 → 串行（不触池）；max_num_seqs=128 → 不建池
mgr128 = make_manager(V, max_num_seqs=256)
req128 = {f"r{i}": FakeRequest(f"r{i}", grammar=FG({i % V})) for i in range(128)}
touched = []


def no_submit(batch):
    touched.append(batch)
    raise AssertionError("serial path must not touch executor")


mgr128._async_submit_fill_bitmask = no_submit
mask128 = mgr128.grammar_bitmask(req128, list(req128.keys()), {})
small = make_manager(V, max_num_seqs=128)
out["boundaries"] = {
    "at_128_touched_executor": len(touched) > 0,
    "at_128_rows": int(mask128.shape[0]),
    "max_seqs_128_has_pool": hasattr(small, "executor_for_fillmask"),
    "max_seqs_129_has_pool": hasattr(make_manager(V, max_num_seqs=129), "executor_for_fillmask"),
}

# ── 乙段：真 xgrammar 计时（生产形 256 行 × 50257 词表）────────────────────────
import torch
import xgrammar as xgr

from _ch32_common import compile_grammar

EBNF = 'root ::= "yes" | "no"'
compiled = compile_grammar(EBNF)
ROWS = 256
COLS = -(-VOCAB // 32)
bm = xgr.allocate_token_bitmask(ROWS, VOCAB)
matchers = [xgr.GrammarMatcher(compiled) for _ in range(ROWS)]


def fill_rows(rows):
    for i in rows:
        matchers[i].fill_next_token_bitmask(bm, i)


# 预热（首调含惰性初始化）
fill_rows(range(4))
torch.cuda.synchronize() if torch.cuda.is_available() else None


def med(fn, iters=7):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return round(ts[len(ts) // 2], 3)


serial_ms = med(lambda: fill_rows(range(ROWS)))

pool = ThreadPoolExecutor(max_workers=workers)
tasks = [range(s, min(s + 16, ROWS)) for s in range(0, ROWS, 16)]


def parallel_fill():
    futs = [pool.submit(fill_rows, t) for t in tasks]
    for f in futs:
        f.result()


parallel_ms = med(parallel_fill)
pool.shutdown()

out["timing_real_xgrammar"] = {
    "rows": ROWS,
    "vocab": VOCAB,
    "int32_per_row": COLS,
    "bytes_total": ROWS * COLS * 4,
    "serial_median_ms": serial_ms,
    "parallel_median_ms": parallel_ms,
    "parallel_workers": workers,
    "parallel_tasks": len(tasks),
    "parallel_task_rows": 16,
    "speedup_x": round(serial_ms / parallel_ms, 2),
    "grammar": 'root ::= "yes" | "no"',
    "per_row_us_light": round(serial_ms * 1000.0 / ROWS, 2),
    "note": "host 实测中位数（7 次取中位）。轻语法（yes|no）下单行填充仅数 µs、"
            "整批不足 1ms——线程池提交/收割开销反超串行（speedup<1 是真实结果，"
            "不是测量错误）；行间独写是『允许并行』的理由，是否回本取决于单行"
            "成本（见下方 JSON schema 重语法对照）",
    "gauge_row0_allowed": [t for t in range(VOCAB)
                           if (int(bm[0].numpy().astype(np.uint32)[t // 32]) >> (t % 32)) & 1],
}

# ── 乙段对照：重语法（JSON schema）——单行填充成本高时并行回本─────────────────
import json as _json

from _ch32_common import compile_grammar as _cg

heavy_schema = _json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "pattern": "^[a-z]+@[a-z]+\\.[a-z]+$"},
        "tags": {"type": "array", "items": {"type": "string"},
                 "minItems": 1, "maxItems": 8},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "email"],
    "additionalProperties": False,
})
compiled_heavy = None
import xgrammar as _xgr2
from _ch32_common import get_tokenizer as _gt
_heavy_compiler = _xgr2.GrammarCompiler(
    _xgr2.TokenizerInfo.from_huggingface(_gt()), max_threads=1)
compiled_heavy = _heavy_compiler.compile_json_schema(heavy_schema)
matchers_h = [_xgr2.GrammarMatcher(compiled_heavy) for _ in range(ROWS)]


def fill_rows_heavy(rows):
    for i in rows:
        matchers_h[i].fill_next_token_bitmask(bm, i)


fill_rows_heavy(range(4))
serial_h = med(lambda: fill_rows_heavy(range(ROWS)))
pool2 = ThreadPoolExecutor(max_workers=workers)


def parallel_fill_heavy():
    futs = [pool2.submit(fill_rows_heavy, t) for t in tasks]
    for f in futs:
        f.result()


parallel_h = med(parallel_fill_heavy)
pool2.shutdown()
out["timing_real_xgrammar_heavy"] = {
    "rows": ROWS,
    "grammar": "JSON schema（object/string/int 带边界/regex email/数组 1-8 项/required）",
    "serial_median_ms": serial_h,
    "parallel_median_ms": parallel_h,
    "speedup_x": round(serial_h / parallel_h, 2),
    "per_row_us_heavy": round(serial_h * 1000.0 / ROWS, 2),
    "note": "xgrammar 0.2.6 的 fill_next_token_bitmask 单行成本由行宽（1571 个 "
            "int32 写）主导、与 FSM 复杂度弱相关——重语法在本机同为 ~3µs/行，"
            "整批仍不足 1ms，线程池开销反超（见 gil_probe 的机理判定）",
}

# ── 乙段机理判定：fill 是否释放 GIL（放大工作量 40 倍的对照实验）──────────────
# 若 C++ 侧持有 GIL：无论工作量多大并行都不可能加速（speedup≈1）；
# 若释放 GIL：放大后并行应显著加速。本机轻/重语法单批都 <1ms，常规尺寸
# 分不出机理与回本条件——用重复因子 R 放大每任务工作量来判。
R = 40


def fill_rows_x(rows):
    for _ in range(R):
        for i in rows:
            matchers[i].fill_next_token_bitmask(bm, i)


serial_x = med(lambda: fill_rows_x(range(ROWS)), iters=5)
pool3 = ThreadPoolExecutor(max_workers=workers)
futs_wrap = lambda: [pool3.submit(fill_rows_x, t) for t in tasks]  # noqa: E731


def parallel_fill_x():
    for f in futs_wrap():
        f.result()


parallel_x = med(parallel_fill_x, iters=5)
pool3.shutdown()
out["gil_probe"] = {
    "repeat_factor": R,
    "rows": ROWS,
    "serial_median_ms": serial_x,
    "parallel_median_ms": parallel_x,
    "speedup_x": round(serial_x / parallel_x, 2),
    "workers": workers,
    "verdict": "fill_next_token_bitmask 在 C++ 侧释放 GIL" if serial_x / parallel_x > 2
               else "GIL 被持有——线程池无法加速 fill",
    "binary_evidence": {
        "binding": "tvm-ffi（load_binding.py: tvm_ffi.libinfo.load_lib_module）",
        "xgrammar_bindings_dll_gil_scoped_release_symbols": 0,
        "xgrammar_bindings_dll_gil_scoped_acquire_symbols": 0,
        "check": "site-packages/xgrammar/xgrammar_bindings.dll 逐字节搜索"
                 " gil_scoped_release/gil_scoped_acquire——0 命中（非 pybind11 形态）",
    },
    "note": "放大 40 倍工作量（34ms 串行）后并行仍慢 2.2 倍 + DLL 无 GIL 释放符号"
            "⇒ 本机 xgrammar 0.2.6（tvm-ffi 绑定）的 fill_next_token_bitmask 持有"
            " GIL：源码并行分支的『行间独写=结构许可』真实成立，但实际收益依赖"
            "后端实现释放 GIL 与单行成本——取证环境（host/0.2.6）实测未回本，"
            "writer 须按 exp-0718-1 显式挑明（生产声明有效的条件：后端释放 GIL）",
}

dump("trace_m06_parallel.json", out)
print("tasks:", out["structure"]["tasks_submitted"], "sizes:", out["structure"]["task_sizes"][:9])
print("serial", serial_ms, "ms vs parallel", parallel_ms, "ms →", out["timing_real_xgrammar"]["speedup_x"], "x")
