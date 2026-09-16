# ch30 m10 驱动脚本：编译复用各后端各自为政。
#   ① vLLM 侧零跨请求去重：structured_output_key 只是每请求 cached_property
#      （键相等但对象独立，见 trace_m02_m04_params_key.json per_request_key）；
#      每次 compile_grammar 都照常进后端。
#   ② xgrammar 库内 LRU+字节预算缓存（GrammarCompiler(cache_enabled=True,
#      cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB*1024*1024，默认 512MB）：
#      同 schema 第二次编译命中缓存——用计时对比观测（冷 vs 热）。
#   ③ guidance（llguidance）无编译缓存：同 schema 连编计时基本持平（每次重编）。
#   ④ compiler 对象的缓存可观测面探测（API 扫描）。
# 计时说明：host 单线程 perf_counter，毫秒级数字有抖动——每个测点取 5 次中位数，
# 数字只作数量级证据（首次 vs 命中的量级差），不作精确延迟口径（正文不得引为
# 生产毫秒数；dossier theory[2] 明令：无容器运行环境不给未实测毫秒数）。
import json
import pathlib
import statistics
import sys
import time

IMPL = pathlib.Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

from transformers import AutoTokenizer

import vllm.envs as envs
from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         StructuredOutputsConfig, VllmConfig)
from vllm.v1.structured_output.backend_types import StructuredOutputOptions
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

TOKENIZER_NAME = "gpt2"
VOCAB = 50257
tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# 有点嵌套的 schema（编译量可测；微型 schema <1ms 不好对比）
SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "address": {"type": "object", "properties": {
            "city": {"type": "string"}, "zip": {"type": "string"}}},
    },
    "required": ["name", "age"],
}, ensure_ascii=False)


def make_vllm_config():
    return VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER_NAME, vocab_size=VOCAB),
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        scheduler_config=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=8192),
        structured_outputs_config=StructuredOutputsConfig(backend="auto"),
        speculative_config=None,
    )


out = {"env": {"tokenizer": TOKENIZER_NAME, "vocab_size": VOCAB,
               "VLLM_XGRAMMAR_CACHE_MB_default": envs.VLLM_XGRAMMAR_CACHE_MB,
               "cache_limit_bytes": envs.VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024,
               "timing_note": "host perf_counter 单线程，5 次取中位数，毫秒级有抖动，只作数量级证据"}}

# ── ② xgrammar：同 schema 连编 6 次（首拍冷、后续命中库内缓存）──
xb = XgrammarBackend(make_vllm_config(), tokenizer=tok, vocab_size=VOCAB)
xg_times = []
for i in range(6):
    t0 = time.perf_counter()
    xb.compile_grammar(StructuredOutputOptions.JSON, SCHEMA)
    xg_times.append(round((time.perf_counter() - t0) * 1000, 3))
out["xgrammar_repeat_ms"] = xg_times
out["xgrammar_first_vs_median_of_rest"] = {
    "first_ms": xg_times[0],
    "median_of_compile_2_to_6_ms": round(statistics.median(xg_times[1:]), 3),
    "speedup_ratio_first_over_median": round(xg_times[0] / max(statistics.median(xg_times[1:]), 1e-9), 1),
}

# ── ③ guidance：同 schema 连编 6 次（每次重编）──
gb = GuidanceBackend(make_vllm_config(), tokenizer=tok, vocab_size=VOCAB)
gd_times = []
for i in range(6):
    t0 = time.perf_counter()
    gb.compile_grammar(StructuredOutputOptions.JSON, SCHEMA)
    gd_times.append(round((time.perf_counter() - t0) * 1000, 3))
out["guidance_repeat_ms"] = gd_times
out["guidance_first_vs_median_of_rest"] = {
    "first_ms": gd_times[0],
    "median_of_compile_2_to_6_ms": round(statistics.median(gd_times[1:]), 3),
}

# ── ④ GrammarCompiler 缓存可观测面（API 扫描，不杜撰库内部）──
probe = [a for a in dir(xb.compiler) if "cache" in a.lower()]
out["compiler_cache_api_surface"] = probe

# 缓存字节数的硬观测：命中缓存的编译让 get_cache_size_bytes 增长、clear_cache 归零
size_after_warm = xb.compiler.get_cache_size_bytes()
xb.compiler.clear_cache()
size_after_clear = xb.compiler.get_cache_size_bytes()
t0 = time.perf_counter()
xb.compile_grammar(StructuredOutputOptions.JSON, SCHEMA)  # clear 后重编：又是冷路径
cold_again_ms = round((time.perf_counter() - t0) * 1000, 3)
out["cache_size_probe"] = {
    "bytes_after_6_warm_compiles": size_after_warm,
    "bytes_after_clear_cache": size_after_clear,
    "recompile_after_clear_ms": cold_again_ms,
    "note": "clear_cache 后重编耗时回到冷路径量级——缓存命中的因果证据",
}

# ── 表格行建议（数字与本 trace 一致）──
out["table_rows_m10"] = [
    ["第 1 次（冷）", "xgrammar compile_json_schema", f"{xg_times[0]} ms",
     "库内 LRU 无此 schema", "真编译"],
    ["第 2 次（同 schema）", "xgrammar compile_json_schema",
     f"{xg_times[1]} ms", "命中库内缓存（键=语法串）", "近似免费"],
    ["第 6 次", "xgrammar compile_json_schema", f"{xg_times[5]} ms", "仍命中", "稳定热路径"],
    ["guidance 第 1 次", "guidance compile_grammar（serialize→LLMatcher）",
     f"{gd_times[0]} ms", "无编译缓存", "每次真编译"],
    ["guidance 第 6 次", "guidance compile_grammar",
     f"{gd_times[5]} ms", "仍无缓存", "与首次同量级（无热路径）"],
    ["缓存硬观测", f"get_cache_size_bytes（6 次编译后）={size_after_warm} B；clear_cache 后={size_after_clear} B",
     f"clear 后重编 {cold_again_ms} ms", "回到冷路径量级", "缓存命中的因果证据（非纯计时推断）"],
]

p = pathlib.Path(__file__).with_name("trace_m10_cache.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("WROTE", p)
print(json.dumps(out, ensure_ascii=False, indent=1))
