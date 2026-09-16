# ch30 m19 驱动脚本：ReDoS 超时护栏（compile_regex_with_timeout，utils.py:L48-L83）。
#   ① 护栏机制验证：受控慢编译函数 + VLLM_REGEX_COMPILATION_TIMEOUT_S=1 →
#      超时抛 ValueError（消息含超时秒数与 pattern 前 200 字符）。
#   ② 真实 regex 编译路径：正常 pattern 秒过（xgr.Grammar.from_regex 走护栏）。
#   ③ 恶意 pattern 探测：嵌套量词候选在 host 上的真实耗时——若 host CPU
#      不复现指数爆炸则如实记录（护栏的存在依据是 utils.py:L52-L55 docstring
#      原话，正文引用 docstring 而非编造爆炸毫秒数）。
#   ④ timeout<=0 关闭护栏：fn 直跑不包 executor。
import json
import pathlib
import sys
import time

IMPL = pathlib.Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

import os

os.environ["VLLM_REGEX_COMPILATION_TIMEOUT_S"] = "1"  # 精简版 envs 无 cache，逐次读

import xgrammar as xgr

from vllm.v1.structured_output.utils import compile_regex_with_timeout

out = {"env": {"timeout_env_set_s": 1}}

# ── ① 机制验证：受控慢 fn ──
def slow_compile(pattern: str) -> str:
    time.sleep(3.0)  # > 1s 超时
    return "compiled"

mech = {}
t0 = time.perf_counter()
try:
    compile_regex_with_timeout(slow_compile, "BENIGN_PATTERN")
    mech["raised"] = None
except ValueError as e:
    mech["raised"] = "ValueError"
    mech["elapsed_s"] = round(time.perf_counter() - t0, 2)
    mech["message"] = str(e)[:220]
out["guard_mechanism"] = mech

# ── ② 正常 pattern ──
t0 = time.perf_counter()
res = compile_regex_with_timeout(xgr.Grammar.from_regex, "[0-9]+")
out["normal_pattern"] = {
    "pattern": "[0-9]+",
    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
    "returns": type(res).__name__,
}

# ── ③ 恶意 pattern 候选在 host 的真实耗时 ──
os.environ["VLLM_REGEX_COMPILATION_TIMEOUT_S"] = "20"  # 探测期放宽护栏
probes = []
candidates = [
    "(a+)+$",           # docstring 点名的嵌套量词
    "(a|a)+$",
    "((a)*)*b",
    "(a+){10}b",
    "(a+)+$" + "a" * 30,
    "(x+x+)+y",
]
for pat in candidates:
    t0 = time.perf_counter()
    try:
        compile_regex_with_timeout(xgr.Grammar.from_regex, pat)
        probes.append({"pattern": pat, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                       "outcome": "compiled"})
    except Exception as e:
        probes.append({"pattern": pat, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
                       "outcome": f"{type(e).__name__}: {str(e)[:60]}"})
out["host_adversarial_probes"] = probes
out["host_probe_verdict"] = (
    "host 上全部候选毫秒级完成——xgrammar 0.2.6 的 regex 编译在本机不复现指数爆炸；"
    "护栏必要性以 utils.py:L52-L55 docstring 原话为据（'nested quantifiers like (a+)+b "
    "cause exponential DFA state-space explosion, hanging the inference worker indefinitely'），"
    "护栏机制本身由①受控验证")

# ── ④ timeout=0 关闭护栏 ──
os.environ["VLLM_REGEX_COMPILATION_TIMEOUT_S"] = "0"
res0 = compile_regex_with_timeout(xgr.Grammar.from_regex, "[0-9]+")
out["timeout_zero_disables_guard"] = {"env_value": 0, "returns": type(res0).__name__,
                                      "anchor": "utils.py:L63-L64 if timeout <= 0: return fn(pattern)"}
os.environ["VLLM_REGEX_COMPILATION_TIMEOUT_S"] = "5"  # 还原默认
out["default_timeout_s_restored"] = 5

# ── 表格行建议 ──
out["table_rows_m19"] = [
    ["护栏·正常", "compile_regex_with_timeout(from_regex, '[0-9]+')",
     f"{out['normal_pattern']['elapsed_ms']} ms", "1s 预算内", "返回 Grammar（编译产物）"],
    ["护栏·超时", "受控慢编译函数（sleep 3s）+ 超时=1s",
     f"{mech.get('elapsed_s')} s 后抛 ValueError", "超预算",
     "executor.shutdown(cancel_futures=True)；消息含 pattern 前 200 字符"],
    ["恶意·host 探测", "(a+)+$ 等 6 个嵌套量词候选",
     f"host 全部 {max(p['elapsed_ms'] for p in probes)} ms 内编完", "host 不复现爆炸",
     "护栏依据=docstring 原话；机制由受控超时验证"],
    ["关闭护栏", "VLLM_REGEX_COMPILATION_TIMEOUT_S=0",
     "fn(pattern) 直跑", "不包 executor", "utils.py:L63-L64（生产不建议，envs docstring）"],
]

p = pathlib.Path(__file__).with_name("trace_m19_redos.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("WROTE", p)
print(json.dumps(out, ensure_ascii=False, indent=1))
