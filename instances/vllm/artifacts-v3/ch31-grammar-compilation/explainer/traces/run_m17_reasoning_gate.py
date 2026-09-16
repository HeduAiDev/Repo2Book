# ch30 m17 驱动脚本：思考模型联动（推进侧思考门）。
#   should_advance 判「思考段结束后语法才生效」；reasoner=None 常量路径；
#   reasoning_ended 缓存；is_reasoning_end_streaming 探测（new_token_ids 直传的
#   delta 窗口，#43388）；reasoning_end_token_index + trim_reasoning_for_advance
#   剔除混在同一步的思考 token（#44006）。
# 诚实标注：ReasoningParser 装配链按减法计划删除（dossier delete[4]），reasoner_cls
#   由本脚本直注 FakeReasoner（等价于装配产物）——这是精简版的既定 HOST SEAM，
#   四个方法本体逐字（impl-notes：L99-L486 全逐字）。
import json
import pathlib
import sys

IMPL = pathlib.Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         StructuredOutputsConfig, VllmConfig)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

TOKENIZER_NAME = "gpt2"


class FakeReasoner:
    """ReasoningParser 最小替身：delta 窗口内出现 99 即判思考结束（同 tests）。"""

    def __init__(self, tokenizer=None, **kwargs):
        pass

    def is_reasoning_end_streaming(self, all_token_ids, delta_ids):
        return 99 in list(delta_ids)


def make_vllm_config():
    return VllmConfig(
        model_config=ModelConfig(tokenizer=TOKENIZER_NAME, vocab_size=50257),
        parallel_config=ParallelConfig(distributed_executor_backend="mp"),
        scheduler_config=SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=8192),
        structured_outputs_config=StructuredOutputsConfig(backend="auto"),
        speculative_config=None,
    )


def make_req(prompt=(1, 2), reasoning_ended=None):
    sp = SamplingParams(structured_outputs=StructuredOutputsParams(regex="a"))
    req = Request(request_id="r", prompt_token_ids=list(prompt),
                  sampling_params=sp, pooling_params=None,
                  reasoning_ended=reasoning_ended)
    req.structured_output_request._grammar = object()  # 已就绪 grammar 替身（只测门）
    return req


out = {"env": {"note": "FakeReasoner=ReasoningParser 最小替身（装配链按减法计划删除，方法本体逐字）"}}

rows = []
# ── 常量路径：reasoner=None（非思考模型，真实主路径）──
m_plain = StructuredOutputManager(make_vllm_config())  # reasoner_cls 停留 None 初值
req = make_req()
rows.append({
    "case": "非思考模型（reasoner=None）",
    "input": "should_advance(new_token_ids=[5])",
    "returns": m_plain.should_advance(req, new_token_ids=[5]),
    "anchor": "__init__.py:L396-L398 快返回 True 的常量路径",
})
# 无结构化输出：False
sp = SamplingParams(structured_outputs=None)
req_n = Request(request_id="p", prompt_token_ids=[1], sampling_params=sp, pooling_params=None)
rows.append({
    "case": "无结构化输出",
    "input": "should_advance(new_token_ids=[5])",
    "returns": m_plain.should_advance(req_n, new_token_ids=[5]),
    "anchor": "__init__.py:L385-L386 use_structured_output 守卫",
})

# ── 思考分支 ──
m = StructuredOutputManager(make_vllm_config())
m.reasoner_cls = FakeReasoner  # 直注（等价于装配产物）
# 思考未结束：False
req1 = make_req()
rows.append({
    "case": "思考未结束",
    "input": "new_token_ids=[7, 8]（delta 窗口无 99）",
    "returns": m.should_advance(req1, new_token_ids=[7, 8]),
    "reasoning_ended_after": req1.structured_output_request.reasoning_ended,
    "anchor": "is_reasoning_end_streaming=False → return False",
})
# 本步结束：设边界（#43388 直传 delta 窗口）
req2 = make_req(prompt=[1, 2])
req2.append_output_token_ids([7, 8, 99])  # 本步输出含思考结束边界 99
rows.append({
    "case": "思考在本步结束（#43388）",
    "input": "new_token_ids=[7,8,99]；all_token_ids=[1,2,7,8,99]",
    "returns": m.should_advance(req2, new_token_ids=[7, 8, 99]),
    "reasoning_ended_after": req2.structured_output_request.reasoning_ended,
    "reasoning_end_token_index": req2.structured_output_request.reasoning_end_token_index,
    "anchor": "边界=99 在 all_token_ids 的绝对索引 4（L434-L436）",
})
# 已结束缓存：短路 True
rows.append({
    "case": "已结束缓存（reasoning_ended=True）",
    "input": "should_advance(new_token_ids=[7])",
    "returns": m.should_advance(req2, new_token_ids=[7]),
    "anchor": "__init__.py:L409-L410 reasoning_ended 缓存短路",
})
# prompt 里早有 99 但不在 delta 窗口：False（#43388 修复的行为）
req3 = make_req(prompt=[1, 99, 2])
rows.append({
    "case": "边界 token 在 prompt 里（不在 delta 窗口）",
    "input": "new_token_ids=[7]；prompt=[1,99,2]",
    "returns": m.should_advance(req3, new_token_ids=[7]),
    "anchor": "#43388：delta 窗口只看本步 token——历史里的 99 不触发",
})

# ── trim_reasoning_for_advance（#44006 混步剔除）──
req4 = make_req(prompt=[1, 2])
req4.structured_output_request.reasoning_end_token_index = 4
req4.append_output_token_ids([7, 8, 99, 50, 51])
rows.append({
    "case": "混步剔除（#44006）",
    "input": "trim_reasoning_for_advance(new=[7,8,99,50,51])；边界 idx=4",
    "returns": m.trim_reasoning_for_advance(req4, [7, 8, 99, 50, 51]),
    "num_reasoning_dropped": 3,
    "anchor": "first_idx=len(all)-len(new)=2；num_reasoning=4+1-2=3 → new[3:]",
})
# 边界之后的整步：原样
req5 = make_req(prompt=[1, 2])
req5.structured_output_request.reasoning_end_token_index = 4
req5.append_output_token_ids([7, 8, 99, 50, 51])
rows.append({
    "case": "边界之后的整步",
    "input": "trim_reasoning_for_advance(new=[50,51])；边界 idx=4",
    "returns": m.trim_reasoning_for_advance(req5, [50, 51]),
    "num_reasoning_dropped": 0,
    "anchor": "first_idx=5；num_reasoning=4+1-5=0 → 原样返回",
})
out["reasoning_gate"] = rows

# ── 表格行建议 ──
out["table_rows_m17"] = [
    ["非思考模型", "reasoner=None", "should_advance([5])", "常量路径快返回", "True（语法从头就生效）"],
    ["思考未结束", "reasoner=FakeReasoner", "should_advance([7,8])，delta 无 99", "探测=思考中", "False（不推进 FSM）"],
    ["本步结束", "同上", "should_advance([7,8,99])", "delta 含 99 → 探测=结束",
     "True；reasoning_end_token_index=4（绝对索引）"],
    ["已结束缓存", "同上（reasoning_ended=True）", "should_advance([7])", "短路", "True（不再探测）"],
    ["混步剔除", "边界 idx=4", "trim([7,8,99,50,51])", "首 3 个是思考 token", "[50,51]（剔 3 个，#44006）"],
    ["边界后整步", "边界 idx=4", "trim([50,51])", "num_reasoning=0", "[50,51]（原样）"],
]

p = pathlib.Path(__file__).with_name("trace_m17_reasoning.json")
with open(p, "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("WROTE", p)
print(json.dumps(out, ensure_ascii=False, indent=1))
