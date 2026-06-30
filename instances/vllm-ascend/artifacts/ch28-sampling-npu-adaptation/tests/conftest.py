"""ch28 测试脚手架：host 无 NPU/CANN/vllm，在 sys.modules 桩掉 torch_npu / vllm 基类 /
vllm_ascend 周边算子，再把（已减法的）implementation/ 模块按**规范模块名**注册进去，
让它们彼此 import 解析到精简版。

可在 host 验证、与真仓一致的纯 Python 控制流（核心算法宝石就在 host 跑得起来）：
  (1) random_sample —— Gumbel-max（probs.div_(q).argmax, q~Exp(1)）与 torch.multinomial 同分布
      （大样本经验频率对照）；有种子 generator 可复现；
  (2) AscendSampler.apply_penalties —— HAS_TRITON 优雅回退：不可用→Sampler.apply_penalties（基类原版），
      可用→apply_all_penalties→apply_penalties_triton（昇腾内核）；
  (3) AscendTopKTopPSampler.forward_native —— VLLM_BATCH_INVARIANT 回退基类；默认走 random_sample；
  (4) _apply_top_k_top_p_pytorch —— sort/cumsum/masked_fill 的 top-k/top-p 截断；
  (5) greedy_sample —— 单卡 argmax；
  (6) rejection_greedy_sample_pytorch —— 首个 mismatch 截断 + 全接受补 bonus；
  (7) rejection_random_sample_pytorch —— target/draft>=u 接受、被拒取 recovered、全接受补 bonus；
  (8) sample_recovered_tokens_pytorch —— 残差 max(0,target-draft)/q 的 argmax 重采；
  (9) rejection_sample —— HAS_TRITON 关时 greedy 路径端到端。
NPU-only（npu_stream_switch / wait_stream / Triton kernel / AscendC）由 nullcontext / no-op / 记录替身承接，
只验控制流分流与入参，不真算（昇腾才有内核）。
"""

import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"

# 基类常量（对齐 vllm/v1/sample/rejection_sampler.py）。
PLACEHOLDER_TOKEN_ID = -1
GREEDY_TEMPERATURE = -1.0
MAX_SPEC_LEN = 32


def _load(relpath, modname):
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    if "." in modname:
        parent = modname.rsplit(".", 1)[0]
        if parent in sys.modules:
            setattr(sys.modules[parent], modname.rsplit(".", 1)[1], mod)
    spec.loader.exec_module(mod)
    return mod


class _Stubs:
    def __init__(self):
        self.added = []

    def mod(self, dotted):
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[: i + 1])
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
                self.added.append(name)
                if i > 0:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        return sys.modules[dotted]

    def cleanup(self):
        for n in reversed(self.added):
            sys.modules.pop(n, None)


class _Recorder:
    def __init__(self):
        self.calls = []

    def rec(self, name):
        self.calls.append(name)
        return name


class _Kernel:
    """Triton kernel 替身：支持 kernel[(grid,)](...) 的下标 + 调用，仅记录不真算。"""

    def __init__(self, rec, name):
        self._rec = rec
        self._name = name

    def __getitem__(self, _grid):
        def _launch(*a, **k):
            self._rec.rec(self._name)

        return _launch


@pytest.fixture
def env():
    stubs = _Stubs()
    rec = _Recorder()

    knobs = types.SimpleNamespace(
        has_triton=False,
        batch_invariant=False,
    )

    # ---- torch.npu 替身（random_sample 的 stream/wait_stream，AscendSampler.__init__ 不再用 Event）---- #
    had_npu = hasattr(torch, "npu")
    saved_npu = getattr(torch, "npu", None)

    class _Stream:
        def wait_stream(self, other):
            return None

    torch.npu = types.SimpleNamespace(
        current_stream=lambda: _Stream(),
        Event=lambda: types.SimpleNamespace(record=lambda: None, synchronize=lambda: None),
        stream=lambda s: contextlib.nullcontext(),
    )

    # ---- vllm 顶层 + 子模块 ---- #
    stubs.mod("vllm")
    venvs = stubs.mod("vllm.envs")
    venvs.VLLM_BATCH_INVARIANT = property  # 占位，下方用 _Envs 替换
    venvs.VLLM_BATCH_INVARIANT = False
    triton_utils = stubs.mod("vllm.triton_utils")
    triton_utils.HAS_TRITON = False

    # vllm.utils.* （penalties 用）
    pu = stubs.mod("vllm.utils.platform_utils")
    pu.is_pin_memory_available = lambda: False

    def _make_tensor_with_pad(lists, pad, device, dtype, pin_memory=False):
        maxlen = max((len(x) for x in lists), default=0)
        padded = [list(x) + [pad] * (maxlen - len(x)) for x in lists]
        return torch.tensor(padded if padded else [[]], dtype=dtype)

    tu = stubs.mod("vllm.utils.torch_utils")
    tu.make_tensor_with_pad = _make_tensor_with_pad

    # vllm.v1.sample.metadata / outputs / spec_decode.metadata
    meta_mod = stubs.mod("vllm.v1.sample.metadata")
    meta_mod.SamplingMetadata = type("SamplingMetadata", (), {})
    out_mod = stubs.mod("vllm.v1.outputs")
    out_mod.SamplerOutput = type(
        "SamplerOutput",
        (),
        {"__init__": lambda self, sampled_token_ids=None, logprobs_tensors=None: setattr(
            self, "sampled_token_ids", sampled_token_ids
        ) or setattr(self, "logprobs_tensors", logprobs_tensors)},
    )
    spec_mod = stubs.mod("vllm.v1.spec_decode.metadata")
    spec_mod.SpecDecodeMetadata = type("SpecDecodeMetadata", (), {})

    # vllm.v1.sample.sampler.Sampler（基类，apply_penalties 是回退目标）
    class _Sampler:
        def __init__(self, logprobs_mode="raw_logprobs"):
            self.logprobs_mode = logprobs_mode

        @staticmethod
        def apply_penalties(logits, sampling_metadata, output_token_ids):
            rec.rec("BASE_Sampler.apply_penalties")
            return logits

    sampler_base = stubs.mod("vllm.v1.sample.sampler")
    sampler_base.Sampler = _Sampler

    # vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler（基类，forward_native 是 BATCH_INVARIANT 回退目标）
    class _TopKTopPSampler:
        def __init__(self, logprobs_mode="raw_logprobs", **kwargs):
            self.logprobs_mode = logprobs_mode

        def forward_native(self, logits, generators, k, p):
            rec.rec("BASE_TopKTopP.forward_native")
            return ("BASE_native", None)

    topk_base = stubs.mod("vllm.v1.sample.ops.topk_topp_sampler")
    topk_base.TopKTopPSampler = _TopKTopPSampler

    # vllm.v1.sample.rejection_sampler（基类常量 + RejectionSampler + generate_uniform_probs）
    class _RejectionSampler:
        def __init__(self, sampler):
            self.sampler = sampler
            self.is_processed_logprobs_mode = False

    def _generate_uniform_probs(num_tokens, num_draft_tokens, generators, device):
        return torch.rand(num_tokens, device=device)

    rs_base = stubs.mod("vllm.v1.sample.rejection_sampler")
    rs_base.RejectionSampler = _RejectionSampler
    rs_base.GREEDY_TEMPERATURE = GREEDY_TEMPERATURE
    rs_base.MAX_SPEC_LEN = MAX_SPEC_LEN
    rs_base.PLACEHOLDER_TOKEN_ID = PLACEHOLDER_TOKEN_ID
    rs_base.generate_uniform_probs = _generate_uniform_probs

    # ---- vllm_ascend 周边 ---- #
    stubs.mod("vllm_ascend")
    ascfg = stubs.mod("vllm_ascend.ascend_config")
    ascfg.get_ascend_config = lambda: types.SimpleNamespace(
        enable_reduce_sample=False,
        enable_async_exponential=False,
    )

    # utils：npu_stream_switch（nullcontext）/ global_stream（哑对象）/ 设备探针
    utils = stubs.mod("vllm_ascend.utils")
    utils.npu_stream_switch = lambda s: contextlib.nullcontext()
    utils.global_stream = lambda: object()
    utils.AscendDeviceType = types.SimpleNamespace(A2="A2", A3="A3", _310P="310P", A5="A5")
    utils.get_ascend_device_type = lambda: "A2"

    # 昇腾 Triton penalty kernel（NPU-only，记录）
    pen = stubs.mod("vllm_ascend.ops.triton.penalty")

    def _apply_penalties_triton(logits, *a, **k):
        rec.rec("apply_penalties_triton")
        return logits

    pen.apply_penalties_triton = _apply_penalties_triton

    # 昇腾 Triton reject_sample kernel（NPU-only，记录）
    rj = stubs.mod("vllm_ascend.ops.triton.reject_sample")
    rj.cal_grid_and_block_size = lambda bs: (bs, 128)
    rj.expand_triton = lambda *a, **k: rec.rec("expand_triton")
    rj.rejection_greedy_sample_with_triton = lambda *a, **k: rec.rec("rejection_greedy_sample_with_triton")
    rj.rejection_random_sample_kernel = _Kernel(rec, "rejection_random_sample_kernel")
    rj.sample_recovered_tokens_kernel = _Kernel(rec, "sample_recovered_tokens_kernel")

    # ---- 加载（已减法的）精简版，按规范模块名注册 ---- #
    stubs.mod("vllm_ascend.sample")
    penalties = _load("penalties.py", "vllm_ascend.sample.penalties")
    sampler = _load("sampler.py", "vllm_ascend.sample.sampler")
    rejection = _load("rejection_sampler.py", "vllm_ascend.sample.rejection_sampler")

    def set_triton(flag: bool):
        knobs.has_triton = flag
        sampler.HAS_TRITON = flag
        rejection.HAS_TRITON = flag
        penalties_has = flag  # penalties 无 HAS_TRITON 分支
        del penalties_has

    def set_batch_invariant(flag: bool):
        knobs.batch_invariant = flag
        venvs.VLLM_BATCH_INVARIANT = flag

    set_triton(False)
    set_batch_invariant(False)

    yield types.SimpleNamespace(
        penalties=penalties,
        sampler=sampler,
        rejection=rejection,
        rec=rec,
        knobs=knobs,
        set_triton=set_triton,
        set_batch_invariant=set_batch_invariant,
        Sampler=_Sampler,
        PLACEHOLDER_TOKEN_ID=PLACEHOLDER_TOKEN_ID,
        GREEDY_TEMPERATURE=GREEDY_TEMPERATURE,
    )

    if had_npu:
        torch.npu = saved_npu
    else:
        del torch.npu
    stubs.cleanup()
