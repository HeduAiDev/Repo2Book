"""ch29 测试脚手架：host 无 NPU/CANN/vllm，在 sys.modules 桩掉 vllm 基类 proposer /
vllm_ascend 周边算子，再把（已减法的）implementation/ 模块按**规范模块名**注册进去，
让 factory.py 的 `from vllm_ascend.spec_decode.* import *` 与各 proposer 互相解析到精简版。

可在 host 验证、与真仓一致的纯 Python 控制流：
  (1) get_spec_decode_method —— 一处 if-elif 把 8 个 method 字符串分发到对应 Ascend*Proposer
      （含非同名映射：ngram_gpu→NPU、draft_model→DraftModel、eagle/eagle3/mtp 共用 Eagle）；
  (2) AscendNgramProposer.propose —— 跳过空/不支持/超长请求、写回 token_ids_cpu、交父类 batch_propose；
  (3) AscendNgramProposerNPU.propose —— no-op 薄壳（裸 pass，不复用父类 GPU kernel，返回 None）；
  (4) AscendSuffixDecodingProposer.propose —— 一行转发父类（补 runner.input_batch）；
  (5) AscendMedusaProposer.propose —— 按已接受 token 数 gather 每请求末位 hidden state 的索引计算；
  (6) AscendEagle/DraftModelProposer.__init__ —— 薄入口转调 base，pass_hidden_states_to_model True/False；
  (7) prepare_inputs —— 按拒绝 token 数重算 query_start_loc / seq_lens、构造 token_indices（纯 host numpy）。
重 NPU 路径（ACLGraph/Triton draft 前向/MLA）不真跑，只验上述控制流分流与索引运算。
"""

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


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


def _permissive(name, **methods):
    """造一个宽松基类：__init__ 吞掉任意参数，附带给定方法。"""
    ns = {"__init__": lambda self, *a, **k: None}
    ns.update(methods)
    return type(name, (), ns)


@pytest.fixture
def env():
    stubs = _Stubs()

    @contextmanager
    def _nullctx(*a, **k):
        yield None

    # ---- vllm.config ---- #
    cfg = stubs.mod("vllm.config")
    cfg.CUDAGraphMode = types.SimpleNamespace(NONE="NONE", FULL="FULL")
    cfg.CompilationMode = types.SimpleNamespace(NONE="EAGER")
    cfg.VllmConfig = type("VllmConfig", (), {})

    # ---- vllm.forward_context ---- #
    fc = stubs.mod("vllm.forward_context")
    fc.BatchDescriptor = type("BatchDescriptor", (), {})
    fc.set_forward_context = _nullctx

    # ---- vllm.distributed.parallel_state ---- #
    ps = stubs.mod("vllm.distributed.parallel_state")
    ps.get_world_group = lambda: types.SimpleNamespace(rank=0, device_group=None)
    ps.init_model_parallel_group = lambda *a, **k: object()
    ps.patch_tensor_parallel_group = lambda g: _nullctx()

    # ---- vllm.model_executor.models.* （_propose 的 isinstance 断言用） ---- #
    for dotted, cls in (
        ("vllm.model_executor.models.deepseek_eagle3", "Eagle3DeepseekV2ForCausalLM"),
        ("vllm.model_executor.models.llama_eagle3", "Eagle3LlamaForCausalLM"),
        ("vllm.model_executor.models.qwen3_dflash", "DFlashQwen3ForCausalLM"),
    ):
        m = stubs.mod(dotted)
        setattr(m, cls, type(cls, (), {}))

    # ---- vllm.utils.platform_utils ---- #
    pu = stubs.mod("vllm.utils.platform_utils")
    pu.is_pin_memory_available = lambda: False

    # ---- vllm.v1.* 杂项类型 ---- #
    stubs.mod("vllm.v1.attention.backends.utils").CommonAttentionMetadata = type("CommonAttentionMetadata", (), {})
    stubs.mod("vllm.v1.core.sched.output").SchedulerOutput = type("SchedulerOutput", (), {})
    stubs.mod("vllm.v1.sample.metadata").SamplingMetadata = type("SamplingMetadata", (), {})
    stubs.mod("vllm.v1.spec_decode.metadata").SpecDecodeMetadata = type("SpecDecodeMetadata", (), {})

    # ---- vllm.v1.spec_decode.* 基类 proposer ---- #
    stubs.mod("vllm.v1.spec_decode.ngram_proposer_gpu").NgramProposerGPU = _permissive("NgramProposerGPU")

    def _base_batch_propose(self, num_reqs, valid_ngram_requests, num_tokens_no_spec, token_ids_cpu):
        # 父类 NgramProposer.batch_propose 的桩：回显「被判为 valid 的请求下标」，便于断言控制流。
        return list(valid_ngram_requests)

    stubs.mod("vllm.v1.spec_decode.ngram_proposer").NgramProposer = _permissive(
        "NgramProposer", batch_propose=_base_batch_propose
    )

    def _suffix_base_propose(self, input_batch, valid_sampled_token_ids):
        return ("BASE_suffix", input_batch, valid_sampled_token_ids)

    stubs.mod("vllm.v1.spec_decode.suffix_decoding").SuffixDecodingProposer = _permissive(
        "SuffixDecodingProposer", propose=_suffix_base_propose
    )

    def _medusa_base_propose(self, target_hidden_states, sampling_metadata):
        return ("BASE_medusa", target_hidden_states)

    stubs.mod("vllm.v1.spec_decode.medusa").MedusaProposer = _permissive(
        "MedusaProposer", propose=_medusa_base_propose
    )
    stubs.mod("vllm.v1.spec_decode.extract_hidden_states").ExtractHiddenStatesProposer = _permissive(
        "ExtractHiddenStatesProposer"
    )
    stubs.mod("vllm.v1.spec_decode.draft_model").DraftModelProposer = _permissive(
        "DraftModelProposer",
        _raise_if_vocab_size_mismatch=lambda self: None,
        _raise_if_draft_tp_mismatch=lambda self: None,
    )
    stubs.mod("vllm.v1.spec_decode.eagle").EagleProposer = _permissive("EagleProposer")
    stubs.mod("vllm.v1.spec_decode.llm_base_proposer").SpecDecodeBaseProposer = _permissive("SpecDecodeBaseProposer")

    # ---- vllm_ascend 周边 ---- #
    stubs.mod("vllm_ascend.ascend_forward_context").set_ascend_forward_context = _nullctx
    stubs.mod("vllm_ascend.attention.attention_mask").AttentionMaskBuilder = _permissive("AttentionMaskBuilder")

    class _AscendCommonAttentionMetadata:
        """记录 prepare_inputs 装箱的字段，便于断言。"""

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    stubs.mod("vllm_ascend.attention.utils").AscendCommonAttentionMetadata = _AscendCommonAttentionMetadata
    stubs.mod("vllm_ascend.compilation.acl_graph").ACLGraphWrapper = _permissive("ACLGraphWrapper")
    asu = stubs.mod("vllm_ascend.utils")
    asu.enable_sp = lambda cfg: False
    asu.shared_expert_dp_enabled = lambda: False

    # ---- 加载（已减法的）精简版，按规范模块名注册 ---- #
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.spec_decode")
    llm_base = _load("llm_base_proposer.py", "vllm_ascend.spec_decode.llm_base_proposer")
    ngram = _load("ngram_proposer.py", "vllm_ascend.spec_decode.ngram_proposer")
    ngram_npu = _load("ngram_proposer_npu.py", "vllm_ascend.spec_decode.ngram_proposer_npu")
    suffix = _load("suffix_proposer.py", "vllm_ascend.spec_decode.suffix_proposer")
    medusa = _load("medusa_proposer.py", "vllm_ascend.spec_decode.medusa_proposer")
    extract = _load("extract_hidden_states_proposer.py", "vllm_ascend.spec_decode.extract_hidden_states_proposer")
    draft = _load("draft_proposer.py", "vllm_ascend.spec_decode.draft_proposer")
    eagle = _load("eagle_proposer.py", "vllm_ascend.spec_decode.eagle_proposer")
    dflash = _load("dflash_proposer.py", "vllm_ascend.spec_decode.dflash_proposer")
    factory = _load("factory.py", "vllm_ascend.spec_decode.factory")

    yield types.SimpleNamespace(
        factory=factory,
        llm_base=llm_base,
        ngram=ngram,
        ngram_npu=ngram_npu,
        suffix=suffix,
        medusa=medusa,
        extract=extract,
        draft=draft,
        eagle=eagle,
        dflash=dflash,
    )

    stubs.cleanup()
