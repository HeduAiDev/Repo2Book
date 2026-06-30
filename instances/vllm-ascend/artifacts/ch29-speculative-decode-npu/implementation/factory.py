# ch29 工厂入口 —— subtract-only 精简版（投机解码 proposer 工厂分发）
#
# 真实源码位于 vllm_ascend/spec_decode/__init__.py（包入口）。精简版改名为 factory.py
# 仅因 lint_fidelity 不扫描名为 __init__.py 的文件（会漏掉 must_keep 的 get_spec_decode_method）；
# 控制流与源码逐字一致，SOURCE 标注指向真实路径。
#
# 只删 dossier.subtraction_plan.delete 批准项：
#   - 删文件头 L1-L19 的 Apache 许可证抬头 + "Adapted from gpu_model_runner.py" 注释（纯版权样板）。
# must_keep：get_spec_decode_method 及其分发到的 8 个 Ascend*Proposer 类全部保留。
#
# SUBTRACTED: __init__.py:L1-L19 Apache 许可证抬头 + "Adapted from vllm gpu_model_runner.py"
#   注释 —— 纯版权样板，不影响控制流。

# SOURCE: vllm_ascend/spec_decode/__init__.py:L21-L30
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
    AscendExtractHiddenStatesProposer,
)
from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer


# SOURCE: vllm_ascend/spec_decode/__init__.py:L33
def get_spec_decode_method(method, vllm_config, device, runner):
    if method == "ngram":
        return AscendNgramProposer(vllm_config, runner)
    elif method == "ngram_gpu":
        return AscendNgramProposerNPU(vllm_config, device, runner)
    elif method == "suffix":
        return AscendSuffixDecodingProposer(vllm_config, runner)
    elif method == "medusa":
        return AscendMedusaProposer(vllm_config, device)
    elif method in ("eagle", "eagle3", "mtp"):
        return AscendEagleProposer(vllm_config, device, runner)
    elif method == "dflash":
        return AscendDflashProposer(vllm_config, device, runner)
    elif method == "draft_model":
        return AscendDraftModelProposer(vllm_config, device, runner)
    elif method == "extract_hidden_states":
        return AscendExtractHiddenStatesProposer(vllm_config, device, runner)
    else:
        raise ValueError(f"Unknown speculative decoding method: {method}")
