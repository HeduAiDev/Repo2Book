# ch29《Sampler 9 步管线》测试配置。
# 行为基准 = 真实 vLLM v0.27.1（6e448d0ea，instances/vllm/source 现核行号）：
#   - vllm/v1/sample/sampler.py:L20-L59（9 步 docstring 契约）
#   - vllm/v1/sample/logits_processor/*（argmax 不变性两列分类）
#   - vllm/v1/sample/ops/*（bad_words / penalties / topk_topp / logprobs）
#   - vllm/model_executor/layers/utils.py:L34-L89（惩罚真算式）
import os
import pathlib
import sys

IMPL_DIR = pathlib.Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))

# HOST SEAM 确定性：构造期 FlashInfer 裁决（topk_topp_sampler.py:L28-L74）
# 默认走真实的显式禁用支路（L45-L50，VLLM_USE_FLASHINFER_SAMPLER=0 →
# info_once + return False → 构造期绑 forward_native，L100-L102 的 else 位）。
# 需要测默认值（True）或容器 flashinfer 真路径的测试自行覆盖此环境变量。
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
