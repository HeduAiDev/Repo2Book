# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/__init__.py —— HOST SEAM：本章只取 vllm 包的身份占位与
# BaseServing（vllm/entrypoints/serve/engine/serving.py:L7）消费的三符号
# re-export（真实文件在此处初始化 torch/envs/大件，与本章服务面无关）。
from vllm.inputs import PromptType  # noqa: F401
from vllm.sampling_params import SamplingParams  # noqa: F401
import vllm.envs as envs  # noqa: F401
