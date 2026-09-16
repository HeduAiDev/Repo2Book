# ch30《约束解码 I：语法编译》测试配置。
# 行为基准 = 真实 vLLM v0.27.1（6e448d0ea，instances/vllm/source 现核行号）：
#   - vllm/sampling_params.py:L72-L142（六选一互斥）/ L923-L1086（auto 阶梯）
#   - vllm/v1/structured_output/*（request 三态 / backend_types 两层 ABC /
#     backend_xgrammar 五分派 / backend_guidance rollback_lag / utils 改写工具箱）
#   - vllm/v1/core/sched/scheduler.py:L2050-L2062（侧队）/ L2678-L2712（晋级）/
#     L1954-L1972（编译失败收账）/ L1817-L1843（采样后推进）
# xgrammar==0.2.6 / llguidance==1.7.6（requirements/common.txt 钉版区间的
# Windows wheel，host 已装）——测试跑真实状态机行为，不用 Fake。
import os
import pathlib
import sys

IMPL_DIR = pathlib.Path(__file__).resolve().parent.parent / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))

import pytest
from transformers import AutoTokenizer

# gpt2 是 HF 缓存里的本地小词表（50257），无需网络。
TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


@pytest.fixture(scope="session")
def vocab_size():
    return VOCAB_SIZE


def make_vllm_config(
    *,
    backend="auto",
    num_speculative_tokens=None,
    distributed_executor_backend="mp",
    max_num_seqs=16,
    max_num_batched_tokens=8192,
    is_diffusion=False,
    skip_tokenizer_init=False,
    tokenizer_name=TOKENIZER_NAME,
    vocab_size=VOCAB_SIZE,
    disable_any_whitespace=False,
    disable_additional_properties=False,
    reasoning_parser="",
    enable_in_reasoning=False,
):
    """构造 HOST SEAM VllmConfig（真实 VllmConfig 是 pydantic 大对象图，归 ch03）。"""
    from vllm.config import (
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        StructuredOutputsConfig,
        VllmConfig,
    )

    model_config = ModelConfig(
        tokenizer=tokenizer_name,
        vocab_size=vocab_size,
        is_diffusion=is_diffusion,
        skip_tokenizer_init=skip_tokenizer_init,
    )
    parallel_config = ParallelConfig(
        distributed_executor_backend=distributed_executor_backend
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens
    )
    structured_outputs_config = StructuredOutputsConfig(
        backend=backend,
        disable_any_whitespace=disable_any_whitespace,
        disable_additional_properties=disable_additional_properties,
        reasoning_parser=reasoning_parser,
        enable_in_reasoning=enable_in_reasoning,
    )
    speculative_config = (
        None
        if num_speculative_tokens is None
        else SpeculativeConfig(num_speculative_tokens=num_speculative_tokens)
    )
    return VllmConfig(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        structured_outputs_config=structured_outputs_config,
        speculative_config=speculative_config,
    )


@pytest.fixture()
def manager_factory():
    """每个测试造独立的 StructuredOutputManager（backend 惰性、互不污染）。"""
    from vllm.v1.structured_output import StructuredOutputManager

    def _factory(**kwargs):
        return StructuredOutputManager(make_vllm_config(**kwargs))

    return _factory


@pytest.fixture()
def manager(manager_factory):
    return manager_factory()


@pytest.fixture()
def spec_manager(manager_factory):
    """投机解码配置（num_speculative_tokens=1）下的 manager。"""
    return manager_factory(num_speculative_tokens=1)
