# ch31 主电池七：V2 落地路径（m18/m19）——use_v2_model_runner 真实选择器 +
# StructuredOutputsWorker（copy_stream 双 H2D + cu_num_logits 行映射）+
# 自写 Triton kernel（(bit&1)==0 → tl.store(-inf)）+ V2 sample() 调用点。
# 基准：vllm/config/vllm.py:L564-L658 / vllm/v1/worker/gpu/structured_outputs.py
# / vllm/v1/worker/gpu/model_runner.py:L1143-L1175。
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import cdiv


V = 100  # 非块对齐词表：检验 kernel 的越界谓词


def make_v2_config(**kw):
    from vllm.config.vllm import VllmConfig

    return VllmConfig(**kw)


class TestUseV2ModelRunner:
    """m18：V1/V2 两条落地路径的真实选择器（v0.27 稠密默认 V2）。"""

    def test_env_override_wins(self, monkeypatch):
        import vllm.envs as envs

        monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
        assert make_v2_config().use_v2_model_runner is True
        monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
        assert make_v2_config().use_v2_model_runner is False

    def test_pcp_forces_v2(self):
        cfg = make_v2_config(
            parallel_config=dict(prefill_context_parallel_size=2)
        )
        assert cfg.use_v2_model_runner is True

    def test_dspark_forces_v2(self):
        cfg = make_v2_config(speculative_config=dict(method="dspark"))
        assert cfg.use_v2_model_runner is True

    def test_diffusion_forces_v2(self):
        cfg = make_v2_config(model_config=dict(is_diffusion=True))
        assert cfg.use_v2_model_runner is True

    def test_dense_generate_model_defaults_v2(self):
        # 『非 MoE（=稠密生成模型）→ V2』：Llama 这类在 v0.27.1 默认就走 V2
        # ——与 v0.21『V2 opt-in』完全反转
        cfg = make_v2_config(
            model_config=dict(architectures=("LlamaForCausalLM",), is_moe=False)
        )
        assert cfg.use_v2_model_runner is True

    def test_moe_not_in_default_list_stays_v1(self):
        cfg = make_v2_config(
            model_config=dict(architectures=("FooMoeForCausalLM",), is_moe=True)
        )
        assert cfg.use_v2_model_runner is False

    def test_moe_in_default_list_goes_v2(self):
        cfg = make_v2_config(
            model_config=dict(architectures=("Qwen2MoeForCausalLM",), is_moe=True)
        )
        assert cfg.use_v2_model_runner is True

    def test_pooling_runner_stays_v1(self):
        cfg = make_v2_config(model_config=dict(runner_type="pooling"))
        assert cfg.use_v2_model_runner is False

    def test_no_triton_falls_back_v1(self, monkeypatch):
        import vllm.config.vllm as cv

        monkeypatch.setattr(cv, "HAS_TRITON", False)
        cfg = make_v2_config(
            model_config=dict(architectures=("LlamaForCausalLM",), is_moe=False)
        )
        assert cfg.use_v2_model_runner is False

    def test_ngram_spec_falls_back_v1(self):
        cfg = make_v2_config(
            model_config=dict(architectures=("LlamaForCausalLM",)),
            speculative_config=dict(method="ngram"),
        )
        assert cfg.use_v2_model_runner is False


class TestNumSpeculativeTokens:
    """m04 前置：num_speculative_tokens property 连 diffusion canvas_length
    一起覆盖（vllm/config/vllm.py:L564-L575）。"""

    def test_none_by_default(self):
        assert make_v2_config().num_speculative_tokens == 0

    def test_spec_config_wins(self):
        cfg = make_v2_config(speculative_config=dict(num_speculative_tokens=4))
        assert cfg.num_speculative_tokens == 4

    def test_diffusion_canvas_length(self):
        cfg = make_v2_config(diffusion_config=dict(canvas_length=64))
        assert cfg.num_speculative_tokens == 64


class FakeV2InputBatch:
    def __init__(self, req_ids, cu_num_logits):
        self.req_ids = list(req_ids)
        self.cu_num_logits_np = np.asarray(cu_num_logits, dtype=np.int32)
        self.logits_indices = torch.tensor(list(range(cu_num_logits[-1])))
        self.num_draft_tokens = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="V2 落地需 CUDA")
class TestStructuredOutputsWorker:
    """V2 落地主体：GPU 常驻缓冲 + copy_stream 双 H2D + cu_num_logits 映射。"""

    def test_buffer_shapes(self):
        from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

        w = StructuredOutputsWorker(max_num_logits=16, vocab_size=V,
                                    device=torch.device("cuda"))
        assert w.grammar_bitmask.shape == (16, cdiv(V, 32))
        assert w.grammar_bitmask.dtype == torch.int32
        assert w.logits_indices.shape == (16,)
        assert w.copy_stream is not None

    def test_apply_end_to_end_with_triton_kernel(self):
        from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

        w = StructuredOutputsWorker(max_num_logits=8, vocab_size=V,
                                    device=torch.device("cuda"))
        # 批序 [B, A]；cu_num_logits 前缀和：B 占 logits 行 0..1、A 占行 2
        ib = FakeV2InputBatch(["B", "A"], [0, 2, 3])
        # 掩码行序（调度序）[A, B]：A 1 行 + B 2 行 = 3 行
        # 映射：A 的行 0 → logits 行 2；B 的行 1/2 → logits 行 0/1
        bm = np.zeros((3, cdiv(V, 32)), dtype=np.int32)
        bm[0, :] = -1                       # A：全允许 → logits 行 2
        bm[1, :] = -1                       # B 行 0：全允许 → logits 行 0
        bm[2, 0] = 0b101                    # B 行 1：只允许 token 0/2 → logits 行 1
        logits = torch.zeros((3, V), device="cuda", dtype=torch.float32)
        logits[1, 5] = 3.0                  # 行 1 上一个显著值
        logits[2, 7] = 2.0                  # 行 2（全允许）上的显著值

        w.apply_grammar_bitmask(logits, ib, ["A", "B"], bm)

        # 行 0（B 第一行全允许）、行 2（A 全允许）不受影响
        assert torch.all(logits[0] == 0.0)
        assert logits[2, 7].item() == 2.0
        assert torch.all(logits[2, :7] == 0.0)
        # 行 1（B 第二行 0b101）：bit=0 → -inf；bit=1（token 0/2）保留
        assert logits[1, 0].item() == 0.0
        assert logits[1, 2].item() == 0.0
        assert logits[1, 1].item() == float("-inf")
        assert logits[1, 5].item() == float("-inf")
        assert logits[1, V - 1].item() == float("-inf")  # 词表尾也在谓词内

    def test_row_count_invariant_assert(self):
        from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

        w = StructuredOutputsWorker(max_num_logits=8, vocab_size=V,
                                    device=torch.device("cuda"))
        ib = FakeV2InputBatch(["A"], [0, 2])  # A 占 2 行
        bm = np.zeros((3, cdiv(V, 32)), dtype=np.int32)  # 掩码 3 行 ≠ 映射 2 行
        logits = torch.zeros((2, V), device="cuda", dtype=torch.float32)
        with pytest.raises(AssertionError):
            w.apply_grammar_bitmask(logits, ib, ["A"], bm)

    def test_empty_req_ids_noop(self):
        from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

        w = StructuredOutputsWorker(max_num_logits=8, vocab_size=V,
                                    device=torch.device("cuda"))
        logits = torch.zeros((2, V), device="cuda", dtype=torch.float32)
        ib = FakeV2InputBatch(["A"], [0, 2])
        w.apply_grammar_bitmask(logits, ib, [], np.zeros((0, cdiv(V, 32)), np.int32))
        assert torch.all(logits == 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="V2 落地需 CUDA")
class TestV2SampleCallPoint:
    """V2 runner 的 sample 调用点：切片 → compute_logits → 先掩码后采样。"""

    def test_mask_before_sampler(self):
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner

        order = []

        class W:
            def apply_grammar_bitmask(self, logits, ib, ids, bm):
                order.append("mask")
                logits.fill_(-1.0)

        class S:
            def __init__(self):
                self.seen = None

            def __call__(self, logits, input_batch):
                order.append("sample")
                self.seen = logits
                import types as _t

                return _t.SimpleNamespace(
                    sampled_token_ids=None, logprobs_tensors=None,
                    num_sampled=1, num_rejected=0,
                )

        class M:
            def compute_logits(self, h):
                order.append("logits")
                return torch.zeros(h.shape[0], V, device=h.device)

        runner = V2Runner.__new__(V2Runner)
        runner.model = M()
        runner.structured_outputs_worker = W()
        runner.sampler = S()
        runner.rejection_sampler = None

        hidden = torch.zeros(4, 8, device="cuda")
        ib = FakeV2InputBatch(["A"], [0, 3])
        ib.num_draft_tokens = 0

        class GO:
            structured_output_request_ids = ["A"]
            grammar_bitmask = np.zeros((3, cdiv(V, 32)), np.int32)

        out = runner.sample(hidden, ib, GO())
        assert order == ["logits", "mask", "sample"]
        assert (out[1], out[2]) == (1, 0)
        assert torch.all(runner.sampler.seen == -1.0)  # sampler 拿到被掩码改过的 logits
