# SpecDecodeBaseProposer 契约骨架（vllm/v1/spec_decode/llm_base_proposer.py）。
# 行为基准 = 真实 v0.27.1：
#   - _greedy_sample（L428-L438）：draft 恒贪心；use_local_argmax_reduction 走
#     get_top_tokens 免全词表 gather；异词表经 vocab_mapping 约束+映射
#   - _sample_draft_tokens（L468-L497）：默认 greedy；概率化路径
#     （rejection_sample_method=="standard" 且 draft_sample_method=="probabilistic"）
#     才走 compute_probs_and_sample_next_token（L1848-L1886）并产 draft_probs
#   - propose（L502-L767）EAGLE 主路径：首前向（input_ids 左移一格、末槽填
#     next_token L846-L851）→ 单步早退（L619-L627）/ 自回归多步链式
#     （L682-L761：上一步草稿 int() 后喂回下一步）→ stack 成 [B, k]（L763-L767）
#   - take_last_draft_probs（L499-L500）：概率化草稿的缓存出口
#   - model_returns_tuple（L1007-L1015）：mtp 按架构判、eagle 系 True、
#     draft_model/dflash False
# 注意：注意力元数据/cudagraph/多模态/M-RoPE/DFlash extra-slots 为引擎栈旁路
# （subtraction_plan.delete 批准/未内嵌），测试用最小 cad 替身（batch_size() +
# query_start_loc）只驱动主路径控制流。
import types

import torch
import torch.nn.functional as F

from conftest import make_sampling_metadata
from vllm.config import (
    DraftModelConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

V, H, K = 16, 8, 3


class _ChainDraftModel:
    """测试替身草稿模型：hidden=embed(input_ids)、logits=hidden@W——
    链式喂回的每一步都可预测（f(x)=argmax(embed(x)@W)）。"""

    def __init__(self, returns_tuple: bool):
        g = torch.Generator().manual_seed(0)
        self.embed = torch.randn(V, H, generator=g)
        self.W = (torch.randn(H, V, generator=g) * 3).softmax(-1)
        self.returns_tuple = returns_tuple
        self.calls: list[torch.Tensor] = []
        self.combine_inputs: list[torch.Tensor] = []
        self.top_tokens: torch.Tensor | None = None

    def __call__(self, input_ids=None, positions=None, inputs_embeds=None,
                 hidden_states=None, **kwargs):
        self.calls.append(input_ids.detach().clone())
        h = self.embed[input_ids.long()]
        if self.returns_tuple:
            return h, h
        return h

    def compute_logits(self, hidden_states):
        return hidden_states @ self.W

    def combine_hidden_states(self, target_hidden_states):
        self.combine_inputs.append(target_hidden_states)
        return target_hidden_states.sum(-1, keepdim=True).expand(-1, H).contiguous()

    def get_top_tokens(self, hidden_states):
        assert self.top_tokens is not None
        return self.top_tokens


def make_proposer(method="eagle", *, k=K, parallel_drafting=False,
                  probabilistic=False, architectures=None, device=None,
                  max_model_len=64, returns_tuple=None):
    device = device or torch.device("cpu")
    draft_cfg = DraftModelConfig(
        hidden_size=H,
        inputs_embeds_size=H,
        hf_config=types.SimpleNamespace(
            architectures=architectures or ["EagleLlamaForCausalLM"]
        ),
    )
    cfg = VllmConfig(
        model_config=ModelConfig(max_model_len=max_model_len),
        scheduler_config=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=32),
        parallel_config=ParallelConfig(),
        speculative_config=SpeculativeConfig(
            method=method,
            num_speculative_tokens=k,
            parallel_drafting=parallel_drafting,
            use_local_argmax_reduction=False,
            use_heterogeneous_vocab=False,
            rejection_sample_method="standard",
            draft_sample_method="probabilistic" if probabilistic else "greedy",
            draft_model_config=draft_cfg,
        ),
    )
    p = SpecDecodeBaseProposer(cfg, device, pass_hidden_states_to_model=False)
    if returns_tuple is None:
        returns_tuple = p.model_returns_tuple()
    p.model = _ChainDraftModel(returns_tuple)
    return p


class _Cad:
    """最小 CommonAttentionMetadata 替身：默认 EAGLE 分支只吃这两个面。"""

    def __init__(self, query_lens, device):
        import numpy as np

        qsl = np.cumsum([0] + list(query_lens), dtype=np.int32)
        self.query_start_loc = torch.from_numpy(qsl).to(device)

    def batch_size(self):
        return len(self.query_start_loc) - 1


def _f(model, token: int) -> int:
    return int(model.compute_logits(model.embed[token]).argmax().item())


def _run_propose(p, next_token_ids, query_lens=(3, 2), target_len=5):
    device = p.device
    g = torch.Generator().manual_seed(2)
    target_token_ids = torch.randint(0, V, (target_len,), generator=g).to(device)
    target_positions = torch.arange(target_len, device=device)
    target_hidden = p.model.embed[target_token_ids.long()]
    cad = _Cad(query_lens, device)
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=cad.batch_size(), device=device
    )
    return p.propose(
        p.num_speculative_tokens,
        target_token_ids,
        target_positions,
        target_hidden,
        next_token_ids.to(device),
        None,
        cad,
        smeta,
    ), cad


def test_greedy_sample_plain_argmax():
    p = make_proposer()
    hidden = torch.randn(4, H)
    assert torch.equal(
        p._greedy_sample(hidden), p.model.compute_logits(hidden).argmax(dim=-1)
    )


def test_greedy_sample_local_argmax_reduction():
    # use_local_argmax_reduction=True → get_top_tokens 免全词表 logits gather
    p = make_proposer()
    p.use_local_argmax_reduction = True
    p.model.top_tokens = torch.tensor([7, 8, 9, 10])
    assert p._greedy_sample(torch.randn(4, H)).tolist() == [7, 8, 9, 10]


def test_greedy_sample_heterogeneous_vocab():
    # 异词表 drafter：logits 经 constrain、采 draft-vocab argmax、再映射回 target 词表
    p = make_proposer()
    p.use_heterogeneous_vocab = True

    class _Mapping:
        def constrain_draft_logits(self, logits):
            return logits

        def map_draft_to_target_ids(self, ids):
            return ids + 100

    p.vocab_mapping = _Mapping()
    hidden = torch.randn(3, H)
    got = p._greedy_sample(hidden)
    expect = p.model.compute_logits(hidden).argmax(dim=-1) + 100
    assert torch.equal(got, expect)


def test_sample_draft_tokens_default_greedy_no_probs():
    p = make_proposer(probabilistic=False)
    ids, probs = p._sample_draft_tokens(torch.randn(2, H), make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=2, device=p.device))
    assert probs is None
    assert ids.shape == (2,)


def test_sample_draft_tokens_probabilistic_greedy_falls_back():
    # 概率化启用 + all_greedy → 仍 greedy、不产 probs（L473-L474）
    p = make_proposer(probabilistic=True)
    hidden = p.model.embed[torch.tensor([1, 2])]
    ids, probs = p._sample_draft_tokens(
        hidden,
        make_sampling_metadata(
            all_greedy=True, temperature=None, batch_size=2, device=p.device
        ),
    )
    assert probs is None
    assert torch.equal(ids, p.model.compute_logits(hidden).argmax(dim=-1))


def test_sample_draft_tokens_probabilistic_produces_probs():
    p = make_proposer(probabilistic=True)
    hidden = p.model.embed[torch.tensor([1, 2])]
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(2, device=p.device),
        batch_size=2,
        device=p.device,
    )
    ids, probs = p._sample_draft_tokens(hidden, smeta)
    assert probs is not None and probs.shape == (2, V)
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-5)  # softmax 分布
    # compute_probs_and_sample_next_token 只用温度（L1872-L1875 注释：
    # 忽略其余采样参数——只影响接受率、不影响拒绝采样后的输出分布）
    assert ((ids >= 0) & (ids < V)).all()


def test_take_last_draft_probs_default_none():
    p = make_proposer()
    assert p.take_last_draft_probs() is None


def test_propose_eagle_autoregressive_chain():
    # 多步链式：d0 = f(next_token)；d_t = f(d_{t-1})——上一步草稿喂回下一步输入
    p = make_proposer(k=K)
    nxt = torch.tensor([3, 5], dtype=torch.int32)
    out, _ = _run_propose(p, nxt)
    assert out.shape == (2, K)
    f0 = [_f(p.model, 3), _f(p.model, 5)]
    assert out[:, 0].tolist() == f0
    assert out[:, 1].tolist() == [_f(p.model, t) for t in f0]
    assert out[:, 2].tolist() == [_f(p.model, t) for t in out[:, 1].tolist()]
    assert p.take_last_draft_probs() is None  # greedy 无 draft_probs
    # 前向次数 = 首拍 1 + 链式 k-1
    assert len(p.model.calls) == 1 + (K - 1)
    # 最后一次前向的输入 = 倒数第二步的草稿（int32 cast，L684-L686）
    assert torch.equal(p.model.calls[-1], out[:, K - 2].to(torch.int32))


def test_propose_single_step_early_exit():
    p = make_proposer(k=1)
    out, _ = _run_propose(p, torch.tensor([3, 5], dtype=torch.int32))
    assert out.shape == (2, 1)
    assert out[:, 0].tolist() == [_f(p.model, 3), _f(p.model, 5)]
    assert len(p.model.calls) == 1  # 单步早退：只跑首前向（L619-L627）


def test_propose_zero_draft_tokens_returns_empty():
    # 动态 K 查出 0（dynamic_sd_lookup 重载缩 K）→ [B, 0] 空草稿（L610-L616）；
    # 首前向照跑（保持 drafter KV cache 同步，注释 L607-L609 原话）
    p = make_proposer(k=2)
    nxt = torch.tensor([3, 5], dtype=torch.int32)
    device = p.device
    g = torch.Generator().manual_seed(2)
    tt = torch.randint(0, V, (5,), generator=g).to(device)
    cad = _Cad((3, 2), device)
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=2, device=device
    )
    out_empty = p.propose(
        0,
        tt,
        torch.arange(5, device=device),
        p.model.embed[tt.long()],
        nxt.to(device),
        None,
        cad,
        smeta,
    )
    assert out_empty.shape == (2, 0)
    assert out_empty.dtype == torch.int64
    assert len(p.model.calls) == 1  # 首前向已跑、链式循环 0 次


def test_propose_eagle3_combines_hidden_states():
    p = make_proposer(method="eagle3", k=1)
    out, _ = _run_propose(p, torch.tensor([3, 5], dtype=torch.int32))
    assert p.model.combine_inputs, "eagle3 首前向前应 combine_hidden_states（L526-L543）"
    assert out.shape == (2, 1)


def test_propose_draft_model_single_tensor_return():
    # model_returns_tuple：draft_model → False（L1015）→ 草稿模型单张量返回
    p = make_proposer(method="draft_model", returns_tuple=False)
    out, _ = _run_propose(p, torch.tensor([3, 5], dtype=torch.int32))
    assert out.shape == (2, K)


def test_model_returns_tuple_dispatch():
    assert make_proposer(method="eagle").model_returns_tuple() is True
    assert make_proposer(method="dflash").model_returns_tuple() is False
    assert make_proposer(method="draft_model").model_returns_tuple() is False
    p = make_proposer(method="mtp", architectures=["DeepSeekMTPModel"])
    assert p.model_returns_tuple() is True
    p = make_proposer(method="mtp", architectures=["SomeOtherMTP"])
    assert p.model_returns_tuple() is False


def test_propose_probabilistic_caches_last_draft_probs():
    # 单步早退 + 概率化：draft_probs 经 view 缓存为 [B, k, V]（L623-L626）
    p = make_proposer(k=1, probabilistic=True)
    nxt = torch.tensor([3, 5], dtype=torch.int32)
    device = p.device
    g = torch.Generator().manual_seed(2)
    tt = torch.randint(0, V, (5,), generator=g).to(device)
    cad = _Cad((3, 2), device)
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(2, device=device),
        batch_size=2,
        device=device,
    )
    out = p.propose(
        1, tt, torch.arange(5, device=device), p.model.embed[tt.long()],
        nxt.to(device), None, cad, smeta,
    )
    assert out.shape == (2, 1)
    probs = p.take_last_draft_probs()
    assert probs is not None
    assert probs.shape == (2, 1, V)


def test_init_derivations_from_spec_config():
    # __init__ 契约面：k/method/dtype/hidden_size 等从 SpeculativeConfig 派生
    p = make_proposer(method="eagle", k=5)
    assert p.num_speculative_tokens == 5
    assert p.method == "eagle"
    assert p.hidden_size == H
    assert p.max_model_len == 64
    assert p.parallel_drafting is False
    assert p._enable_probabilistic_draft_probs is False
    assert p.use_local_argmax_reduction is False
    p2 = make_proposer(probabilistic=True)
    assert p2._enable_probabilistic_draft_probs is True
    assert p2.use_fp64_gumbel is False
