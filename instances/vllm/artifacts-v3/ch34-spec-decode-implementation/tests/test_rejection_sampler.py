# RejectionSampler（vllm/v1/sample/rejection_sampler.py:L38-L953）——本章算法心脏。
# 行为基准 = 真实 v0.27.1（严格按 arXiv:2211.17192，docstring L39-L59）：
#   - greedy 准则：draft==target_argmax 接受；拒绝位写 argmax（=one-hot 残差的
#     唯一 token）并早停；全接受补 bonus（kernel L715-L769）
#   - random 准则：draft_prob>0 且 target_prob/draft_prob >= uniform_prob
#     （kernel L829，u∈[0,1) 隐式 min(1,·)）；拒绝位取 recovered、早停；
#     全接受补 bonus（kernel L772-L845）
#   - NO_DRAFT_PROBS（ngram）：draft_prob=1 退化以 p_t(x) 接受（L816-L817）
#   - padded draft(-1) 直接拒（L809-L811）
#   - recovered = argmax((p_t−p_d)+ · inv_q) 免归一化、每请求一份 q（L663-L710/
#     L872-L953）；ngram 残差 = 屏蔽 draft token 的 p_t（L913-L918）
#   - uniform float64（L639-L648，pytorch#16706）
#   - 输出 buffer [B, max_spec_len+1] 预填 PLACEHOLDER=-1（L428-L433）
#   - parse_output valid_mask（≠-1 且 <vocab_size）还原变长（L252-L287）
#   - apply_sampling_constraints/expand_batch_to_tokens 逐草稿位扩展（L510-L605）
#   - bonus 位外采（组合持有的普通 Sampler，可带 top_p/top_k；L133-L147）
import torch
import pytest

from conftest import make_sampling_metadata, make_spec_metadata
from vllm.v1.sample.rejection_sampler import (
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    RejectionSampler,
    apply_sampling_constraints,
    expand_batch_to_tokens,
    generate_uniform_probs,
    rejection_sample,
)
from vllm.v1.sample.sampler import Sampler

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel 需 CUDA"
)


def run_rejection(
    num_draft, drafts_2d, target_logits, smeta, device, draft_probs=None, vocab=8
):
    """摊平输入并调 rejection_sample（真实 L394-L507 的调用面）。

    target_logits/draft_probs 接受 2-D 张量或 1-D 行的列表（每草稿位一行）。
    """
    md = make_spec_metadata(num_draft, drafts_2d, device)
    if isinstance(target_logits, torch.Tensor):
        target_logits = target_logits.to(device=device, dtype=torch.float32)
    else:
        target_logits = torch.stack(target_logits, dim=0).to(
            device=device, dtype=torch.float32
        )
    bonus = torch.full(
        (len(num_draft), 1), 999, dtype=torch.int32, device=device
    )  # bonus 哨兵：全接受时行尾必为 999
    if draft_probs is not None:
        if isinstance(draft_probs, torch.Tensor):
            draft_probs = draft_probs.to(device)
        else:
            draft_probs = torch.stack(draft_probs, dim=0).to(device)
        draft_probs = draft_probs.contiguous()
    return rejection_sample(
        md.draft_token_ids,
        md.num_draft_tokens,
        md.max_spec_len,
        md.cu_num_draft_tokens,
        draft_probs,
        target_logits,
        bonus,
        smeta,
    )


# ── greedy kernel（L715-L769）────────────────────────────────────────────


@requires_cuda
def test_greedy_all_accept_appends_bonus(device):
    # argmax 恒等于 draft → 三位全收，末位补 bonus 哨兵 999
    V = 8
    drafts = [[3, 5], [1, 0]]
    logits_rows = []
    for row in drafts:
        for t in row:
            lg = torch.zeros(V)
            lg[t] = 10.0
            logits_rows.append(lg)
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=2, device=device
    )
    out = run_rejection([2, 2], drafts, logits_rows, smeta, device)
    assert out.shape == (2, 3)
    assert out.dtype == torch.int32
    assert out.tolist() == [[3, 5, 999], [1, 0, 999]]


@requires_cuda
def test_greedy_first_reject_truncates_and_writes_argmax(device):
    # req0：pos1 草稿错 → pos1 写 target argmax（greedy 语义的 recovered），
    # pos2 保持 -1（早停后不写）、不补 bonus
    V = 8
    drafts = [[3, 5]]
    l0 = torch.zeros(V)
    l0[3] = 10.0
    l1 = torch.zeros(V)
    l1[7] = 10.0  # argmax=7，draft=5 ≠ 7 → 拒
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=1, device=device
    )
    out = run_rejection([2], drafts, [l0, l1], smeta, device)
    assert out.tolist() == [[3, 7, -1]]


@requires_cuda
def test_greedy_padded_draft_rejected_immediately(device):
    # pad 到 1+K 的新 decode 请求：[-1] 占位草稿第一位就拒（draft=-1 ≠ argmax），
    # 输出 = [argmax, -1]，不补 bonus
    V = 8
    l0 = torch.zeros(V)
    l0[4] = 10.0
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=1, device=device
    )
    out = run_rejection([1], [[-1]], [l0], smeta, device)
    assert out.tolist() == [[4, -1]]


@requires_cuda
def test_greedy_zero_draft_tokens_request_gets_bonus_slot(device):
    # ngram 场景常见：某请求一个草稿都没猜出（num_draft=0）→ kernel 循环 0 次、
    # not rejected 成立 → bonus 写在第 0 位。真实语义：0 草稿请求由 bonus 位
    # 供 token（num_draft=0 请求的 logits 行数为 0，target_logits 为空矩阵）。
    V = 8
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=1, device=device
    )
    out = run_rejection([0], [[]], torch.zeros(0, V), smeta, device)
    assert out.tolist() == [[999]]


# ── random kernel（L772-L845）：统计律（接受率/分布等价）────────────────


@requires_cuda
def test_random_no_draft_probs_acceptance_rate_is_target_prob(device):
    # NO_DRAFT_PROBS：draft_prob=1 → 接受概率 = p_t(x)（退化准则）
    V = 8
    B = 20000
    p = torch.zeros(V)
    p[3] = 0.3
    p[0] = 0.7
    logits_rows = [p.log() for _ in range(B)]
    drafts = [[3]] * B
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(B, device=device),
        batch_size=B,
        device=device,
    )
    out = run_rejection([1] * B, drafts, logits_rows, smeta, device)
    accepted = (out[:, 0] == 3).float().mean().item()
    assert abs(accepted - 0.3) < 0.02, accepted


@requires_cuda
def test_random_acceptance_ratio_min_one(device):
    # p_d=0.6, p_t=0.3 → 接受率 ≈ min(1, 0.5)=0.5；p_d=0.3, p_t=0.6 → min(1,2)=1 全接受
    V = 4
    B = 20000
    p_t = torch.tensor([0.6, 0.3, 0.05, 0.05])
    p_d = torch.tensor([0.3, 0.6, 0.05, 0.05])
    logits_rows = [p_t.log() for _ in range(2 * B)]
    dprobs_rows = [p_d for _ in range(2 * B)]
    drafts = [[0]] * B + [[1]] * B
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(2 * B, device=device),
        batch_size=2 * B,
        device=device,
    )
    out = run_rejection(
        [1] * (2 * B), drafts, logits_rows, smeta, device, draft_probs=dprobs_rows
    )
    accepted_hi = (out[:B, 0] == 0).float().mean().item()  # min(1, 0.6/0.3)=1
    accepted_lo = (out[B:, 0] == 1).float().mean().item()  # min(1, 0.3/0.6)=0.5
    assert accepted_hi > 0.999
    assert abs(accepted_lo - 0.5) < 0.02, accepted_lo


@requires_cuda
def test_random_output_distribution_is_lossless(device):
    # 无损性（本章核心论断）：k=1 时输出 token 的经验分布 = 直接从 p_t 采样
    V = 6
    B = 30000
    p_t = torch.tensor([0.05, 0.1, 0.2, 0.25, 0.3, 0.1])
    p_d = torch.tensor([0.3, 0.3, 0.2, 0.1, 0.05, 0.05])
    torch.manual_seed(0)
    draft_choice = torch.multinomial(p_d, B, replacement=True)
    logits_rows = [p_t.log() for _ in range(B)]
    dprobs_rows = [p_d for _ in range(B)]
    drafts = [[int(x)] for x in draft_choice.tolist()]
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(B, device=device),
        batch_size=B,
        device=device,
    )
    out = run_rejection(
        [1] * B, drafts, logits_rows, smeta, device, draft_probs=dprobs_rows
    )
    counts = torch.bincount(out[:, 0].long(), minlength=V).float().cpu() / B
    tv = 0.5 * (counts - p_t).abs().sum().item()
    assert tv < 0.03, (counts.tolist(), tv)


@requires_cuda
def test_random_matches_python_reference_with_seeded_generators(device):
    # 确定性全链对拍：per-request generator 种子固定 → uniform/q 均确定，
    # 与纯 torch 参考实现（min(1,p_t/p_d) 接受 + 残差 argmax(p·inv_q) 恢复）
    # 逐位一致。参考侧与被测侧各用一套同种子 generator（随机流互不干扰）。
    from vllm.v1.sample.rejection_sampler import sample_recovered_tokens

    V = 10
    num_draft = [2, 3]
    torch.manual_seed(7)
    p_t64 = torch.softmax(torch.randn(sum(num_draft), V, dtype=torch.float64), -1)
    p_d64 = torch.softmax(torch.randn(sum(num_draft), V, dtype=torch.float64), -1)
    drafts = [[4, 9], [1, 0, 7]]
    flat_draft = [4, 9, 1, 0, 7]
    # 被测侧输入：float32 logits/概率（与真实数据流同 dtype）
    target_logits = torch.stack([p_t64[i].log() for i in range(sum(num_draft))]).float()
    draft_probs = p_d64.float()
    p_t32 = torch.softmax(target_logits, -1)  # 与 rejection_sample 内部一致
    p_d32 = draft_probs

    def fresh_gens(prefix_seed):
        return {
            i: torch.Generator(device=device).manual_seed(prefix_seed + i)
            for i in range(len(num_draft))
        }

    # ── 参考实现（纯 torch，复刻真实随机源生成顺序）──
    gens_ref = fresh_gens(100)
    uniform = generate_uniform_probs(sum(num_draft), num_draft, gens_ref, device)
    q = torch.empty((len(num_draft), V), dtype=torch.float32, device=device)
    q.exponential_()
    for i, g in gens_ref.items():
        if num_draft[i] > 0:
            q[i].exponential_(generator=g)
    inv_q = q.reciprocal().cpu()
    recovered = torch.empty(sum(num_draft), dtype=torch.int32)
    s = 0
    for i, n in enumerate(num_draft):
        for pos in range(n):
            t = s + pos
            prob = torch.clamp(p_t32[t] - p_d32[t], min=0.0)
            recovered[t] = int((prob * inv_q[i]).argmax().item())
        s += n

    expected = torch.full((len(num_draft), 4), -1, dtype=torch.int32)
    bonus = torch.tensor([[500], [501]], dtype=torch.int32, device=device)
    s = 0
    for i, n in enumerate(num_draft):
        rejected = False
        for pos in range(n):
            t = s + pos
            x = flat_draft[t]
            u = uniform[t].item()
            accepted = (p_d32[t][x].item() > 0) and (
                p_t32[t][x].item() / p_d32[t][x].item() >= u
            )
            if accepted:
                expected[i, pos] = x
            else:
                rejected = True
                expected[i, pos] = recovered[t].item()
                break
        if not rejected:
            expected[i, n] = bonus[i, 0].item()
        s += n

    # ── 被测侧：同种子的另一套 generator ──
    gens_run = fresh_gens(100)
    md = make_spec_metadata(num_draft, drafts, device)
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(len(num_draft), device=device),
        generators=gens_run,
        batch_size=len(num_draft),
        device=device,
    )
    out = rejection_sample(
        md.draft_token_ids,
        md.num_draft_tokens,
        md.max_spec_len,
        md.cu_num_draft_tokens,
        draft_probs.to(device),
        target_logits.to(device),
        bonus,
        smeta,
    )
    assert out.tolist() == expected.tolist()


@requires_cuda
def test_mixed_greedy_random_batch_coexists(device):
    # is_greedy mask 同批共存（L435-L438）：greedy 请求走 greedy kernel（确定性），
    # random 请求走 random kernel——一次调用、两 kernel 各自早退
    V = 6
    B = 500
    drafts = [[2, 3]] + [[2]] * (B - 1)
    logits_rows = []
    lg = torch.zeros(V)
    lg[2], lg[3] = 10.0, 5.0  # argmax=2：req0 pos0 接受(draft 2)、pos1 拒(draft 3≠2)
    logits_rows += [lg, lg]
    p_t = torch.tensor([0.05, 0.05, 0.8, 0.02, 0.04, 0.04])
    logits_rows += [p_t.log()] * (B - 1)
    temperature = torch.zeros(B, device=device)
    temperature[1:] = 1.0
    smeta = make_sampling_metadata(temperature=temperature, batch_size=B, device=device)
    out = run_rejection([2] + [1] * (B - 1), drafts, logits_rows, smeta, device)
    # greedy 行：draft==argmax 接受、首拒写 argmax、停（无 bonus）
    assert out[0].tolist() == [2, 2, -1]
    # random 行：接受率 ≈ p_t(2)=0.8（NO_DRAFT_PROBS）
    accepted = (out[1:, 0] == 2).float().mean().item()
    assert abs(accepted - 0.8) < 0.03, accepted


# ── recovered 残差（L663-L710/L872-L953）────────────────────────────────


@requires_cuda
def test_recovered_residual_argmax_no_normalization(device):
    # score=prob·inv_q 免归一化（L931-L932 注释）：Σprob 不影响 argmax 胜者。
    # 固定 q（seed）→ recovered = argmax((p_t−p_d)+ · inv_q) 逐位可算。
    from vllm.v1.sample.rejection_sampler import sample_recovered_tokens

    V = 12
    torch.manual_seed(3)
    p_t = torch.softmax(torch.randn(3, V, dtype=torch.float64), -1)
    p_d = torch.softmax(torch.randn(3, V, dtype=torch.float64), -1)
    gens = {0: torch.Generator(device=device).manual_seed(11)}
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(1, device=device),
        generators=gens,
        batch_size=1,
        device=device,
    )
    md = make_spec_metadata([3], [[1, 5, 8]], device)
    got = sample_recovered_tokens(
        md.max_spec_len,
        md.num_draft_tokens,
        md.cu_num_draft_tokens,
        md.draft_token_ids,
        p_d.float().to(device),
        p_t.float().to(device),
        smeta,
        device,
    )
    # 参考侧用同种子的另一支 generator（随机流与被测侧互不干扰）
    g_ref = torch.Generator(device=device).manual_seed(11)
    q = torch.empty((1, V), dtype=torch.float32, device=device)
    q.exponential_(generator=g_ref)
    prob = torch.clamp(p_t.float() - p_d.float(), min=0.0).to(device)
    expect = (prob * q.reciprocal()[0]).argmax(dim=-1).to(torch.int32)
    assert got.tolist() == expect.tolist()
    assert got.dtype == torch.int32


@requires_cuda
def test_recovered_no_draft_probs_masks_draft_token(device):
    # NO_DRAFT_PROBS 残差 = 屏蔽 draft token 的 p_t（L913-L918）：
    # 除 draft token 外 p_t 均匀 → recovered 均匀分布于其余 token（统计）
    from vllm.v1.sample.rejection_sampler import sample_recovered_tokens

    V = 5
    B = 4000
    p_t = torch.full((B, V), 0.25)
    p_t[:, 2] = 0.0  # 屏蔽位
    p_t /= p_t.sum(-1, keepdim=True)
    md = make_spec_metadata([1] * B, [[2]] * B, device)
    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(B, device=device),
        batch_size=B,
        device=device,
    )
    got = sample_recovered_tokens(
        md.max_spec_len,
        md.num_draft_tokens,
        md.cu_num_draft_tokens,
        md.draft_token_ids,
        None,
        p_t.to(device),
        smeta,
        device,
    )
    assert (got != 2).all()
    counts = torch.bincount(got.long(), minlength=V).float() / B
    for v in [0, 1, 3, 4]:
        assert abs(counts[v].item() - 0.25) < 0.03, counts.tolist()


def test_generate_uniform_probs_float64_and_reproducible(device):
    n = [2, 0, 3, 1]
    num_tokens = sum(n)
    gens = {0: torch.Generator(device=device).manual_seed(42)}
    u = generate_uniform_probs(num_tokens, n, gens, device)
    assert u.dtype == torch.float64  # pytorch#16706：float32 可能采到精确 0.0
    assert u.shape == (num_tokens,)
    assert (u >= 0).all() and (u < 1).all()
    # n=0 请求跳过生成（可复现性，L651-L654）：req0 的两值与没有空请求时同种子一致
    g_ref = torch.Generator(device=device).manual_seed(42)
    ref = generate_uniform_probs(2, [2], {0: g_ref}, device)
    assert u[:2].tolist() == ref.tolist()


def test_generate_uniform_probs_unseeded_fill(device):
    # 无 generator 的请求用全局 torch.rand 填充（L644-L648），仍在 [0,1)
    u = generate_uniform_probs(5, [2, 3], {}, device)
    assert u.shape == (5,)
    assert (u >= 0).all() and (u < 1).all()
    assert u.dtype == torch.float64


# ── parse_output（L252-L287）─────────────────────────────────────────────


def test_parse_output_restores_variable_lengths(device):
    out = torch.tensor(
        [[1, 2, 3], [4, -1, -1], [5, -1, 7]], dtype=torch.int32, device=device
    )
    # 注：同一行内 -1 之后不会再有有效位（早停保证），[5,-1,7] 只测过滤器语义。
    # 返回二元组（真实调用面 gpu_model_runner.py:L3791 解包两个值）。
    outputs, output_logprobs = RejectionSampler.parse_output(out, vocab_size=100)
    assert outputs == [[1, 2, 3], [4], [5, 7]]
    assert output_logprobs is None


def test_parse_output_filters_out_of_vocab_and_discard_rows(device):
    out = torch.tensor([[1, 200, 3], [4, 5, 6]], dtype=torch.int32, device=device)
    outputs, _ = RejectionSampler.parse_output(
        out, vocab_size=100, discard_req_indices=(1,)
    )
    assert outputs == [[1, 3], []]


def test_placeholder_token_id_sentinel():
    assert PLACEHOLDER_TOKEN_ID == -1
    assert MAX_SPEC_LEN == 128


# ── expand_batch_to_tokens / apply_sampling_constraints（L510-L605）─────


@requires_cuda
def test_expand_batch_to_tokens_docstring_example(device):
    # docstring 工例（L578-L579）：x=[a,b,c], cu=[2,5,6] → [a,a,b,b,b,c]
    x = torch.tensor([10.0, 20.0, 30.0], device=device)
    cu = torch.tensor([2, 5, 6], dtype=torch.int32, device=device)
    out = expand_batch_to_tokens(x, cu, 6)
    assert out.tolist() == [10.0, 10.0, 20.0, 20.0, 20.0, 30.0]
    # greedy 温度 0 → 1 防除零（apply_sampling_constraints L537-L543 的 replace 语义）；
    # cu=[1,3] = 请求0 占 1 位、请求1 占 2 位
    out2 = expand_batch_to_tokens(
        torch.tensor([0.0, 0.7], device=device),
        torch.tensor([1, 3], dtype=torch.int32, device=device),
        3,
        replace_from=0,
        replace_to=1,
    )
    assert out2.tolist() == pytest.approx([1.0, 0.7, 0.7])


@requires_cuda
def test_apply_sampling_constraints_greedy_identity(device):
    logits = torch.randn(5, 7, device=device)
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=2, device=device
    )
    cu = torch.tensor([2, 3], dtype=torch.int32, device=device)
    out = apply_sampling_constraints(logits, cu, smeta)
    assert out is logits  # all_greedy 原样返回（L533-L534）


@requires_cuda
def test_apply_sampling_constraints_temperature_expansion(device):
    # [batch] 温度扩到逐草稿位：greedy 行 0→1 不除，random 行除自身
    logits = torch.ones(4, 6, device=device)
    temperature = torch.tensor([0.0, 2.0], device=device)
    smeta = make_sampling_metadata(
        temperature=temperature, top_k=None, top_p=None, batch_size=2, device=device
    )
    cu = torch.tensor([1, 4], dtype=torch.int32, device=device)  # req0:1 位、req1:3 位
    out = apply_sampling_constraints(logits, cu, smeta)
    assert torch.allclose(out[0], torch.ones(6, device=device))
    assert torch.allclose(out[1:], torch.full((3, 6), 0.5, device=device))


@requires_cuda
def test_apply_sampling_constraints_top_k_per_request(device):
    # top_k=1 的行只剩 argmax；top_k=3 的行剩前 3（[batch] 参数逐草稿位扩展）
    torch.manual_seed(1)
    logits = torch.randn(3, 10, device=device)
    smeta = make_sampling_metadata(
        temperature=torch.ones(2, device=device),
        top_k=torch.tensor([1, 3], dtype=torch.int64, device=device),
        top_p=None,
        batch_size=2,
        device=device,
    )
    cu = torch.tensor([1, 3], dtype=torch.int32, device=device)  # req0:1 位、req1:2 位
    out = apply_sampling_constraints(logits, cu, smeta)
    assert out[0].argmax().item() == logits[0].argmax().item()
    assert (out[0] == float("-inf")).sum().item() == 9
    assert (out[1] == float("-inf")).sum().item() == 7
    assert (out[2] == float("-inf")).sum().item() == 7


# ── RejectionSampler.forward：组合 Sampler 外采 bonus（L92-L201）────────


@requires_cuda
def test_forward_bonus_sampled_by_composed_sampler_with_topk(device):
    # m5：bonus 位交给组合持有的普通 Sampler（可带 top_k/top_p——spec 主路径
    # 不支持的策略 bonus 位能用）；target 位 top_k=1 → 输出确定化。
    V = 8
    torch.manual_seed(5)
    target_rows = torch.randn(2, V)
    bonus_row = torch.randn(1, V)
    logits = torch.cat([target_rows, bonus_row], 0).to(device)  # [num_tokens+B, V]
    md = make_spec_metadata([2], [[2, 4]], device)
    # bonus_logits_indices=[2]、target_logits_indices=[0,1]、logits_indices=[0,1,2]
    md.bonus_logits_indices = torch.tensor([2], dtype=torch.int32, device=device)
    md.target_logits_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    md.logits_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    smeta = make_sampling_metadata(
        all_random=True,
        temperature=torch.ones(1, device=device),
        top_k=torch.tensor([1], dtype=torch.int64, device=device),
        batch_size=1,
        device=device,
    )
    rs = RejectionSampler(Sampler())
    out = rs(md, None, logits, smeta)
    # target 位经 top_k=1 → one-hot → 概率 0/1：接受 iff draft==argmax（否则拒、
    # recovered=argmax、早停）——按确定性语义推期望输出
    arg0 = target_rows[0].argmax().item()
    arg1 = target_rows[1].argmax().item()
    expected = [-1, -1, -1]
    if 2 == arg0:
        expected[0] = 2
        if 4 == arg1:
            expected[1] = 4  # 全接受 → bonus 补在末位
            expected[2] = bonus_row[0].argmax().item()
        else:
            expected[1] = arg1  # pos1 拒 → 写 argmax、停
    else:
        expected[0] = arg0  # pos0 拒 → 写 argmax、停
    assert out.sampled_token_ids.shape == (1, 3)
    assert out.sampled_token_ids[0].tolist() == expected
    # bonus 位（若全接受）= 组合 Sampler 以 top_k=1 采出的 argmax
    assert out.logprobs_tensors is None  # logprobs 装配已按计划删除（恒 None）


@requires_cuda
def test_forward_returns_int32_padded_buffer(device):
    V = 4
    logits = torch.randn(3, V).to(device)
    md = make_spec_metadata([2], [[0, 1]], device)
    md.bonus_logits_indices = torch.tensor([2], dtype=torch.int32, device=device)
    md.target_logits_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    md.logits_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=1, device=device
    )
    out = RejectionSampler(Sampler())(md, None, logits, smeta)
    assert out.sampled_token_ids.dtype == torch.int32
    assert out.sampled_token_ids.shape == (1, 3)


def test_max_spec_len_guard():
    md = make_spec_metadata(
        [MAX_SPEC_LEN + 1], [[1] * (MAX_SPEC_LEN + 1)], torch.device("cpu")
    )
    smeta = make_sampling_metadata(
        all_greedy=True, temperature=None, batch_size=1, device=torch.device("cpu")
    )
    rs = RejectionSampler(Sampler())
    with pytest.raises(AssertionError):
        rs(md, None, torch.zeros(1, 4), smeta)  # assert max_spec_len <= 128（L123）


def test_use_fp64_gumbel_transmitted():
    # v0.27 组合结构：use_fp64_gumbel 从组合的 Sampler 透传（L69）
    rs = RejectionSampler(Sampler(use_fp64_gumbel=True))
    assert rs.use_fp64_gumbel is True
    assert RejectionSampler(Sampler()).use_fp64_gumbel is False


def test_bonus_logits_slice_is_new_storage():
    # L128-L131/L149-L151 注释：张量索引产生新存储——对 bonus/target 的
    # in-place 修改不会污染原 logits（forward 的 clone 语义前提）。
    logits = torch.randn(4, 6)
    bonus = logits[torch.tensor([0, 3])]
    bonus += 100.0
    assert logits[0, 0] != bonus[0, 0]


# ── apply_logits_processors 的 spec 特化（L289-L391）─────────────────────


def _make_rs():
    return RejectionSampler(Sampler())


def test_combine_outputs_with_spec_tokens_builds_prefix_rows():
    # RejectionSampler 版（L376-L391）：为每个草稿位造『到此位为止的前缀历史』行；
    # 空草稿请求整段跳过（它的 target logits 行数为 0）。
    out = _make_rs()._combine_outputs_with_spec_tokens([[5], [6, 7]], [[10, 11], []])
    assert out == [[5], [5, 10]]


def test_combine_outputs_no_spec_passthrough():
    out = _make_rs()._combine_outputs_with_spec_tokens([[5], [6]], None)
    assert out == [[5], [6]]


def test_sampler_combine_outputs_appends_whole_spec():
    # 对照：Sampler 里的同名方法语义不同（拼整段 spec、逐请求一行）——读代码陷阱
    out = Sampler._combine_outputs_with_spec_tokens([[5], [6, 7]], [[10, 11], []])
    assert out == [[5, 10, 11], [6, 7]]


def test_min_tokens_apply_with_spec_decode_masks_leading_rows():
    # docstring 工例（builtin.py:L240-L244）：num_draft_tokens=[2,3,1] →
    # logits [6,V]、cumsum=[0,2,5,6]、req0 占 0-1、req1 占 2-4、req2 占 5。
    # req0 已出 3 token、min_tokens=5 → remaining=2 → n_mask=min(2,2)=2：0-1 行封 stop。
    # req1 已出 4、min=6 → remaining=2 → n_mask=min(2,3)=2：2-3 行封、第 4 行不封。
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
    from vllm.v1.sample.logits_processor.interface import BatchUpdate

    device = torch.device("cpu")
    proc = MinTokensLogitsProcessor(VllmConfig(), device, False)
    proc.update_state(
        BatchUpdate(
            batch_size=3,
            removed=(),
            added=(
                (
                    0,
                    SamplingParams(min_tokens=5, all_stop_token_ids={90}),
                    None,
                    [1, 2, 3],
                ),
                (
                    1,
                    SamplingParams(min_tokens=6, all_stop_token_ids={90, 91}),
                    None,
                    [4, 5, 6, 7],
                ),
            ),
            moved=(),
        )
    )
    logits = torch.zeros(6, 100)
    out = proc.apply_with_spec_decode(logits, [2, 3, 1])
    # token 90 同时在两请求的 stop 集：req0 行 0-1 + req1 行 2-3
    assert (out[:, 90] == float("-inf")).nonzero().flatten().tolist() == [0, 1, 2, 3]
    assert (out[:, 91] == float("-inf")).nonzero().flatten().tolist() == [2, 3]
    # req1 的第 3 行（remaining 已耗尽）与 req2 无约束
    assert out[4, 90] == 0 and out[4, 91] == 0 and out[5, 90] == 0


def test_min_tokens_apply_with_spec_decode_no_entries_passthrough():
    from vllm.config import VllmConfig
    from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

    logits = torch.randn(3, 8)
    proc = MinTokensLogitsProcessor(VllmConfig(), torch.device("cpu"), False)
    out = proc.apply_with_spec_decode(logits, [1, 1, 1])
    assert out is logits  # min_toks 空 → 原样返回（L246-L247）


@requires_cuda
def test_apply_logits_processors_penalties_per_draft_row(device):
    # repeat_indices 把 [batch] 惩罚参数展开到草稿位；output_token_ids 已被
    # _combine 造出逐位前缀行 → 频率惩罚按各草稿位『到此为止』的历史计数。
    V = 10
    rs = _make_rs()
    md = make_spec_metadata([2], [[3, 3]], device)
    logits = torch.zeros(2, V, device=device)
    smeta = make_sampling_metadata(
        no_penalties=False,
        prompt_token_ids=torch.zeros(1, 4, dtype=torch.long, device=device),
        frequency_penalties=torch.tensor([1.0], device=device),
        presence_penalties=torch.tensor([0.0], device=device),
        repetition_penalties=torch.tensor([1.0], device=device),
        output_token_ids=[[3]],
        spec_token_ids=[[3, 3]],
        batch_size=1,
        device=device,
    )
    out = rs.apply_logits_processors(logits, smeta, md)
    # 前缀历史：行0=[3]（token 3 计数 1）、行1=[3,3]（计数 2）；
    # OpenAI 频率式 logit -= freq*count；repetition 1.0 对 0 logit 无变化
    assert out[0, 3].item() == pytest.approx(-1.0)
    assert out[1, 3].item() == pytest.approx(-2.0)
    assert out[0, 5].item() == 0.0


@requires_cuda
def test_apply_logits_processors_bad_words_with_drafts(device):
    # bad_words 吃『草稿当已输出』的前缀历史：禁词 [3,7] 只在以 3 结尾的行被掩。
    V = 10
    rs = _make_rs()
    md = make_spec_metadata([2], [[3, 3]], device)
    logits = torch.zeros(2, V, device=device)
    smeta = make_sampling_metadata(
        no_penalties=True,
        output_token_ids=[[3]],
        spec_token_ids=[[3, 3]],
        bad_words_token_ids={0: [[3, 7]]},
        batch_size=1,
        device=device,
    )
    out = rs.apply_logits_processors(logits, smeta, md)
    # 前缀行：行0=[3]、行1=[3,3]（均以 3 结尾 → 前缀匹配 → 掩 7）
    assert out[0, 7].item() == float("-inf")
    assert out[1, 7].item() == float("-inf")
    assert out[0, 3].item() == 0.0
