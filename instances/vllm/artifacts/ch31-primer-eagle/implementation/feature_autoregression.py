"""
PAPER: arXiv:2401.15077 "EAGLE: Speculative Sampling Requires Rethinking
Feature Uncertainty" -- §2 Preliminaries (vanilla token-level autoregression
notation T -> E -> F -> p -> t), §3.1 (Autoregression Head: FC(2h->h) fusing
[token_embed (+) feature] + a decoder layer, Fig.6; "feature & shifted-token"
input, Fig.3-5), §3.2 (training objective: Smooth-L1 regression + cross-
entropy classification, w_cls=0.1).

Small torch (CPU-only) reference of the *drafting-phase building blocks*:
ToyTargetLLM stands in for "the target LLM" whose second-to-top-layer
features and shared Embedding/LM Head the draft model reuses (design
decision in the dossier: EAGLE trains no new embedding/LM head, only a
lightweight Autoregression Head). AutoregressionHead is that head.

This is NOT vllm/model_executor/models/llama_eagle.py: no ReplicatedLinear,
no tensor-parallel/quantization plumbing, no checkpoint weight-name remap,
no real multi-layer LLaMA backbone. hidden_dim and vocab_size are toy
(single digits) so a reader can hand-trace every number; the *shape and
control flow* fc(2h->h) -> decoder layer -> shared LM head is faithful to
the paper (§3.1, Fig.6) and to llama_eagle.py's forward() (L100-116).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# PAPER: §2 Preliminaries (Notations: T_{1:j} -> E_{1:j} -> f_j -> p_{j+1} -> t_{j+1})
class ToyTargetLLM(nn.Module):
    """
    Deterministic stand-in for the paper's "target LLM": owns the Embedding
    layer and LM Head that the EAGLE draft model reuses without additional
    training (§3.1: "The Embedding layer and LM Head employ the parameters
    of the target LLM and do not necessitate additional training"), plus a
    tiny causal "backbone" turning a token history into a second-to-top-
    layer feature sequence F_{1:j}. The backbone's internals are not
    EAGLE's subject -- only the T -> E -> F -> p -> t interface is (§2).
    """

    # PAPER: §3.1 ("The Embedding layer and LM Head employ the parameters of
    # the target LLM and do not necessitate additional training")
    def __init__(self, vocab_size: int, hidden_dim: int, seed: int = 0) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.backbone = nn.GRUCell(hidden_dim, hidden_dim)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)

    # PAPER: §2 (T_{1:j} -> E_{1:j} -> f_j, run causally token by token)
    def forward_prefix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (seq_len,) int64. Returns F_{1:seq_len}, shape
        (seq_len, hidden_dim) -- f_j is the feature *after* consuming t_j,
        matching the paper's f_j notation.
        """
        h = torch.zeros(self.hidden_dim)
        feats = []
        for t in token_ids.tolist():
            e = self.embed_tokens(torch.tensor(t))
            h = self.backbone(e.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            feats.append(h)
        return torch.stack(feats, dim=0)

    # PAPER: §2 (p_{j+1} = LM_Head(f_j))
    def next_token_distribution(self, feature: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.lm_head(feature), dim=-1)


# PAPER: §3.1 Fig.6 (Autoregression Head: FC(2h->h) + one decoder layer)
class AutoregressionHead(nn.Module):
    """
    EAGLE's draft model core. Takes a fused [token_embed (+) feature] of
    shape (..., 2*hidden_dim), reduces it to hidden_dim via `fc`, then
    predicts the next feature through one decoder-layer-like transform. The
    Embedding and LM Head are NOT owned here -- they are shared with the
    target LLM (§3.1 design decision) -- so callers pass embeddings in and
    take the returned feature to the target LLM's `next_token_distribution`.
    """

    def __init__(self, hidden_dim: int, seed: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        # PAPER: §3.1 Fig.6 ("The FC layer reduces the dimensionality of the
        # fused sequence to (bs, seq_len, hidden_dim)")
        self.fc = nn.Linear(2 * hidden_dim, hidden_dim, bias=False)
        # One decoder-layer stand-in for llama_eagle.py's single
        # LlamaDecoderLayer (whose first input_layernorm is Identity, per
        # the dossier note) -- a full transformer block is not the point
        # here, only that fc's output goes through one more nonlinear
        # transform, with a residual add, before being read as the next
        # feature (mirrors "hidden_states = hidden_states + residual").
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)

    # PAPER: §3.1 (fc(cat(token_embed, feature)) -> decoder -> next feature)
    def forward(self, token_embed: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([token_embed, feature], dim=-1)
        hidden = self.fc(fused)
        residual = hidden
        hidden = self.decoder(hidden)
        return hidden + residual


# PAPER: §1 / §3.1 Fig.3-5 (token sequence advanced by one time step)
def build_shifted_token_input(token_ids: torch.Tensor, sampled_next_token: int) -> torch.Tensor:
    """
    PAPER §3.1: "EAGLE inputs the token sequence from one time step ahead,
    which includes the sampling outcomes, into the draft model" (Fig.3:
    f_I's successor is t_am or t_always depending on which was actually
    sampled). Given the feature sequence F_{1:j} (aligned to t_1..t_j) and
    the newly-sampled token t_{j+1}, builds the shifted token input T'_{1:j}
    with T'_i = t_{i+1} (for i<j) and T'_j = t_{j+1} -- i.e. shift left by
    one and splice the freshly-sampled token into the last slot, matching
    vLLM's set_inputs_first_pass (llm_base_proposer.py:L664-L669) for a
    single request (batched multi-request boundary handling there is
    engineering plumbing, not part of the paper's mechanism, and is left
    out here).
    """
    shifted = torch.empty_like(token_ids)
    shifted[:-1] = token_ids[1:]
    shifted[-1] = sampled_next_token
    return shifted


# PAPER: §3.2 (L_reg = Smooth_L1(f_{i+1}, Draft_Model(...)))
def regression_loss(pred_feature: torch.Tensor, target_feature: torch.Tensor) -> torch.Tensor:
    """Feature-prediction regression loss (§3.2). Not used at inference --
    training only; included because the dossier marks this mechanism as
    needing a worked numeric example."""
    return F.smooth_l1_loss(pred_feature, target_feature)


# PAPER: §3.2 (p_{i+2}=Softmax(LM_Head(f)), p_hat=Softmax(LM_Head(f_hat)), L_cls=CE(p,p_hat))
def classification_loss(pred_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """
    Soft cross-entropy between the target distribution p_{i+2} (from the
    true next feature) and the predicted distribution p_hat_{i+2} (from the
    draft's predicted feature): L_cls = CE(p_{i+2}, p_hat_{i+2}) =
    -sum_x p(x) log(p_hat(x)).
    """
    p = F.softmax(target_logits, dim=-1)
    log_p_hat = F.log_softmax(pred_logits, dim=-1)
    return -(p * log_p_hat).sum(dim=-1).mean()


# PAPER: §3.2 (L = L_reg + w_cls * L_cls, w_cls = 0.1)
def combined_loss(
    pred_feature: torch.Tensor,
    target_feature: torch.Tensor,
    pred_logits: torch.Tensor,
    target_logits: torch.Tensor,
    w_cls: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (L, L_reg, L_cls) so a numeric worked example can show both
    terms and the paper's fixed weighting (w_cls=0.1, chosen because
    "the classification loss is an order of magnitude larger than the
    regression loss in numerical terms")."""
    l_reg = regression_loss(pred_feature, target_feature)
    l_cls = classification_loss(pred_logits, target_logits)
    return l_reg + w_cls * l_cls, l_reg, l_cls
