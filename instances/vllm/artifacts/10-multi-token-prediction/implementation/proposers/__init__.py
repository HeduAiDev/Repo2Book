"""Pedagogical mirrors of vLLM's spec-decode proposers.

Each module corresponds to one file in `vllm/v1/spec_decode/`:

    base.py            <- llm_base_proposer.py    (1820 lines; the scaffolding)
    eagle.py           <- eagle.py                (22 lines; pure inheritance)
    medusa.py          <- medusa.py               (78 lines; K independent MLP heads, NOT inheriting)
    ngram.py           <- ngram_proposer.py       (285 lines; NO draft probs)
    draft_model.py     <- draft_model.py          (88 lines; same architecture not required)
    extract_hidden.py  <- extract_hidden_states.py (382 lines; assert num_speculative_tokens == 1)

These illustrate the (cost, accuracy, coupling) trade-off space:

  | Proposer           | Draft cost | Acceptance | Param overhead       | Needs target hidden |
  |--------------------|------------|------------|----------------------|---------------------|
  | NgramProposer      | ≈0         | low (0.3)  | 0                    | NO                  |
  | MedusaProposer     | low        | low-med    | K * MLP block        | YES                 |
  | DeepSeek MTP       | medium     | high (.85) | K * transformer block| YES                 |
  | DraftModelProposer | high       | med (.5-.7)| whole small model    | NO                  |
  | EAGLE/EAGLE3       | low-med    | highest    | small fc + transformer| YES                 |
"""
