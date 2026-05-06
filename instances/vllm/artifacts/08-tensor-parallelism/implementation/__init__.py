"""Ch08 — Tensor Parallelism.

Modules mirror vLLM's TP surface at commit 98661fe:

    tp_math.py         <-> instances/vllm/source/vllm/distributed/utils.py
                              (L60-L92 divide, split_tensor_along_last_dim)
                            and the docstring math from
                              vllm/model_executor/layers/linear.py:L410-L432
                              vllm/model_executor/layers/linear.py:L1394-L1425
                            — pure-math derivations of column/row parallel.

    comm_primitives.py <-> instances/vllm/source/vllm/distributed/parallel_state.py
                              (L502-L530 GroupCoordinator.all_reduce)
                            and instances/vllm/source/vllm/distributed/communication_op.py
                              (L12-L14 tensor_model_parallel_all_reduce)
                            — α-β cost model + step-by-step ring all-reduce.

    column_parallel.py <-> instances/vllm/source/vllm/model_executor/layers/linear.py
                              (L410-L608 ColumnParallelLinear,
                               L609-L976 MergedColumnParallelLinear)

    row_parallel.py    <-> instances/vllm/source/vllm/model_executor/layers/linear.py
                              (L1394-L1577 RowParallelLinear)

    qkv_parallel.py    <-> instances/vllm/source/vllm/model_executor/layers/linear.py
                              (L977-L1393 QKVParallelLinear, with GQA branch
                               at L1031-L1036 num_kv_head_replicas)

    mlp_block.py       <-> instances/vllm/source/vllm/model_executor/models/llama.py
                              (L81-L121 LlamaMLP — Megatron col→row pair)

    demo.py            — five demos producing the verbatim numerics that the
                         writer quotes character-for-character per the
                         demo-numerics-verbatim hard gate (K17).

All "ranks" are simulated in a single process. There is NO real
torch.distributed call here; every collective is reproduced as a numpy op.
"""
