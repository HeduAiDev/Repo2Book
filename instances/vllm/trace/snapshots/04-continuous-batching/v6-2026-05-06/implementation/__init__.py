"""Continuous-batching reimplementation, faithful to vLLM v1 scheduler.

Module layout mirrors `vllm/v1/`:
    request.py         <-> vllm/v1/request.py
    output.py          <-> vllm/v1/core/sched/output.py
    request_queue.py   <-> vllm/v1/core/sched/request_queue.py
    kv_cache_manager.py<-> vllm/v1/core/kv_cache_manager.py (heavily simplified)
    scheduler.py       <-> vllm/v1/core/sched/scheduler.py
    demo.py            (pedagogical, runnable trace)
"""
