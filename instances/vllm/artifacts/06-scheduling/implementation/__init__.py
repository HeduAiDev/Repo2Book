"""Ch06 — request scheduling POLICY layer (not mechanics).

Ch04 already covered the schedule()/step()/preempt mechanics. Ch06 is the
strategy layer that sits on top:

    request_queue.py       <-> vllm/v1/core/sched/request_queue.py
    policy.py              <-> vllm/v1/core/sched/scheduler.py (policy bits only)
    preemption_strategy.py (recompute vs swap vs abort; analytical comparator)
    starvation_analysis.py (head-of-line blocking under FCFS; aging compensator)
    pareto.py              (schedule-latency vs throughput trade-off model)
    demo.py                (runnable numerics for all five sections)

For schedule() loop mechanics, refer back to Ch04
(`instances/vllm/artifacts/04-continuous-batching/implementation/scheduler.py`).
"""
