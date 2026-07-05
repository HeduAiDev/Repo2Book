# vllm_ascend/eplb/core/policy/policy_abstract.py —— subtract-only companion（ch09 主线④）
#
# 策略层的唯一接口契约：rebalance_experts(current_expert_table, expert_workload)
# → (change, priority, new_table)。EplbWorker 只依赖这个抽象基类，不感知具体算法。
# 源码顶部 TODO：待 vLLM PR 24069 合入即删除本套实现。
from abc import abstractmethod


# SOURCE: vllm_ascend/eplb/core/policy/policy_abstract.py:L6-L30
class EplbPolicy:
    @abstractmethod
    def rebalance_experts(self, current_expert_table, expert_workload):
        # SOURCE: vllm_ascend/eplb/core/policy/policy_abstract.py:L7-L30
        """
        Pass in the weights and return expert replication and placement under relevant constraints.
        INPUT:
        current_expert_table: [layerId, rankId, expert_num_i]
        expert_workload = expert_table[layer0][rankId][expert_num_i]

        RETURNED: (res, expert_table)
        res:
        1 -- table_changed
        0 -- not_changed

        expert_table: [layerId, rankId, expert_num_i]
        expert_num_i --- [0, MaxExpertPerRank]
        expertID = expert_table[layer0][rankId][expert_num_i]
        array_values:
        [0, 1, 2, 3, 248]
        [4, 5, 6, 7, 254]
        [8, 9, 10, 11, 71]
        ...
        [252, 253, 254, 255, 0]
        """
        pass
