# 「点到为止」的其它策略占位 —— 本章只把 DefaultEplb 走完，Random/Swift/FlashLB 仅
# 为讲清『策略多态分发』而保留类名与接口，本体不展开（subtract-only：删本体，留契约）。
from policy_abstract import EplbPolicy


# SOURCE: vllm_ascend/eplb/core/policy/policy_random.py:RandomLoadBalance
# SUBTRACTED: 本体（shuffle 末位物理 expert）—— PolicyFactory 未知 policy_type 的回退策略，
#   非本章主线，仅保留类名供工厂 dispatch。原 policy_random.py:RandomLoadBalance
class RandomLoadBalance(EplbPolicy):
    def rebalance_experts(self, current_expert_table, expert_workload):
        # SOURCE: vllm_ascend/eplb/core/policy/policy_random.py:RandomLoadBalance.rebalance_experts
        raise NotImplementedError("点到为止：见 vllm_ascend/eplb/core/policy/policy_random.py")


# SOURCE: vllm_ascend/eplb/core/policy/policy_swift_balancer.py:SwiftBalanceEplb
# SUBTRACTED: 本体（V2：受约束的有限专家洗牌）。原 policy_swift_balancer.py:SwiftBalanceEplb
class SwiftBalanceEplb(EplbPolicy):
    def rebalance_experts(self, current_expert_table, expert_workload):
        # SOURCE: vllm_ascend/eplb/core/policy/policy_swift_balancer.py:SwiftBalanceEplb.rebalance_experts
        raise NotImplementedError("点到为止：见 vllm_ascend/eplb/core/policy/policy_swift_balancer.py")


# SOURCE: vllm_ascend/eplb/core/policy/policy_flashlb.py:FlashLB
# SUBTRACTED: 本体（Joint Optimization / Multi-Shot / 增量调整）+ 模块级 warm_up()。
#   policy_type==3 才走，依赖编译/预热，非主线。原 policy_flashlb.py:FlashLB / warm_up
class FlashLB(EplbPolicy):
    def rebalance_experts(self, current_expert_table, expert_workload):
        # SOURCE: vllm_ascend/eplb/core/policy/policy_flashlb.py:FlashLB.rebalance_experts
        raise NotImplementedError("点到为止：见 vllm_ascend/eplb/core/policy/policy_flashlb.py")
