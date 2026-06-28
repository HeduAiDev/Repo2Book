# vllm_ascend/eplb/core/policy/policy_factory.py —— subtract-only companion（ch09 主线④ 分发点）
#
# 策略多态的唯一分发点：int policy_type → 策略类实例。0 Random / 1 DefaultEplb /
# 2 Swift / 3 FlashLB，未知回退 RandomLoadBalance。EplbWorker 只拿到 EplbPolicy 抽象。
# 源码顶部 TODO：待 vLLM PR 24069 合入即删除本工厂。
from eplb_runtime_stub import logger

from policy_abstract import EplbPolicy
from policy_default_eplb import DefaultEplb
from policy_other import FlashLB, RandomLoadBalance, SwiftBalanceEplb

# SUBTRACTED: from vllm.logger import logger —— host 经 eplb_runtime_stub 接住。原 policy_factory.py:L3
# SUBTRACTED: from .policy_flashlb import FlashLB, warm_up —— 仅保留 FlashLB 类名（点到为止），
#   warm_up 是 policy_type==3 专属预热（delete 批准）。原 policy_factory.py:L7


# SOURCE: vllm_ascend/eplb/core/policy/policy_factory.py:L12-L41
class PolicyFactory:
    @staticmethod
    def generate_policy(policy_type: int) -> EplbPolicy:
        # SOURCE: vllm_ascend/eplb/core/policy/policy_factory.py:L13-L41
        policy: dict[int, type[EplbPolicy]] = {
            # Constraint applying Dynamic EPLB policy V2:
            # If there exists redundant expert:
            # only one redundant expert can be placed in one NPU and its physical expert index must be 0
            # Applying greedy d2d expert weight update composing
            0: RandomLoadBalance,  # RandomLoadBalance: shuffle last physical expert on NPU 1 and 3
            1: DefaultEplb,  # Dynamic EPLB policy: overall expert replacement based on current moe load
            # Dynamic EPLB policy V2: expert replacement with constrained number of expert shuffle
            2: SwiftBalanceEplb,
            # FlashLB EPLB policy: expert replacement based on Joint Optimization,
            # Multi-Shot Enhancement and Incremental Adjustment
            3: FlashLB,
        }
        policy_class = policy.get(policy_type)
        if policy_class is None:
            policy_class = RandomLoadBalance
            logger.warning(
                "[eplb/policy] Unrecognized policy_type=%s, falling back to %s",
                policy_type,
                policy_class.__name__,
            )
        else:
            logger.info("[eplb/policy] Policy: %s (type=%s)", policy_class.__name__, policy_type)
        policy_instance = policy_class()
        # SUBTRACTED: if policy_type == 3: warm_up() —— FlashLB 专属预热（点到为止，delete 批准）。
        #   原 policy_factory.py:L39-L40
        return policy_instance
