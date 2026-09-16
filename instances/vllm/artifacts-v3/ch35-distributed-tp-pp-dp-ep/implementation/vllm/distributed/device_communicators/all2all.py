# SOURCE: vllm/distributed/device_communicators/all2all.py
# ch34 切面（m22）：AgRsAll2AllManager——默认 EP 后端（dispatch=all_gatherv /
# combine=reduce_scatterv）。删除（dossier 删除项 3）：DeepEPAll2AllManagerBase /
# DeepEPHT / DeepEPLL / DeepEPV2 / NixlEP / MoRI / FlashInfer* 全家族（L156-L1092）
# 与 AgRs 内 extra_tensors 的 gathered_extras 拼接细节由消费方（naive_dp_ep 切面）
# 保留主干。全部保留行对 pin 现核。

from __future__ import annotations

from typing import Any

import torch

from vllm.distributed import get_dp_group, get_ep_group, get_pcp_group
from vllm.forward_context import get_forward_context

# SUBTRACTED: threading/dataclass/envs/dist/StatelessProcessGroup/flashinfer/
#   has_deep_ep 等十余个 import（L3-L39）——被裁家族的依赖面（删除项 3）。

from .base_device_communicator import All2AllManagerBase

# SUBTRACTED: logger = init_logger(__name__)（L41）——被裁家族的日志面。


# SOURCE: vllm/distributed/device_communicators/all2all.py:L44-L153 AgRsAll2AllManager
class AgRsAll2AllManager(All2AllManagerBase):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

    # SOURCE: vllm/distributed/device_communicators/all2all.py:L50-L51 __init__
    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    # SOURCE: vllm/distributed/device_communicators/all2all.py:L53-L58 _get_comm_group
    def _get_comm_group(self, is_sequence_parallel: bool) -> Any:
        if is_sequence_parallel:
            return get_ep_group()
        if self.dp_world_size > 1:
            return get_dp_group()
        return get_pcp_group()

    # SOURCE: vllm/distributed/device_communicators/all2all.py:L60-L68 _get_sizes
    def _get_sizes(self, num_local_tokens: int, comm_group: Any) -> list[int]:
        if self.dp_world_size == 1:
            return [num_local_tokens] * comm_group.world_size

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        return sizes

    # SOURCE: vllm/distributed/device_communicators/all2all.py:L70-L99
    #   dispatch_router_logits —— 逐字
    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        # SOURCE: vllm/distributed/device_communicators/all2all.py:L70-L99（锚点双置）
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

    # SOURCE: vllm/distributed/device_communicators/all2all.py:L101-L136 dispatch
    #   —— 逐字（extra_tensors 的 gathered_extras[3:] 透传保留）
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        # SOURCE: vllm/distributed/device_communicators/all2all.py:L101-L136（锚点双置）
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

    # SOURCE: vllm/distributed/device_communicators/all2all.py:L138-L150 combine
    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all dp ranks.
        """
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(
            hidden_states.shape[0] // dist_group.world_size,
            dist_group,
        )
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states

    # SOURCE: vllm/distributed/device_communicators/all2all.py:L152-L153 destroy
    def destroy(self):
        pass


# SUBTRACTED: DeepEPAll2AllManagerBase / DeepEPHTAll2AllManager /
#   DeepEPLLAll2AllManager / DeepEPV2All2AllManager / MoriAll2AllManager /
#   NixlEPAll2AllManager / FlashInferNVLinkTwoSidedManager /
#   FlashInferNVLinkOneSidedManager（L156-L1092）——删除项 3：真 A2A 后端家族，
#   替换同一 dispatch/combine 接口，正文一段带过。
