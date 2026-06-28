"""验证注入点①：一个 classmethod 返回一个字符串 qualname 就换了底座
（must_keep: get_device_communicator_cls）。"""
import platform_injection as p


def test_base_default_returns_base_communicator_qualname():
    assert p._BasePlatform.get_device_communicator_cls() == (
        "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"
    )


def test_npu_override_returns_npu_communicator_qualname():
    assert p.NPUPlatform.get_device_communicator_cls() == (
        "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"
    )


def test_override_differs_from_base():
    # OOT 换底座的全部「动作」：覆写返回的字符串，基座据此 resolve 出 NPU 通信器类。
    assert (
        p.NPUPlatform.get_device_communicator_cls()
        != p._BasePlatform.get_device_communicator_cls()
    )
    # 覆写后的 qualname 指向昇腾 OOT 包路径，基座零硬依赖。
    assert p.NPUPlatform.get_device_communicator_cls().startswith("vllm_ascend.")
