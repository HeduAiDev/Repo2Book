# v0.21.0 更新摘要 — G 组：调度器 + KV-connector 工厂

基线 `f3fef1235` → `v0.21.0`。文件组：`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/sched/async_scheduler.py`、`vllm/distributed/kv_transfer/kv_connector/factory.py`、`vllm/distributed/kv_transfer/kv_connector/v1/base.py`。目标章节：ch13、ch14、ch29。

涉及提交：
- `8189a1591` [Core] Replace routing replay with device cache and async D2H pipeline (#39917)
- `ea0e501bb` [KV Connector] Remove compat support for pre-v0.12.0 constructor signatures without `KVCacheConfig` (#39832)
- `cbaa80fed` [KV Transfer] Add MooncakeStoreConnector for KV cache offloading via Mooncake distributed store (#40900)

---

## 1. KV-connector 强制三参构造，废弃 `compat_sig` 兼容分支

- **class**: API-CHANGE
- **anchor**: `vllm/distributed/kv_transfer/kv_connector/factory.py` — `KVConnectorFactory.create_connector` / `KVConnectorFactory.get_connector_class`；`vllm/distributed/kv_transfer/kv_connector/v1/base.py` — `KVConnectorBase_V1.__init__`
- **target**: ch29
- **变化要点**：基线里 `create_connector` 的 `kv_cache_config` 是可选的（`"KVCacheConfig | None" = None`），并通过 `_get_connector_class_with_compat` 探测旧式两参构造器、走 `if compat_sig:` 分支按 `connector_cls(config, role)` 实例化，对旧 connector 仅打 warning 放行。v0.21.0 彻底删除这条兼容路径：`kv_cache_config` 变为**必填**，`create_connector` 一律 `return connector_cls(config, role, kv_cache_config)`；探测函数更名并收紧为公开的 `get_connector_class`，对仍用两参签名的外部 connector 直接 `raise ValueError`。`KVConnectorBase_V1.__init__` 同步把 `kv_cache_config` 改为必填、删掉"未传则告警弃用"那段。
- **集成（书声线）**：29.3 节正文逐字引用了 `create_connector` 里的 `compat_sig` 分支与 `kv_cache_config: "KVCacheConfig | None" = None` 这条**可选签名**——这正是 v0.21.0 删除的兼容垫片。可在该处补一句版本注：自 v0.21.0 起 vLLM 彻底废除 pre-v0.12.0 的两参构造器，`kv_cache_config` 成为 connector 构造的第三个必填参数，工厂不再做签名探测兜底，外部 v1 connector 若不接 `kv_cache_config` 会在 `get_connector_class` 处直接报错。这其实强化了本章主旨："工厂按 role 各造一份"现在三个参数（`config, role, kv_cache_config`）齐备，进程隔离的契约更硬。
- **diagram impact**：无。29.3 的 role-split / 工厂图不涉及构造器签名细节，标签无需改。仅正文代码片段与一句版本注。

---

## 2. 新增 MooncakeStoreConnector 注册项

- **class**: NEW-FEATURE（边际，注册表条目）
- **anchor**: `vllm/distributed/kv_transfer/kv_connector/factory.py` — `KVConnectorFactory.register_connector("MooncakeStoreConnector", ...)`
- **target**: ch29
- **变化要点**：在工厂尾部的注册块新增一行，把面向 Mooncake 分布式存储做 KV offload 的 `MooncakeStoreConnector` 登记进懒加载注册表（模块 `...v1.mooncake.store.connector`）。纯注册表新增，不改 connector 接口与调度逻辑。
- **集成（书声线）**：29.3 节讲"懒加载注册表只存 `(name → 模块路径 + 类名)`、不 import"时，可在列举内置后端时顺带提一句 v0.21.0 又多了一个 `MooncakeStoreConnector`（KV cache 卸载到 Mooncake 分布式存储），作为"十几种后端各拖重依赖、故懒加载"论点的又一佐证。无需展开其内部实现。
- **diagram impact**：无。

---

## 3. scheduler.py 的 -70 行：routed_experts 同步采集逻辑移走（非删除）

- **class**: SKIP（移动 + 书中无覆盖）
- **anchor**: `vllm/v1/core/sched/scheduler.py` — 删除 `Scheduler._get_routed_experts`、`__init__` 中 `enable_return_routed_experts` 的 `RoutedExpertsReader` 初始化块
- **target**: 无（ch13/ch14 均不涉及）
- **裁定**：scheduler.py 的 +7/-70 全部来自单个提交 `8189a1591`，主题是 MoE "routing replay"。逻辑并非删除而是**搬迁**：原先调度器在 stop 时同步调 `_get_routed_experts`（自己算 slot mapping、从 `RoutedExpertsReader` 设备缓存里读），v0.21.0 改为由 worker 侧 D2H 流水线产出，调度器只在 `update_from_output` 里从 `model_runner_output.routed_experts_dict[req_id]` **读取**结果（+7 行即这段）。属性质上的"逻辑搬到别处"，按规则 SKIP。
- **额外理由**：`enable_return_routed_experts` 是 MoE 路由回放的小众可观测性特性，**全书 ch13/ch14/ch29 均未提及**（grep `routed_experts` 无命中）。即便算"行为变化"也无落点章节。`async_scheduler.py` 在本区间**零改动**。

---

## 结论

三项可教变化中，**仅 ch29 需要更新**（API-CHANGE + 一条注册表新增）；ch13/ch14 无需改动——scheduler.py 的 -70 是 routed_experts 逻辑搬迁、且书中无覆盖。无任何图需要重绘或改标签。
