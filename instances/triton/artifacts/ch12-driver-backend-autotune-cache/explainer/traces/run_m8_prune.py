#!/usr/bin/env python3
"""m8 — prune_configs 两级裁剪的忠实复刻（host 纯控制流，无需 CUDA）。

复刻 python/triton/runtime/autotuner.py:L211-L229：
    pruned_configs = self.configs
    if self.early_config_prune:
        pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
    if self.perf_model:
        top_k = self.configs_top_k
        if isinstance(top_k, float) and top_k <= 1.0:
            top_k = int(len(self.configs) * top_k)
        if len(pruned_configs) > top_k:
            est_timing = {c: perf_model(c) for c in pruned_configs}
            pruned_configs = sorted(..., key=est)[:top_k]

演示：8 个候选 config，early_config_prune 硬筛掉非法项，perf_model + top_k 估时精选。
配置对象用整数编号 stand-in，perf_model 用其编号当估时（越小越快）。
"""

CONFIGS = list(range(8))   # 8 个候选：编号 0..7，编号即估时（0 最快）


def prune(configs, early_config_prune, perf_model, configs_top_k):
    pruned_configs = configs
    if early_config_prune:
        pruned_configs = early_config_prune(configs)
    if perf_model:
        top_k = configs_top_k
        if isinstance(top_k, float) and top_k <= 1.0:
            top_k = int(len(configs) * top_k)   # 注意：基数是 len(self.configs) 全集，不是 pruned
        if len(pruned_configs) > top_k:
            est_timing = {c: perf_model(c) for c in pruned_configs}
            pruned_configs = sorted(est_timing.keys(), key=lambda x: est_timing[x])[:top_k]
    return pruned_configs, (top_k if perf_model else None)


scenarios = [
    ("无 prune_configs_by（默认）",
     None, None, 1.0),
    ("early_config_prune 删 2 项 + perf_model, top_k=0.5",
     lambda cs: [c for c in cs if c not in (6, 7)],   # 删掉编号 6、7（模拟非法/超资源）
     lambda c: c, 0.5),
    ("仅 perf_model, top_k=2（整数）",
     None, lambda c: c, 2),
]

print("m8 prune_configs 两级裁剪\n" + "=" * 60)
print(f"候选全集 configs = {CONFIGS}  (共 {len(CONFIGS)} 个)\n")
for label, ecp, pm, tk in scenarios:
    pruned, top_k = prune(CONFIGS, ecp, pm, tk)
    print(f"[{label}]")
    if pm:
        print(f"    top_k 解析 = {top_k}")
    print(f"    实际下场 _bench 的 config = {pruned}  (共 {len(pruned)} 个)")
    print(f"    相对全集 {len(CONFIGS)} 个，编译+计时成本压到 {len(pruned)}/{len(CONFIGS)}")
    print()
