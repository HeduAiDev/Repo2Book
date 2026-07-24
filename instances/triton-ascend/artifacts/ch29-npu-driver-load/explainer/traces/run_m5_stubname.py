#!/usr/bin/env python3
"""m5 忠实模拟：registerKernel 的 stubName 计数器逻辑（纯控制流，host 可跑）。

对照 C++ 源（third_party/ascend/backend/npu_utils.cpp:L37-38,L68-72）:
    static std::unordered_map<std::string, size_t> registered_names;   # 默认值 0
    static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;
    std::string stubName = name;
    stubName += "_" + std::to_string(registered_names[name]);   # 读当前计数拼后缀
    registered_names[name]++;                                    # 再自增
    auto registered = func_stubs.emplace(stubName, make_unique<size_t>(0));
    void *func_stub_handle = registered.first->second.get();     # 稳定堆地址

本脚本只复刻**计数器 + 后缀 + 唯一性**这部分纯逻辑（语言无关、确定性）。
真实 func stub 句柄是 CANN 运行时里 unique_ptr<size_t> 的堆地址（每次进程不同、
非确定），不属可复现数值，故此处只断言「互异」，不打印具体地址。
输出 JSON 存 run_m5_stubname.json，供 explainer 表格数字核对。
"""
import json
from collections import defaultdict

# 忠实镜像两个全局 map
registered_names = defaultdict(int)          # name -> 已注册次数 (size_t 默认 0)
func_stubs = {}                              # stubName -> 独立堆对象(用 object() 模拟 unique_ptr)

# 注册序列：同名 add_kernel 来自两个不同 binary(A/B)，再来一个不同名 mul_kernel(C)
sequence = [("add_kernel", "binA"), ("add_kernel", "binB"), ("mul_kernel", "binC")]

rows = []
handles = []
for i, (name, binlabel) in enumerate(sequence, start=1):
    read_count = registered_names[name]                 # 读当前计数
    stub_name = f"{name}_{read_count}"                   # 拼后缀
    registered_names[name] += 1                          # 自增
    new_count = registered_names[name]
    obj = object()                                       # 模拟 make_unique<size_t>(0) 新堆对象
    func_stubs[stub_name] = obj
    handle = id(obj)                                     # .get() 的稳定地址（非确定，仅测唯一）
    unique = handle not in handles
    handles.append(handle)
    rows.append({
        "round": i,
        "name": name,
        "binary": binlabel,
        "read_count": read_count,
        "stub_name": stub_name,
        "count_after_incr": new_count,
        "handle_unique": unique,
    })

result = {
    "sequence": [f"{n}({b})" for n, b in sequence],
    "rows": rows,
    "distinct_stub_names": len(set(r["stub_name"] for r in rows)),
    "total_registrations": len(rows),
    "registered_names_final": dict(registered_names),   # 键数 = 去重后的 name 数
    "func_stubs_keys": len(func_stubs),
    "all_handles_distinct": len(set(handles)) == len(handles),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
