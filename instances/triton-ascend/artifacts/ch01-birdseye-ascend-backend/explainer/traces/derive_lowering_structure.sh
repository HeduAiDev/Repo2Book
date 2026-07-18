#!/usr/bin/env bash
# ch01 meta 章的「结构性 trace」驱动脚本。
#
# 本章 trace_source=manual：host 无 NPU/CANN/bishengir-compile，无法真机运行 add()
# 或触发端到端 npubin 编译，因此不存在可跑的运行时数值轨迹。本脚本改为从 pin 源码
# 直接抠出「下降链有几段、每段挂哪个函数、在哪一行、对照基座 GPU 路几段」的结构性事实，
# 让 explainer.json 逐轮表里的每个行号/段数都能被逐字复核。
#
# 用法：在 instances/triton-ascend/source 下运行，输出即 derive_lowering_structure.txt。
set -euo pipefail
SRC="third_party/ascend/backend/compiler.py"
BASE="../../triton/source/third_party/nvidia/backend/compiler.py"  # 姊妹篇基座 v3.2.0

echo "=== [基座 GPU 路] CUDABackend.add_stages 五段 ==="
grep -n 'stages\["' "$BASE" 2>/dev/null || echo "(基座源码不在此工作树，行号见姊妹篇 dossier)"

echo "=== [本书 NPU 路] AscendBackend.add_stages 段键 ==="
grep -n 'stages\["' "$SRC"

echo "=== [本书 NPU 路] 各段实现函数定义行 ==="
grep -n 'def make_ttir\|def ttir_to_linalg\|def linalg_to_bin_enable_npu_compile_A2_A3\|def linalg_to_bin_enable_npu_compile_910_95\|def ttir_to_npubin' "$SRC"

echo "=== [下降链走向开关] NPUOptions 默认值 ==="
grep -n 'force_simt_only: bool\|compile_on_910_95: bool' "$SRC"

echo "=== [ttadapter 内部 pass 链] ttir_to_linalg 的 add_triton_to_* 编排 ==="
grep -n 'ascend.passes.ttir.add_' "$SRC" | sed -n '1,20p'

echo "=== [支柱③锚点] CoreType / AddressSpace 枚举 ==="
grep -n '"CUBE"\|"VECTOR"\|"L1"\|"UB"\|"L0A"\|"L0B"\|"L0C"' third_party/ascend/ascend_ir.cc

echo "=== [支柱①活证据] vector-add grid 示例注释（256 / 64 → 4 program） ==="
grep -n 'length 256\|block_size of 64\|0:64' third_party/ascend/tutorials/01-vector-add.py
