#!/usr/bin/env bash
# m10：用 gcc -E 实测 FOR_EACH_P 宏链——4 元正常展开、5 元把第5个后端名撞成未定义宏。
# 宏本体逐字取自 pin: python/src/main.cc:L7-L32（DECLARE_BACKEND 见 L34）。
set -e
SRC=/mnt/e/Laboratory/Repo2Book/instances/triton/source/python/src/main.cc
echo "# 宏本体来源: python/src/main.cc:L7-L34 (pin v3.2.0)"
sed -n '7,34p' "$SRC"
echo
echo "======================================================================"

macro_prelude() {
cat <<'EOF'
#define FOR_EACH_1(MACRO, X) MACRO(X)
#define FOR_EACH_2(MACRO, X, ...) MACRO(X) FOR_EACH_1(MACRO, __VA_ARGS__)
#define FOR_EACH_3(MACRO, X, ...) MACRO(X) FOR_EACH_2(MACRO, __VA_ARGS__)
#define FOR_EACH_4(MACRO, X, ...) MACRO(X) FOR_EACH_3(MACRO, __VA_ARGS__)
#define FOR_EACH_NARG(...) FOR_EACH_NARG_(__VA_ARGS__, FOR_EACH_RSEQ_N())
#define FOR_EACH_NARG_(...) FOR_EACH_ARG_N(__VA_ARGS__)
#define FOR_EACH_ARG_N(_1, _2, _3, _4, N, ...) N
#define FOR_EACH_RSEQ_N() 4, 3, 2, 1, 0
#define CONCATENATE(x, y) CONCATENATE1(x, y)
#define CONCATENATE1(x, y) x##y
#define FOR_EACH(MACRO, ...) \
  CONCATENATE(FOR_EACH_, FOR_EACH_NARG_HELPER(__VA_ARGS__))(MACRO, __VA_ARGS__)
#define FOR_EACH_NARG_HELPER(...) FOR_EACH_NARG(__VA_ARGS__)
#define REMOVE_PARENS(...) __VA_ARGS__
#define FOR_EACH_P_INTERMEDIATE(MACRO, ...) FOR_EACH(MACRO, __VA_ARGS__)
#define FOR_EACH_P(MACRO, ARGS_WITH_PARENS) \
  FOR_EACH_P_INTERMEDIATE(MACRO, REMOVE_PARENS ARGS_WITH_PARENS)
#define DECLARE_BACKEND(name) void init_triton_##name();
EOF
}

echo "==== 4 个后端 (nvidia,amd,ascend,proton)：正常展开 ===="
{ macro_prelude; echo 'FOR_EACH_P(DECLARE_BACKEND, (nvidia,amd,ascend,proton))'; } \
  | gcc -E -P -x c - | grep -v '^$'
echo
echo "==== 5 个后端 (nvidia,amd,ascend,proton,fifth)：撞未定义宏 ===="
{ macro_prelude; echo 'FOR_EACH_P(DECLARE_BACKEND, (nvidia,amd,ascend,proton,fifth))'; } \
  | gcc -E -P -x c - | grep -v '^$'
