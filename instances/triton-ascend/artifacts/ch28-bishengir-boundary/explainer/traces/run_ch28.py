#!/usr/bin/env python3
"""ch28 explainer driver — exercise the REAL source constants on a synthetic Linalg IR.

The closed-source bishengir-compile binary cannot be run in this environment, so
the subprocess call / binary compilation is NOT reproduced here (that is the
closed boundary the chapter honestly stops at). But three sub-mechanisms are pure
Python and ARE runnable with the verbatim source constants:

  1. regex-metadata-extraction  — the 6 regexes from compiler.py L197-L212
  2. tensor-kind-regex          — TENSOR_KIND_REGEX deep-dive (compiler.py L209/L227)
  3. cmdline-conditional-assembly — the `if metadata[x] is not None` if-ladder shape
  4. kernel-name-mix-mode-encoding — name = kernel_name + '_' + mix_mode / rsplit + truncate

Regex/encoding constants below are copied verbatim from
third_party/ascend/backend/compiler.py so the trace reflects real behavior.
"""
import re
import json

out = {}

# ---------------------------------------------------------------------------
# Verbatim regex constants — compiler.py:_parse_linalg_metadata L197-L212
# ---------------------------------------------------------------------------
DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX = r'hivm.disable_auto_tile_and_bind_subblock'          # L197
MIX_MODE_REGEX      = r'mix_mode\s*=\s*"([^"]+)"'                                                 # L200
PARALLEL_MODE_REGEX = r'parallel_mode\s*=\s*"([^"]+)"'                                            # L203
KERNEL_NAME_REGEX   = r"func\.func\s+@(\w+)"                                                      # L206
TENSOR_KIND_REGEX   = r'%arg(\d+):[^,)]*?\{[^}]*?tt\.tensor_kind\s*=\s*([^:\s}]+)\s*:[^}]*?\}'    # L209
BITCODES_REGEX      = r'bitcode\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\w+))'                            # L212

# ---------------------------------------------------------------------------
# Synthetic Linalg IR (the *text* representation ttadapter emits, str(mod)).
# Attribute shapes match the //Example comments in the source.
#   - 2 input tensors (tensor_kind=0), 1 output tensor (tensor_kind=1)
#   - a scalar %arg3 with NO tt.tensor_kind attr (must NOT match)
#   - one bitcode path
# ---------------------------------------------------------------------------
linalg = (
    'module attributes {mix_mode = "aiv", parallel_mode = "mix_simd_simt", '
    'bitcode = "libdevice.bc"} {\n'
    '  func.func @add_kernel('
    '%arg0: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, '
    '%arg1: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, '
    '%arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, '
    '%arg3: i32) {\n'
    '    linalg.generic ...\n'
    '    return\n'
    '  }\n'
    '}\n'
)

# ---- mechanism 1: regex-metadata-extraction (verbatim L216-L233 logic) ----
metadata = {}
metadata["shared"] = 1                                                                    # L216 hardcoded
metadata["auto_tile_and_bind_subblock"] = not re.search(
    DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg)                                     # L218
metadata["mix_mode"] = re.search(MIX_MODE_REGEX, linalg).group(1)                         # L220
metadata["parallel_mode"] = re.search(PARALLEL_MODE_REGEX, linalg).group(1)               # L221
metadata["kernel_name"] = re.search(KERNEL_NAME_REGEX, linalg).group(1)                   # L222
metadata["name"] = metadata["kernel_name"] + "_" + metadata["mix_mode"]                   # L225
metadata["tensor_kinds"] = [int(kind) for _, kind in re.findall(TENSOR_KIND_REGEX, linalg)]  # L227
metadata["required_ub_bits"] = 0                                                          # L229 hardcoded
bitcodes = re.findall(BITCODES_REGEX, linalg)                                             # L232
metadata["bitcodes"] = [val for group in bitcodes for val in group if val]               # L233

out["regex_metadata_extraction"] = {
    "input_ir_excerpt": linalg,
    "metadata": metadata,
    "per_regex": [
        {"regex": "MIX_MODE_REGEX", "method": "re.search().group(1)",
         "matched_text": 'mix_mode = "aiv"', "result": metadata["mix_mode"], "sink": "metadata['mix_mode']"},
        {"regex": "PARALLEL_MODE_REGEX", "method": "re.search().group(1)",
         "matched_text": 'parallel_mode = "mix_simd_simt"', "result": metadata["parallel_mode"], "sink": "metadata['parallel_mode']"},
        {"regex": "KERNEL_NAME_REGEX", "method": "re.search().group(1)",
         "matched_text": "func.func @add_kernel", "result": metadata["kernel_name"], "sink": "metadata['kernel_name']"},
        {"regex": "TENSOR_KIND_REGEX", "method": "re.findall() -> list",
         "matched_text": "3x {tt.tensor_kind = k : i32}", "result": metadata["tensor_kinds"], "sink": "metadata['tensor_kinds']"},
        {"regex": "BITCODES_REGEX", "method": "re.findall() flatten",
         "matched_text": 'bitcode = "libdevice.bc"', "result": metadata["bitcodes"], "sink": "metadata['bitcodes']"},
        {"regex": "DISABLE_AUTO_TILE_..._REGEX", "method": "not re.search() (None->True)",
         "matched_text": "(absent)", "result": metadata["auto_tile_and_bind_subblock"], "sink": "metadata['auto_tile_and_bind_subblock']"},
    ],
}

# variant B: DISABLE attribute PRESENT -> auto_tile flips to False
linalg_B = linalg.replace("mix_simd_simt",
                          "mix_simd_simt\", hivm.disable_auto_tile_and_bind_subblock")
out["auto_tile_variant"] = {
    "absent": not re.search(DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg),
    "present": not re.search(DISABLE_AUTO_TILE_AND_BIND_SUBBLOCK_REGEX, linalg_B),
}

# ---- mechanism 2: tensor-kind-regex per-argument deep dive ----
tk_rows = []
for m in re.finditer(TENSOR_KIND_REGEX, linalg):
    tk_rows.append({"arg_index": m.group(1), "kind_raw": m.group(2), "kind_int": int(m.group(2))})
# %arg3 (i32 scalar, no tt.tensor_kind) must be absent from matches:
matched_args = {r["arg_index"] for r in tk_rows}
out["tensor_kind_regex"] = {
    "matches": tk_rows,
    "arg3_matched": "3" in matched_args,   # expect False: scalar has no tt.tensor_kind attr
    "final_list": metadata["tensor_kinds"],
}

# ---- mechanism 3: cmdline-conditional-assembly (if-ladder shape, 910_95) ----
# Small metadata subset with the exact keys the source reads. Values chosen to
# hit both branches: some set (append), some None (skip), some bool flags.
md = {
    "target_arch": "Ascend910B",                 # -> get_common_bishengir_compile_options --target=
    "multibuffer": 2,                            # not None -> append
    "disable_tightly_coupled_buffer_reuse": False,  # falsy -> skip
    "auto_tile_and_bind_subblock": True,
    "enable_auto_bind_sub_block": None,          # None -> falls back to auto_tile (True)
    "sync_solver": None,                         # None -> skip
    "enable_vf_fusion": True,                    # truthy flag -> append bare flag
    "unit_flag": 1,                              # not None -> append
    "bitcodes": ["libdevice.bc"],                # loop -> one --link-aicore-bitcode
}
cmd = []
# get_common_bishengir_compile_options (L263-L266)
cmd.append(f"--target={md['target_arch']}")
trace_steps = [{"switch": "get_common(target)", "metadata_val": md["target_arch"],
                "decision": "always", "emitted": f"--target={md['target_arch']}"}]
# multibuffer (L312)
if md["multibuffer"] is not None:
    cmd.append(f"--enable-auto-multi-buffer={md['multibuffer']}")
    trace_steps.append({"switch": "multibuffer", "metadata_val": md["multibuffer"],
                        "decision": "is not None -> append", "emitted": f"--enable-auto-multi-buffer={md['multibuffer']}"})
# disable_tightly_coupled_buffer_reuse (L318)
if md["disable_tightly_coupled_buffer_reuse"]:
    cmd.append("--disable-tightly-coupled-buffer-reuse")
else:
    trace_steps.append({"switch": "disable_tightly_coupled_buffer_reuse", "metadata_val": md["disable_tightly_coupled_buffer_reuse"],
                        "decision": "falsy -> skip", "emitted": "(none)"})
# auto-bind-sub-block (L322) — get_auto_bind_sub_block_option: enable is None -> auto_tile
abs_val = md["auto_tile_and_bind_subblock"] if md["enable_auto_bind_sub_block"] is None else md["enable_auto_bind_sub_block"]
cmd.append(f"--enable-auto-bind-sub-block={abs_val}")
trace_steps.append({"switch": "auto-bind-sub-block", "metadata_val": f"user=None, module={md['auto_tile_and_bind_subblock']}",
                    "decision": "user None -> use module attr", "emitted": f"--enable-auto-bind-sub-block={abs_val}"})
# sync_solver (L341)
if md["sync_solver"] is not None:
    cmd.append(f"--enable-hivm-graph-sync-solver={md['sync_solver']}")
else:
    trace_steps.append({"switch": "sync_solver", "metadata_val": md["sync_solver"],
                        "decision": "is None -> skip", "emitted": "(none)"})
# unit_flag (L346)
if md["unit_flag"] is not None:
    cmd.append(f"--enable-hivm-unit-flag-sync={md['unit_flag']}")
    trace_steps.append({"switch": "unit_flag", "metadata_val": md["unit_flag"],
                        "decision": "is not None -> append", "emitted": f"--enable-hivm-unit-flag-sync={md['unit_flag']}"})
# enable_vf_fusion (L391) — bare flag if truthy
if md["enable_vf_fusion"]:
    cmd.append("--enable-vf-fusion")
    trace_steps.append({"switch": "enable_vf_fusion", "metadata_val": md["enable_vf_fusion"],
                        "decision": "truthy -> bare flag", "emitted": "--enable-vf-fusion"})
# bitcodes loop (L422)
for bc in md["bitcodes"]:
    cmd.append(f"--link-aicore-bitcode={bc}")
    trace_steps.append({"switch": "bitcodes[loop]", "metadata_val": bc,
                        "decision": "for each -> append", "emitted": f"--link-aicore-bitcode={bc}"})
out["cmdline_conditional_assembly"] = {
    "steps": trace_steps,
    "final_cmd_options": cmd,
    "n_emitted": len(cmd),
    "n_switches_considered": len(md),
}

# ---- mechanism 4: kernel-name / mix-mode encoding + runtime rsplit ----
def encode(kernel_name, mix_mode):
    return kernel_name + "_" + mix_mode
def decode(name):  # pack_metadata L913
    return name.rsplit("_", 1)
KERNEL_NAME_MAX_LEN = 49  # L912
enc_rows = []
for kn, mm in [("add_kernel", "aiv"), ("gather_sorted_kernel", "mix"),
               ("a" * 55, "aic")]:
    name = encode(kn, mm)
    orig, mode = decode(name)
    truncated = orig[-KERNEL_NAME_MAX_LEN:] if len(orig) > KERNEL_NAME_MAX_LEN else orig
    enc_rows.append({
        "kernel_name_in": kn if len(kn) <= 30 else f"{kn[:6]}...(len={len(kn)})",
        "mix_mode": mm,
        "encoded_name": name if len(name) <= 40 else f"{name[:10]}...(len={len(name)})",
        "rsplit_orig": orig if len(orig) <= 30 else f"...(len={len(orig)})",
        "rsplit_mode": mode,
        "final_kernel_name_len": len(truncated),
        "truncated": len(orig) > KERNEL_NAME_MAX_LEN,
    })
out["kernel_name_mix_mode_encoding"] = {"rows": enc_rows, "KERNEL_NAME_MAX_LEN": KERNEL_NAME_MAX_LEN}

print(json.dumps(out, indent=2, ensure_ascii=False))
