"""IR 算子名不得写成「方言前缀 + C++ 类名」(exp-2026-07-21-04)。

来源:同一病种三章连发——
  ch06 `tt.indirect_load`(错在方言前缀,真名 `ascend.indirect_load`)
  ch08 `ascend.CustomOp`(错在拿 C++ 类名当 IR 名,真名 `ascend.custom`)
  ch07 `hivm.CustomOp` + Book Bible glossary 同款(真名 `hivm.custom`)

判据:**IR 算子名 = 方言 `let name` + ODS 助记符**,两处都要回 .td 查,
**不能从 C++ 类名推**。MLIR 助记符惯例是小写起头(本仓 119 个助记符无一以大写开头),
而 C++ 类名一律 CamelCase 且以 `Op` 结尾 —— 故 `<小写方言>.<大写开头...Op>` 这个形态
几乎必然是把 C++ 类名当成了 IR 名。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_ir_opname import scan_text


# 单测一律显式传方言集:不依赖活动实例的源码树(hermetic)。
# 方言集本身的抽取由 test_known_dialects_reads_td_let_name 单独覆盖。
DIA = {"hivm", "ascend", "annotation", "scope", "func", "tensor"}


def test_flags_dialect_plus_cpp_class_name():
    hits = scan_text("emit 一条 `hivm.CustomOp`——结果转回 tensor", dialects=DIA)
    assert hits and hits[0][0] == "hivm.CustomOp"


def test_flags_ascend_variant():
    assert scan_text("落成通用的 `ascend.CustomOp`(只带 sender 与 id)", dialects=DIA)


def test_real_ir_names_pass():
    """真实 IR 名(小写助记符)不得误报。"""
    for ok in ("hivm.custom", "ascend.custom", "hivm.sync_block_set",
               "annotation.mark", "scope.scope", "func.func", "tensor::ExtractOp"):
        assert not scan_text(f"落成 `{ok}`", dialects=DIA), ok


def test_cpp_class_reference_passes():
    """规矩的 C++ 类名引用(带 :: 作用域)是对的,不该报。"""
    for ok in ("triton::ascend::CustomOp", "hivm::SyncBlockSetOp", "self.create<hivm::SyncBlockSetOp>"):
        assert not scan_text(ok, dialects=DIA), ok


def test_mixed_case_mnemonic_not_flagged():
    """本仓确有含大写的助记符(如 batchMmadL1/broadcastableOTF),但都小写起头,不得误报。"""
    for ok in ("hivm.batchMmadL1", "hivm.broadcastableOTF", "hivm.mmadL1"):
        assert not scan_text(f"算子 `{ok}`", dialects=DIA), ok


def test_reports_suggested_form():
    hits = scan_text("`ascend.CustomOp`", dialects=DIA)
    assert "ascend.custom" in hits[0][1] or "助记符" in hits[0][1]


# ---------- 方言白名单:防误伤 Python 的「模块.类名」 ----------

DIALECTS = {"hivm", "ascend", "annotation", "scope", "func", "tensor"}


def test_python_module_class_not_flagged():
    """真实误报样本(oracle 对表抓到的):`ast.BinOp` 是 Python ast 模块的类,
    `distributed.ReduceOp` 是 torch.distributed 的类——都不是 MLIR 方言,不得报。"""
    for ok in ("ast.BinOp", "distributed.ReduceOp", "typing.NamedTupleOp"):
        assert not scan_text(f"`{ok}`", dialects=DIALECTS), ok


def test_real_dialect_still_flagged_with_whitelist():
    assert scan_text("`hivm.CustomOp`", dialects=DIALECTS)
    assert scan_text("`ascend.CustomOp`", dialects=DIALECTS)


def test_known_dialects_reads_td_let_name(tmp_path):
    """方言白名单应从 .td 的 `let name = "..."` 抽出来。"""
    from lint_ir_opname import known_dialects
    (tmp_path / "d.td").write_text('def Foo_Dialect : Dialect {\n  let name = "mydialect";\n}\n', encoding="utf-8")
    assert "mydialect" in known_dialects(tmp_path)
    assert "func" in known_dialects(tmp_path)   # 内建方言恒在


# ---------- 评审记录豁免 ----------

def test_blind_review_notes_are_exempt(tmp_path):
    """`figure-manifest.json` 的 `blind_review.notes` 是**评审记录**——它必须能原样引用
    「被判错的写法」来说明问题(『页脚写着 ascend.CustomOp,这是错的』)。若不豁免,
    门禁会对这类章永远红,而**永远红不掉的门禁等于没有门禁**(同 warn 噪音的失效模式)。
    豁免只针对 notes;figure_spec/claim 等**断言性**字段照常严查。
    """
    from lint_ir_opname import lint
    ch = tmp_path / "ch99"
    (ch / "diagrams").mkdir(parents=True)
    (ch / "diagrams" / "figure-manifest.json").write_text(json.dumps({"figures": [
        {"id": "f1",
         "claim": "落成 hivm.CustomOp",                       # 断言字段 → 必须报
         "blind_review": {"verdict": "FAIL",
                          "notes": "页脚写着 ascend.CustomOp,这是错的,应为 ascend.custom"}},
    ]}, ensure_ascii=False), encoding="utf-8")
    issues = lint(str(ch), dialects={"hivm", "ascend"})
    joined = " ".join(issues)
    assert "hivm.CustomOp" in joined, "断言字段里的错名必须报"
    assert "ascend.CustomOp" not in joined, "评审记录 notes 不该报"
