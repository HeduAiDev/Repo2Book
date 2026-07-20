"""开篇块(H1 + hook + roadmap + 本章地图)不该被要求内嵌源码引用(exp-2026-07-20-05)。

来源:全书 39 章里 9 章报 `Sections without vLLM source references: ['# 第 NN 章 …']`。
根因是 meta 豁免正则写作 `^#\s*第\d+章` —— 要求「第」后紧跟数字、数字后紧跟「章」,
而全书 H1 实际一律是 `# 第 39 章　标题`(**带空格**),故从来没匹配上。
凡开篇 hook 里恰好没提到某个源码文件的章,就吃一个假 BLOCKING;
这与 exp-2026-07-19-01(连字符路径)同类:正则与语料的真实写法对不上。

修法一并泛化:开篇块由「以单个 # 开头的标题」唯一标识(后续小节都是 ## 起),
与是否叫「第 N 章」无关——triton-ascend 用自然标题(`# 显式内存层级——…`)同样适用。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_source_grounding import lint_source_grounding

BODY = ("正文引用 vllm/v1/core/sched/scheduler.py:L100-L120 的实现。\n"
        "```python\n# vllm/v1/core/sched/scheduler.py:L100-L120\npass\n```\n")


def _mk(tmp, narrative):
    ch = tmp / "instances" / "vllm" / "artifacts" / "ch39-x"
    (ch / "narrative").mkdir(parents=True)
    (ch / "narrative" / "chapter.md").write_text(narrative, encoding="utf-8")
    (ch / "implementation").mkdir()
    (ch / "implementation" / "m.py").write_text(
        "# SOURCE: vllm/a.py:L1\n# SOURCE: vllm/b.py:L2\n# SOURCE: vllm/c.py:L3\n", encoding="utf-8")
    return ch


def test_spaced_chapter_heading_intro_is_exempt(tmp_path):
    """`# 第 39 章　标题`(带空格)的开篇块不提源码文件,不该报。"""
    ch = _mk(tmp_path, "# 第 39 章　高级引擎运维\n\n开篇钩子,不提任何源码文件。\n\n## 39.1 起\n\n" + BODY)
    res = lint_source_grounding(str(ch))
    assert res["narrative_vllm_refs"] == [], res["narrative_vllm_refs"]


def test_natural_title_intro_is_exempt(tmp_path):
    """自然标题章(triton-ascend 体例)的开篇块同样豁免。"""
    ch = _mk(tmp_path, "# 显式内存层级——门牌号与两条搬运边\n\n开篇钩子。\n\n## 起步\n\n" + BODY)
    res = lint_source_grounding(str(ch))
    assert res["narrative_vllm_refs"] == []


def test_real_section_without_refs_still_flagged(tmp_path):
    """真正的内容小节缺源码引用,仍必须报——豁免只针对开篇块,不得放宽到正文。"""
    ch = _mk(tmp_path, "# 第 39 章　标题\n\n钩子。\n\n## 39.1 有引用\n\n" + BODY + "\n## 39.2 空谈\n\n只有抽象叙述。\n")
    res = lint_source_grounding(str(ch))
    assert res["narrative_vllm_refs"], "内容小节缺引用却没报"
    assert "39.2" in str(res["narrative_vllm_refs"])
