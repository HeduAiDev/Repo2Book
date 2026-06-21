import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lint_chapter_structure import lint_structure


def _w(tmp, text):
    p = tmp / "chapter.md"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_missing_roadmap_blocking(tmp_path):
    text = ("# 第四章\n正文\n```python\n# vllm/v1/engine/async_llm.py:L280\nx=1\n```\n"
            "```python\n# vllm/v1/engine/async_llm.py:L637\ny=2\n```\n")
    assert lint_structure(_w(tmp_path, text))["no_roadmap"]


def test_missing_embedded_source_blocking(tmp_path):
    text = "## Roadmap 你在这里\n正文没有源码块\n"
    assert lint_structure(_w(tmp_path, text))["no_embedded_source"]


def test_good_chapter_passes(tmp_path):
    text = ("## Roadmap：你在这里\n地图\n正文\n"
            "```python\n# vllm/v1/engine/async_llm.py:L280\nasync def add_request(): ...\n```\n"
            "解读\n```python\n# vllm/v1/engine/async_llm.py:L637\nasync def _run_output_handler(): ...\n```\n")
    res = lint_structure(_w(tmp_path, text))
    assert not res["no_roadmap"] and not res["no_embedded_source"] and not res["scaffold_leak"]


def test_scaffold_leak_blocking(tmp_path):
    text = ("## Roadmap 你在这里\n"
            "```python\n# instances/vllm/source/vllm/v1/engine/async_llm.py:L280\nx=1\n```\n"
            "## Cell 3 源码走读\n详见 impl-notes.md\n"
            "```python\n# vllm/v1/engine/async_llm.py:L637\ny=2\n```\n")
    res = lint_structure(_w(tmp_path, text))
    assert res["scaffold_leak"]
