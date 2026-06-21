#!/usr/bin/env python3
"""Book Bible — cross-chapter continuity store (terms, interfaces, foreshadow/payoff).

The book's cross-chapter coherence does NOT rely on a long-lived agent's (degrading)
memory; it lives in these explicit, durable files, kept by the Archivist.

Commands:
  bible.py due <chapter_id>                       # foreshadows to PLANT + payoffs DUE here
  bible.py foreshadow --add --plant chN --payoff chM --what "..."
  bible.py payoff --resolve <id> --in chN
  bible.py term --add <zh> <en>
  bible.py iface --add <chapter> <signature>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "instances/vllm/book/bible"
ARC = ROOT / "arc-map.json"
GLOSS = ROOT / "glossary.json"
IFACE = ROOT / "interfaces.json"


def _load(p, default):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _save(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def due(chapter_id: str, arc_path=ARC) -> dict:
    arc = _load(arc_path, [])
    return {
        "plant": [a for a in arc if a.get("plant") == chapter_id],
        "payoff": [a for a in arc if a.get("payoff") == chapter_id and a.get("status") != "resolved"],
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("due")
    d.add_argument("chapter_id")
    f = sub.add_parser("foreshadow")
    f.add_argument("--add", action="store_true")
    f.add_argument("--plant", required=True)
    f.add_argument("--payoff", required=True)
    f.add_argument("--what", required=True)
    pf = sub.add_parser("payoff")
    pf.add_argument("--resolve", required=True)
    pf.add_argument("--in", dest="in_ch", required=True)
    t = sub.add_parser("term")
    t.add_argument("--add", nargs=2, metavar=("ZH", "EN"))
    i = sub.add_parser("iface")
    i.add_argument("--add", nargs=2, metavar=("CHAPTER", "SIG"))
    a = ap.parse_args()

    if a.cmd == "due":
        res = due(a.chapter_id)
        print(f"== {a.chapter_id} 应埋伏笔 ==")
        for x in res["plant"]:
            print(f"  [{x['id']}] {x['what']} → 回收于 {x['payoff']}")
        print(f"== {a.chapter_id} 应回收 ==")
        for x in res["payoff"]:
            print(f"  [{x['id']}] {x['what']} （埋于 {x['plant']}）")
    elif a.cmd == "foreshadow" and a.add:
        arc = _load(ARC, [])
        nid = f"f{len(arc) + 1}"
        arc.append({"id": nid, "what": a.what, "plant": a.plant, "payoff": a.payoff, "status": "open"})
        _save(ARC, arc)
        print(f"added {nid}")
    elif a.cmd == "payoff":
        arc = _load(ARC, [])
        for x in arc:
            if x.get("id") == a.resolve:
                x["status"] = "resolved"
                x["resolved_in"] = a.in_ch
        _save(ARC, arc)
        print(f"resolved {a.resolve}")
    elif a.cmd == "term" and a.add:
        g = _load(GLOSS, {})
        g[a.add[0]] = a.add[1]
        _save(GLOSS, g)
        print("ok")
    elif a.cmd == "iface" and a.add:
        it = _load(IFACE, {})
        it.setdefault(a.add[0], []).append(a.add[1])
        _save(IFACE, it)
        print("ok")


if __name__ == "__main__":
    main()
