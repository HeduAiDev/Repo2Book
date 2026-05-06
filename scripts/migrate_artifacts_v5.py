"""One-shot migration: align artifacts/ dir IDs with current outline + snapshot legacy content.

Idempotent. Bottom-up rename order avoids collisions. Run from repo root.
"""
import json
import shutil
import sys
from pathlib import Path

INST = Path(__file__).resolve().parents[1] / "instances" / "vllm"
ART = INST / "artifacts"
OUTLINE = json.loads((INST / "book" / "book-outline.json").read_text())

new_ids = []
for part in OUTLINE["parts"].values():
    for ch in part["chapters"]:
        new_ids.append(ch["id"])


def slug(d: str) -> str:
    return d.split("-", 1)[1] if "-" in d else d


def main():
    existing = sorted([p.name for p in ART.iterdir() if p.is_dir() and not p.name.startswith("_")])
    new_by_slug = {slug(n): n for n in new_ids}

    plan_archive = []
    plan_rename = []
    for old in existing:
        s = slug(old)
        if s in new_by_slug:
            new = new_by_slug[s]
            if old != new:
                plan_rename.append((old, new))
        else:
            plan_archive.append(old)

    # Step 1: archive
    archive_dir = ART / "_archive"
    archive_dir.mkdir(exist_ok=True)
    for a in plan_archive:
        src = ART / a
        dst = archive_dir / a
        if dst.exists():
            print(f"  [skip-archived] {a} already in _archive/")
            continue
        shutil.move(str(src), str(dst))
        print(f"  [archived]  {a} → _archive/{a}")

    # Step 2: rename ascending (06→05 first, then 07→06, ...)
    plan_rename.sort(key=lambda t: int(t[0].split("-")[0]))
    for old, new in plan_rename:
        src = ART / old
        dst = ART / new
        if dst.exists():
            print(f"  [skip-collision] {new} exists; cannot rename {old}")
            continue
        if not src.exists():
            print(f"  [skip-missing] {old} no longer exists")
            continue
        src.rename(dst)
        print(f"  [renamed]   {old} → {new}")

    # Step 3: snapshot legacy content for Ch04..Ch28 (everything to be rewritten)
    rewrite_targets = new_ids[3:]  # Ch04..Ch28 (index 3 onward)
    print(f"\n  Snapshotting legacy content for {len(rewrite_targets)} chapters → _legacy/")
    for nid in rewrite_targets:
        chdir = ART / nid
        if not chdir.exists():
            print(f"  [no-prior-content]  {nid} has no existing artifact dir; will be FRESH")
            continue
        moved_any = False
        for sub in ("narrative", "implementation", "tests"):
            subdir = chdir / sub
            if not subdir.exists():
                continue
            legacy_root = subdir / "_legacy"
            if legacy_root.exists():
                continue  # already snapshotted, idempotent
            legacy_root.mkdir()
            for item in subdir.iterdir():
                if item.name == "_legacy":
                    continue
                shutil.move(str(item), str(legacy_root / item.name))
            moved_any = True
        if moved_any:
            print(f"  [snapshot]  {nid} → narrative/_legacy, implementation/_legacy, tests/_legacy")
        else:
            print(f"  [empty]     {nid} had no narrative/impl/tests to snapshot")

    print("\nMigration complete.")


if __name__ == "__main__":
    main()
