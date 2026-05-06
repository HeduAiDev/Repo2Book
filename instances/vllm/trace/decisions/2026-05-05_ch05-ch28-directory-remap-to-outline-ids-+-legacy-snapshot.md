# Ch05-Ch28 directory remap to outline IDs + legacy snapshot

- **Type**: design_change
- **Chapter**: N/A
- **Date**: 2026-05-05
- **Timestamp**: 2026-05-05T12:26:46Z
- **Agents involved**: archivist
- **User present**: False
- **Tags**: 

## What happened

24 renames (06-mem→05-mem, 07-sched→06-sched, …, 29-v4-pro→28-v4-pro). 05-chunked-prefill archived. 14 chapters had narrative/impl/tests snapshotted to _legacy/ for reference. vLLM source cloned at 98661fe.

## Why it matters

User decided A1 (full rewrite of Ch04-Ch28 per current writer.md/wisdom standards) and B1 (migrate-then-rewrite, keep legacy as reference). Old artifact IDs were +1 offset due to chunked-prefill being inserted-then-merged-back.

## What to remember

24 renames (06-mem→05-mem, 07-sched→06-sched, …, 29-v4-pro→28-v4-pro). 05-chunked-prefill archived. 14 chapters had narrative/impl/tests snapshotted to _legacy/ for reference. vLLM source cloned at 98...
