# Outline Changelog

Every structural change to the book outline, detected by archivist.

## Change Log

### 2026-03-01 — Chapter insertion: 04-chunked-prefill

- **Detected by**: (not recorded — before archivist existed)
- **Type**: insertion
- **Description**: Chunked Prefill chapter inserted after 04-continuous-batching
- **Impact**: Chapters 05-13 shifted right (now 06-14)
- **Migration**: (not recorded)

### 2026-05-04 — Archivist system initialized

- **Detected by**: archivist
- **Type**: metadata
- **Description**: Archivist and trace system created. Chapter lineage tracking begins.
- **Current outline snapshot**: `cross-chapter/outline-snapshot-2026-05-04.json`
- **Note**: All prior lineage is approximate (reconstructed from git history and context).

---

## Detection Protocol

When `book/book-outline.json` changes:

1. Archivist loads current snapshot from `cross-chapter/outline-snapshot-{date}.json`
2. Archivist loads new outline from `book/book-outline.json`
3. Archivist diffs: new chapters, deleted chapters, reordered chapters, renamed chapters
4. Archivist sends structured message to book-editor: detected changes + proposed migrations
5. Book-editor approves → archivist executes migrations
6. Archivist records changelog entry with before/after snapshots
7. Archivist updates state.json chapter_lineage for affected chapters
