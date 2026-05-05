# Trace Index — vLLM Book Project

Master index of all significant events. Query this first before any work.

## Recent Activity (last 10 events)

<!-- ARCHIVIST: Update on every recorded event. Keep to last 10. -->

| Date | Type | Chapter | Summary | File |
| 2026-05-05 | delivery | 04-continuous-batching | Ch04 v5 | [2026-05-05_ch04-continuous-batching-ch04-v5.md](deliverys/2026-05-05_ch04-continuous-batching-ch04-v5.md) |
| 2026-05-04 | delivery | 03-flashattention-pagedattention | Ch03 Creative Rewrite | [2026-05-04_ch03-flashattention-pagedattention-ch03-creative-rewrite.md](deliverys/2026-05-04_ch03-flashattention-pagedattention-ch03-creative-rewrite.md) |
| 2026-05-04 | delivery | 02-kv-cache | Ch02 v5 Creative Rewrite | [2026-05-04_ch02-kv-cache-ch02-v5-creative-rewrite.md](deliverys/2026-05-04_ch02-kv-cache-ch02-v5-creative-rewrite.md) |
| 2026-05-04 | delivery | 01-self-attention-fundamentals | Ch01 v5 Creative Rewrite | [2026-05-04_ch01-self-attention-fundamentals-ch01-v5-creative-rewrite.md](deliveries/2026-05-04_ch01-self-attention-fundamentals-ch01-v5-creative-rewrite.md) |
| 2026-05-04 | delivery | 01-self-attention-fundamentals | Ch01 v5 Rewrite Complete | [2026-05-04_ch01-self-attention-fundamentals-ch01-v5-rewrite-complete.md](deliveries/2026-05-04_ch01-self-attention-fundamentals-ch01-v5-rewrite-complete.md) |
| 2026-05-04 | decision | 01-self-attention-fundamentals | v5 Rewrite Started — Ch01 Self-Attention | [2026-05-04_ch01-self-attention-fundamentals-v5-rewrite-started-—-ch01-self-attention.md](decisions/2026-05-04_ch01-self-attention-fundamentals-v5-rewrite-started-—-ch01-self-attention.md) |
|------|------|---------|---------|------|
| 2026-05-04 | delivery | 04 | Ch4 v5 rewrite complete: 14/14 tests, 0 blocking formula issues | [delivery-04-v5.md](deliveries/delivery-04-v5.md) |
| 2026-05-04 | decision | framework | Migrated to repo2book architecture: agent teams, knowledge/wisdom, topology | [2026-05-04_repo2book-migration.md](decisions/2026-05-04_repo2book-migration.md) |

## By Chapter

| Chapter | Status | Last Delivery | Key Decisions | Open Issues |
|---------|--------|--------------|---------------|-------------|
| 01 | published | 2026-05-04 (v5 creative rewrite) | Intuition-first, 4 diagrams, Bahdanau→Dao | — |
| 02 | published | 2026-04 (v3) | — | — |
| 03 | published | 2026-05 (v5 rewrite) | Proof structure, diagram method | — |
| 04 | published | 2026-05-04 (v5 rewrite) | Preemption bug fix, bubble proof | — |
| 05-13 | published | 2026-04 (v1-v3) | — | Need v5 compliance scan |
| 14 | not_started | — | — | — |
| 15-28 | not_started | — | — | — |

## By Type

### Decisions
- [2026-05-04_repo2book-migration.md](decisions/2026-05-04_repo2book-migration.md) — Framework architecture

### Deliveries
- [2026-05-04_ch01-self-attention-fundamentals-ch01-v5-creative-rewrite.md](deliveries/2026-05-04_ch01-self-attention-fundamentals-ch01-v5-creative-rewrite.md) — Ch01 v5 Creative Rewrite: intuition-first, 4 diagrams, Bahdanau→Dao, 9/9 pass
- [2026-05-04_ch01-self-attention-fundamentals-ch01-v5-rewrite-complete.md](deliveries/2026-05-04_ch01-self-attention-fundamentals-ch01-v5-rewrite-complete.md) — Ch01 v5 Rewrite Complete (implementation + tests)
- [delivery-04-v5.md](deliveries/delivery-04-v5.md) — Ch4 v5 rewrite complete

### User Interactions
- (none recorded yet — first session after trace system creation)

### Context Summaries
- (none yet — will be created after each session)

## Query Protocol

Before starting work on chapter `{id}`:
1. Check this INDEX for recent related events
2. Read `deliveries/delivery-{id}-*.md` for chapter delivery history
3. Read `decisions/` for any decisions affecting this chapter
4. Read `state.json` for current project status
5. If user feedback exists for this chapter, read `user_interactions/` entries
6. If resuming after a break, read the latest `context_summaries/` entry
