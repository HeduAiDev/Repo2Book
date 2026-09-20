# v3 ch14 混合组化 + DSV4 重写归档（retrofit 补记）

- **Type**: retrofit 归档 · **Chapter**: v3 ch14《显存账本》 · **Date**: 2026-09-21 · **Agents**: pipeline 各站 + archivist（本记录）
- **一句话**: ch14 完成混合组化 + DeepSeek-V4 缓存账本重写——4 张新图（v4-layer-anatomy / v4-census / v4-compress-accounting / grouping-lineage）落地并替换被实跑 trace 证错的旧图 v4-ledger-census（六形态口径漏 indexer_state 与 c1 层）；archivist 同步 Book Bible 图登记（+4 / −1 孤儿），`lint_figures_registered` 显式章目录复跑通过（exit 0）。
- **涉及文件**: `instances/vllm/book/bible/figures.json`（本次改动：+4 登记、删孤儿 ch14-fig-v4-ledger-census，ch14 条目 11→14）；`instances/vllm/artifacts-v3/ch14-memory-ledger/`（diagrams/figure-manifest.json 与 4 组新 gen/svg/png、narrative、dossier/supplement-hybrid-v4.json、explainer 等——重写产物，非本次改动）；本记录。
