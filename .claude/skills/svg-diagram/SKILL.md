---
name: svg-diagram
description: >
  Generate technical diagrams as valid SVG+PNG using Python scripts. Use this
  skill whenever you need to create diagrams for technical documentation,
  book chapters, or educational content — especially for dense many-to-many
  graphs (tiling patterns, state evolution tables, architecture overviews)
  where Mermaid auto-layout fails and Excalidraw manual coordinates cause
  misalignment. Triggers on: "create a diagram", "draw this", "visualize
  the tiling", "make a figure", "需要画图", "画一个图", or any request to
  illustrate a technical concept. DO NOT use Excalidraw or Mermaid for
  diagrams with >3 connected elements — use this skill instead.
---

# SVG Diagram Generator

Generate technical diagrams with precise, programmatic SVG — then validate and convert to PNG for reliable rendering in any markdown viewer.

## Why This Skill Exists

After multiple failed attempts with Excalidraw (manual coordinate bugs, text overlap, arrow misalignment) and Mermaid (dense-graph tangling), we converged on a proven workflow: **Python script → xmllint validation → ImageMagick PNG**. This skill encapsulates that workflow so any agent can produce reliable diagrams.

**Use this skill when:**
- The diagram has >3 connected elements (Mermaid tangles)
- You need precise coordinate control (Excalidraw is too manual)
- Text labels must not clip or overlap
- Arrows must connect to exact element edges
- The output must render reliably in any markdown viewer

**Do NOT use this skill when:**
- A simple 3-node flowchart suffices (use Mermaid)
- A numerical comparison table suffices (use Markdown tables)
- The diagram is purely decorative (skip it)

## Required Tools

All tools are pre-installed on standard Linux. Verify before first use:
- `xmllint` — XML validation (from libxml2)
- `convert` — ImageMagick SVG→PNG conversion
- `python3` — with xml.sax.saxutils (stdlib)

## Workflow (Follow Exactly)

### Step 1: Design the layout in your head

Before writing Python, sketch the diagram structure:
- What are the visual groups? (columns, rows, regions)
- What elements are in each group? (rectangles, text, lines)
- How do arrows connect elements? (which edges to which edges)

**Key principle: ALL coordinates are computed by Python loops. ZERO manual x/y values.**

### Step 2: Generate the SVG with Python

Use the template at `scripts/generate_diagram.py` as your starting point. Adapt it for your specific diagram.

**CRITICAL rules for the Python script:**

1. **Escape ALL text with `xml.sax.saxutils.escape()`** — unescaped `<`, `>`, `&` make the SVG invalid XML
2. **Pad viewBox for text labels** — if labels extend left of x=0 (via `text-anchor="end"`), they clip. Ensure label x > label_text_width
3. **Compute arrow endpoints from element edges** — never hardcode. Arrow start = source.right_edge, arrow end = target.left_edge
4. **Use marker-end for arrowheads** — define once in `<defs>`, reference as `marker-end="url(#arrowId)"`
5. **Use `<text>` elements for multiline** — SVG does NOT support `<br/>`. Use separate `<text>` elements with increasing y offsets

**SVG boilerplate:**
```python
import xml.sax.saxutils as xs
def esc(s): return xs.escape(s)

w, h = 700, 400  # canvas size — pad generously
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')
# ... build elements ...
L.append('</svg>')
svg = '\n'.join(L)
```

### Step 3: Validate with xmllint + semantic check

```bash
xmllint --noout diagram.svg           # XML syntax validation
python3 scripts/validate_svg.py diagram.svg  # Semantic validation
```

`xmllint` catches syntax errors (unescaped `<`, unclosed tags).
`validate_svg.py` catches rendering bugs that xmllint can't:
- **Double-escaping**: `&amp;lt;` means `esc()` was called on already-escaped text
- **Text clipping**: `text-anchor="end"` at x < 30 → text extends beyond viewBox
- **Missing arrowheads**: `<line>` with stroke but no `marker-end`

If EITHER fails, fix before proceeding. Both must pass.

### Step 4: Convert to PNG

```bash
convert -density 150 diagram.svg diagram.png
```

Reference the `.png` file in markdown, NOT the `.svg`. SVG rendering varies by viewer; PNG is universal.

### Step 5: Reference in markdown

```markdown
![Diagram caption](../diagrams/diagram-name.png)

> *图注：Chinese caption explaining what the diagram shows.*
```

## Diagram Type Templates

### Type A: Tiling / Many-to-Many Connections

Use when showing connections between two groups of tiles (e.g., FlashAttention Q→KV).

Pattern: left column (source tiles, one color per tile) + right column (target tiles) + arrows from each source to each target. Each source tile gets its own arrow color for visual tracking.

See `references/example-fa-tiling.py` for the complete example.

### Type B: State Evolution / Numerical Trace

Use when showing values evolving across iterations or steps.

Pattern: columns = iterations/time steps, rows = variables, highlighted cells for key transitions. Colored boxes (green=stable, red=changed) for the most important row.

See `references/example-softmax-trace.py` for the complete example.

### Type C: Simple Flow / Pipeline

Use when showing a linear sequence of steps with one-way arrows.

Pattern: horizontal or vertical chain of boxes, single arrow between each pair. For simple flows (<5 nodes), Mermaid is acceptable as an alternative.

## Validation Checklist

Before marking a diagram as complete, verify ALL of:
- [ ] `xmllint --noout file.svg` passes
- [ ] `convert -density 150 file.svg file.png` produces a valid PNG
- [ ] No text extends beyond viewBox boundaries (check left edge especially)
- [ ] All arrow endpoints touch element edges (not floating in empty space)
- [ ] Every `<text>` that might contain `<`, `>`, `&` is escaped
- [ ] PNG file is referenced in markdown (not SVG)

## Common Pitfalls

1. **`<-` in text** — write RAW `<-` in Python source, `esc()` converts to `&lt;-`. NEVER pre-escape as `&lt;` in Python strings — `esc()` will double-escape the `&` to `&amp;lt;`
2. **`text-anchor="end"` at small x** — text extends LEFT from x. If x < text_width, text clips. Solution: use x >= 50 for row labels
3. **Arrows starting/ending in empty space** — compute endpoints from element coordinates, never hardcode
4. **`<br/>` in SVG text** — not supported. Use separate `<text>` elements with y offsets
5. **`font-weight:bold` (CSS)** — SVG uses `font-weight="bold"` (attribute syntax, not CSS)
6. **`fill="#374151" color:#b91c1c`** — `color:` is CSS, invalid in SVG attributes. Use `fill="#b91c1c"`
7. **Forgetting `xmlns`** — SVG must have `xmlns="http://www.w3.org/2000/svg"` on root element
