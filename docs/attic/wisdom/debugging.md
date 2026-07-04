# Wisdom: Debugging Patterns

Universal debugging patterns discovered across repo2book projects. Gated: must appear in 2+ repos before promotion.

---

## W01: F.linear weight shape is [out, in]

**Discovered by**: implementer, tester
**Confirmed in**: vllm (03-flashattention), pytorch (general)
**Severity**: BLOCKER — causes silent wrong results

### Symptom
- Dimension mismatch errors when using `F.linear(x, weight)`
- Output values are silently wrong (no error, just incorrect results)
- Tests pass but numerical values are subtly off

### Root Cause
`nn.Linear(in_features, out_features)` stores its weight tensor as `[out_features, in_features]`.
`F.linear(x, weight)` computes `x @ weight^T`.
If you create a weight tensor with shape `[in, out]` and pass it to `F.linear`, the output
dimensions will be wrong.

### Detection
```python
# Always verify weight shape before using F.linear
assert weight.shape[0] == out_features, f"Expected [out, in], got {weight.shape}"
assert weight.shape[1] == in_features
```

### Solution
- Prefer `nn.Linear` directly when possible (handles shape correctly)
- When using `F.linear`, pass `nn.Linear(...).weight` which is guaranteed correct
- Document weight shape expectations in function signatures

### Affected Roles
- **implementer**: Check this BEFORE writing any linear layer code
- **tester**: Add shape assertions to all linear layer tests
- **reviewer**: Flag any bare `F.linear` call that doesn't verify weight shape

---

## W05: Docker vs local — CUDA version mismatch

**Discovered by**: tester, implementer
**Confirmed in**: vllm
**Severity**: HIGH — tests pass locally but fail in Docker, or vice versa

### Symptom
- `All tests passed` locally → `ImportError: undefined symbol` in Docker
- PyTorch version mismatch causes different behavior between environments
- GPU kernels compile differently in Docker vs local

### Root Cause
The vLLM Docker image has a specific PyTorch/CUDA combination. Local environments
may have different versions. Some operators are only available in certain versions.

### Solution
- Always test in the Docker container before claiming tests pass
- Document the exact Docker image tag used for testing
- When adding new dependencies, verify they exist in the Docker image

### Affected Roles
- **tester**: NEVER claim tests pass based on local run alone
- **implementer**: Check Docker compatibility of any new imports
- **writer**: Reference the Docker command in Cell 10 (verification)

---

## W11: SVG text clipping with text-anchor="end"

**Discovered by**: writer
**Confirmed in**: vllm (03-flashattention, 04-continuous-batching)
**Severity**: HIGH — text labels disappear in rendered diagrams

### Symptom
- Text labels at the left edge of SVG diagrams are invisible
- `text-anchor="end"` at x < 30 causes text to extend beyond viewBox
- Diagram looks correct in source but labels are cut off in PNG

### Detection
Run `python3 .claude/skills/svg-diagram/scripts/validate_svg.py diagram.svg`

### Solution
- Row labels with `text-anchor="end"` must have x >= 50
- Use `xml.sax.saxutils.escape()` for ALL text content (prevents `&lt;` double-escaping)
- Always validate: `xmllint --noout && validate_svg.py` before converting to PNG

### Affected Roles
- **writer**: Always validate SVG before referencing in narrative
- **reviewer**: Check diagram text is fully visible in review

---

## Other Debugging Patterns

### Silent shape bugs in reshape/transpose
Always add shape assertions before and after reshape/transpose operations.
Dimension order mistakes (BHLC vs BLHC, etc.) cause silent wrong results.

### In-place ops on tensors that share storage
`x = y.view(-1)` shares storage. Modifying x modifies y. Use `.clone()` when needed.

### Numerical instability in softmax with large values
Subtract max before exp. Online softmax is NOT optional — it's required for large tensors.
