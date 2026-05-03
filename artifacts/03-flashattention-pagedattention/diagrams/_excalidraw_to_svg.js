#!/usr/bin/env node
/**
 * Excalidraw JSON → SVG converter
 * Converts .excalidraw files to .svg for embedding in Jupyter notebooks.
 *
 * Usage: node _excalidraw_to_svg.js [file.excalidraw | --all]
 *
 * Supported element types: rectangle, diamond, ellipse, text, arrow, line
 */

const fs = require('fs');
const path = require('path');

// ─── Helpers ───────────────────────────────────────────────

function escapeXml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/** Compute bounding box of all elements */
function computeViewBox(elements) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const el of elements) {
    if (el.isDeleted) continue;
    const x = el.x || 0;
    const y = el.y || 0;
    const w = el.width || 0;
    const h = el.height || 0;

    if (el.type === 'arrow' || el.type === 'line') {
      for (const [px, py] of (el.points || [])) {
        minX = Math.min(minX, x + px);
        minY = Math.min(minY, y + py);
        maxX = Math.max(maxX, x + px);
        maxY = Math.max(maxY, y + py);
      }
    } else {
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + w);
      maxY = Math.max(maxY, y + h);
    }
  }
  const pad = 20;
  return {
    x: minX - pad,
    y: minY - pad,
    w: maxX - minX + 2 * pad,
    h: maxY - minY + 2 * pad,
  };
}

/** Build element lookup map by id */
function buildElementMap(elements) {
  const map = {};
  for (const el of elements) {
    if (el.id) map[el.id] = el;
  }
  return map;
}

// ─── SVG Renderers ─────────────────────────────────────────

function renderRect(el) {
  const rx = el.roundness ? Math.min(8, el.width / 4, el.height / 4) : 0;
  return `<rect x="${el.x}" y="${el.y}" width="${el.width}" height="${el.height}" `
    + `rx="${rx}" `
    + `fill="${el.backgroundColor === 'transparent' ? 'none' : el.backgroundColor}" `
    + `stroke="${el.strokeColor}" `
    + `stroke-width="${el.strokeWidth}" `
    + `${el.strokeStyle === 'dashed' ? 'stroke-dasharray="8 4"' : ''} `
    + `opacity="${el.opacity / 100}"/>`;
}

function renderDiamond(el) {
  const cx = el.x + el.width / 2;
  const cy = el.y + el.height / 2;
  const hw = el.width / 2;
  const hh = el.height / 2;
  const pts = [
    `${cx},${cy - hh}`,
    `${cx + hw},${cy}`,
    `${cx},${cy + hh}`,
    `${cx - hw},${cy}`,
  ].join(' ');
  return `<polygon points="${pts}" `
    + `fill="${el.backgroundColor === 'transparent' ? 'none' : el.backgroundColor}" `
    + `stroke="${el.strokeColor}" `
    + `stroke-width="${el.strokeWidth}" `
    + `${el.strokeStyle === 'dashed' ? 'stroke-dasharray="8 4"' : ''} `
    + `opacity="${el.opacity / 100}"/>`;
}

function renderEllipse(el) {
  const cx = el.x + el.width / 2;
  const cy = el.y + el.height / 2;
  return `<ellipse cx="${cx}" cy="${cy}" rx="${el.width / 2}" ry="${el.height / 2}" `
    + `fill="${el.backgroundColor === 'transparent' ? 'none' : el.backgroundColor}" `
    + `stroke="${el.strokeColor}" `
    + `stroke-width="${el.strokeWidth}" `
    + `opacity="${el.opacity / 100}"/>`;
}

function renderText(el) {
  const text = el.text || el.originalText || '';
  const lines = text.split('\n');
  const fontSize = el.fontSize || 14;
  const lineHeight = fontSize * (el.lineHeight || 1.25);

  // Determine anchor from textAlign
  let anchor = 'middle';
  if (el.textAlign === 'left') anchor = 'start';
  else if (el.textAlign === 'right') anchor = 'end';

  // If this text is contained in a parent, center within parent bounds
  // The excalidraw text elements already have correct x/y positioning

  let x;
  if (anchor === 'start') x = el.x;
  else if (anchor === 'end') x = el.x + (el.width || 0);
  else x = el.x + (el.width || 0) / 2;

  // Compute starting y: vertically center within element height
  const totalTextH = lines.length * lineHeight;
  let startY;
  if (el.verticalAlign === 'middle') {
    startY = el.y + (el.height || totalTextH) / 2 - totalTextH / 2 + fontSize * 0.85;
  } else {
    startY = el.y + fontSize * 0.85;
  }

  const tspans = lines.map((line, i) =>
    `<tspan x="${x}" dy="${i === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`
  ).join('');

  return `<text x="${x}" y="${startY}" `
    + `text-anchor="${anchor}" `
    + `font-size="${fontSize}" `
    + `font-family="'Cascadia Code', 'Source Code Pro', 'Courier New', monospace" `
    + `fill="${el.strokeColor}" `
    + `opacity="${el.opacity / 100}">`
    + tspans
    + `</text>`;
}

function renderArrow(el, markerMap) {
  const points = el.points || [[0, 0], [0, 0]];
  const pathData = points.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L';
    return `${cmd}${el.x + p[0]},${el.y + p[1]}`;
  }).join(' ');

  // Register marker color
  const color = el.strokeColor || '#000';
  const markerId = `arrow_${color.replace('#', '')}`;
  if (!markerMap[markerId]) {
    markerMap[markerId] = color;
  }

  let markerEnd = '';
  if (el.endArrowhead === 'arrow') {
    markerEnd = `marker-end="url(#${markerId})"`;
  }

  return `<path d="${pathData}" `
    + `fill="none" `
    + `stroke="${el.strokeColor}" `
    + `stroke-width="${el.strokeWidth}" `
    + `${el.strokeStyle === 'dashed' ? 'stroke-dasharray="8 4"' : ''} `
    + `${markerEnd} `
    + `opacity="${el.opacity / 100}"/>`;
}

function renderLine(el) {
  const points = el.points || [[0, 0], [0, 0]];
  const pathData = points.map((p, i) => {
    const cmd = i === 0 ? 'M' : 'L';
    return `${cmd}${el.x + p[0]},${el.y + p[1]}`;
  }).join(' ');

  return `<path d="${pathData}" `
    + `fill="none" `
    + `stroke="${el.strokeColor}" `
    + `stroke-width="${el.strokeWidth}" `
    + `${el.strokeStyle === 'dashed' ? 'stroke-dasharray="8 4"' : ''} `
    + `opacity="${el.opacity / 100}"/>`;
}

// ─── Main Conversion ───────────────────────────────────────

function excalidrawToSvg(inputPath) {
  const raw = fs.readFileSync(inputPath, 'utf-8');
  const data = JSON.parse(raw);
  const elements = data.elements.filter(e => !e.isDeleted);
  const elMap = buildElementMap(elements);
  const vb = computeViewBox(elements);

  // Collect arrow marker colors
  const markerMap = {};

  // Render elements in order (shapes first, then text on top)
  const shapes = [];
  const texts = [];

  for (const el of elements) {
    if (el.isDeleted) continue;

    switch (el.type) {
      case 'rectangle':
        shapes.push(renderRect(el));
        break;
      case 'diamond':
        shapes.push(renderDiamond(el));
        break;
      case 'ellipse':
        shapes.push(renderEllipse(el));
        break;
      case 'text':
        texts.push(renderText(el));
        break;
      case 'arrow':
        shapes.push(renderArrow(el, markerMap));
        break;
      case 'line':
        shapes.push(renderLine(el));
        break;
    }
  }

  // Build marker defs
  const markerDefs = Object.entries(markerMap).map(([id, color]) =>
    `<marker id="${id}" viewBox="0 0 10 6" refX="9" refY="3" `
    + `markerWidth="8" markerHeight="6" orient="auto-start-reverse">`
    + `<path d="M0,0 L10,3 L0,6 Z" fill="${color}"/>`
    + `</marker>`
  ).join('\n    ');

  // Assemble SVG
  const width = Math.ceil(vb.w);
  const height = Math.ceil(vb.h);

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     viewBox="${vb.x} ${vb.y} ${width} ${height}"
     width="${width}" height="${height}"
     style="background: ${data.appState?.viewBackgroundColor || '#ffffff'}">
  <defs>
    ${markerDefs}
  </defs>
  ${shapes.join('\n  ')}
  ${texts.join('\n  ')}
</svg>`;

  return svg;
}

// ─── CLI ───────────────────────────────────────────────────

const args = process.argv.slice(2);
const diagramsDir = __dirname;

if (args[0] === '--all') {
  const files = fs.readdirSync(diagramsDir)
    .filter(f => f.endsWith('.excalidraw'));
  console.log(`Converting ${files.length} excalidraw files...`);
  for (const f of files) {
    const input = path.join(diagramsDir, f);
    const output = path.join(diagramsDir, f.replace('.excalidraw', '.svg'));
    const svg = excalidrawToSvg(input);
    fs.writeFileSync(output, svg);
    console.log(`  ✓ ${f} → ${path.basename(output)} (${svg.length} bytes)`);
  }
  console.log('Done!');
} else if (args.length > 0) {
  const input = args[0];
  const output = args[1] || input.replace('.excalidraw', '.svg');
  const svg = excalidrawToSvg(input);
  fs.writeFileSync(output, svg);
  console.log(`✓ ${path.basename(input)} → ${path.basename(output)} (${svg.length} bytes)`);
} else {
  console.log('Usage: node _excalidraw_to_svg.js [file.excalidraw | --all]');
}
