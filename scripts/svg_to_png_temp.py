#!/usr/bin/env python
"""Temporary SVG-to-PNG converter using Pillow.
Used when rsvg-convert is not available on the system.
"""
import xml.etree.ElementTree as ET
import math
import sys
from PIL import Image, ImageDraw, ImageFont


def svg_to_png(svg_path, png_path, scale=2):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    viewbox = root.get('viewBox', '0 0 800 600')
    _, _, svg_w, svg_h = [float(x) for x in viewbox.split()]

    img_w = int(svg_w * scale)
    img_h = int(svg_h * scale)

    img = Image.new('RGBA', (img_w, img_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load CJK-capable fonts
    font_paths = [
        ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc'),
        ('C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/simhei.ttf'),
        ('C:/Windows/Fonts/simsun.ttc', 'C:/Windows/Fonts/simsun.ttc'),
        ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'),
    ]

    font_reg = None
    font_bold = None

    for reg_path, bold_path in font_paths:
        try:
            font_reg = ImageFont.truetype(reg_path, int(10.5 * scale))
            font_small = ImageFont.truetype(reg_path, int(8.8 * scale))
            font_title = ImageFont.truetype(bold_path if 'bd' in bold_path or 'Bold' in bold_path else reg_path, int(16 * scale))
            font_h2 = ImageFont.truetype(bold_path if 'bd' in bold_path or 'Bold' in bold_path else reg_path, int(11.5 * scale))
            font_bold = ImageFont.truetype(bold_path if 'bd' in bold_path or 'Bold' in bold_path else reg_path, int(10.5 * scale))
            font_leg = ImageFont.truetype(reg_path, int(10.2 * scale))
            print(f'Using font: {reg_path}')
            break
        except Exception:
            continue

    if font_reg is None:
        font_reg = ImageFont.load_default()
        font_small = font_reg
        font_title = font_reg
        font_h2 = font_reg
        font_bold = font_reg
        font_leg = font_reg
        print('Using default font (no CJK support)')

    ns = {'svg': 'http://www.w3.org/2000/svg'}

    NAMED_COLORS = {
        'white': '#ffffff', 'black': '#000000', 'red': '#ff0000',
        'green': '#008000', 'blue': '#0000ff', 'none': None,
        'transparent': None,
    }

    def hex_to_rgb(h):
        if h in NAMED_COLORS:
            if NAMED_COLORS[h] is None:
                return None
            h = NAMED_COLORS[h]
        h = h.lstrip('#')
        if len(h) == 3:
            h = ''.join([c * 2 for c in h])
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def get_float(el, attr, default=0):
        v = el.get(attr)
        if v is None:
            return default
        return float(v)

    # Collect all drawable elements preserving document order
    for el in root.iter():
        tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag

        if tag == 'rect':
            x = get_float(el, 'x') * scale
            y = get_float(el, 'y') * scale
            w = get_float(el, 'width') * scale
            h = get_float(el, 'height') * scale
            rx = get_float(el, 'rx') * scale
            fill = el.get('fill', 'none')
            stroke = el.get('stroke', 'none')
            sw = get_float(el, 'stroke-width') * scale
            dash = el.get('stroke-dasharray')

            color = hex_to_rgb(fill)
            if color:
                if rx > 0:
                    draw.rounded_rectangle([x, y, x + w, y + h], radius=rx, fill=color)
                else:
                    draw.rectangle([x, y, x + w, y + h], fill=color)

            sc = hex_to_rgb(stroke)
            if sc:
                if dash:
                    dashes = [float(d) * scale for d in dash.replace(',', ' ').split()]
                    if len(dashes) >= 2:
                        dash_len, gap_len = dashes[0], dashes[1]
                        # Approximate dashed border with line segments
                        # Top edge
                        cx = x
                        toggle = True
                        while cx < x + w:
                            seg_end = min(cx + dash_len, x + w)
                            if toggle:
                                draw.line([(cx, y), (seg_end, y)], fill=sc, width=int(sw))
                            cx += dash_len if toggle else gap_len
                            toggle = not toggle
                        # Bottom edge
                        cx = x
                        toggle = True
                        while cx < x + w:
                            seg_end = min(cx + dash_len, x + w)
                            if toggle:
                                draw.line([(cx, y + h), (seg_end, y + h)], fill=sc, width=int(sw))
                            cx += dash_len if toggle else gap_len
                            toggle = not toggle
                        # Left edge
                        cy = y
                        toggle = True
                        while cy < y + h:
                            seg_end = min(cy + dash_len, y + h)
                            if toggle:
                                draw.line([(x, cy), (x, seg_end)], fill=sc, width=int(sw))
                            cy += dash_len if toggle else gap_len
                            toggle = not toggle
                        # Right edge
                        cy = y
                        toggle = True
                        while cy < y + h:
                            seg_end = min(cy + dash_len, y + h)
                            if toggle:
                                draw.line([(x + w, cy), (x + w, seg_end)], fill=sc, width=int(sw))
                            cy += dash_len if toggle else gap_len
                            toggle = not toggle
                    else:
                        if rx > 0:
                            draw.rounded_rectangle([x, y, x + w, y + h], radius=rx, outline=sc, width=int(sw))
                        else:
                            draw.rectangle([x, y, x + w, y + h], outline=sc, width=int(sw))
                else:
                    if rx > 0:
                        draw.rounded_rectangle([x, y, x + w, y + h], radius=rx, outline=sc, width=int(sw))
                    else:
                        draw.rectangle([x, y, x + w, y + h], outline=sc, width=int(sw))

        elif tag == 'path':
            d = el.get('d', '')
            stroke = el.get('stroke', '#000')
            sw = get_float(el, 'stroke-width') * scale
            fill = el.get('fill', 'none')
            marker_end = el.get('marker-end', '')

            sc = hex_to_rgb(stroke)

            # Simple path parser: M x,y L x,y ...
            parts = d.replace(',', ' ').split()
            points = []
            i = 0
            while i < len(parts):
                cmd = parts[i]
                if cmd in ('M', 'L'):
                    i += 1
                    px = float(parts[i]) * scale
                    py = float(parts[i + 1]) * scale
                    points.append((cmd, px, py))
                    i += 2
                elif cmd == 'C':
                    # Bezier curve - extract endpoints
                    i += 5
                    px = float(parts[i]) * scale
                    py = float(parts[i + 1]) * scale
                    points.append(('C_end', px, py))
                    i += 2
                elif cmd == 'Q':
                    i += 3
                    px = float(parts[i]) * scale
                    py = float(parts[i + 1]) * scale
                    points.append(('Q_end', px, py))
                    i += 2
                else:
                    i += 1

            if len(points) >= 2:
                for j in range(len(points) - 1):
                    x1, y1 = points[j][1], points[j][2]
                    x2, y2 = points[j + 1][1], points[j + 1][2]
                    draw.line([(x1, y1), (x2, y2)], fill=sc, width=int(sw))

                # Arrowhead for marker-end
                if marker_end and len(points) >= 2:
                    x1, y1 = points[-2][1], points[-2][2]
                    x2, y2 = points[-1][1], points[-1][2]
                    dx, dy = x2 - x1, y2 - y1
                    length = math.sqrt(dx * dx + dy * dy)
                    if length > 0:
                        dx, dy = dx / length, dy / length
                        arrow_len = 8 * scale
                        arrow_w = 4 * scale
                        tip = (x2, y2)
                        base_x = x2 - dx * arrow_len
                        base_y = y2 - dy * arrow_len
                        perp_x = -dy * arrow_w
                        perp_y = dx * arrow_w
                        left = (base_x + perp_x, base_y + perp_y)
                        right = (base_x - perp_x, base_y - perp_y)
                        # Use marker color if specified
                        marker_color = marker_end
                        if marker_color and not marker_color.startswith('url'):
                            afill = hex_to_rgb(marker_color)
                        else:
                            afill = sc
                        draw.polygon([tip, left, right], fill=afill)

        elif tag == 'text':
            # Collect all text content
            text_content = el.text or ''
            for child in el:
                if child.tail:
                    text_content += child.tail
            if not text_content.strip():
                continue

            x = get_float(el, 'x') * scale
            y = get_float(el, 'y') * scale
            fill = el.get('fill', '#000')
            fs = get_float(el, 'font-size', 12) * scale
            fw = el.get('font-weight', 'normal')
            ta = el.get('text-anchor', 'start')

            # Select font
            font = font_reg
            if fw == 'bold' or fw == '700':
                if abs(fs / scale - 16) < 1.5:
                    font = font_title
                elif abs(fs / scale - 11.5) < 1.5:
                    font = font_h2
                else:
                    font = font_bold
            elif abs(fs / scale - 8.8) < 1.5:
                font = font_small
            elif abs(fs / scale - 10.2) < 1.5:
                font = font_leg

            color = hex_to_rgb(fill) if fill else (0, 0, 0)

            # Measure text
            try:
                bbox = draw.textbbox((0, 0), text_content, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw = len(text_content) * fs * 0.55
                th = fs

            # Adjust for text-anchor
            if ta == 'middle':
                x = x - tw / 2
            elif ta == 'end':
                x = x - tw

            # Baseline adjustment
            y = y - th * 0.85

            draw.text((x, y), text_content, fill=color, font=font)

    img.save(png_path, 'PNG')
    print(f'Saved {png_path} ({img_w}x{img_h})')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <input.svg> <output.png>')
        sys.exit(1)
    svg_to_png(sys.argv[1], sys.argv[2], scale=2)
