#!/usr/bin/env python3
"""Generate 3 themed mockups of the Draw tab (ui-2a layout) at iPhone 15 Pro @3x."""

from PIL import Image, ImageDraw, ImageFont
import os

# === Constants ===
W, H = 1179, 2556
S = 3  # @3x scale factor

# Try to load SF Pro or fall back
def load_font(size, bold=False):
    """Load a good font at given size (in pixels)."""
    candidates = [
        # macOS SF Pro
        "/System/Library/Fonts/SFProText-Bold.otf" if bold else "/System/Library/Fonts/SFProText-Regular.otf",
        "/System/Library/Fonts/SFProDisplay-Bold.otf" if bold else "/System/Library/Fonts/SFProDisplay-Regular.otf",
        "/Library/Fonts/SF-Pro-Text-Bold.otf" if bold else "/Library/Fonts/SF-Pro-Text-Regular.otf",
        "/System/Library/Fonts/SFProText-Semibold.otf" if bold else "/System/Library/Fonts/SFProText-Regular.otf",
        # Helvetica fallback
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

def load_semibold_font(size):
    candidates = [
        "/System/Library/Fonts/SFProText-Semibold.otf",
        "/System/Library/Fonts/SFProDisplay-Semibold.otf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return load_font(size, bold=True)

def load_medium_font(size):
    candidates = [
        "/System/Library/Fonts/SFProText-Medium.otf",
        "/System/Library/Fonts/SFProDisplay-Medium.otf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return load_font(size, bold=False)

# Pre-load fonts
font_title = load_font(17 * S, bold=True)           # Nav title
font_seg = load_semibold_font(14 * S)               # Segmented control
font_palette_label = load_font(11 * S, bold=False)   # Color labels
font_brush_label = load_font(13 * S, bold=False)     # "Brush Size"
font_brush_val = load_semibold_font(14 * S)          # Size value
font_btn = load_semibold_font(15 * S)                # Action buttons
font_tab = load_medium_font(10 * S)                  # Tab bar labels
font_status = load_semibold_font(14 * S)             # Status bar time
font_status_small = load_font(12 * S, bold=False)    # Status bar extras


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def draw_rounded_rect(draw, xy, radius, fill=None, outline=None, width=1):
    """Draw a rounded rectangle."""
    x0, y0, x1, y1 = xy
    r = radius
    if fill:
        # Center rectangles
        draw.rectangle([x0 + r, y0, x1 - r, y1], fill=fill)
        draw.rectangle([x0, y0 + r, x1, y1 - r], fill=fill)
        # Corners
        draw.pieslice([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=fill)
        draw.pieslice([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=fill)
        draw.pieslice([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=fill)
        draw.pieslice([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=fill)
    if outline:
        # Top edge
        draw.line([x0 + r, y0, x1 - r, y0], fill=outline, width=width)
        # Bottom edge
        draw.line([x0 + r, y1, x1 - r, y1], fill=outline, width=width)
        # Left edge
        draw.line([x0, y0 + r, x0, y1 - r], fill=outline, width=width)
        # Right edge
        draw.line([x1, y0 + r, x1, y1 - r], fill=outline, width=width)
        # Corner arcs
        draw.arc([x0, y0, x0 + 2*r, y0 + 2*r], 180, 270, fill=outline, width=width)
        draw.arc([x1 - 2*r, y0, x1, y0 + 2*r], 270, 360, fill=outline, width=width)
        draw.arc([x0, y1 - 2*r, x0 + 2*r, y1], 90, 180, fill=outline, width=width)
        draw.arc([x1 - 2*r, y1 - 2*r, x1, y1], 0, 90, fill=outline, width=width)


def text_center_x(draw, text, font, y, fill, img_width=W):
    """Draw text centered horizontally."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (img_width - tw) // 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_status_bar(draw, theme, y_top):
    """Draw iOS status bar with time, signal, battery."""
    text_color = hex_to_rgb(theme['text_primary'])

    # Time "9:41" centered
    text_center_x(draw, "9:41", font_status, y_top + 5 * S, text_color)

    # Signal bars (left side)
    bar_x = 50 * S
    bar_y_base = y_top + 20 * S
    for i in range(4):
        bh = (8 + i * 3) * S
        bw = 5 * S
        bx = bar_x + i * (bw + 3 * S)
        draw.rectangle([bx, bar_y_base - bh, bx + bw, bar_y_base], fill=text_color)

    # WiFi icon (simple arc representation)
    wifi_x = bar_x + 4 * (5 * S + 3 * S) + 8 * S
    draw.ellipse([wifi_x, bar_y_base - 6*S, wifi_x + 6*S, bar_y_base], fill=text_color)

    # Battery (right side)
    bat_x = W - 80 * S
    bat_y = y_top + 12 * S
    bat_w, bat_h = 25 * S, 12 * S
    draw_rounded_rect(draw, [bat_x, bat_y, bat_x + bat_w, bat_y + bat_h],
                       3 * S, outline=text_color, width=2)
    # Battery fill (green)
    fill_pad = 2 * S
    draw_rounded_rect(draw, [bat_x + fill_pad, bat_y + fill_pad,
                              bat_x + bat_w - fill_pad, bat_y + bat_h - fill_pad],
                       2 * S, fill=(52, 199, 89))
    # Battery nub
    draw_rounded_rect(draw, [bat_x + bat_w + 1, bat_y + 3*S,
                              bat_x + bat_w + 4*S, bat_y + bat_h - 3*S],
                       1 * S, fill=text_color)


def draw_segmented_control(draw, theme, y_top, pad_x):
    """Draw Brush/Eraser segmented control."""
    seg_bg = hex_to_rgb(theme['segmented_bg'])
    accent = hex_to_rgb(theme['accent'])
    text_sec = hex_to_rgb(theme['text_secondary'])

    seg_h = 36 * S
    seg_pad = 2 * S
    seg_radius = 8 * S
    inner_radius = 7 * S

    # Outer background
    draw_rounded_rect(draw, [pad_x, y_top, W - pad_x, y_top + seg_h],
                       seg_radius, fill=seg_bg)

    # Active "Brush" button (left half)
    mid_x = W // 2
    draw_rounded_rect(draw, [pad_x + seg_pad, y_top + seg_pad,
                              mid_x - seg_pad, y_top + seg_h - seg_pad],
                       inner_radius, fill=accent)

    # Text
    brush_bbox = draw.textbbox((0, 0), "Brush", font=font_seg)
    brush_tw = brush_bbox[2] - brush_bbox[0]
    brush_th = brush_bbox[3] - brush_bbox[1]
    left_center = (pad_x + mid_x) // 2
    draw.text((left_center - brush_tw // 2, y_top + (seg_h - brush_th) // 2),
              "Brush", font=font_seg, fill=(255, 255, 255))

    eraser_bbox = draw.textbbox((0, 0), "Eraser", font=font_seg)
    eraser_tw = eraser_bbox[2] - eraser_bbox[0]
    eraser_th = eraser_bbox[3] - eraser_bbox[1]
    right_center = (mid_x + W - pad_x) // 2
    draw.text((right_center - eraser_tw // 2, y_top + (seg_h - eraser_th) // 2),
              "Eraser", font=font_seg, fill=text_sec)

    return y_top + seg_h


def draw_palette(draw, theme, y_top):
    """Draw color palette circles with labels."""
    accent = hex_to_rgb(theme['accent'])
    text_sec = hex_to_rgb(theme['text_secondary'])

    colors = [
        ("#000000", "Black"),
        ("#E94560", "Red"),
        ("#3498DB", "Blue"),
        ("#2ECC71", "Green"),
        ("#F39C12", "Orange"),
    ]

    circle_r = 20 * S  # radius
    gap = 16 * S
    total_w = len(colors) * circle_r * 2 + (len(colors) - 1) * gap
    start_x = (W - total_w) // 2 + circle_r

    for i, (hex_color, label) in enumerate(colors):
        cx = start_x + i * (circle_r * 2 + gap)
        cy = y_top + circle_r

        color = hex_to_rgb(hex_color)

        # Active indicator (first color = Black is active)
        if i == 0:
            # Outer ring + glow
            draw.ellipse([cx - circle_r - 6*S, cy - circle_r - 6*S,
                          cx + circle_r + 6*S, cy + circle_r + 6*S],
                         outline=accent, width=3*S)

        # Circle fill
        draw.ellipse([cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
                     fill=color)

        # Red is shown as ring only in reference
        if hex_color == "#E94560":
            draw.ellipse([cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
                         fill=None, outline=color, width=3*S)
            # Fill interior with container color to make it look hollow
            inner_r = circle_r - 4*S
            draw.ellipse([cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
                         fill=hex_to_rgb(theme['background']))
            # Re-draw outline
            draw.ellipse([cx - circle_r, cy - circle_r, cx + circle_r, cy + circle_r],
                         outline=color, width=3*S)

        # Label
        lbl_bbox = draw.textbbox((0, 0), label, font=font_palette_label)
        lbl_tw = lbl_bbox[2] - lbl_bbox[0]
        draw.text((cx - lbl_tw // 2, cy + circle_r + 4 * S),
                  label, font=font_palette_label, fill=text_sec)

    return y_top + circle_r * 2 + 4 * S + 14 * S  # Below labels


def draw_brush_size(draw, theme, y_top, pad_x):
    """Draw Brush Size label + slider + value."""
    accent = hex_to_rgb(theme['accent'])
    text_sec = hex_to_rgb(theme['text_secondary'])

    # Label
    draw.text((pad_x + 4*S, y_top), "Brush Size", font=font_brush_label, fill=text_sec)

    # Value "3" on right
    val_bbox = draw.textbbox((0, 0), "3", font=font_brush_val)
    val_tw = val_bbox[2] - val_bbox[0]
    draw.text((W - pad_x - 4*S - val_tw, y_top), "3", font=font_brush_val, fill=accent)

    # Slider track
    slider_y = y_top + 22 * S
    track_h = 4 * S
    track_left = pad_x
    track_right = W - pad_x
    thumb_x = track_left + int((track_right - track_left) * 0.15)  # ~15% position

    # Filled portion (accent)
    draw_rounded_rect(draw, [track_left, slider_y, thumb_x, slider_y + track_h],
                       track_h // 2, fill=accent)
    # Unfilled portion (muted)
    draw_rounded_rect(draw, [thumb_x, slider_y, track_right, slider_y + track_h],
                       track_h // 2, fill=hex_to_rgb(theme['text_muted']))

    # Thumb
    thumb_r = 10 * S
    thumb_cy = slider_y + track_h // 2
    draw.ellipse([thumb_x - thumb_r, thumb_cy - thumb_r,
                  thumb_x + thumb_r, thumb_cy + thumb_r],
                 fill=(255, 255, 255))

    return slider_y + track_h + 8 * S


def draw_canvas_area(draw, img, theme, y_top, pad_x, canvas_h):
    """Draw white canvas with sample sketch."""
    canvas_left = pad_x
    canvas_right = W - pad_x
    canvas_radius = 10 * S

    # White canvas background with rounded corners
    draw_rounded_rect(draw, [canvas_left, y_top, canvas_right, y_top + canvas_h],
                       canvas_radius, fill=(255, 255, 255))

    # Draw sample house sketch on canvas
    cx_off = canvas_left
    cy_off = y_top
    cw = canvas_right - canvas_left
    ch = canvas_h

    # Scale everything relative to canvas
    def cx(frac): return int(cx_off + cw * frac)
    def cy(frac): return int(cy_off + ch * frac)

    stroke_color = (40, 40, 40)
    stroke_w = 3 * S

    # House body
    house_left = cx(0.25)
    house_right = cx(0.55)
    house_top = cy(0.38)
    house_bottom = cy(0.72)
    draw.rectangle([house_left, house_top, house_right, house_bottom],
                    outline=stroke_color, width=stroke_w)

    # Roof (triangle)
    roof_peak_x = (house_left + house_right) // 2
    roof_peak_y = cy(0.22)
    draw.line([house_left, house_top, roof_peak_x, roof_peak_y],
              fill=stroke_color, width=stroke_w)
    draw.line([roof_peak_x, roof_peak_y, house_right, house_top],
              fill=stroke_color, width=stroke_w)

    # Door (rectangle inside house)
    door_left = cx(0.32)
    door_right = cx(0.40)
    door_top = cy(0.55)
    door_bottom = house_bottom
    draw.rectangle([door_left, door_top, door_right, door_bottom],
                    outline=stroke_color, width=stroke_w)

    # Window (circle)
    win_cx = cx(0.47)
    win_cy = cy(0.50)
    win_r = int(cw * 0.04)
    draw.ellipse([win_cx - win_r, win_cy - win_r, win_cx + win_r, win_cy + win_r],
                 outline=stroke_color, width=stroke_w)

    # Tree (green circle + brown trunk) on left
    tree_cx = cx(0.15)
    tree_top_cy = cy(0.42)
    tree_r = int(cw * 0.07)
    # Trunk (brown line)
    draw.line([tree_cx, tree_top_cy + tree_r, tree_cx, cy(0.72)],
              fill=(101, 67, 33), width=3*S)
    # Foliage (green circle)
    draw.ellipse([tree_cx - tree_r, tree_top_cy - tree_r,
                  tree_cx + tree_r, tree_top_cy + tree_r],
                 outline=(0, 128, 0), width=stroke_w)

    # Red curve on right
    red_color = (233, 69, 96)
    rc_cx = cx(0.70)
    rc_cy = cy(0.55)
    rc_r = int(cw * 0.06)
    draw.arc([rc_cx - rc_r, rc_cy - rc_r, rc_cx + rc_r, rc_cy + rc_r],
             30, 300, fill=red_color, width=stroke_w)

    return y_top + canvas_h


def draw_action_buttons(draw, theme, y_top, pad_x):
    """Draw Undo, Clear, Upload buttons."""
    container = hex_to_rgb(theme['container'])
    border = hex_to_rgb(theme['border'])
    accent = hex_to_rgb(theme['accent'])
    text_pri = hex_to_rgb(theme['text_primary'])

    btn_gap = 10 * S
    btn_h = 48 * S
    btn_radius = 10 * S
    available_w = W - 2 * pad_x
    btn_w = (available_w - 2 * btn_gap) // 3

    buttons = [
        ("Undo", "outline"),
        ("Clear", "outline"),
        ("Upload", "primary"),
    ]

    for i, (label, style) in enumerate(buttons):
        bx = pad_x + i * (btn_w + btn_gap)

        if style == "outline":
            draw_rounded_rect(draw, [bx, y_top, bx + btn_w, y_top + btn_h],
                               btn_radius, fill=container, outline=border, width=2)
            # Draw unicode icon + text
            icon = "\u25A1 " if label == "Undo" else "\u25A1 "
            text = f"{label}"
            text_bbox = draw.textbbox((0, 0), text, font=font_btn)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            draw.text((bx + (btn_w - tw) // 2, y_top + (btn_h - th) // 2),
                      text, font=font_btn, fill=text_pri)
        else:
            draw_rounded_rect(draw, [bx, y_top, bx + btn_w, y_top + btn_h],
                               btn_radius, fill=accent)
            text = f"{label}"
            text_bbox = draw.textbbox((0, 0), text, font=font_btn)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            draw.text((bx + (btn_w - tw) // 2, y_top + (btn_h - th) // 2),
                      text, font=font_btn, fill=(255, 255, 255))

    # Add small icons before text
    # Undo arrow icon (simplified)
    undo_bx = pad_x
    icon_y = y_top + btn_h // 2
    icon_size = 8 * S
    # Small undo arrow
    undo_cx = undo_bx + btn_w // 2 - 25 * S

    # Upload arrow
    upload_bx = pad_x + 2 * (btn_w + btn_gap)
    upload_cx = upload_bx + btn_w // 2 - 25 * S
    arrow_top = icon_y - icon_size
    arrow_bot = icon_y + icon_size // 2

    return y_top + btn_h


def draw_tab_icon_draw(draw, cx, cy, size, color):
    """Draw a paintbrush icon."""
    # Simplified brush: angled rectangle + tip
    s = size
    # Brush body (angled line)
    draw.line([cx - s//3, cy + s//3, cx + s//4, cy - s//3], fill=color, width=max(3, s//5))
    # Brush tip
    draw.ellipse([cx - s//3 - s//6, cy + s//3 - s//6,
                  cx - s//3 + s//6, cy + s//3 + s//6], fill=color)


def draw_tab_icon_settings(draw, cx, cy, size, color):
    """Draw a gear icon (simplified)."""
    s = size
    r_outer = s // 2
    r_inner = s // 3
    # Outer circle
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
                 outline=color, width=max(2, s//8))
    # Inner circle
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
                 outline=color, width=max(2, s//8))
    # Gear teeth (small lines radiating out) — simplified as points on circle
    import math
    for angle_deg in range(0, 360, 45):
        angle = math.radians(angle_deg)
        x1 = int(cx + r_inner * math.cos(angle))
        y1 = int(cy + r_inner * math.sin(angle))
        x2 = int(cx + (r_outer + s//8) * math.cos(angle))
        y2 = int(cy + (r_outer + s//8) * math.sin(angle))
        draw.line([x1, y1, x2, y2], fill=color, width=max(2, s//7))


def draw_tab_icon_preview(draw, cx, cy, size, color):
    """Draw a 3D cube icon."""
    s = size // 2
    # Simple cube outline
    # Front face
    pts_front = [(cx - s, cy), (cx, cy - s), (cx + s, cy), (cx, cy + s)]
    draw.polygon(pts_front, outline=color)
    # Top
    draw.line([cx - s, cy, cx, cy - s], fill=color, width=max(2, s//5))
    draw.line([cx, cy - s, cx + s, cy], fill=color, width=max(2, s//5))
    # Bottom
    draw.line([cx - s, cy, cx, cy + s], fill=color, width=max(2, s//5))
    draw.line([cx, cy + s, cx + s, cy], fill=color, width=max(2, s//5))
    # Center vertical
    draw.line([cx, cy - s, cx, cy + s], fill=color, width=max(1, s//8))


def draw_tab_bar(draw, theme, y_top):
    """Draw bottom tab bar with Draw, Settings, Preview."""
    container = hex_to_rgb(theme['container'])
    border = hex_to_rgb(theme['border'])
    accent = hex_to_rgb(theme['accent'])
    text_muted = hex_to_rgb(theme['text_muted'])

    # Background
    draw.rectangle([0, y_top, W, H], fill=container)
    # Top border
    draw.line([0, y_top, W, y_top], fill=border, width=2)

    tabs = [
        ("Draw", True),
        ("Settings", False),
        ("Preview", False),
    ]

    tab_w = W // 3
    icon_size = 24 * S

    for i, (label, is_active) in enumerate(tabs):
        tx_center = i * tab_w + tab_w // 2
        color = accent if is_active else text_muted

        # Icon area
        icon_cy = y_top + 20 * S

        if label == "Draw":
            # Paint brush — simple diagonal line + dot
            draw.line([tx_center - 4*S, icon_cy + 6*S, tx_center + 6*S, icon_cy - 6*S],
                      fill=color, width=3*S)
            draw.ellipse([tx_center - 6*S, icon_cy + 4*S, tx_center - 2*S, icon_cy + 8*S],
                         fill=color)
        elif label == "Settings":
            # Gear - simplified as circle with radiating lines
            gr = 8 * S
            draw.ellipse([tx_center - gr, icon_cy - gr, tx_center + gr, icon_cy + gr],
                         outline=color, width=2*S)
            import math
            for a in range(0, 360, 45):
                rad = math.radians(a)
                x1 = int(tx_center + (gr - 2*S) * math.cos(rad))
                y1 = int(icon_cy + (gr - 2*S) * math.sin(rad))
                x2 = int(tx_center + (gr + 3*S) * math.cos(rad))
                y2 = int(icon_cy + (gr + 3*S) * math.sin(rad))
                draw.line([x1, y1, x2, y2], fill=color, width=2*S)
            # Center dot
            draw.ellipse([tx_center - 3*S, icon_cy - 3*S, tx_center + 3*S, icon_cy + 3*S],
                         fill=color)
        elif label == "Preview":
            # 3D box
            s = 8 * S
            # Diamond/cube shape
            pts = [(tx_center, icon_cy - s), (tx_center + s, icon_cy),
                   (tx_center, icon_cy + s), (tx_center - s, icon_cy)]
            draw.polygon(pts, outline=color)
            draw.line([tx_center - s, icon_cy, tx_center + s, icon_cy], fill=color, width=1)
            draw.line([tx_center, icon_cy - s, tx_center, icon_cy + s], fill=color, width=1)

        # Label
        lbl_bbox = draw.textbbox((0, 0), label, font=font_tab)
        lbl_tw = lbl_bbox[2] - lbl_bbox[0]
        draw.text((tx_center - lbl_tw // 2, icon_cy + 14 * S),
                  label, font=font_tab, fill=color)

    # Home indicator
    indicator_w = 134 * S
    indicator_h = 5 * S
    indicator_y = H - 20 * S
    indicator_x = (W - indicator_w) // 2
    draw_rounded_rect(draw, [indicator_x, indicator_y, indicator_x + indicator_w, indicator_y + indicator_h],
                       indicator_h // 2, fill=hex_to_rgb(theme['text_muted']))


def generate_mockup(theme, output_path):
    """Generate a full Draw tab mockup with the given theme."""
    bg = hex_to_rgb(theme['background'])
    container = hex_to_rgb(theme['container'])
    border = hex_to_rgb(theme['border'])
    text_pri = hex_to_rgb(theme['text_primary'])

    img = Image.new('RGB', (W, H), bg)
    draw = ImageDraw.Draw(img)

    pad_x = 16 * S  # Horizontal padding

    # --- Status Bar ---
    status_bar_h = 54 * S
    draw_status_bar(draw, theme, 0)

    # --- Nav Bar ---
    nav_y = status_bar_h
    nav_h = 44 * S
    draw.rectangle([0, nav_y, W, nav_y + nav_h], fill=container)
    draw.line([0, nav_y + nav_h, W, nav_y + nav_h], fill=border, width=2)
    text_center_x(draw, "sketch2stl", font_title, nav_y + (nav_h - 17*S) // 2, text_pri)

    # --- Content area ---
    content_y = nav_y + nav_h + 12 * S

    # Segmented Control
    seg_bottom = draw_segmented_control(draw, theme, content_y, pad_x)

    # Color Palette
    palette_y = seg_bottom + 12 * S
    palette_bottom = draw_palette(draw, theme, palette_y)

    # Brush Size
    brush_y = palette_bottom + 8 * S
    brush_bottom = draw_brush_size(draw, theme, brush_y, pad_x)

    # Canvas
    canvas_y = brush_bottom + 8 * S
    # Tab bar starts from bottom
    tab_bar_h = 50 * S + 34 * S  # tab bar + safe area
    tab_bar_y = H - tab_bar_h
    # Action buttons height
    action_h = 48 * S
    action_gap = 12 * S
    # Canvas fills remaining space
    canvas_bottom_limit = tab_bar_y - action_gap - action_h - action_gap
    canvas_h = canvas_bottom_limit - canvas_y

    canvas_bottom = draw_canvas_area(draw, img, theme, canvas_y, pad_x, canvas_h)

    # Action Buttons
    action_y = canvas_bottom + action_gap
    draw_action_buttons(draw, theme, action_y, pad_x)

    # --- Tab Bar ---
    draw_tab_bar(draw, theme, tab_bar_y)

    img.save(output_path, 'PNG')
    print(f"Saved: {output_path} ({img.size[0]}x{img.size[1]})")


# === Theme Definitions ===

theme_clay = {
    'name': 'Clay Studio',
    'background': '#1A1410',
    'container': '#2A2118',
    'border': '#3D3228',
    'accent': '#D4A574',
    'accent_dark': '#B88A5E',
    'text_primary': '#F5F1ED',
    'text_secondary': '#CAB8A8',
    'text_muted': '#8B7D73',
    'segmented_bg': '#12100C',
}

theme_moonlight = {
    'name': 'Moonlight',
    'background': '#0F0F12',
    'container': '#1A1A1F',
    'border': '#2A2A32',
    'accent': '#7C9EFF',
    'accent_dark': '#6484E0',
    'text_primary': '#E8E8EB',
    'text_secondary': '#A8A8B3',
    'text_muted': '#6B6B75',
    'segmented_bg': '#0A0A0E',
}

theme_electric = {
    'name': 'Electric Forge',
    'background': '#0A0E27',
    'container': '#1A1F3A',
    'border': '#2D3B5F',
    'accent': '#00D9FF',
    'accent_dark': '#00B8D9',
    'text_primary': '#FFFFFF',
    'text_secondary': '#B8BFFF',
    'text_muted': '#7580B8',
    'segmented_bg': '#070A1E',
}

# === Generate ===
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend')

generate_mockup(theme_clay, os.path.join(out_dir, 'opus-ui-1.png'))
generate_mockup(theme_moonlight, os.path.join(out_dir, 'opus-ui-2.png'))
generate_mockup(theme_electric, os.path.join(out_dir, 'opus-ui-3.png'))

print("\nAll 3 mockups generated!")
