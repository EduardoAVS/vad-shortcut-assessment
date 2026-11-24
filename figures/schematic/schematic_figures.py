#!/usr/bin/env python3
"""Schematic figures of the paper as SVG: Figs. 1, 2, 3, 4, 7 and 9.

  fig1_modules.svg        Fig. 1  the three-module procedure
  fig2_event_control.svg  Fig. 2  event control (uses drawings/event_control_frames.svg)
  fig3_interventions.svg  Fig. 3  pixel interventions (uses drawings/background_sufficiency.svg, event_necessity.svg)
  fig4_architecture.svg   Fig. 4  GRUMobileNet architecture
  fig7_null_models.svg    Fig. 7  µw against the two null models (the numbers are typed below; they come from
                                  sotad_diagnosis/results/box_attribution/20261006-100109/summary.json)
  fig9_diagnosis.svg      Fig. 9  two-axis diagnosis

Style: IBM Plex Sans, one font size (FS) in all figures; titles only change the weight. Common scale: 1600 SVG units
= \\textwidth; narrower figures have a smaller W and go into LaTeX with width = W/1600 \\textwidth, so the text has the
same size in all of them. Text wrapping uses the font metrics of the TTF files (Pillow).

Usage (from the repo root):  python figures/schematic/schematic_figures.py [--fonts DIR] [--out DIR]
  defaults: --fonts figures/fonts, --out figures/output. The PNGs are rendered by render_svg.js (resvg).
"""
import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import ImageFont

HERE = Path(__file__).resolve().parent
DRAWINGS = HERE / "drawings"
FONTS_DIR = HERE.parent / "fonts"
OUT_DIR = HERE.parent / "output"

FONT = "IBM Plex Sans, DejaVu Sans, sans-serif"
INK, INK2, MUTED = "#1C2633", "#47515E", "#5E6773"
BLUE, ORANGE, GRAY = "#2A5F96", "#A54F18", "#9AA3AE"
CARD, BORDER, BG = "#FBFAF7", "#DDD7CB", "#FFFFFF"
FS, LH = 34, 1.3
W = 1600

_WEIGHTS = {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold"}
_fonts = {}


def text_width(txt, weight=400):
    if weight not in _fonts:
        _fonts[weight] = ImageFont.truetype(str(FONTS_DIR / f"IBMPlexSans-{_WEIGHTS[weight]}.ttf"), FS)
    return _fonts[weight].getlength(txt)


def wrap(txt, maxw, weight=400):
    lines, cur = [], ""
    for word in txt.split(" "):
        trial = (cur + " " + word).strip()
        if text_width(trial, weight) <= maxw or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def text(x, y, lines, weight=400, fill=INK, anchor="start"):
    """lines: str or list of segments (text, weight, colour) per line."""
    out = [f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{FS}" font-weight="{weight}" '
           f'fill="{fill}" text-anchor="{anchor}">']
    for i, ln in enumerate(lines):
        dy = 0 if i == 0 else round(FS * LH, 1)
        if isinstance(ln, str):
            out.append(f'<tspan x="{x}" dy="{dy}">{escape(ln)}</tspan>')
        else:
            segs = "".join(f'<tspan font-weight="{w}" fill="{f}">{escape(s)}</tspan>' for s, w, f in ln)
            out.append(f'<tspan x="{x}" dy="{dy}">{segs}</tspan>')
    out.append("</text>")
    return "".join(out)


def height(n):
    return n * FS * LH


def svg(h, body, label, w=W):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{escape(label, {chr(34): "&quot;"})}">'
            f'<rect width="{w}" height="{h}" fill="{BG}"/>{body}</svg>\n')


def card(x, y, w, h, fill=CARD, stroke=BORDER, sw=2):
    return (f'<rect x="{x + sw / 2}" y="{y + sw / 2}" width="{w - sw}" height="{h - sw}" rx="12" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x, y):
    return (f'<polygon points="{x},{y - 7} {x + 26},{y - 7} {x + 26},{y - 15} {x + 40},{y} '
            f'{x + 26},{y + 15} {x + 26},{y + 7} {x},{y + 7}" fill="{GRAY}"/>')


def drawing(name):
    """Inner content of drawings/<name>.svg (inserted into a figure with a transform)."""
    t = (DRAWINGS / f"{name}.svg").read_text(encoding="utf-8")
    return re.findall(r"<svg[^>]*>(.*?)</svg>", t, re.S)[0]


# --- Fig. 1: the three-module procedure ------------------------------------------------------------
def fig_modules():
    aw, pad = 56, 32
    cw = (W - 2 * aw) / 3
    mods = [("1. Data assessment", "Can the label be predicted without the event?"),
            ("2. Attribution against null models", "Does attribution concentrate on the event?"),
            ("3. Pixel interventions", "Does the decision depend on the event?")]
    blocks = [(wrap(t, cw - 2 * pad, 600), wrap(q, cw - 2 * pad)) for t, q in mods]
    ch = max(pad + FS + height(len(t) - 1) + 24 + height(len(q)) + pad for t, q in blocks)
    c, y0 = [], 2
    for i, (title, q) in enumerate(blocks):
        x = i * (cw + aw)
        c.append(card(x, y0, cw, ch))
        c.append(text(x + pad, y0 + pad + FS - 4, title, 600, INK))
        c.append(text(x + pad, y0 + pad + FS - 4 + height(len(title) - 1) + 24 + height(1), q, 400, INK2))
        if i < 2:
            c.append(arrow(x + cw + 8, y0 + ch / 2))
    yb = y0 + ch + 22
    for x1, x2, label in [(16, cw - 16, "data level"), (cw + aw + 16, W - 16, "model level")]:
        c.append(f'<path d="M{x1},{yb} v10 H{x2} v-10" fill="none" stroke="{GRAY}" stroke-width="2"/>')
        c.append(text((x1 + x2) / 2, yb + 52, [label], 500, MUTED, "middle"))
    h = yb + 66
    return svg(h, "".join(c), "Three modules in sequence: 1, data assessment (can the label be predicted without the "
               "event?), at the data level; 2, attribution against null models (does attribution concentrate on the "
               "event?) and 3, pixel interventions (does the decision depend on the event?), at the model level.")


# --- Fig. 2: event control -------------------------------------------------------------------------
def fig_event_control():
    dx, ys = -32, 56
    c = [f'<g transform="translate({dx},{ys})">{drawing("event_control_frames")}</g>']
    xk = 692 + dx
    c.append(text(xk, 38, ["keyframe"], 600, ORANGE, "middle"))
    yl = ys + 240 + 50
    left, right = (80 + 600) / 2, (720 + 1520) / 2
    c.append(text(left, yl, [[("Before the keyframe", 600, INK)], "real video, no mark",
                             [("label ", 400, INK2), ("0", 600, INK)]], 400, INK2, "middle"))
    c.append(text(right, yl, [[("After the keyframe", 600, INK)], "real accident + overlaid checkerboard",
                              [("label ", 400, INK2), ("1", 600, INK)]], 400, INK2, "middle"))
    h = yl + height(2) + 16
    return svg(h, "".join(c), "Five frames of a real accident video with the 4 by 4 grid. Before the keyframe, two "
               "cars approach and there is no mark (label 0). After the keyframe, the cars collide and a checkerboard "
               "appears in one cell of the road, overlaid on the video (label 1).")


# --- Fig. 3: interventions -------------------------------------------------------------------------
def fig_interventions():
    pw, gap, sc, pad = 780, 40, 1.15, 24
    panels = [("Background sufficiency", "Is the video without the event enough?",
               "static background in all 16 frames", "background_sufficiency",
               [[("low", 600, INK), (" retention: needs the event", 400, INK2)],
                [("high", 600, INK), (" retention: the video suffices", 400, INK2)]]),
              ("Event necessity", "Is the event needed?",
               "only the event region is replaced by the background", "event_necessity",
               [[("large", 600, INK), (" drop: the event is needed", 400, INK2)],
                [("small", 600, INK), (" drop: the event is not needed", 400, INK2)]])]
    captions = [wrap(p[2], 340, 600) for p in panels]
    nr = max(len(r) for r in captions)
    yi = pad + FS + height(1) + 30
    yr = yi + 200 * sc + 50
    yl = yr + height(nr) + 30
    h = yl + height(1) + FS / 2 + pad + 8
    c = []
    for i, ((title, sub, _, name, reading), cap) in enumerate(zip(panels, captions)):
        x0 = i * (pw + gap)
        cx = x0 + pw / 2
        c.append(card(x0, 0, pw, h))
        c.append(text(cx, pad + FS, [title], 600, INK, "middle"))
        c.append(text(cx, pad + FS + height(1), [sub], 400, INK2, "middle"))
        gx = x0 + (pw - 520 * sc) / 2
        c.append(f'<g transform="translate({gx},{yi}) scale({sc})">{drawing(name)}</g>')
        c.append(text(gx + 100 * sc, yr, ["original"], 400, INK2, "middle"))
        c.append(text(gx + 420 * sc, yr, cap, 600, INK, "middle"))
        c.append(text(cx, yl, reading, 400, INK2, "middle"))
    return svg(h, "".join(c), "Two panels. Background sufficiency: the accident frame becomes a static background image, "
               "without cars, in all 16 frames; low retention means the model needs the event, high retention means "
               "the video suffices. Event necessity: in the same frame, only the event region is replaced by the "
               "background and the other car remains; a large drop means the event is needed.")


# --- Fig. 4: GRUMobileNet architecture -------------------------------------------------------------
def fig_architecture():
    aw = 60                      # room for each arrow
    y0, ch = 2, 230              # top and height of the cards
    cy = y0 + ch / 2
    c = []

    # 1. window of 16 frames (stack of frames)
    x = 0
    fw, fh = 150, 112
    for k in range(3, -1, -1):
        fx, fy = x + 10 + 14 * k, cy - fh / 2 - 34 + 14 * k
        c.append(f'<rect x="{fx}" y="{fy}" width="{fw}" height="{fh}" rx="4" fill="#E4E8ED" stroke="{GRAY}" stroke-width="2"/>')
    c.append(text(x + 10 + fw / 2 + 21, y0 + ch + 2, ["16 frames"], 600, INK, "middle"))
    c.append(text(x + 10 + fw / 2 + 21, y0 + ch + 2 + height(1), ["128×128 px"], 400, INK2, "middle"))
    x = 10 + fw + 42 + 16
    c.append(arrow(x + 12, cy))
    x += aw

    # 2. MobileNetV3-Small, frozen
    w2 = 370
    c.append(card(x, y0, w2, ch))
    c.append(text(x + 32, y0 + 32 + FS - 4, ["MobileNetV3-Small"], 600, INK))
    c.append(text(x + 32, y0 + 32 + FS - 4 + height(1) + 16, ["pretrained, frozen", "applied to each frame"], 400, INK2))
    x += w2
    c.append(arrow(x + 12, cy))
    x += aw

    # 3. sequence of feature vectors
    bw, gap, bh = 9, 5, 120
    for k in range(16):
        bx = x + 6 + k * (bw + gap)
        c.append(f'<rect x="{bx}" y="{cy - bh / 2}" width="{bw}" height="{bh}" rx="2" fill="{GRAY}"/>')
    vw = 16 * (bw + gap) - gap + 12
    c.append(text(x + vw / 2, y0 + ch + 2, ["16 vectors"], 600, INK, "middle"))
    c.append(text(x + vw / 2, y0 + ch + 2 + height(1), ["576 values each"], 400, INK2, "middle"))
    x += vw + 30
    c.append(arrow(x + 12 - 30 + 18, cy))
    x += aw - 12

    # 4. GRU + linear layer, trained
    w4 = 360
    c.append(card(x, y0, w4, ch, stroke=BLUE))
    c.append(text(x + 32, y0 + 32 + FS - 4, ["GRU (128 units)"], 600, INK))
    c.append(text(x + 32, y0 + 32 + FS - 4 + height(1) + 16, ["+ linear output", "layer, trained"], 400, INK2))
    x += w4
    c.append(arrow(x + 12, cy))
    x += aw

    # 5. score
    c.append(text(x + 4, cy - 6, ["accident"], 600, INK))
    c.append(text(x + 4, cy - 6 + height(1), ["score"], 600, INK))
    c.append(text(x + 4, cy - 6 + height(2), ["(0 to 1)"], 400, INK2))
    assert x + 150 <= W, x

    h = y0 + ch + 2 + height(1) + 20
    return svg(round(h, 1), "".join(c),
               "GRUMobileNet: a window of 16 frames of 128 by 128 pixels passes, frame by frame, through "
               "MobileNetV3-Small, pretrained and frozen, which produces 16 vectors of 576 values; a GRU with 128 "
               "units and a linear output layer, both trained, produce the accident score of the window, from 0 to 1.")


# --- Fig. 7: µw against the two null models --------------------------------------------------------
def fig_null_models():
    w = 1100
    yb, s, bw = 340, 160, 150
    bars = [(190, 1.64, (1.46, 1.82), BLUE, "GRUMobileNet", "1.64 [1.46, 1.82]"),
            (550, 0.67, (0.56, 0.78), GRAY, "Null 1", "0.67 [0.56, 0.78]"),
            (910, 0.85, (0.71, 1.00), GRAY, "Null 2", "0.85 [0.71, 1.00]")]
    c = []
    xl, xr = 40, w - 40
    c.append(f'<line x1="{xl}" y1="{yb - s}" x2="{xr}" y2="{yb - s}" stroke="{ORANGE}" stroke-width="3" stroke-dasharray="12 8"/>')
    for cx, v, (lo, hi), colour, name, val in bars:
        x0, top, r = cx - bw / 2, yb - v * s, 4
        c.append(f'<path d="M{x0},{yb} V{top + r} Q{x0},{top} {x0 + r},{top} H{x0 + bw - r} '
                 f'Q{x0 + bw},{top} {x0 + bw},{top + r} V{yb} Z" fill="{colour}"/>')
        y1, y2 = yb - hi * s, yb - lo * s
        c.append(f'<g stroke="{INK}" stroke-width="3"><line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}"/>'
                 f'<line x1="{cx - 14}" y1="{y1}" x2="{cx + 14}" y2="{y1}"/>'
                 f'<line x1="{cx - 14}" y1="{y2}" x2="{cx + 14}" y2="{y2}"/></g>')
        c.append(text(cx, yb + 50, [name], 400, INK2, "middle"))
        c.append(text(cx, yb + 50 + height(1), [val], 600, INK, "middle"))
    c.append(text(xr, yb - s - 14, ["chance"], 600, INK2, "end"))
    c.append(f'<line x1="{xl}" y1="{yb}" x2="{xr}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    h = yb + 50 + height(1) + 16
    return svg(h, "".join(c), "Bars of Grad-CAM µw on the annotated accident box: GRUMobileNet 1.64 [1.46, 1.82]; "
               "null 1 0.67 [0.56, 0.78]; null 2 0.85 [0.71, 1.00]. Dashed line at chance, µw equal to 1.", w)


# --- Fig. 9: two-axis diagnosis --------------------------------------------------------------------
def fig_diagnosis():
    c0w, gap, pad = 420, 20, 30
    cw = (W - c0w - 2 * gap) / 2
    x1, x2 = c0w + gap, c0w + gap + cw + gap
    c = [text(x1 + cw / 2, 40, ["Decision uses the event"], 600, INK, "middle"),
         text(x2 + cw / 2, 40, ["Decision uses other", "characteristics of the video"], 600, INK, "middle")]
    rows = [("Attribution on the event", "µw above the null models",
             [("Event control", "Map and decision on the event", False),
              ("GRUMobileNet on SO-TAD", "µw 1.64; 94% of the score with the static background", True)]),
            ("Attribution not on the event", "µw at the null-model level",
             [("None of the evaluated models", "This case would indicate a poorly estimated event region", False),
              ("Video control; GRUMobileNet on TAD", "µw at the null-model level; at least 90% of the score with the static background", False)])]
    y = 72 + height(1)
    for label, sub, cells in rows:
        bodies = [wrap(body, cw - 2 * pad) for _, body, _ in cells]
        titles = [wrap(title, cw - 2 * pad, 600) for title, _, _ in cells]
        nt = max(len(t) for t in titles)
        rh = pad + FS + 12 + height(nt - 1) + height(max(len(b) for b in bodies)) + pad - 8
        rl = wrap(label, c0w - 10, 600)
        sl = wrap(sub, c0w - 10)
        yr = y + rh / 2 - height(len(rl) + len(sl)) / 2 + FS / 2
        c.append(text(0, yr, rl, 600, INK))
        c.append(text(0, yr + height(len(rl)), sl, 400, INK2))
        for k, ((_, _, highlight), title, body) in enumerate(zip(cells, titles, bodies)):
            x = x1 if k == 0 else x2
            c.append(card(x, y, cw, rh, fill=BG, stroke=INK if highlight else BORDER, sw=4 if highlight else 2))
            c.append(text(x + pad, y + pad + FS - 4, title, 600, INK))
            c.append(text(x + pad, y + pad + FS - 4 + 12 + height(len(title)), body, 400, INK2))
        y += rh + gap
    c = [re.sub(r"µw(\s| )", lambda m: '<tspan font-style="italic">µ</tspan><tspan dy="8" font-size="24" '
                f'font-style="italic">w</tspan><tspan dy="-8">{m.group(1)}</tspan>', t) for t in c]
    return svg(y, "".join(c), "Two by two grid. Rows: attribution on the event (µw above the null models) or not on the "
               "event (µw at the level of the null models). Columns: according to the interventions, the decision uses "
               "the event or other characteristics of the video. Attribution and decision on the event: the event "
               "control. Attribution on the event, decision on other characteristics of the video: GRUMobileNet on "
               "SO-TAD, with µw 1.64 and a score of 94% of the original with the static background. Attribution not on "
               "the event, decision on the event: none of the evaluated models; this case would indicate a poorly estimated event region. "
               "Attribution not on the event, decision on other characteristics of the video: the video control and "
               "GRUMobileNet on TAD, with µw at the level of the null models and at least 90% of the score with "
               "the static background.")


FIGS = {"fig1_modules": fig_modules, "fig2_event_control": fig_event_control, "fig3_interventions": fig_interventions,
        "fig4_architecture": fig_architecture, "fig7_null_models": fig_null_models, "fig9_diagnosis": fig_diagnosis}


def main():
    global FONTS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts", default=str(FONTS_DIR), help="folder with IBMPlexSans-*.ttf")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder")
    args = ap.parse_args()
    FONTS_DIR = Path(args.fonts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, f in FIGS.items():
        p = out / f"{name}.svg"
        p.write_text(f(), encoding="utf-8")
        print("ok", p)


if __name__ == "__main__":
    main()
