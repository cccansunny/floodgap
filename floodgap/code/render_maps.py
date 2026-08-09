"""
FloodGap - map rendering
========================
Reads output/houston_robust.csv (written by floodgap_compound.py) and renders:

  output/scheme_A.png ... scheme_E.png   one map per weighting scheme
  output/scheme_overlay.png              all five schemes overlaid (robustness)
  output/zoom_texas_city.png             cluster zoom-ins with street labels,
  output/zoom_gulfton.png                north arrow and scale bar; each cluster
  output/zoom_east_end.png               is shown under its most favorable scheme
  output/zoom_kashmere.png

The basemap is the C-CAP land-cover raster recolored to look satellite-like
(water dark blue, developed grey, vegetation green), so everything renders
offline from local data.

Run:   cd floodgap
       python code/render_maps.py

Requires: pandas, numpy, rasterio, pyproj, matplotlib
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ===========================================================================
# CONFIG
# ===========================================================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "output")
ROBUST_CSV = os.path.join(OUT, "houston_robust.csv")
CCAP_TIF   = os.path.join(DATA, "Impervious", "2021_CCAP_J1414090tR0_C0.tif")

# C-CAP class value -> display RGB (satellite-like recoloring)
LUT = np.zeros((26, 3), dtype=np.uint8)
LUT[0]  = (12, 18, 26)      # nodata -> near-black
LUT[1]  = (30, 34, 40)
LUT[2]  = (205, 205, 203)   # developed high -> light grey
LUT[3]  = (165, 165, 163)
LUT[4]  = (128, 128, 122)
LUT[5]  = (120, 134, 98)    # developed open -> grey-green
LUT[6]  = (96, 82, 56)      # cultivated -> brown
LUT[7]  = (128, 118, 80)
LUT[8]  = (114, 114, 76)
LUT[9]  = (48, 78, 36)      # forest -> green
LUT[10] = (32, 60, 28)
LUT[11] = (44, 72, 34)
LUT[12] = (76, 76, 48)
LUT[13] = (38, 70, 58)      # wetlands -> dark green-teal
LUT[14] = (44, 76, 60)
LUT[15] = (58, 88, 76)
LUT[16] = (42, 70, 70)
LUT[17] = (48, 76, 76)
LUT[18] = (54, 86, 84)
LUT[19] = (110, 102, 88)
LUT[20] = (92, 92, 82)
LUT[21] = (20, 38, 58)      # open water -> dark blue
LUT[22] = (26, 48, 68)
LUT[23] = (28, 52, 72)
LUT[24] = (122, 122, 122)
LUT[25] = (205, 222, 232)

SNAMES = {"A": "Tidal-only", "B": "Equal sub-weights", "C": "Harvey-informed",
          "D": "Flood-dominant 80/20", "E": "Equity-forward"}

# Regions used for map annotations (lat_min, lat_max, lon_min, lon_max)
REGIONS = {
    "Gulfton / Sharpstown":    (29.69, 29.735, -95.56, -95.445),
    "Galveston Island":        (29.25, 29.33, -94.88, -94.76),
    "Texas City / La Marque":  (29.35, 29.42, -94.98, -94.88),
    "Dickinson":               (29.42, 29.50, -95.10, -94.98),
    "Greenspoint":             (29.92, 29.98, -95.45, -95.38),
    "East End / Ship Channel": (29.70, 29.76, -95.30, -95.22),
    "Kashmere / Fifth Ward":   (29.78, 29.83, -95.35, -95.28),
    "Clear Lake / Webster":    (29.52, 29.58, -95.15, -95.05),
}

# Cluster zoom maps: each cluster under its MOST FAVORABLE scheme, with a
# tight bounding box around its high-scoring tracts and local street labels.
# labels: (text, lat, lon, kind) with kind in {city, water, road}
CLUSTERS = {
    "texas_city": {
        "name": "Texas City / La Marque", "scheme": "A", "frame": "#2E6DA4",
        "box": (29.335, 29.425, -95.00, -94.885),
        "labels": [("Texas City", 29.383, -94.902, "city"),
                   ("La Marque", 29.365, -94.965, "city"),
                   ("Galveston Bay", 29.415, -94.895, "water"),
                   ("Moses Lake", 29.352, -94.908, "water"),
                   ("Hwy 146", 29.395, -94.892, "road")]},
    "gulfton": {
        "name": "Gulfton / Sharpstown", "scheme": "C", "frame": "#B5532E",
        "box": (29.672, 29.748, -95.585, -95.468),
        "labels": [("Gulfton", 29.717, -95.497, "city"),
                   ("Sharpstown", 29.706, -95.545, "city"),
                   ("Bellaire Blvd", 29.706, -95.518, "road"),
                   ("US-59", 29.73, -95.528, "road"),
                   ("Brays Bayou", 29.699, -95.508, "water"),
                   ("Hillcroft Ave", 29.722, -95.478, "road")]},
    "east_end": {
        "name": "East End / Ship Channel", "scheme": "D", "frame": "#534AB7",
        "box": (29.683, 29.772, -95.325, -95.195),
        "labels": [("East End", 29.735, -95.298, "city"),
                   ("Manchester", 29.72, -95.265, "city"),
                   ("Buffalo Bayou", 29.755, -95.258, "water"),
                   ("Ship Channel", 29.735, -95.235, "water"),
                   ("Broadway St", 29.712, -95.272, "road")]},
    "kashmere": {
        "name": "Kashmere / Fifth Ward", "scheme": "B", "frame": "#B5532E",
        "box": (29.768, 29.838, -95.362, -95.278),
        "labels": [("Kashmere Gardens", 29.812, -95.315, "city"),
                   ("Fifth Ward", 29.783, -95.33, "city"),
                   ("Hunting Bayou", 29.807, -95.30, "water"),
                   ("Lockwood Dr", 29.80, -95.322, "road"),
                   ("I-610", 29.828, -95.31, "road")]},
}
LABEL_COLORS = {"city": "#0C3D5E", "water": "#1C6EA4", "road": "#7A4A10"}

TR = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)


# ===========================================================================
# Shared drawing helpers
# ===========================================================================

def north_arrow(ax, X0, X1, Y0, Y1):
    x = X1 - (X1 - X0) * 0.08
    y0 = Y1 - (Y1 - Y0) * 0.18; y1 = Y1 - (Y1 - Y0) * 0.07
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=3.5, mutation_scale=22), zorder=16)
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color="#0C3D5E", lw=1.6, mutation_scale=20), zorder=17)
    ax.text(x, y1 + (Y1 - Y0) * 0.015, "N", fontsize=13, fontweight="bold", color="white",
            ha="center", va="bottom", zorder=17,
            bbox=dict(boxstyle="circle,pad=0.18", fc="#0C3D5E", ec="white", lw=1.2))


def scale_bar(ax, X0, X1, Y0, Y1):
    """EPSG:5070 units are meters, so the bar length is direct."""
    x0 = X0 + (X1 - X0) * 0.06; y0 = Y0 + (Y1 - Y0) * 0.06
    bar = 1000 if (X1 - X0) / 1000 < 6 else 2000
    ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=4, solid_capstyle="butt", zorder=15)
    ax.plot([x0, x0 + bar], [y0, y0], color="#0C3D5E", lw=2, solid_capstyle="butt", zorder=16)
    ax.text(x0 + bar / 2, y0 + (Y1 - Y0) * 0.024, f"{bar // 1000} km", fontsize=10,
            fontweight="bold", color="white", ha="center", va="bottom", zorder=16,
            bbox=dict(boxstyle="round,pad=0.15", fc="#0C3D5E", ec="none", alpha=0.85))


def clip_basemap(src, full, X0, X1, Y0, Y1):
    r0, c0 = src.index(X0, Y1); r1, c1 = src.index(X1, Y0)
    r0, r1 = max(0, r0), min(full.shape[0], r1)
    c0, c1 = max(0, c0), min(full.shape[1], c1)
    return LUT[np.clip(full[r0:r1, c0:c1], 0, 25)]


def place_labels(items, X0, X1, Y0, Y1, clamp):
    """Iterative repulsion so annotation boxes never overlap each other."""
    HW = (X1 - X0) * 0.098; HH = (Y1 - Y0) * 0.040
    pos = [list(p[3]) for p in items]
    for _ in range(120):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dx = pos[j][0] - pos[i][0]; dy = pos[j][1] - pos[i][1]
                ox = 2 * HW - abs(dx); oy = 2 * HH - abs(dy)
                if ox > 0 and oy > 0:
                    moved = True
                    if oy / (2 * HH) < ox / (2 * HW):
                        sft = (oy / 2 + 1) * (1 if dy >= 0 else -1)
                        pos[i][1] -= sft; pos[j][1] += sft
                    else:
                        sft = (ox / 2 + 1) * (1 if dx >= 0 else -1)
                        pos[i][0] -= sft; pos[j][0] += sft
        for p in pos:
            p[0], p[1] = clamp(p[0], p[1])
        if not moved:
            break
    return pos


# ===========================================================================
# Regional scheme maps + overlay
# ===========================================================================

def render_scheme_maps(df, src, full):
    # Regional frame: the study area in EPSG:5070, aspect 1.25
    X0, X1 = 1922.0, 144375.0
    Hh = (X1 - X0) / 1.25
    cy = (df["Y"].min() + df["Y"].max()) / 2
    Y0, Y1 = cy - Hh / 2, cy + Hh / 2
    W_, H_ = X1 - X0, Y1 - Y0
    rgb = clip_basemap(src, full, X0, X1, Y0, Y1)

    def clamp(tx, ty):
        mx0, mx1 = X0 + 0.115 * W_, X1 - 0.125 * W_
        my0, my1 = Y0 + 0.075 * H_, Y1 - 0.155 * H_
        tx = min(max(tx, mx0), mx1); ty = min(max(ty, my0), my1)
        if tx < X0 + 0.32 * W_ and ty < Y0 + 0.22 * H_:   # legend keep-out
            ty = Y0 + 0.22 * H_
        return tx, ty

    # Region centers in projected coordinates
    rc = {}
    for reg, (a, b, c, d) in REGIONS.items():
        x, y = TR.transform((c + d) / 2, (a + b) / 2)
        rc[reg] = (x, y)

    OFF = {"Gulfton / Sharpstown": (-0.25, 0.12), "Galveston Island": (0.12, -0.06),
           "Texas City / La Marque": (-0.21, -0.03), "Dickinson": (0.18, 0.10),
           "Greenspoint": (0.15, 0.06), "East End / Ship Channel": (0.19, -0.09),
           "Kashmere / Fifth Ward": (0.16, 0.11), "Clear Lake / Webster": (0.20, 0.08)}

    cm = plt.get_cmap("autumn_r")
    for key in "ABCDE":
        sc = df[f"sc_{key}"]
        rank = sc.rank(ascending=False)
        top50 = df[rank <= 50]
        fig, ax = plt.subplots(figsize=(12.03, 9.62))
        ax.imshow(rgb, extent=(X0, X1, Y0, Y1), origin="upper", interpolation="bilinear")
        v = (sc - sc.min()) / (sc.max() - sc.min())
        ax.scatter(df["X"], df["Y"], c=v, cmap="autumn_r", s=42, edgecolors="#101010",
                   linewidths=0.4, alpha=0.92, zorder=4)
        t20 = df[rank <= 20]
        ax.scatter(t20["X"], t20["Y"], c=v[rank <= 20], cmap="autumn_r", s=130,
                   edgecolors="white", linewidths=2.2, zorder=6, vmin=0, vmax=1)

        # Count top-50 sites per region and annotate the four leading ones
        counts = {}
        for reg, (a, b, c, d) in REGIONS.items():
            n = len(top50[(top50["lat"] >= a) & (top50["lat"] <= b)
                          & (top50["lon"] >= c) & (top50["lon"] <= d)])
            if n > 0:
                counts[reg] = n
        counts = dict(sorted(counts.items(), key=lambda x: -x[1]))
        items = []
        for reg in list(counts)[:4]:
            cx, cy2 = rc[reg]; ox, oy = OFF.get(reg, (0.18, 0.08))
            items.append((reg, f"{reg}\n{counts[reg]} of top 50", (cx, cy2),
                          clamp(cx + ox * W_, cy2 + oy * H_), "#B5532E"))
        for (reg, txt, anc, _, cc), (tx, ty) in zip(items, place_labels(items, X0, X1, Y0, Y1, clamp)):
            ax.annotate(txt, xy=anc, xytext=(tx, ty), fontsize=11.5, fontweight="bold",
                        color="#0C3D5E", ha="center", va="center", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=cc, lw=1.6, alpha=0.94),
                        arrowprops=dict(arrowstyle="-|>", color=cc, lw=1.9, shrinkA=2, shrinkB=8,
                                        connectionstyle="arc3,rad=0.12"))

        ax.text(0.012, 0.978, f"Scheme {key} - {SNAMES[key]}", transform=ax.transAxes,
                fontsize=15, fontweight="bold", color="#0C3D5E", va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#0C3D5E", lw=1.4, alpha=0.95))
        hh_ = int(top50["COUNTY"].str.contains("Harris").sum())
        ax.text(0.988, 0.978, f"Top 50:   Harris {hh_}   |   Galveston {50 - hh_}",
                transform=ax.transAxes, fontsize=13, fontweight="bold", color="white",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="#0C3D5E", ec="white", lw=1.6, alpha=0.96))
        lg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cm(0.0),
                     markeredgecolor="#101010", markeredgewidth=0.5, markersize=9, label="lower score"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=cm(0.55),
                     markeredgecolor="#101010", markeredgewidth=0.5, markersize=9, label="mid"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=cm(1.0),
                     markeredgecolor="#101010", markeredgewidth=0.5, markersize=9, label="higher score"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor=cm(0.95),
                     markeredgecolor="white", markeredgewidth=2.0, markersize=13, label="top 20 site")]
        ax.legend(handles=lg, loc="lower left", fontsize=10, framealpha=0.93, borderpad=0.7,
                  handletextpad=0.5, labelspacing=0.45)
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        plt.tight_layout(pad=0.2)
        plt.savefig(os.path.join(OUT, f"scheme_{key}.png"), dpi=100,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"scheme_{key}.png")

    # ---- Overlay map: color by how many schemes chose each tract ----
    fig, ax = plt.subplots(figsize=(12.03, 9.62))
    ax.imshow(rgb, extent=(X0, X1, Y0, Y1), origin="upper", interpolation="bilinear")
    colors = {1: "#C99A00", 2: "#FFB03A", 3: "#FF7A1A", 4: "#F03C10", 5: "#C2001E"}
    sizes  = {1: 40, 2: 55, 3: 75, 4: 110, 5: 240}
    for n in [1, 2, 3, 4]:
        g = df[df["n50"] == n]
        ax.scatter(g["X"], g["Y"], c=colors[n], s=sizes[n], edgecolors="white",
                   linewidths=0.8, zorder=4 + n, alpha=0.95)
    g5 = df[df["n50"] == 5]
    ax.scatter(g5["X"], g5["Y"], marker="*", c=colors[5], s=520, edgecolors="white",
               linewidths=1.2, zorder=10)

    def clamp2(tx, ty):
        return clamp(tx, ty)
    ANN = [("Texas City / La Marque", "2 sites in ALL 5 schemes\nthe robust core", (-0.21, -0.03), "#C2001E"),
           ("Gulfton / Sharpstown", "largest robust cluster\nat 4 / 5 schemes", (-0.25, 0.12), "#F03C10"),
           ("Galveston Island", "tidal-driven, drops\nout above 3 / 5", (0.12, -0.06), "#FFB03A"),
           ("East End / Ship Channel", "appears once equity\nweight rises", (0.19, -0.09), "#FF7A1A"),
           ("Kashmere / Fifth Ward", "highest SVI (0.91)\nin the study area", (0.16, 0.11), "#FF7A1A")]
    it2 = [(reg, f"{reg}\n{txt}", rc[reg],
            clamp2(rc[reg][0] + ox * W_, rc[reg][1] + oy * H_), cc)
           for reg, txt, (ox, oy), cc in ANN]
    for (reg, txt, anc, _, cc), (tx, ty) in zip(it2, place_labels(it2, X0, X1, Y0, Y1, clamp2)):
        ax.annotate(txt, xy=anc, xytext=(tx, ty), fontsize=10.8, fontweight="bold",
                    color="#0C3D5E", ha="center", va="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=cc, lw=1.9, alpha=0.95),
                    arrowprops=dict(arrowstyle="-|>", color=cc, lw=1.9, shrinkA=2, shrinkB=8,
                                    connectionstyle="arc3,rad=0.12"))
    ax.text(0.012, 0.978, "All Five Schemes Overlaid - how often each tract makes the top 50",
            transform=ax.transAxes, fontsize=14, fontweight="bold", color="#0C3D5E", va="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#0C3D5E", lw=1.4, alpha=0.95))
    tallies = {n: int((df["n50"] == n).sum()) for n in range(5, 0, -1)}
    lg = ([Line2D([0], [0], marker="*", color="w", markerfacecolor=colors[5], markeredgecolor="white",
                  markersize=17, label=f"5 / 5 schemes  ({tallies[5]} tracts)")] +
          [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[n], markeredgecolor="white",
                  markersize=12 - (5 - n), label=f"{n} / 5  ({tallies[n]} tracts)") for n in [4, 3, 2, 1]] +
          [Line2D([0], [0], marker="o", color="w", markerfacecolor="#9BA0A6", markersize=6,
                  label=f"never in a top 50  ({int((df['n50'] == 0).sum())})")])
    ax.legend(handles=lg, loc="lower left", fontsize=9.5, framealpha=0.93, borderpad=0.7,
              title="Robustness across schemes", title_fontsize=10)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(OUT, "scheme_overlay.png"), dpi=100,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("scheme_overlay.png")


# ===========================================================================
# Cluster zoom-in maps
# ===========================================================================

def render_zooms(df, src, full):
    for k in "ABCDE":
        df[f"r{k}"] = df[f"sc_{k}"].rank(ascending=False).astype(int)

    for key, cfg in CLUSTERS.items():
        sch = cfg["scheme"]; rcname = f"r{sch}"
        a, b, c, d = cfg["box"]
        xs, ys = [], []
        for la, lo in [(a, c), (a, d), (b, c), (b, d)]:
            x, y = TR.transform(lo, la); xs.append(x); ys.append(y)
        X0, X1 = min(xs), max(xs); Y0, Y1 = min(ys), max(ys)
        m = max(X1 - X0, Y1 - Y0)
        cx, cy = (X0 + X1) / 2, (Y0 + Y1) / 2
        X0, X1, Y0, Y1 = cx - m / 2, cx + m / 2, cy - m / 2, cy + m / 2
        rgb = clip_basemap(src, full, X0, X1, Y0, Y1)

        fig, ax = plt.subplots(figsize=(8.4, 8.4))
        ax.imshow(rgb, extent=(X0, X1, Y0, Y1), origin="upper", interpolation="bilinear")

        g = df[(df["lat"] >= a) & (df["lat"] <= b) & (df["lon"] >= c) & (df["lon"] <= d)].copy()
        cat = g[rcname].apply(lambda r: 0 if r <= 20 else 1 if r <= 50 else 2 if r <= 100 else 3)
        styles = {0: ("#E8140C", 430, "white", 2.8), 1: ("#FF7A1A", 250, "white", 2.0),
                  2: ("#FFC53A", 150, "#333333", 1.0)}
        for ct in [2, 1, 0]:                       # only top-100; no grey dots
            gg = g[cat == ct]
            if len(gg) == 0:
                continue
            col, sz, ec, lw = styles[ct]
            ax.scatter(gg["X"], gg["Y"], c=col, s=sz, edgecolors=ec,
                       linewidths=lw, alpha=0.97, zorder=4 + (3 - ct))

        for lab, la, lo, kind in cfg["labels"]:
            x, y = TR.transform(lo, la)
            if not (X0 < x < X1 and Y0 < y < Y1):
                continue
            ax.text(x, y, lab, fontsize=(13 if kind == "city" else 11),
                    fontweight=("bold" if kind in ("city", "water") else "normal"),
                    fontstyle=("italic" if kind == "water" else "normal"),
                    color=LABEL_COLORS[kind], ha="center", va="center", zorder=12,
                    bbox=dict(boxstyle="round,pad=0.28", fc="white",
                              ec=LABEL_COLORS[kind], lw=1.1, alpha=0.9))

        north_arrow(ax, X0, X1, Y0, Y1)
        scale_bar(ax, X0, X1, Y0, Y1)
        ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_edgecolor(cfg["frame"]); sp.set_linewidth(3)
        lg = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#E8140C",
                     markeredgecolor="white", markeredgewidth=1.5, markersize=16, label="Top 20 site"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF7A1A",
                     markeredgecolor="white", markeredgewidth=1.2, markersize=13, label="Top 50 site"),
              Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFC53A",
                     markeredgecolor="#333333", markeredgewidth=0.6, markersize=11, label="Top 100")]
        ax.legend(handles=lg, loc="lower right", fontsize=10, framealpha=0.93, borderpad=0.7,
                  title=f"Scheme {sch} ({SNAMES[sch].split(' ')[0]}) rank", title_fontsize=9.5)
        ax.text(0.5, 1.015, cfg["name"], transform=ax.transAxes, fontsize=16,
                fontweight="bold", color="#0C3D5E", ha="center", va="bottom")
        plt.tight_layout(pad=0.3)
        plt.savefig(os.path.join(OUT, f"zoom_{key}.png"), dpi=120,
                    bbox_inches="tight", facecolor="white")
        plt.close()
        t20 = int((g[rcname] <= 20).sum()); t50 = int((g[rcname] <= 50).sum())
        print(f"zoom_{key}.png  (scheme {sch}: {t20} in top 20, {t50} in top 50)")


# ===========================================================================
# Main
# ===========================================================================

def main():
    df = pd.read_csv(ROBUST_CSV)
    # Projected coordinates for plotting on the EPSG:5070 raster
    if "X" not in df.columns:
        xy = [TR.transform(lo, la) for lo, la in zip(df["lon"], df["lat"])]
        df["X"] = [p[0] for p in xy]; df["Y"] = [p[1] for p in xy]
    src = rasterio.open(CCAP_TIF)
    full = src.read(1)
    render_scheme_maps(df, src, full)
    render_zooms(df, src, full)
    print("\nAll maps written to", OUT)


if __name__ == "__main__":
    main()
