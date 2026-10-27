"""Turn results/*.json into SVG charts and Markdown tables (stdlib only).

    ring-ttft-report --results results --assets assets
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

INK, MUTED, GRID, BG = "#1f2328", "#59636e", "#d8dee4", "#ffffff"
SERIES = ["#2f6fdb", "#d1661f", "#2a9d6f", "#8a4fd0", "#c43d5b"]
FONT = "font-family='-apple-system,Segoe UI,Helvetica,Arial,sans-serif'"


def _nice_step(raw: float) -> float:
    """Round a tick step up to 1, 2, 2.5 or 5 times a power of ten."""
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Chart:
    def __init__(self, title: str, subtitle: str, w=720, h=420, left=64, right=150, top=70, bottom=56):
        self.w, self.h, self.l, self.r, self.t, self.b = w, h, left, right, top, bottom
        self.parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' {FONT}>",
            f"<rect width='{w}' height='{h}' fill='{BG}'/>",
            f"<text x='{left}' y='30' font-size='17' font-weight='600' fill='{INK}'>{_esc(title)}</text>",
            f"<text x='{left}' y='50' font-size='12.5' fill='{MUTED}'>{_esc(subtitle)}</text>",
        ]

    @property
    def pw(self):
        return self.w - self.l - self.r

    @property
    def ph(self):
        return self.h - self.t - self.b

    def y_axis(self, vmax: float, label: str, ticks: int = 5, fmt="{:g}", vmin: float = 0.0):
        step = _nice_step((vmax - vmin) / ticks)
        ticks = math.ceil((vmax - vmin) / step - 1e-9)
        self.vmin, self.vmax = vmin, vmin + step * ticks
        for i in range(ticks + 1):
            v = vmin + step * i
            y = self.t + self.ph * (1 - i / ticks)
            self.parts.append(f"<line x1='{self.l}' x2='{self.l + self.pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}'/>")
            self.parts.append(f"<text x='{self.l - 8}' y='{y + 4:.1f}' font-size='11' fill='{MUTED}' text-anchor='end'>{fmt.format(v)}</text>")
        cy = self.t + self.ph / 2
        self.parts.append(f"<text x='16' y='{cy}' font-size='12' fill='{MUTED}' transform='rotate(-90 16 {cy})' text-anchor='middle'>{_esc(label)}</text>")

    def y(self, v):
        return self.t + self.ph * (1 - (v - self.vmin) / (self.vmax - self.vmin))

    def x_labels(self, labels, label):
        self.n = len(labels)
        for i, lab in enumerate(labels):
            self.parts.append(f"<text x='{self.xc(i):.1f}' y='{self.t + self.ph + 18}' font-size='11.5' fill='{MUTED}' text-anchor='middle'>{_esc(lab)}</text>")
        self.parts.append(f"<text x='{self.l + self.pw / 2}' y='{self.h - 12}' font-size='12' fill='{MUTED}' text-anchor='middle'>{_esc(label)}</text>")

    def xc(self, i):
        return self.l + self.pw * (i + 0.5) / self.n

    def line(self, values, color, name, dashed=False, row=0, labels=True, below=False, fmt="{:.1f}"):
        pts = [(self.xc(i), self.y(v)) for i, v in enumerate(values) if v is not None]
        dash = " stroke-dasharray='5 4'" if dashed else ""
        self.parts.append(f"<polyline points='{' '.join(f'{x:.1f},{y:.1f}' for x, y in pts)}' fill='none' stroke='{color}' stroke-width='2.4'{dash}/>")
        if not dashed:
            for (x, y), v in zip(pts, [v for v in values if v is not None]):
                self.parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'/>")
                if labels:
                    ty = y + 18 if below else y - 9
                    self.parts.append(f"<text x='{x:.1f}' y='{ty:.1f}' font-size='10.5' fill='{color}' font-weight='600' text-anchor='middle'>{fmt.format(v)}</text>")
        self.legend(color, name, row, dashed)

    def bars(self, groups, series_names, colors):
        n_s = len(series_names)
        gw = self.pw / self.n * 0.78
        bw = gw / n_s
        for i, vals in enumerate(groups):
            x0 = self.xc(i) - gw / 2
            for j, v in enumerate(vals):
                x, y = x0 + j * bw, self.y(v)
                self.parts.append(f"<rect x='{x + 1:.1f}' y='{y:.1f}' width='{bw - 2:.1f}' height='{self.t + self.ph - y:.1f}' fill='{colors[j]}' rx='2'/>")
                self.parts.append(f"<text x='{x + bw / 2:.1f}' y='{y - 4:.1f}' font-size='10' fill='{INK}' text-anchor='middle'>{v:.1f}</text>")
        for j, name in enumerate(series_names):
            self.legend(colors[j], name, j)

    def legend(self, color, name, row, dashed=False):
        x, y = self.l + self.pw + 16, self.t + 6 + row * 20
        dash = " stroke-dasharray='5 4'" if dashed else ""
        self.parts.append(f"<line x1='{x}' x2='{x + 18}' y1='{y}' y2='{y}' stroke='{color}' stroke-width='3'{dash}/>")
        self.parts.append(f"<text x='{x + 24}' y='{y + 4}' font-size='11.5' fill='{INK}'>{_esc(name)}</text>")

    def save(self, path: Path):
        path.write_text("\n".join(self.parts + ["</svg>"]))


def _k(n):
    return f"{round(n / 1024)}k" if n >= 1000 else str(n)


def chart_modes(data, assets: Path) -> str:
    runs = data["runs"]
    lengths = sorted({r["tokens"] for r in runs})
    modes = [m for m in ("chunked", "full", "ring") if any(r["mode"] == m for r in runs)]
    get = {(r["tokens"], r["mode"]): r for r in runs}
    groups = [[get[(L, m)]["ttft_s"] for m in modes] for L in lengths]
    c = Chart("Single Mac: TTFT by prefill mode",
              f"{data['model'].split('/')[-1]} on {data['env']['chip']} - ring = {data['devices']} simulated devices run one after another")
    c.y_axis(max(max(g) for g in groups) * 1.12, "TTFT (seconds)")
    c.x_labels([f"{_k(L)} tokens" for L in lengths], "RAG prompt length")
    names = {"chunked": "chunked (mlx-lm)", "full": "full (one pass)", "ring": f"ring ({data['devices']} sim.)"}
    c.bars(groups, [names[m] for m in modes], SERIES)
    c.save(assets / "single_mac_modes.svg")

    rows = ["| Tokens | " + " | ".join(names[m] for m in modes) + " | Peak memory (chunked / full) | Needle found |",
            "|---|" + "---|" * len(modes) + "---|---|"]
    for L in lengths:
        cells = [f"{get[(L, m)]['ttft_s']:.2f} s" for m in modes]
        mem = f"{get[(L, 'chunked')]['peak_gb']:.1f} / {get[(L, 'full')]['peak_gb']:.1f} GB" if "full" in modes and "chunked" in modes else "-"
        found = all(get[(L, m)]["needle_found"] for m in modes)
        rows.append(f"| {get[(L, modes[0])]['tokens']:,} | " + " | ".join(cells) + f" | {mem} | {'yes, all modes' if found else 'NO'} |")
    return "\n".join(rows)


def chart_parallel(data, assets: Path) -> str:
    runs = data["runs"]
    lengths = sorted({r["tokens"] for r in runs})
    devs = sorted({r["n_dev"] for r in runs})
    layouts = [l for l in ("zigzag", "contiguous") if any(r["layout"] == l for r in runs)]

    def find(L, n, layout):
        for r in runs:
            if r["tokens"] == L and r["n_dev"] == n and (n == 1 or r["layout"] == layout):
                return r
        return None

    L = lengths[-1]
    c = Chart(f"Measured TTFT vs devices ({_k(L)}-token RAG prompt)",
              f"slowest device per layer + {data['link']} K/V transfer ({data['link_gbps']} GB/s), output matches normal prefill")
    tt = {lay: [find(L, n, lay)["ttft_overlap_s"] if find(L, n, lay) else None for n in devs] for lay in layouts}
    c.y_axis(max(v for vals in tt.values() for v in vals if v) * 1.1, "TTFT (seconds)")
    c.x_labels([f"{n} device{'s' if n > 1 else ''}" for n in devs], "Ring size")
    for i, lay in enumerate(layouts):
        # zigzag is the lower line: label it below the points so labels don't collide
        c.line(tt[lay], SERIES[i], lay, row=i, below=(lay == "zigzag"))
    c.save(assets / "ttft_vs_devices.svg")

    c = Chart("Scaling efficiency (zigzag layout)",
              "speedup / devices: 100% = perfect linear scaling; longer prompts scale better")
    c.y_axis(100, "Efficiency (%)", ticks=4, vmin=60)
    c.x_labels([f"{n} device{'s' if n > 1 else ''}" for n in devs], "Ring size")
    c.line([100.0] * len(devs), MUTED, "ideal", dashed=True, row=0)
    for i, Lx in enumerate(lengths):
        eff = [100 * find(Lx, n, "zigzag")["speedup"] / n if find(Lx, n, "zigzag") else None for n in devs]
        c.line(eff, SERIES[i], f"{_k(Lx)} tokens", row=i + 1,
               labels=Lx in (lengths[0], lengths[-1]), below=(Lx == lengths[0]), fmt="{:.0f}%")
    c.save(assets / "speedup.svg")

    rows = ["| Tokens | Layout | " + " | ".join(f"{n} dev" for n in devs) + " |",
            "|---|---|" + "---|" * len(devs)]
    for Lx in lengths:
        for lay in layouts:
            cells = []
            for n in devs:
                r = find(Lx, n, lay)
                cells.append("-" if r is None else (f"{r['ttft_overlap_s']:.2f} s" if n == 1 else f"{r['ttft_overlap_s']:.2f} s ({r['speedup']:.1f}x)"))
            rows.append(f"| {Lx:,} | {lay} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--assets", default="assets")
    args = ap.parse_args(argv)
    res, assets = Path(args.results), Path(args.assets)
    assets.mkdir(parents=True, exist_ok=True)
    md = []
    if (res / "bench_modes.json").exists():
        md += ["### Single Mac: prefill modes", chart_modes(json.loads((res / "bench_modes.json").read_text()), assets), ""]
    if (res / "parallel_emulation.json").exists():
        md += ["### Measured multi-device TTFT", chart_parallel(json.loads((res / "parallel_emulation.json").read_text()), assets), ""]
    (res / "tables.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"Charts written to {assets}/")


if __name__ == "__main__":
    main()
