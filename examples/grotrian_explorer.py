#!/usr/bin/env python
"""Interactive Plotly Grotrian diagram explorer for Fe II.

Builds an interactive HTML page with hover-to-highlight transitions,
wavelength tooltips, full zoom/pan, and a toggle between the curated
term selection and the full Kurucz table.  Opens in the default browser.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from astropy.table import Table

from dotfit import GrotrianDiagram

EXAMPLE_DIR = Path(__file__).resolve().parent

# Combined JavaScript for hover highlighting and view toggle.
# Hover: highlight single transition, show wavelength label, dim others.
# Toggle: swap between "selected terms" and "all terms" figures.
INTERACTIVE_JS = """
<script>
(function() {
    var plot = document.getElementById('grotrian');
    if (!plot) return;

    var origOpacities = [];
    var ready = false;
    var plotData;
    var termToTrans = {};

    // Floating wavelength-label container
    var lblBox = document.createElement('div');
    lblBox.style.cssText =
        'position:absolute;top:0;left:0;pointer-events:none;z-index:1000;';

    function resetState() {
        ready = false;
        origOpacities = [];
        termToTrans = {};
        lblBox.innerHTML = '';
    }

    function init() {
        if (ready) return;
        plotData = plot.data;

        termToTrans = {};
        plotData.forEach(function(tr, i) {
            if (tr.meta && tr.meta.type === 'transition') {
                [tr.meta.lo, tr.meta.up].forEach(function(t) {
                    if (!termToTrans[t]) termToTrans[t] = [];
                    termToTrans[t].push(i);
                });
            }
        });

        origOpacities = [];
        var trEls = plot.querySelectorAll('.scatterlayer .trace');
        trEls.forEach(function(el) {
            origOpacities.push(el.style.opacity || '');
        });

        plot.style.position = 'relative';
        if (!lblBox.parentNode) plot.appendChild(lblBox);
        ready = true;
    }

    function showLabels(indices) {
        lblBox.innerHTML = '';
        var xa = plot._fullLayout.xaxis;
        var ya = plot._fullLayout.yaxis;

        indices.forEach(function(idx) {
            var tr = plotData[idx];
            if (!tr || !tr.meta || tr.meta.type !== 'transition') return;
            var xs = tr.x, ys = tr.y;
            var mx = (xs[0] + xs[xs.length - 1]) / 2;
            var my = (ys[0] + ys[ys.length - 1]) / 2;
            var px = xa.l2p(mx) + xa._offset;
            var py = ya.l2p(my) + ya._offset;
            var col = (tr.line && tr.line.color) || '#333';

            var d = document.createElement('div');
            d.textContent = tr.meta.wave + ' \\u00c5';
            d.style.cssText =
                'position:absolute;left:' + px + 'px;top:' + py + 'px;' +
                'transform:translate(-50%,-100%) translateY(-3px);' +
                'font:600 10px/1 sans-serif;color:' + col + ';' +
                'background:rgba(255,255,255,0.92);padding:1px 4px;' +
                'border-radius:2px;white-space:nowrap;';
            lblBox.appendChild(d);
        });
    }

    function clearLabels() { lblBox.innerHTML = ''; }

    plot.on('plotly_afterplot', init);

    plot.on('plotly_hover', function(data) {
        init();
        if (!data.points || !data.points.length) return;

        var ci = data.points[0].curveNumber;
        var meta = plotData[ci] && plotData[ci].meta;
        if (!meta) return;

        var hl = new Set();

        if (meta.type === 'transition') {
            var pn = data.points[0].pointNumber;
            var npts = plotData[ci].x.length;
            if (pn === 0 || pn === npts - 1) {
                // At endpoint (near level bar) -> highlight all connected
                var term = (pn === 0) ? meta.lo : meta.up;
                var conns = termToTrans[term];
                if (conns) conns.forEach(function(idx) { hl.add(idx); });
            } else {
                // Along the line -> highlight just this transition
                hl.add(ci);
            }
        } else if (meta.type === 'level') {
            // Hovering a level bar -> highlight all connected transitions
            var conns = termToTrans[meta.term];
            if (conns) conns.forEach(function(idx) { hl.add(idx); });
        }

        if (hl.size === 0) return;

        var trEls = plot.querySelectorAll('.scatterlayer .trace');
        trEls.forEach(function(el, i) {
            var tm = plotData[i] && plotData[i].meta;
            if (!tm) return;

            var orig = parseFloat(origOpacities[i]);

            if (tm.type === 'level') {
                el.style.opacity = '1';  // never dim level bars
            } else if (hl.has(i)) {
                el.style.opacity = '1';
                el.querySelectorAll('path').forEach(function(p) {
                    p.style.strokeWidth = '3.5px';
                });
            } else if (orig > 0.5) {
                // Detected lines — keep visible, never dim
                el.style.opacity = origOpacities[i];
            } else {
                el.style.opacity = '0.04';
            }
        });

        showLabels(Array.from(hl));
    });

    plot.on('plotly_unhover', function() {
        var trEls = plot.querySelectorAll('.scatterlayer .trace');
        trEls.forEach(function(el, i) {
            el.style.opacity = origOpacities[i] || '';
            el.querySelectorAll('path').forEach(function(p) {
                p.style.strokeWidth = '';
            });
        });
        clearLabels();
    });

    // --- View toggle: selected terms <-> all terms ---
    var figAll = JSON.parse(
        document.getElementById('fig-all-data').textContent);
    var figSelected = null;
    var showingAll = false;
    var btn = document.getElementById('toggle-view');

    btn.addEventListener('click', function() {
        if (!figSelected) {
            figSelected = {
                data: JSON.parse(JSON.stringify(plot.data)),
                layout: JSON.parse(JSON.stringify(plot.layout))
            };
        }
        showingAll = !showingAll;
        var fig = showingAll ? figAll : figSelected;
        Plotly.react('grotrian', fig.data, fig.layout, {scrollZoom: true});
        btn.textContent = showingAll ? 'Selected terms' : 'All terms';
        resetState();
    });
})();
</script>
"""

# Toggle button (fixed position, top-right)
TOGGLE_BTN = """
<div style="position:fixed;top:10px;right:20px;z-index:2000;">
    <button id="toggle-view"
        style="padding:8px 16px;font:600 13px/1.3 sans-serif;
        cursor:pointer;border:1px solid #999;border-radius:4px;
        background:#f5f5f5;box-shadow:0 1px 3px rgba(0,0,0,0.15);">
        All terms</button>
</div>
"""


def main():
    gd = GrotrianDiagram.from_kurucz(ion="Fe II")
    print(f"Full diagram: {len(gd.levels)} levels, " f"{len(gd.transitions)} transitions")

    # --- Load detected Fe II wavelengths ---
    detected_waves = None
    forbidden_waves = None
    for name in ("monster_v6.csv", "monster_v5.csv"):
        csv_path = EXAMPLE_DIR / name
        if csv_path.exists():
            cat = Table.read(csv_path, format="csv")
            ions = [str(r["ion"]).strip() for r in cat]
            fe_mask = [("Fe II" in ion or "Fe III" in ion) for ion in ions]
            detected_waves = list(cat["wave_vac"][fe_mask])
            forb_mask = [ion.startswith("[Fe") for ion in ions]
            forbidden_waves = list(cat["wave_vac"][forb_mask])
            print(
                f"Loaded {name}: {len(detected_waves)} Fe II lines " f"({len(forbidden_waves)} forbidden)"
            )
            break

    # --- Select terms ---
    ground_terms = ["a6D", "a6S", "a4F", "a4D", "a4G", "a4P", "a4H", "a2D2"]
    mid_terms = ["b4P", "b4F", "b4D", "b2H", "b4G"]
    upper_terms = [
        "z6P",
        "z6F",
        "z6D",
        "z4F",
        "z4D",
        "z4P",
        "y4D",
        "y4G",
        "y6P",
        "y4P",
        "x4D",
        "w4F",
        "c4D",
        "d4P",
        "e4D",
        "e6D",
        "e4F",
        "f4D",
    ]
    lya_terms = ["t4G", "u4G", "v4F", "u4D", "u4P", "b6D"]
    all_terms = ground_terms + mid_terms + upper_terms + lya_terms

    sub = gd.select(terms=all_terms, gf_min=0.001)
    print(f"Selected: {len(sub.levels)} levels, " f"{len(sub.transitions)} transitions")

    # Full table — all terms passing gf_min, no term cap
    full = gd.select(gf_min=0.001, max_terms=None)
    print(f"Full (gf>0.001): {len(full.levels)} levels, " f"{len(full.transitions)} transitions")

    # --- Ly-alpha pumping colour overrides ---
    purple = "#8B00FF"
    color_overrides = {("a4D", t): purple for t in ["4F", "4D", "4P", "4G", "4S"]}

    # Common plot kwargs
    common_kw = dict(
        layout="crossing",
        min_gap_eV=0.9,
        col_spacing=1.6,
        detected_waves=detected_waves,
        forbidden_waves=forbidden_waves,
        color_overrides=color_overrides,
        title="Fe II Grotrian Diagram — Interactive Explorer",
        height=900,
    )

    # --- Build both figures ---
    fig_selected = sub.plot_plotly(
        pin_left=["a4P", "a4D", "a6D"], pin_right=["a4F", "z4P", "b4D"], **common_kw
    )
    fig_all = full.plot_plotly(**common_kw)

    # --- Write HTML ---
    out_path = EXAMPLE_DIR / "grotrian_explorer.html"
    html = fig_selected.to_html(
        full_html=True, include_plotlyjs=True, div_id="grotrian", config={"scrollZoom": True}
    )

    # Inject: toggle button + hidden JSON blob for "all" figure + JS
    inject = (
        TOGGLE_BTN
        + '\n<script id="fig-all-data" type="application/json">\n'
        + fig_all.to_json()
        + '\n</script>\n'
        + INTERACTIVE_JS
    )
    html = html.replace("</body>", inject + "</body>")
    out_path.write_text(html)
    print(f"Saved {out_path}")

    # Open in browser
    webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    main()
