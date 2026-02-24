#!/usr/bin/env python
"""3-panel Fe II Grotrian diagram: Optical / UV / Ly-alpha cascade.

Panel 1 (Optical):  transitions from (4,6) eV terms down to (2,4) eV terms.
Panel 2 (UV):       transitions from (4,6) eV terms down to (0,2) eV tperms.
Panel 3 (Ly-alpha): transitions down from Ly-alpha pumped levels (~10-12 eV),
                    with the pumping channel shown in purple.

Produces PDF/PNG outputs in the current directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

from dotfit import GrotrianDiagram

EXAMPLE_DIR = Path(__file__).resolve().parent


def main():
    gd = GrotrianDiagram.from_kurucz(ion="Fe II")
    print(f"Full diagram: {len(gd.levels)} levels, " f"{len(gd.transitions)} transitions")

    # --- Load detected Fe II wavelengths ---
    detected_waves = None
    forbidden_waves = None
    csv_path = EXAMPLE_DIR / "monster_v6.csv"
    if csv_path.exists():
        cat = Table.read(csv_path, format="csv")
        ions = [str(r["ion"]).strip() for r in cat]
        fe_mask = [("Fe II" in ion or "Fe III" in ion) for ion in ions]
        detected_waves = list(cat["wave_vac"][fe_mask])
        forb_mask = [ion.startswith("[Fe") for ion in ions]
        forbidden_waves = list(cat["wave_vac"][forb_mask])
        print(f"Detected Fe II lines: {len(detected_waves)} " f"({len(forbidden_waves)} forbidden)")

    # --- Common terms ---
    ground_terms = ["a6D", "a6S", "a4F", "a4D", "a4G", "a4P", "a4H", "a2D2", "a2G"]
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
    # Ly-alpha fluorescence terms (NIST aliases resolved by select())
    lya_terms = ["t4G", "u4G", "v4F", "u4D", "u4P", "b6D"]

    all_terms = ground_terms + mid_terms + upper_terms + lya_terms
    sub = gd.select(terms=all_terms, gf_min=0.01)
    print(f"Selected: {len(sub.levels)} levels, " f"{len(sub.transitions)} transitions")

    # --- Ly-alpha pumping colour overrides ---
    # a4D (~1 eV) absorbs Ly-alpha photons, reaching ~11.2 eV upper levels.
    # Kurucz names after alias resolution: 4F, 4D, 4P (strongest channels).
    pump_upper = ["4F", "4D", "4P", "4G", "4S"]
    purple = "#8B00FF"
    color_overrides = {}
    for up in pump_upper:
        #        color_overrides[("a4D", up)] = purple
        color_overrides[("a4D", up)] = purple

    # --- Shared plot kwargs ---
    common = dict(
        layout="crossing",
        #   layout="compact",
        color_by="wavelength",
        aggregate=True,
        min_gap_eV=0.9,
        col_spacing=1.6,
        #        pin_left=["a4D", "a6D", "a4P", "a4F"],
        #        pin_right=["z4P", "b4D"],
        pin_left=["a4D", "a4P", "4P", "4G", "4D"],  # include Ly-alpha pumping lower term
        pin_right=["z4P", "b4D", "a4F"],
        detected_waves=detected_waves,
        forbidden_waves=forbidden_waves,
        bg_alpha=0.2,
        semi_alpha=0.25,
    )

    # --- 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    # Panel 1: Optical — (4,6) eV -> (2,4) eV
    sub.plot(
        ax=axes[0],
        title="Optical: 4-6 eV \u2192 2-4 eV",
        show_strongest_eV=[
            ((1.5, 4), (4, 5), (4000, 6000)),
            ((0, 3), (2, 4), (4000, 7000)),
            ((0.5, 1.5), (10.5, 12.5), (1213, 1219)),
        ],
        color_overrides=color_overrides,
        **common,
    )

    # Panel 2: UV — (4,6) eV -> (0,2) eV
    sub.plot(
        ax=axes[1],
        title="UV: 4-6 eV \u2192 0-2 eV",
        show_strongest_eV=[((4, 6), (0, 2))],
        show_legend=False,
        **common,
    )

    # Panel 3: Ly-alpha pumping — a4D (~1 eV) absorbs near 1216 Å
    # The wave_range filter (3rd element) restricts to ±3 Å of Ly-alpha
    # Upper levels are at 11.2-12.2 eV so we use a wide target range.
    sub.plot(
        ax=axes[2],
        title="Ly\u03b1 pumping: a4D \u2192 11\u201312 eV (1216\u00b13 \u00c5)",
        show_strongest_eV=[((0.5, 1.5), (10.5, 12.5), (1213, 1219))],
        color_overrides=color_overrides,
        show_legend=False,
        highlight_semi_only=True,
        **common,
    )

    fig.tight_layout()

    out_dir = EXAMPLE_DIR
    fig.savefig(out_dir / "grotrian_feii_panels.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "grotrian_feii_panels.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'grotrian_feii_panels.pdf'} / .png")


if __name__ == "__main__":
    main()
