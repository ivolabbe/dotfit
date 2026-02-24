#!/usr/bin/env python
"""Generate Fe II Grotrian diagrams from ground state up.

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
    print(f"Full diagram: {len(gd.levels)} levels, {len(gd.transitions)} transitions")

    # Key terms from ground state up — matching classic Fe II diagrams
    terms = [
        "a6D", "a6S", "a4F", "a4D", "a4G", "a4H", "a2D2",
        "b4P", "b4F", "b4D", "b2H", "b4G",
        "z6P", "z6F", "z6D", "z4F", "z4D", "z4P",
        "y4D", "y4G", "y6P", "y4P",
        "x4D",
        "w4F", "c4D", "d4P",
        "e4D", "e6D", "e4F", "f4D",
        # Ly-alpha fluorescence (NIST aliases resolved to Kurucz labels)
        "t4G", "u4G", "v4F", "u4D", "u4P",  # -> w4G, 4G, v4F, 4D, 4P
        "b6D",  # -> 6D
    ]

    # --- Load detected Fe II wavelengths from monster_v5.csv ---
    detected_waves = None
    forbidden_waves = None
    csv_path = EXAMPLE_DIR / "monster_v5.csv"
    if csv_path.exists():
        cat = Table.read(csv_path, format="csv")
        ions = [str(r["ion"]).strip() for r in cat]
        fe_mask = [("Fe II" in ion or "Fe III" in ion) for ion in ions]
        detected_waves = list(cat["wave_vac"][fe_mask])
        # Bracket notation [Fe II] = forbidden lines only
        forb_mask = [ion.startswith("[Fe") for ion in ions]
        forbidden_waves = list(cat["wave_vac"][forb_mask])
        print(f"Detected Fe II lines: {len(detected_waves)} "
              f"({len(forbidden_waves)} forbidden)")

    # --- Full diagram with all wavelength regimes ---
    sub = gd.select(terms=terms, gf_min=0.001)
    print(f"Selected: {len(sub.levels)} levels, {len(sub.transitions)} transitions")
    print(sub.summary())

    fig = sub.plot(
        layout="compact", color_by="wavelength", aggregate=True,
        figsize=(10, 13), min_gap_eV=0.9,
        pin_left=["a6D", "a4F", "a4D"],
        title="Fe II Grotrian Diagram",
        detected_waves=detected_waves,
        forbidden_waves=forbidden_waves,
        show_strongest_eV=(4, 6),
    )
    fig.savefig("grotrian_feii.pdf", bbox_inches="tight")
    fig.savefig("grotrian_feii.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved grotrian_feii.pdf / .png")


if __name__ == "__main__":
    main()
