"""Grotrian (energy-level / term) diagram module for spectral line visualisation.

Builds and plots Grotrian diagrams from Kurucz-format line tables,
showing energy levels grouped by term with transitions colour-coded
by wavelength regime (UV / optical / NIR) or classification.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.table import Table

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data" / "emission_lines"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_L_VALUES = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6}

# Alias mapping: NIST literature names -> Kurucz table names.
# The Kurucz compilation uses different prefix conventions for high-excitation
# levels compared to NIST.  Ly-alpha fluorescence levels are particularly
# affected.
TERM_ALIASES: dict[str, str] = {
    "t4G": "w4G",   # NIST t 4G (11.65 eV) = Kurucz w4G
    "u4G": "4G",    # NIST u 4G (11.18 eV) = Kurucz bare 4G
    "u4D": "4D",    # NIST u 4D (11.46 eV) subset within Kurucz bare 4D
    "u4P": "4P",    # NIST u 4P (11.44 eV) subset within Kurucz bare 4P
    "b6D": "6D",    # NIST b 6D (10.9 eV) subset within Kurucz bare 6D
}

# Wavelength regime boundaries (Angstrom) and colours
WAVE_COLORS = {
    "UV": ("#3366CC", (0.0, 3500.0)),
    "visible": ("#009688", (3500.0, 7000.0)),
    "NIR": ("#CC3333", (7000.0, 1e8)),
}


def _get_parity(config: str) -> str:
    """Determine parity ('even' or 'odd') from an electron configuration string.

    Sums the orbital angular momentum quantum numbers weighted by electron
    counts.  Even total l -> even parity, odd -> odd.
    """
    total_l = 0
    for _count_str, orbital, exponent_str in re.findall(
        r"([0-9]*)([spdfghi])([0-9]*)", config
    ):
        exponent = int(exponent_str) if exponent_str else 1
        if orbital in _L_VALUES:
            total_l += _L_VALUES[orbital] * exponent
    return "even" if total_l % 2 == 0 else "odd"


def _term_to_latex(term: str) -> str:
    r"""Convert a term label to LaTeX, e.g. ``"a6D"`` -> ``$a\,^{6}\!D$``."""
    m = re.match(r"^([a-z]?)(\d+)([A-Z]\[?\d*/?\d*\]?)(\*?)$", term)
    if not m:
        return f"${term}$"
    prefix, mult, L_sym, star = m.groups()
    circ = r"^\circ" if star else ""
    if prefix:
        return rf"${prefix}^{{{mult}}}\!{L_sym}{circ}$"
    return rf"$^{{{mult}}}\!{L_sym}{circ}$"


def _j_to_str(j: float) -> str:
    """Format a J quantum number: 4.5 -> '9/2', 2.0 -> '2'."""
    if j == int(j):
        return str(int(j))
    return f"{int(2 * j)}/2"


def _wave_color(wave_vac: float) -> str:
    """Return colour hex for a wavelength based on UV/optical/NIR regime."""
    for _name, (color, (lo, hi)) in WAVE_COLORS.items():
        if lo <= wave_vac < hi:
            return color
    return WAVE_COLORS["visible"][0]


def _wave_regime(wave_vac: float) -> str:
    """Return regime name for a wavelength."""
    for name, (_color, (lo, hi)) in WAVE_COLORS.items():
        if lo <= wave_vac < hi:
            return name
    return "visible"


def _desaturate(color: str, strength: float) -> str:
    """Mix *color* toward grey.  strength=1 keeps original, 0 gives pure grey."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    grey = 0.75
    t = strength
    return mcolors.to_hex((r * t + grey * (1 - t),
                           g * t + grey * (1 - t),
                           b * t + grey * (1 - t)))


# ---------------------------------------------------------------------------
# Wavelength-regime presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "uv": dict(wave_range=(2000.0, 3500.0)),
    "optical": dict(wave_range=(3700.0, 7500.0)),
    "nir": dict(wave_range=(8000.0, 13000.0)),
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyLevel:
    """A single energy level identified by term, J, and energy."""

    term: str
    J: float
    energy_eV: float
    parity: str  # "even" or "odd"
    config: str  # electron configuration

    @property
    def key(self) -> tuple[str, float, float]:
        return (self.term, self.J, round(self.energy_eV, 6))


@dataclass
class Transition:
    """A radiative transition between two energy levels."""

    lower: EnergyLevel
    upper: EnergyLevel
    wave_vac: float  # Angstrom
    gf: float
    classification: str  # "permitted", "semi-forbidden", "forbidden"
    multiplet: int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@dataclass
class GrotrianDiagram:
    """Grotrian (energy-level) diagram built from a line table.

    Attributes:
        levels: Deduplicated energy levels keyed by ``(term, J, energy_eV)``.
        transitions: List of transitions linking levels.
        ion: Ion label (e.g. ``"Fe II"``).
    """

    levels: dict[tuple[str, float, float], EnergyLevel] = field(default_factory=dict)
    transitions: list[Transition] = field(default_factory=list)
    ion: str = "Fe II"

    # ---- constructors ----------------------------------------------------

    @classmethod
    def from_table(cls, table: Table, ion: str = "Fe II") -> GrotrianDiagram:
        """Build a diagram from an astropy ``Table`` in dotfit format.

        Args:
            table: Table with columns ``ion``, ``wave_vac``, ``Ei``, ``Ek``,
                ``gf``, ``configuration``, ``terms``, ``Ji-Jk``,
                ``classification``, ``multiplet``.
            ion: Ion to select (bracket notation is stripped automatically).
        """
        ion_clean = ion.replace("[", "").replace("]", "").strip()
        ion_col = np.array([s.replace("[", "").replace("]", "").strip()
                            for s in table["ion"]])
        mask = ion_col == ion_clean
        tab = table[mask]
        if len(tab) == 0:
            logger.warning("No rows for ion '%s' — returning empty diagram.", ion)
            return cls(ion=ion_clean)

        levels: dict[tuple[str, float, float], EnergyLevel] = {}
        transitions: list[Transition] = []

        for row in tab:
            terms_str = str(row["terms"]).strip()
            ji_jk_str = str(row["Ji-Jk"]).strip()
            config_str = str(row["configuration"]).strip()
            if not terms_str or terms_str == "--" or not ji_jk_str or ji_jk_str == "--":
                continue

            parts = terms_str.split("-")
            if len(parts) != 2:
                continue
            lower_term, upper_term = parts[0].strip(), parts[1].strip()
            if not lower_term or not upper_term:
                continue

            j_parts = ji_jk_str.split("-")
            if len(j_parts) != 2:
                continue
            try:
                lower_J = float(eval(j_parts[0].strip()))  # noqa: S307
                upper_J = float(eval(j_parts[1].strip()))  # noqa: S307
            except Exception:
                continue

            cfg_parts = config_str.split("-")
            lower_cfg = cfg_parts[0].strip() if len(cfg_parts) >= 1 else ""
            upper_cfg = cfg_parts[1].strip() if len(cfg_parts) >= 2 else ""
            lower_parity = _get_parity(lower_cfg)
            upper_parity = _get_parity(upper_cfg)

            Ei = float(row["Ei"])
            Ek = float(row["Ek"])
            gf = float(row["gf"])
            wave_vac = float(row["wave_vac"])
            classification = str(row["classification"]).strip()
            try:
                multiplet = int(row["multiplet"])
            except (ValueError, TypeError):
                multiplet = -1

            lkey = (lower_term, lower_J, round(Ei, 6))
            if lkey not in levels:
                levels[lkey] = EnergyLevel(
                    term=lower_term, J=lower_J, energy_eV=Ei,
                    parity=lower_parity, config=lower_cfg,
                )
            ukey = (upper_term, upper_J, round(Ek, 6))
            if ukey not in levels:
                levels[ukey] = EnergyLevel(
                    term=upper_term, J=upper_J, energy_eV=Ek,
                    parity=upper_parity, config=upper_cfg,
                )

            transitions.append(Transition(
                lower=levels[lkey],
                upper=levels[ukey],
                wave_vac=wave_vac,
                gf=gf,
                classification=classification,
                multiplet=multiplet,
            ))

        logger.info(
            "Built GrotrianDiagram for %s: %d levels, %d transitions",
            ion_clean, len(levels), len(transitions),
        )
        return cls(levels=levels, transitions=transitions, ion=ion_clean)

    @classmethod
    def from_kurucz(cls, ion: str = "Fe II") -> GrotrianDiagram:
        """Load the bundled Kurucz CSV and build a diagram.

        Args:
            ion: Ion to select (default ``"Fe II"``).
        """
        path = DATA_DIR / "feti_kurucz.csv"
        if not path.exists():
            raise FileNotFoundError(f"Kurucz table not found: {path}")
        table = Table.read(path, format="csv")
        return cls.from_table(table, ion=ion)

    # ---- filtering -------------------------------------------------------

    def select(
        self,
        *,
        preset: str | None = None,
        terms: list[str] | None = None,
        wave_range: tuple[float, float] | None = None,
        energy_range: tuple[float, float] | None = None,
        gf_min: float = 0.01,
        classification: str = "all",
        max_terms: int = 12,
    ) -> GrotrianDiagram:
        """Return a filtered copy of the diagram.

        Args:
            preset: One of ``"uv"``, ``"optical"``, ``"nir"`` — overrides
                *terms* and *wave_range* with sensible defaults.
            terms: Explicit list of term labels to keep.
            wave_range: ``(lam_min, lam_max)`` in Angstrom.
            energy_range: ``(E_min, E_max)`` in eV.
            gf_min: Minimum oscillator strength.
            classification: ``"all"``, ``"permitted"``, ``"semi-forbidden"``,
                or ``"forbidden"``.
            max_terms: When *terms* is ``None`` (and no preset), auto-select
                the most transition-rich terms up to this count.

        Returns:
            A new ``GrotrianDiagram`` with the filtered subset.
        """
        if preset is not None:
            key = preset.lower()
            if key not in PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. Choose from: {list(PRESETS)}"
                )
            p = PRESETS[key]
            if terms is None:
                terms = p.get("terms")
            if wave_range is None:
                wave_range = p.get("wave_range")

        filtered: list[Transition] = []
        for tr in self.transitions:
            if wave_range is not None:
                if tr.wave_vac < wave_range[0] or tr.wave_vac > wave_range[1]:
                    continue
            if energy_range is not None:
                if (tr.lower.energy_eV < energy_range[0]
                        or tr.upper.energy_eV > energy_range[1]):
                    continue
            if tr.gf < gf_min:
                continue
            if classification != "all" and tr.classification != classification:
                continue
            filtered.append(tr)

        if terms is not None:
            # Resolve aliases (e.g. "t4G" -> "4G")
            resolved = []
            for t in terms:
                mapped = TERM_ALIASES.get(t, t)
                if mapped != t:
                    logger.info("Term alias: %s -> %s", t, mapped)
                resolved.append(mapped)
            term_set = set(resolved)

            # Warn about terms not found in any transition
            all_terms = {tr.lower.term for tr in filtered} | {
                tr.upper.term for tr in filtered}
            missing = term_set - all_terms
            if missing:
                logger.warning(
                    "Requested terms not found in data: %s. "
                    "Available terms: %s",
                    sorted(missing),
                    sorted(all_terms),
                )
            filtered = [
                tr for tr in filtered
                if tr.lower.term in term_set and tr.upper.term in term_set
            ]
        elif max_terms is not None:
            counts: Counter[str] = Counter()
            for tr in filtered:
                counts[tr.lower.term] += 1
                counts[tr.upper.term] += 1

            even_terms = sorted(
                [t for t in counts if any(
                    lv.parity == "even" for lv in self.levels.values()
                    if lv.term == t
                )],
                key=lambda t: -counts[t],
            )
            odd_terms = sorted(
                [t for t in counts if any(
                    lv.parity == "odd" for lv in self.levels.values()
                    if lv.term == t
                )],
                key=lambda t: -counts[t],
            )
            half = max_terms // 2
            picked = set(even_terms[:half]) | set(odd_terms[:half])
            remaining = sorted(
                [t for t in counts if t not in picked],
                key=lambda t: -counts[t],
            )
            for t in remaining:
                if len(picked) >= max_terms:
                    break
                picked.add(t)
            filtered = [
                tr for tr in filtered
                if tr.lower.term in picked and tr.upper.term in picked
            ]

        new_levels: dict[tuple[str, float, float], EnergyLevel] = {}
        for tr in filtered:
            new_levels[tr.lower.key] = tr.lower
            new_levels[tr.upper.key] = tr.upper

        return GrotrianDiagram(levels=new_levels, transitions=filtered, ion=self.ion)

    # ---- summary ---------------------------------------------------------

    def summary(self) -> Table:
        """Return a summary table of terms with level/transition counts."""
        term_levels: Counter[str] = Counter()
        term_trans: Counter[str] = Counter()
        term_energies: dict[str, list[float]] = {}

        for lv in self.levels.values():
            term_levels[lv.term] += 1
            term_energies.setdefault(lv.term, []).append(lv.energy_eV)
        for tr in self.transitions:
            term_trans[tr.lower.term] += 1
            term_trans[tr.upper.term] += 1

        terms = sorted(term_levels.keys())
        rows = []
        for t in terms:
            energies = term_energies.get(t, [])
            parity = "?"
            for lv in self.levels.values():
                if lv.term == t:
                    parity = lv.parity
                    break
            rows.append({
                "term": t,
                "parity": parity,
                "n_levels": term_levels[t],
                "n_transitions": term_trans.get(t, 0),
                "E_min": min(energies) if energies else np.nan,
                "E_max": max(energies) if energies else np.nan,
            })
        return Table(rows=rows)

    # ---- term geometry ---------------------------------------------------

    def _term_mean_energies(self) -> dict[str, float]:
        """Compute mean energy per term from all J sub-levels."""
        energies: dict[str, list[float]] = {}
        for lv in self.levels.values():
            energies.setdefault(lv.term, []).append(lv.energy_eV)
        return {t: float(np.mean(e)) for t, e in energies.items()}

    def _term_parity(self) -> dict[str, str]:
        """Map term -> parity."""
        out: dict[str, str] = {}
        for lv in self.levels.values():
            out[lv.term] = lv.parity
        return out

    # ---- compact layout --------------------------------------------------

    @staticmethod
    def _layout_compact(
        term_energies: dict[str, float],
        min_gap_eV: float = 0.8,
        col_spacing: float = 1.0,
        pin_left: list[str] | None = None,
        pin_right: list[str] | None = None,
    ) -> dict[str, float]:
        """Assign x positions to terms using greedy column packing.

        Places each term (sorted by energy) in the first column where no
        existing term is within *min_gap_eV*, producing a compact layout
        that avoids label overlap.

        Args:
            pin_left: Terms to place in the leftmost columns first,
                reducing line crossings for highly-connected ground-state
                terms.
            pin_right: Terms to place in exclusive rightmost columns,
                after all other terms have been packed.
        """
        pinned_left = [t for t in (pin_left or []) if t in term_energies]
        pinned_right = [t for t in (pin_right or []) if t in term_energies]
        pinned_set = set(pinned_left) | set(pinned_right)

        def _pack_into_columns(
            terms: list[str],
            cols: list[list[tuple[str, float]]],
            tx: dict[str, float],
            col_offset: int = 0,
        ) -> None:
            """Pack *terms* into *cols* starting at *col_offset*, using
            min_gap_eV to allow vertical stacking."""
            for term in terms:
                energy = term_energies[term]
                placed = False
                for col_idx in range(col_offset, len(cols)):
                    if all(abs(energy - e) > min_gap_eV for _, e in cols[col_idx]):
                        cols[col_idx].append((term, energy))
                        tx[term] = float(col_idx) * col_spacing
                        placed = True
                        break
                if not placed:
                    cols.append([(term, energy)])
                    tx[term] = float(len(cols) - 1) * col_spacing

        # Reserve exclusive leftmost columns for pinned-left terms
        columns: list[list[tuple[str, float]]] = []
        term_x: dict[str, float] = {}
        _pack_into_columns(pinned_left, columns, term_x, col_offset=0)
        n_pinned_left = len(columns)

        # Pack remaining terms, skipping the reserved pinned columns
        remaining = sorted(
            [t for t in term_energies if t not in pinned_set],
            key=lambda t: term_energies[t],
        )
        for term in remaining:
            energy = term_energies[term]
            placed = False
            for col_idx, col in enumerate(columns):
                if col_idx < n_pinned_left:
                    continue  # pinned columns are exclusive
                if all(abs(energy - e) > min_gap_eV for _, e in col):
                    col.append((term, energy))
                    term_x[term] = float(col_idx) * col_spacing
                    placed = True
                    break
            if not placed:
                columns.append([(term, energy)])
                term_x[term] = float(len(columns) - 1) * col_spacing

        # Append exclusive rightmost columns for pinned-right terms
        n_before_right = len(columns)
        _pack_into_columns(pinned_right, columns, term_x,
                           col_offset=n_before_right)

        return term_x

    def _layout_crossing(
        self,
        term_energies: dict[str, float],
        min_gap_eV: float = 0.8,
        col_spacing: float = 1.0,
        pin_left: list[str] | None = None,
        pin_right: list[str] | None = None,
        max_iter: int = 10,
    ) -> dict[str, float]:
        """Assign x positions minimising transition line crossings.

        Starts from compact layout, then iteratively repositions each
        free (non-pinned) term toward the barycenter of its connected
        neighbours.  Pinned terms stay in exclusive edge columns and
        act as anchors.

        Args:
            max_iter: Maximum barycenter iterations.
        """
        # Build adjacency: unique term pairs from transitions
        neighbors: dict[str, set[str]] = {}
        for tr in self.transitions:
            lo, up = tr.lower.term, tr.upper.term
            if lo not in term_energies or up not in term_energies:
                continue
            neighbors.setdefault(lo, set()).add(up)
            neighbors.setdefault(up, set()).add(lo)

        # Initial positions from compact layout
        x_pos = self._layout_compact(
            term_energies, min_gap_eV=min_gap_eV,
            col_spacing=col_spacing,
            pin_left=pin_left, pin_right=pin_right)

        pinned_left = [t for t in (pin_left or []) if t in term_energies]
        pinned_right = [t for t in (pin_right or []) if t in term_energies]
        pinned_set = set(pinned_left) | set(pinned_right)
        free_terms = [t for t in term_energies if t not in pinned_set]

        # Identify which free terms have connections (others keep compact pos)
        connected_free = [t for t in free_terms if t in neighbors]
        isolated_free = [t for t in free_terms if t not in neighbors]

        prev_order: list[str] = []
        for _ in range(max_iter):
            # Compute barycenter for each connected free term
            bary = {}
            for t in connected_free:
                nbrs = [n for n in neighbors[t] if n in x_pos]
                if nbrs:
                    bary[t] = sum(x_pos[n] for n in nbrs) / len(nbrs)
                else:
                    bary[t] = x_pos[t]

            sorted_free = sorted(connected_free, key=lambda t: bary[t])
            if sorted_free == prev_order:
                break
            prev_order = list(sorted_free)

            # Re-pack into columns: pin_left | sorted_free+isolated | pin_right
            # Isolated terms are appended after connected terms
            pack_order = sorted_free + sorted(
                isolated_free, key=lambda t: term_energies[t])

            columns: list[list[tuple[str, float]]] = []
            new_x: dict[str, float] = {}

            # Pack pinned-left into exclusive columns
            for term in pinned_left:
                energy = term_energies[term]
                placed = False
                for col in columns:
                    if all(abs(energy - e) > min_gap_eV for _, e in col):
                        col.append((term, energy))
                        new_x[term] = float(columns.index(col)) * col_spacing
                        placed = True
                        break
                if not placed:
                    columns.append([(term, energy)])
                    new_x[term] = float(len(columns) - 1) * col_spacing
            n_pinned_left = len(columns)

            # Pack free terms (connected sorted by bary, then isolated)
            for term in pack_order:
                energy = term_energies[term]
                placed = False
                for col_idx in range(n_pinned_left, len(columns)):
                    col = columns[col_idx]
                    if all(abs(energy - e) > min_gap_eV for _, e in col):
                        col.append((term, energy))
                        new_x[term] = float(col_idx) * col_spacing
                        placed = True
                        break
                if not placed:
                    columns.append([(term, energy)])
                    new_x[term] = float(len(columns) - 1) * col_spacing

            # Pack pinned-right into exclusive rightmost columns
            n_before_right = len(columns)
            for term in pinned_right:
                energy = term_energies[term]
                placed = False
                for col_idx in range(n_before_right, len(columns)):
                    col = columns[col_idx]
                    if all(abs(energy - e) > min_gap_eV for _, e in col):
                        col.append((term, energy))
                        new_x[term] = float(col_idx) * col_spacing
                        placed = True
                        break
                if not placed:
                    columns.append([(term, energy)])
                    new_x[term] = float(len(columns) - 1) * col_spacing

            x_pos = new_x

        return x_pos

    @staticmethod
    def _layout_columns(
        term_energies: dict[str, float],
        term_parity: dict[str, str],
        gap: float = 1.5,
    ) -> dict[str, float]:
        """Assign x positions in strict parity-separated columns."""
        even = sorted(
            [t for t in term_energies if term_parity.get(t) == "even"],
            key=lambda t: term_energies[t],
        )
        odd = sorted(
            [t for t in term_energies if term_parity.get(t) == "odd"],
            key=lambda t: term_energies[t],
        )
        x_pos: dict[str, float] = {}
        for i, t in enumerate(even):
            x_pos[t] = float(i)
        offset = (len(even) + gap) if even else gap
        for i, t in enumerate(odd):
            x_pos[t] = offset + float(i)
        return x_pos

    # ---- plotting --------------------------------------------------------

    def _match_detected(
        self,
        detected_waves: list[float],
        tol_ang: float = 2.0,
        forbidden_waves: list[float] | None = None,
    ) -> tuple[dict[tuple[str, str], float], list[Transition]]:
        """Match detected wavelengths to transitions, returning highlighted pairs.

        Permitted detected wavelengths are matched against Kurucz transitions.
        Forbidden detected wavelengths are matched against the NIST forbidden
        Fe II lines in ``emission_lines.csv`` (which have proper term
        assignments).

        Args:
            detected_waves: Vacuum wavelengths in Angstrom (all detected lines).
            tol_ang: Matching tolerance in Angstrom.
            forbidden_waves: Vacuum wavelengths of detected *forbidden* lines
                only (e.g. ``[Fe II]``).  Used for synthesising forbidden
                transitions.  If ``None``, no forbidden lines are synthesised.

        Returns:
            Tuple of (highlighted term pairs → matched Kurucz wavelength,
            list of new forbidden transitions to add to the diagram).
        """
        # pair → (best_gf, matched_kurucz_wavelength)
        _hl: dict[tuple[str, str], tuple[float, float]] = {}

        # Only match *permitted* detected wavelengths against Kurucz
        # transitions.  Forbidden wavelengths (e.g. [Fe II]) are handled
        # separately below and must NOT be matched here — they would
        # otherwise falsely highlight upper-term permitted transitions
        # at coincidentally similar wavelengths.
        forb_set = set(forbidden_waves) if forbidden_waves else set()
        permitted_det = [w for w in detected_waves if w not in forb_set]
        det = np.array(permitted_det) if permitted_det else np.array([])

        # For each permitted detected wavelength, find its true identity
        # in the *full* Kurucz catalog (closest match regardless of whether
        # its terms are in the diagram).  Only highlight the term pair if
        # both terms are plottable.  This prevents false matches where a
        # line's true identity (e.g. a4P-z4D) isn't in the diagram but a
        # nearby plottable line (e.g. b4D-y4D) would otherwise match.
        full_tab = Table.read(DATA_DIR / "feti_kurucz.csv", format="csv")
        ion_col = np.array([s.replace("[", "").replace("]", "").strip()
                            for s in full_tab["ion"]])
        ion_clean = self.ion.replace("[", "").replace("]", "").strip()
        full_tab = full_tab[ion_col == ion_clean]

        diagram_terms = {lv.term for lv in self.levels.values()}

        # Parse term pairs and gf for ALL Fe II rows
        all_waves = np.array(full_tab["wave_vac"], dtype=float)
        all_gf = np.array(full_tab["gf"], dtype=float)
        term_strs = [str(r["terms"]).strip() for r in full_tab]
        all_pairs: list[tuple[str, str] | None] = []
        for ts in term_strs:
            parts = ts.split("-")
            if len(parts) == 2:
                all_pairs.append((parts[0].strip(), parts[1].strip()))
            else:
                all_pairs.append(None)

        if len(det) > 0 and len(all_waves) > 0:
            for dw in det:
                residuals = np.abs(all_waves - dw)
                idx = int(np.argmin(residuals))
                if residuals[idx] < tol_ang:
                    pair = all_pairs[idx]
                    if pair is not None:
                        lo_t, up_t = pair
                        if lo_t in diagram_terms and up_t in diagram_terms:
                            gf = float(all_gf[idx])
                            wv = float(all_waves[idx])
                            prev = _hl.get(pair)
                            if prev is None or gf > prev[0]:
                                _hl[pair] = (gf, wv)

        # Match forbidden detected wavelengths against the forbidden Fe II
        # lines in the bundled emission_lines.csv (NIST data with proper term
        # assignments), rather than synthesising from energy gaps.
        forbidden_new: list[Transition] = []

        if forbidden_waves is not None and len(forbidden_waves) > 0:
            det_forb = np.array(forbidden_waves)
            forb_tab = Table.read(DATA_DIR / "emission_lines.csv", format="csv")
            # Filter to Fe II forbidden lines only
            ions = np.array([str(r["ion"]).replace("[", "").replace("]", "").strip()
                             for r in forb_tab])
            cls_col = np.array([str(r["classification"]).strip() for r in forb_tab])
            fmask = (ions == "Fe II") & (cls_col == "forbidden")
            forb_tab = forb_tab[fmask]

            # For each detected forbidden wavelength, find single best match
            seen_pairs: dict[tuple[str, str], tuple[float, Transition]] = {}
            for dw in det_forb:
                best_res = tol_ang
                best_row = None
                for row in forb_tab:
                    res = abs(float(row["wave_vac"]) - dw)
                    if res < best_res:
                        best_res = res
                        best_row = row
                if best_row is not None:
                    terms_str = str(best_row["terms"]).strip()
                    parts = terms_str.split("-")
                    if len(parts) != 2:
                        continue
                    lo_term, up_term = parts[0].strip(), parts[1].strip()
                    pk = (lo_term, up_term)

                    # Use existing diagram levels if available, else create
                    Ei = float(best_row["Ei"])
                    Ek = float(best_row["Ek"])
                    j_parts = str(best_row["Ji-Jk"]).split("-")
                    try:
                        lo_J = float(eval(j_parts[0].strip()))  # noqa: S307
                        up_J = float(eval(j_parts[1].strip()))  # noqa: S307
                    except Exception:
                        continue
                    lkey = (lo_term, lo_J, round(Ei, 6))
                    ukey = (up_term, up_J, round(Ek, 6))

                    lower = self.levels.get(lkey) or EnergyLevel(
                        term=lo_term, J=lo_J, energy_eV=Ei,
                        parity="even", config="",
                    )
                    upper = self.levels.get(ukey) or EnergyLevel(
                        term=up_term, J=up_J, energy_eV=Ek,
                        parity="even", config="",
                    )
                    tr = Transition(
                        lower=lower, upper=upper,
                        wave_vac=float(best_row["wave_vac"]),
                        gf=float(best_row["gf"]),
                        classification="forbidden",
                        multiplet=-1,
                    )
                    # Keep best match per term pair
                    if pk not in seen_pairs or best_res < seen_pairs[pk][0]:
                        seen_pairs[pk] = (best_res, tr)

            for pk, (_, tr) in seen_pairs.items():
                _hl[pk] = (tr.gf, tr.wave_vac)
                forbidden_new.append(tr)

        # Build final dict: pair → matched wavelength
        highlighted = {pair: wv for pair, (_, wv) in _hl.items()}
        logger.info(
            "Matched %d detected wavelengths to %d term pairs "
            "(%d forbidden synthesised).",
            len(detected_waves), len(highlighted), len(forbidden_new),
        )
        return highlighted, forbidden_new

    def plot(
        self,
        ax=None,
        *,
        layout: str = "compact",
        color_by: str = "wavelength",
        aggregate: bool = True,
        show_j: bool = False,
        figsize: tuple[float, float] = (10, 12),
        title: str | None = None,
        min_gap_eV: float = 0.8,
        col_spacing: float = 1.3,
        bar_hw: float = 0.35,
        label_fontsize: float = 9.5,
        pin_left: list[str] | None = None,
        pin_right: list[str] | None = None,
        detected_waves: list[float] | None = None,
        forbidden_waves: list[float] | None = None,
        detected_tol: float = 2.0,
        show_strongest_eV: list[tuple] | None = None,
        bg_alpha: float = 0.20,
        semi_alpha: float | None = None,
        style: str = "lines",
        color_overrides: dict[tuple[str, str], str] | None = None,
        show_legend: bool = True,
        highlight_semi_only: bool = False,
    ):
        """Plot the Grotrian diagram.

        Args:
            ax: Optional matplotlib ``Axes``. Created if ``None``.
            layout: ``"compact"``, ``"crossing"``, or ``"columns"``.
            color_by: ``"wavelength"`` or ``"classification"``.
            aggregate: Draw one line per term pair if ``True``.
            show_j: Show individual J sub-levels (``"columns"`` only).
            figsize: Figure size when creating a new figure.
            title: Plot title. Auto-generated if ``None``.
            min_gap_eV: Minimum energy gap between terms in same column.
            bar_hw: Half-width of energy level bars.
            label_fontsize: Font size for term labels.
            pin_left: Terms to place in exclusive leftmost columns.
            pin_right: Terms to place in exclusive rightmost columns.
            detected_waves: Vacuum wavelengths (Angstrom) of all detected
                lines. Matching transitions are highlighted.
            forbidden_waves: Vacuum wavelengths (Angstrom) of detected
                *forbidden* lines only (e.g. ``[Fe II]``). Used to
                synthesise forbidden transitions from same-parity level
                pairs. If ``None``, no forbidden lines are synthesised.
            detected_tol: Wavelength tolerance in Angstrom for matching.
            show_strongest_eV: List of tuples, each either
                ``((src_lo, src_hi), (tgt_lo, tgt_hi))`` or
                ``((src_lo, src_hi), (tgt_lo, tgt_hi), (lam_lo, lam_hi))``.
                For each entry, the strongest permitted transition from
                every term in the source energy range to every connected
                term in the target range is semi-highlighted.  The optional
                third element restricts to transitions whose wavelength
                falls in ``(lam_lo, lam_hi)`` Angstrom (e.g. Ly-alpha
                pumping at ``(1214, 1218)``).
            bg_alpha: Opacity of non-detected background lines (0–1).
            semi_alpha: Opacity of semi-highlighted lines (0–1).
                Defaults to ``min(bg_alpha * 2.5, 1.0)`` when ``None``.
            style: ``"lines"`` (default) draws thin transition lines;
                ``"sankey"`` draws thick semi-transparent flow bands
                with width encoding transition strength.
            color_overrides: Mapping of ``(lower_term, upper_term)`` pairs
                to colour strings.  Overrides the default wavelength/
                classification colour for those transitions.

        Returns:
            ``matplotlib.figure.Figure``
        """
        import matplotlib.pyplot as plt

        if not self.transitions:
            logger.warning("No transitions to plot.")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No transitions", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            return fig

        term_energies = self._term_mean_energies()
        term_par = self._term_parity()

        # --- match detected wavelengths ---
        highlighted: dict[tuple[str, str], float] | None = None
        extra_transitions: list[Transition] = []
        if detected_waves is not None:
            highlighted, extra_transitions = self._match_detected(
                detected_waves, detected_tol,
                forbidden_waves=forbidden_waves)

        # --- show strongest permitted lines for terms in energy range ---
        # Search the full Kurucz catalog (not just self.transitions) so
        # we find connections to every diagram term even when all
        # transitions for a pair were below gf_min.  For pairs that have
        # no transitions in self.transitions we inject a synthetic
        # Transition so the drawing code can render them.
        semi_highlighted: dict[tuple[str, str], float] | None = None
        if show_strongest_eV is not None:
            # Load full Kurucz catalog once for all source→target groups
            full_k = Table.read(DATA_DIR / "feti_kurucz.csv", format="csv")
            k_ions = np.array([s.replace("[", "").replace("]", "").strip()
                               for s in full_k["ion"]])
            k_clean = self.ion.replace("[", "").replace("]", "").strip()
            full_k = full_k[k_ions == k_clean]

            best: dict[tuple[str, str], tuple[float, object]] = {}

            for group in show_strongest_eV:
                (src_lo, src_hi), (tgt_lo, tgt_hi) = group[0], group[1]
                wave_filter = group[2] if len(group) > 2 else None
                src_terms = {t for t, e in term_energies.items()
                             if src_lo <= e <= src_hi}
                for row in full_k:
                    cls = str(row.get("classification", "")).strip()
                    if cls == "forbidden":
                        continue
                    if wave_filter is not None:
                        wv = float(row["wave_vac"])
                        if wv < wave_filter[0] or wv > wave_filter[1]:
                            continue
                    ts = str(row["terms"]).strip()
                    parts = ts.split("-")
                    if len(parts) != 2:
                        continue
                    lo, up = parts[0].strip(), parts[1].strip()
                    if lo not in term_energies or up not in term_energies:
                        continue
                    # One term in source range, the other in target range
                    lo_E = term_energies[lo]
                    up_E = term_energies[up]
                    if lo in src_terms and tgt_lo <= up_E <= tgt_hi:
                        pass  # ok
                    elif up in src_terms and tgt_lo <= lo_E <= tgt_hi:
                        pass  # ok
                    else:
                        continue
                    gf_val = float(row["gf"])
                    pair = (lo, up)
                    if pair not in best or gf_val > best[pair][0]:
                        best[pair] = (gf_val, row)

            semi_highlighted = {
                pair: float(row["wave_vac"])
                for pair, (_gf, row) in best.items()
            }

            # Find which pairs are missing from self.transitions and
            # inject synthetic Transitions so the drawing code renders them.
            existing_pairs: set[tuple[str, str]] = set()
            for tr in self.transitions:
                existing_pairs.add((tr.lower.term, tr.upper.term))
            for pair, (gf_val, row) in best.items():
                if pair in existing_pairs:
                    continue
                lo_term, up_term = pair
                j_parts = str(row["Ji-Jk"]).split("-")
                try:
                    lo_J = float(eval(j_parts[0].strip()))  # noqa: S307
                    up_J = float(eval(j_parts[1].strip()))  # noqa: S307
                except Exception:
                    continue
                Ei = float(row["Ei"])
                Ek = float(row["Ek"])
                lkey = (lo_term, lo_J, round(Ei, 6))
                ukey = (up_term, up_J, round(Ek, 6))
                lower = self.levels.get(lkey) or EnergyLevel(
                    term=lo_term, J=lo_J, energy_eV=Ei,
                    parity="even", config="",
                )
                upper = self.levels.get(ukey) or EnergyLevel(
                    term=up_term, J=up_J, energy_eV=Ek,
                    parity="odd", config="",
                )
                extra_transitions.append(Transition(
                    lower=lower, upper=upper,
                    wave_vac=float(row["wave_vac"]),
                    gf=gf_val,
                    classification=str(row.get("classification", "permitted")),
                    multiplet=-1,
                ))
            logger.info("Semi-highlighted %d term pairs (%d groups), "
                        "%d injected.", len(semi_highlighted),
                        len(show_strongest_eV),
                        len(semi_highlighted) - len(
                            semi_highlighted.keys() & existing_pairs))

        # --- x-position layout ---
        if layout == "compact":
            x_pos = self._layout_compact(
                term_energies, min_gap_eV=min_gap_eV, col_spacing=col_spacing,
                pin_left=pin_left, pin_right=pin_right)
        elif layout == "crossing":
            x_pos = self._layout_crossing(
                term_energies, min_gap_eV=min_gap_eV, col_spacing=col_spacing,
                pin_left=pin_left, pin_right=pin_right)
        elif layout == "columns":
            x_pos = self._layout_columns(term_energies, term_par)
        else:
            raise ValueError(
                f"Unknown layout '{layout}'. "
                "Use 'compact', 'crossing', or 'columns'.")

        # --- create figure ---
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            _owns_fig = True
        else:
            fig = ax.get_figure()
            _owns_fig = False

        # --- set axis limits early (needed for display-coordinate gap calc) ---
        all_energies = list(term_energies.values())
        ylo, yhi = min(all_energies), max(all_energies)
        span = max(yhi - ylo, 1.0)
        ax.set_ylim(max(ylo - 0.06 * span, -0.1), yhi + 0.06 * span)
        if x_pos:
            ax.set_xlim(min(x_pos.values()) - 1.5,
                        max(x_pos.values()) + 1.5)

        # --- optionally restrict highlights to semi-highlighted population ---
        if highlight_semi_only and highlighted and semi_highlighted:
            semi_terms: set[str] = set()
            for lo, up in semi_highlighted:
                semi_terms.add(lo)
                semi_terms.add(up)
            highlighted = {
                pair: wave for pair, wave in highlighted.items()
                if pair[0] in semi_terms and pair[1] in semi_terms
            }

        # --- draw transitions (behind levels) ---
        if semi_alpha is None:
            semi_alpha = min(bg_alpha * 2.5, 1.0)
        self._draw_transitions(ax, x_pos, term_energies, layout, color_by,
                               aggregate, bar_hw=bar_hw,
                               highlighted=highlighted,
                               extra_transitions=extra_transitions,
                               semi_highlighted=semi_highlighted,
                               bg_alpha=bg_alpha,
                               semi_alpha=semi_alpha,
                               style=style,
                               color_overrides=color_overrides)

        # --- draw energy levels (on top) ---
        if layout == "compact":
            self._draw_levels_compact(ax, x_pos, term_energies, bar_hw,
                                      label_fontsize)
        else:
            self._draw_levels_columns(ax, x_pos, bar_hw, show_j,
                                      label_fontsize)

        # --- styling ---
        self._style_axes(ax, x_pos, term_energies, term_par, layout,
                         color_by, title, detected_waves=detected_waves,
                         extra_transitions=extra_transitions,
                         owns_fig=_owns_fig, show_legend=show_legend)

        return fig

    # ---- drawing helpers -------------------------------------------------

    def _draw_levels_compact(
        self, ax, x_pos: dict[str, float],
        term_energies: dict[str, float],
        bar_hw: float, label_fontsize: float,
    ) -> None:
        """Draw one thick bar per term with label on a connector line."""
        x_vals = sorted(set(x_pos.values()))
        x_max = max(x_vals) if x_vals else 0

        # Group terms by column for label de-conflicting
        col_terms: dict[float, list[str]] = defaultdict(list)
        for term, x in x_pos.items():
            col_terms[x].append(term)

        # For rightmost ~30% of columns, place labels on left; rest on right
        right_threshold = x_max * 0.72 if x_max > 0 else 0

        for term, x in x_pos.items():
            e = term_energies[term]

            # Thick black bar
            ax.hlines(e, x - bar_hw, x + bar_hw,
                      color="0.15", linewidth=2.8, zorder=5,
                      capstyle="round")

            # Short connector line + label
            label_on_right = x <= right_threshold
            conn = 0.15
            if label_on_right:
                lx = x + bar_hw + conn
                ha = "left"
                ax.hlines(e, x + bar_hw, lx, color="0.4", linewidth=0.6,
                          zorder=4)
            else:
                lx = x - bar_hw - conn
                ha = "right"
                ax.hlines(e, lx, x - bar_hw, color="0.4", linewidth=0.6,
                          zorder=4)

            ax.text(lx, e, _term_to_latex(term),
                    fontsize=label_fontsize, va="center", ha=ha,
                    fontweight="bold", zorder=6)

    def _draw_levels_columns(
        self, ax, x_pos: dict[str, float],
        bar_hw: float, show_j: bool, label_fontsize: float,
    ) -> None:
        """Draw individual J sub-level bars in parity-separated columns."""
        for lv in self.levels.values():
            if lv.term not in x_pos:
                continue
            x = x_pos[lv.term]
            ax.hlines(lv.energy_eV, x - bar_hw, x + bar_hw,
                      color="0.15", linewidth=1.5, zorder=5,
                      capstyle="round")
            if show_j:
                ax.text(x + bar_hw + 0.06, lv.energy_eV, _j_to_str(lv.J),
                        fontsize=6, va="center", color="0.4")

        # Term labels below each column
        term_min_e: dict[str, float] = {}
        for lv in self.levels.values():
            if lv.term in x_pos:
                term_min_e[lv.term] = min(
                    term_min_e.get(lv.term, 1e9), lv.energy_eV)

        for term, x in x_pos.items():
            e = term_min_e.get(term, 0)
            ax.text(x, e - 0.15, _term_to_latex(term),
                    fontsize=label_fontsize, ha="center", va="top",
                    fontweight="bold")

    def _draw_transitions(
        self, ax, x_pos: dict[str, float],
        term_energies: dict[str, float],
        layout: str, color_by: str,
        aggregate: bool,
        bar_hw: float = 0.35,
        highlighted: dict[tuple[str, str], float] | None = None,
        extra_transitions: list[Transition] | None = None,
        semi_highlighted: dict[tuple[str, str], float] | None = None,
        bg_alpha: float = 0.20,
        semi_alpha: float = 0.50,
        style: str = "lines",
        color_overrides: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Draw transition lines between levels."""
        if style == "sankey":
            self._draw_transitions_sankey(
                ax, x_pos, term_energies, color_by, bar_hw=bar_hw,
                highlighted=highlighted,
                extra_transitions=extra_transitions or [],
                semi_highlighted=semi_highlighted,
                bg_alpha=bg_alpha, semi_alpha=semi_alpha,
                color_overrides=color_overrides)
        elif aggregate and layout in ("compact", "crossing"):
            self._draw_transitions_aggregated(
                ax, x_pos, term_energies, color_by, bar_hw=bar_hw,
                highlighted=highlighted,
                extra_transitions=extra_transitions or [],
                semi_highlighted=semi_highlighted,
                bg_alpha=bg_alpha, semi_alpha=semi_alpha,
                color_overrides=color_overrides)
        else:
            self._draw_transitions_individual(
                ax, x_pos, term_energies, layout, color_by)

    def _draw_transitions_sankey(
        self, ax, x_pos: dict[str, float],
        term_energies: dict[str, float],
        color_by: str,
        bar_hw: float = 0.35,
        highlighted: dict[tuple[str, str], float] | None = None,
        extra_transitions: list[Transition] | None = None,
        semi_highlighted: dict[tuple[str, str], float] | None = None,
        bg_alpha: float = 0.20,
        semi_alpha: float = 0.50,
        width_range: tuple[float, float] = (0.02, 0.15),
        color_overrides: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Draw Sankey-style flow bands between term pairs.

        Each transition pair is drawn as a thick semi-transparent ribbon
        with width proportional to ``log10(gf)``.  Ribbons are stacked at
        each term bar so they don't overlap.

        Args:
            width_range: ``(min, max)`` band half-width in data (eV)
                coordinates.
        """
        import matplotlib.patches as mpatches
        from matplotlib.path import Path as MplPath

        # Group by term pair (same as aggregated)
        pair_data: dict[tuple[str, str], list[Transition]] = defaultdict(list)
        all_transitions = list(self.transitions) + (extra_transitions or [])
        for tr in all_transitions:
            if tr.lower.term in x_pos and tr.upper.term in x_pos:
                pair_data[(tr.lower.term, tr.upper.term)].append(tr)

        if not pair_data:
            return

        # Per-pair: strongest gf, representative wavelength, classification
        pair_info: dict[tuple[str, str], dict] = {}
        log_gfs = []
        for pair, trs in pair_data.items():
            strongest = max(trs, key=lambda t: t.gf)
            is_forbidden = all(t.classification == "forbidden" for t in trs)
            is_detected = highlighted is not None and pair in highlighted
            is_semi = (semi_highlighted is not None
                       and pair in semi_highlighted)
            if is_detected:
                repr_wave = highlighted[pair]
            elif is_semi:
                repr_wave = semi_highlighted[pair]
            else:
                repr_wave = strongest.wave_vac
            pair_info[pair] = dict(
                gf=strongest.gf, wave=repr_wave,
                forbidden=is_forbidden, detected=is_detected,
            )
            log_gfs.append(np.log10(max(strongest.gf, 1e-10)))

        # Scale log(gf) linearly into width_range
        log_gfs = np.array(log_gfs)
        lgf_min, lgf_max = log_gfs.min(), log_gfs.max()
        lgf_span = max(lgf_max - lgf_min, 1.0)
        w_min, w_max = width_range

        pair_hw: dict[tuple[str, str], float] = {}
        for pair, lgf in zip(pair_data.keys(), log_gfs):
            frac = (lgf - lgf_min) / lgf_span
            pair_hw[pair] = w_min + frac * (w_max - w_min)

        # Stack bands at each term: allocate y-offsets along the bar
        # Collect all pairs touching each term, sorted by partner x
        term_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for pair in pair_data:
            lo, up = pair
            term_pairs[lo].append(pair)
            term_pairs[up].append(pair)

        def _partner(term: str, pair: tuple[str, str]) -> str:
            return pair[1] if pair[0] == term else pair[0]

        # For each term, sort connected pairs by partner x-position and
        # compute cumulative y-offset (centre of each band's slot)
        term_offsets: dict[str, dict[tuple[str, str], float]] = {}
        for term, pairs in term_pairs.items():
            # Sort by partner x so bands fan out naturally
            sorted_pairs = sorted(
                pairs, key=lambda p: x_pos.get(_partner(term, p), 0))
            # Total width consumed
            total_w = sum(2 * pair_hw[p] for p in sorted_pairs)
            # Centre the stack on the bar x-position
            cursor = -total_w / 2
            offsets: dict[tuple[str, str], float] = {}
            for p in sorted_pairs:
                hw = pair_hw[p]
                offsets[p] = cursor + hw  # centre of this band
                cursor += 2 * hw
            term_offsets[term] = offsets

        # Draw each band as a Bezier ribbon
        for pair, trs in pair_data.items():
            lo_term, up_term = pair
            info = pair_info[pair]
            hw = pair_hw[pair]

            x_lo = x_pos[lo_term]
            x_up = x_pos[up_term]
            y_lo = term_energies[lo_term]
            y_up = term_energies[up_term]

            # x-offsets from stacking (in data eV, applied to x)
            x_off_lo = term_offsets[lo_term][pair]
            x_off_up = term_offsets[up_term][pair]

            # Band start/end centres
            x0 = x_lo + x_off_lo
            x1 = x_up + x_off_up

            # Colour
            if color_overrides and pair in color_overrides:
                color = color_overrides[pair]
            elif color_by == "wavelength":
                color = _wave_color(info["wave"])
            else:
                color = "#2060B0"

            has_override = color_overrides and pair in color_overrides
            is_semi = (semi_highlighted is not None
                       and pair in semi_highlighted)

            # Alpha and edge
            if highlighted is not None and not info["detected"]:
                if has_override or is_semi:
                    fill_alpha = semi_alpha
                    edge_alpha = fill_alpha
                    edge_lw = 0.5
                else:
                    fill_alpha = bg_alpha * 0.5
                    edge_alpha = bg_alpha * 0.3
                    edge_lw = 0.3
                    color = _desaturate(color, bg_alpha)
            elif info["detected"]:
                fill_alpha = 0.7
                edge_alpha = 0.9
                edge_lw = 1.0
            else:
                fill_alpha = 0.35
                edge_alpha = 0.5
                edge_lw = 0.5

            if info["forbidden"]:
                edge_ls = "--"
                if not info["detected"]:
                    edge_lw *= 0.6
            else:
                edge_ls = "-"

            zorder = 3 if info["detected"] else (1 if is_semi else 0)

            # Build Bezier ribbon: two cubic curves (top + bottom edges)
            # Control points at 1/3 and 2/3 of vertical distance
            dy = y_up - y_lo
            cp_y1 = y_lo + dy / 3
            cp_y2 = y_lo + 2 * dy / 3

            # Top edge (left to right: lo_top -> up_top)
            # Bottom edge (right to left: up_bot -> lo_bot)
            verts = [
                (x0 - hw, y_lo),       # start bottom-left
                (x0 - hw, cp_y1),      # control 1
                (x1 - hw, cp_y2),      # control 2
                (x1 - hw, y_up),       # end top-left
                (x1 + hw, y_up),       # top-right
                (x1 + hw, cp_y2),      # control 2 (return)
                (x0 + hw, cp_y1),      # control 1 (return)
                (x0 + hw, y_lo),       # back to start
                (x0 - hw, y_lo),       # close
            ]
            codes = [
                MplPath.MOVETO,
                MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.LINETO,
                MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                MplPath.CLOSEPOLY,
            ]
            path = MplPath(verts, codes)
            patch = mpatches.PathPatch(
                path,
                facecolor=color, alpha=fill_alpha,
                edgecolor=color, linewidth=edge_lw,
                linestyle=edge_ls, zorder=zorder,
            )
            patch.set_edgecolor((*patch.get_edgecolor()[:3], edge_alpha))
            ax.add_patch(patch)

    @staticmethod
    def _spread_endpoints(
        term_connections: dict[str, list[tuple[str, str]]],
        x_pos: dict[str, float],
        bar_hw: float,
    ) -> dict[tuple[str, str, str], float]:
        """Compute spread x-offsets for transition endpoints along each bar.

        For each term, lines connecting to other terms are sorted by the
        partner's x-position so that lines fan out naturally without crossing.
        Endpoints are distributed evenly across the bar width.

        Args:
            term_connections: ``{term: [(partner_term, pair_key), ...]}``
                where *pair_key* identifies the transition pair.
            x_pos: Term x-positions.
            bar_hw: Half-width of the energy-level bar.

        Returns:
            ``{(term, partner_term, pair_key): x_offset}``
        """
        offsets: dict[tuple[str, str, str], float] = {}
        margin = 0.15  # fraction of bar kept free at each end

        for term, partners in term_connections.items():
            if not partners:
                continue
            x_self = x_pos[term]
            # Sort partners by their x-position for natural fan-out
            sorted_partners = sorted(
                partners,
                key=lambda p: x_pos.get(p[0], x_self),
            )
            n = len(sorted_partners)
            usable = bar_hw * 2 * (1 - 2 * margin)
            start = x_self - bar_hw + bar_hw * 2 * margin
            for i, (partner, pair_key) in enumerate(sorted_partners):
                frac = (i + 0.5) / n if n > 1 else 0.5
                offsets[(term, partner, pair_key)] = start + frac * usable

        return offsets

    def _draw_transitions_aggregated(
        self, ax, x_pos: dict[str, float],
        term_energies: dict[str, float],
        color_by: str,
        bar_hw: float = 0.35,
        show_wavelengths: bool = True,
        wave_fontsize: float = 7.5,
        highlighted: dict[tuple[str, str], float] | None = None,
        extra_transitions: list[Transition] | None = None,
        semi_highlighted: dict[tuple[str, str], float] | None = None,
        bg_alpha: float = 0.20,
        semi_alpha: float = 0.50,
        color_overrides: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Draw one line per term pair from bar centres, with labels.

        Lines that receive a wavelength label are drawn in two segments
        with a gap at the midpoint so the label text sits cleanly inside
        the gap (no opaque background needed).
        """
        # Group by term pair (include extra synthesised transitions)
        pair_data: dict[tuple[str, str], list[Transition]] = defaultdict(list)
        all_transitions = list(self.transitions) + (extra_transitions or [])
        for tr in all_transitions:
            if tr.lower.term in x_pos and tr.upper.term in x_pos:
                pair_data[(tr.lower.term, tr.upper.term)].append(tr)

        if not pair_data:
            return

        # Collect drawing specs; lines drawn after label positions decided.
        # Each entry: (x_lo, y_lo, x_up, y_up, color, ls, lw, alpha, zorder,
        #              wave_str | None, gf)
        draw_specs: list[tuple] = []

        for pair, trs in pair_data.items():
            lo_term, up_term = pair
            y_lo = term_energies[lo_term]
            y_up = term_energies[up_term]
            x_lo = x_pos[lo_term]
            x_up = x_pos[up_term]

            strongest = max(trs, key=lambda t: t.gf)
            is_forbidden = all(t.classification == "forbidden" for t in trs)
            is_detected = (highlighted is not None
                           and pair in highlighted)
            # For detected pairs, use the matched Kurucz wavelength;
            # for semi-highlighted, use the wavelength from the filter match.
            is_semi = (semi_highlighted is not None
                       and pair in semi_highlighted)
            if is_detected:
                repr_wave = highlighted[pair]
            elif is_semi:
                repr_wave = semi_highlighted[pair]
            else:
                repr_wave = strongest.wave_vac

            # Colour by wavelength regime (forbidden same as permitted)
            if color_overrides and pair in color_overrides:
                color = color_overrides[pair]
            elif color_by == "wavelength":
                color = _wave_color(repr_wave)
            else:
                color = "#2060B0"
            ls = "--" if is_forbidden else "-"

            # Singlets (only one J->J' transition) always stay visible
            is_singlet = len(trs) == 1

            has_override = color_overrides and pair in color_overrides

            # Dim non-detected lines when highlights are active
            if highlighted is not None and not is_detected:
                if has_override or is_semi:
                    lw, alpha, zorder = 1.0, semi_alpha, 1
                else:
                    lw = 0.3 + 0.5 * bg_alpha
                    alpha, zorder = bg_alpha, 0
                    color = _desaturate(color, bg_alpha)
            elif is_detected:
                lw, alpha, zorder = 1.8, 0.95, 3
            else:
                lw, alpha, zorder = 0.8, 0.6, 1

            # Forbidden (dashed) lines thinner
            if is_forbidden:
                lw *= 0.5 if not is_detected else 0.7

            # Label text (or None if not labelled)
            wave_str = None
            should_label = show_wavelengths and (
                highlighted is None or is_detected or is_semi)
            if should_label:
                wave_str = f"{repr_wave:.0f}"

            draw_specs.append((
                x_lo, y_lo, x_up, y_up, color, ls, lw, alpha, zorder,
                wave_str, strongest.gf, is_semi,
            ))

        # Draw lines + inline labels
        if show_wavelengths:
            self._draw_lines_with_inline_labels(
                ax, draw_specs, wave_fontsize)
        else:
            for spec in draw_specs:
                x_lo, y_lo, x_up, y_up, color, ls, lw, alpha, zorder = spec[:9]
                ax.plot([x_lo, x_up], [y_lo, y_up],
                        color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder,
                        solid_capstyle="round")

    @staticmethod
    def _draw_lines_with_inline_labels(
        ax,
        draw_specs: list[tuple],
        fontsize: float,
    ) -> None:
        """Draw transition lines with inline wavelength labels.

        For labelled lines the segment is split into two parts with a gap
        at the midpoint so the text sits cleanly inside the gap.  The gap
        size is computed from the text extent in display coordinates.
        Strongest transitions are labelled first; a grid-based occupancy
        check prevents overlapping labels.
        """
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties

        fig = ax.get_figure()
        fig.canvas.draw()
        trans = ax.transData
        inv_trans = ax.transData.inverted()

        # Approximate label width in display pixels using TextPath
        fp = FontProperties(size=fontsize)

        def _text_width_px(text: str) -> float:
            tp = TextPath((0, 0), text, prop=fp)
            bb = tp.get_extents()
            # TextPath extents are in points; convert to pixels (≈ 1:1 at 72 dpi)
            dpi = fig.get_dpi()
            return bb.width * dpi / 72.0

        # Sort by gf descending so strongest get labels first
        indexed = list(enumerate(draw_specs))
        indexed.sort(key=lambda x: -x[1][10])  # gf is index 10

        cell_px = 50.0
        occupied: set[tuple[int, int]] = set()
        # Track which specs get a label and the gap fraction
        label_info: dict[int, tuple[str, float, float]] = {}  # idx -> (text, rot, gap_frac)

        for orig_idx, spec in indexed:
            (x_lo, y_lo, x_up, y_up, color, ls, lw, alpha, zorder,
             wave_str, gf) = spec[:11]
            if wave_str is None:
                continue

            mx = 0.5 * (x_lo + x_up)
            my = 0.5 * (y_lo + y_up)

            # Display-space collision check
            p_mid = trans.transform((mx, my))
            cx = int(round(p_mid[0] / cell_px))
            cy = int(round(p_mid[1] / cell_px))
            if (cx, cy) in occupied:
                continue
            occupied.add((cx, cy))

            # Angle in display coordinates
            p_lo = trans.transform((x_lo, y_lo))
            p_up = trans.transform((x_up, y_up))
            dx_px = p_up[0] - p_lo[0]
            dy_px = p_up[1] - p_lo[1]
            line_len_px = np.hypot(dx_px, dy_px)
            angle = np.degrees(np.arctan2(dy_px, dx_px))
            rot = angle
            if rot > 90:
                rot -= 180
            elif rot < -90:
                rot += 180

            # Gap as fraction of segment length
            text_w_px = _text_width_px(wave_str) * 1.4 + 8
            if line_len_px > 0:
                gap_frac = max(min(text_w_px / line_len_px, 0.40), 0.06)
            else:
                gap_frac = 0.0

            label_info[orig_idx] = (wave_str, rot, gap_frac)

        # Now draw all lines
        for idx, spec in enumerate(draw_specs):
            (x_lo, y_lo, x_up, y_up, color, ls, lw, alpha, zorder,
             wave_str, gf) = spec[:11]
            is_semi = spec[11] if len(spec) > 11 else False

            if idx in label_info:
                text, rot, gap_frac = label_info[idx]
                # Split line into two segments around midpoint
                t_lo = 0.5 - gap_frac / 2
                t_hi = 0.5 + gap_frac / 2
                # Segment 1: start -> gap start
                ax.plot(
                    [x_lo, x_lo + t_lo * (x_up - x_lo)],
                    [y_lo, y_lo + t_lo * (y_up - y_lo)],
                    color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder,
                    solid_capstyle="round")
                # Segment 2: gap end -> end
                ax.plot(
                    [x_lo + t_hi * (x_up - x_lo), x_up],
                    [y_lo + t_hi * (y_up - y_lo), y_up],
                    color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder,
                    solid_capstyle="round")
                # Label in the gap — semi-highlighted labels are smaller/fainter
                mx = 0.5 * (x_lo + x_up)
                my = 0.5 * (y_lo + y_up)
                lbl_fs = fontsize * 0.85 if is_semi else fontsize
                lbl_alpha = 0.6 if is_semi else 1.0
                ax.text(
                    mx, my, text,
                    fontsize=lbl_fs, color=color, alpha=lbl_alpha,
                    ha="center", va="center",
                    rotation=rot, rotation_mode="anchor",
                    zorder=zorder + 1,
                )
            else:
                # No label — draw as single line
                ax.plot([x_lo, x_up], [y_lo, y_up],
                        color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder,
                        solid_capstyle="round")

    def _draw_transitions_individual(
        self, ax, x_pos: dict[str, float],
        term_energies: dict[str, float],
        layout: str, color_by: str,
    ) -> None:
        """Draw every transition as an individual line."""
        gf_arr = np.array([tr.gf for tr in self.transitions])
        gf_arr = np.clip(gf_arr, 1e-10, None)
        log_gf = np.log10(gf_arr)
        lgf_min, lgf_max = log_gf.min(), log_gf.max()
        lgf_range = max(lgf_max - lgf_min, 1.0)

        for tr, lgf in zip(self.transitions, log_gf):
            x_lo = x_pos.get(tr.lower.term)
            x_up = x_pos.get(tr.upper.term)
            if x_lo is None or x_up is None:
                continue

            if layout == "compact":
                y_lo = term_energies[tr.lower.term]
                y_up = term_energies[tr.upper.term]
            else:
                y_lo = tr.lower.energy_eV
                y_up = tr.upper.energy_eV

            frac = (lgf - lgf_min) / lgf_range
            lw = 0.3 + 1.8 * frac
            alpha = 0.2 + 0.6 * frac

            is_forbidden = tr.classification == "forbidden"
            if is_forbidden:
                color, ls = "#999999", ":"
                alpha, lw = 0.4, max(lw * 0.5, 0.3)
            elif color_by == "wavelength":
                color = _wave_color(tr.wave_vac)
                ls = "-"
            else:
                color, ls = "#2060B0", "-"

            ax.plot([x_lo, x_up], [y_lo, y_up],
                    color=color, ls=ls, lw=lw, alpha=alpha, zorder=1,
                    solid_capstyle="round")

    def _style_axes(
        self, ax, x_pos: dict[str, float],
        term_energies: dict[str, float],
        term_par: dict[str, str],
        layout: str, color_by: str,
        title: str | None,
        detected_waves: list[float] | None = None,
        extra_transitions: list[Transition] | None = None,
        owns_fig: bool = True,
        show_legend: bool = True,
    ) -> None:
        """Apply final styling, labels, and legend."""
        from matplotlib.lines import Line2D

        ax.set_ylabel("Excitation energy (eV)", fontsize=12)
        ax.tick_params(axis="y", labelsize=10, direction="in")

        ax.set_xlabel("Atomic term", fontsize=12)
        ax.set_xticks([])

        # Spines — minimal
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("0.4")
        ax.spines["bottom"].set_visible(layout == "compact")
        if layout == "compact":
            ax.spines["bottom"].set_color("0.6")

        # Axis limits
        all_energies = list(term_energies.values())
        ylo, yhi = min(all_energies), max(all_energies)
        span = max(yhi - ylo, 1.0)
        ax.set_ylim(max(ylo - 0.06 * span, -0.1), yhi + 0.06 * span)

        if x_pos:
            xlo = min(x_pos.values()) - 1.5
            xhi = max(x_pos.values()) + 1.5
            ax.set_xlim(xlo, xhi)

        # Title
        if title is None:
            title = f"{self.ion} Grotrian Diagram"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Legend — regimes present + forbidden + detected
        handles = []
        all_trs = list(self.transitions) + (extra_transitions or [])
        if color_by == "wavelength":
            regimes_present: set[str] = set()
            for tr in all_trs:
                regimes_present.add(_wave_regime(tr.wave_vac))
            for name, (color, _) in WAVE_COLORS.items():
                if name in regimes_present:
                    handles.append(
                        Line2D([0], [0], color=color, lw=2.0, label=name))
        else:
            handles.append(
                Line2D([0], [0], color="#2060B0", lw=2.0, label="permitted"))

        has_forbidden = any(
            tr.classification == "forbidden" for tr in all_trs)
        if has_forbidden:
            handles.append(
                Line2D([0], [0], color="0.4", lw=1.5, ls="--",
                       label="forbidden"))

        if detected_waves is not None:
            handles.append(
                Line2D([0], [0], color="0.15", lw=2.5, alpha=0.9,
                       label="detected"))
            handles.append(
                Line2D([0], [0], color="#bbbbbb", lw=0.5, alpha=0.3,
                       label="not detected"))

        if handles and show_legend:
            ax.legend(handles=handles, loc="upper right", fontsize=8,
                      frameon=False, handlelength=2.0)

        if owns_fig:
            fig = ax.get_figure()
            fig.tight_layout()

    # ---- interactive plotly explorer -----------------------------------------

    def plot_plotly(
        self,
        *,
        layout: str = "compact",
        color_by: str = "wavelength",
        min_gap_eV: float = 0.8,
        col_spacing: float = 1.3,
        bar_hw: float = 0.35,
        pin_left: list[str] | None = None,
        pin_right: list[str] | None = None,
        detected_waves: list[float] | None = None,
        forbidden_waves: list[float] | None = None,
        detected_tol: float = 2.0,
        title: str | None = None,
        color_overrides: dict[tuple[str, str], str] | None = None,
        height: int = 800,
    ):
        """Create an interactive Plotly Grotrian diagram.

        Returns a ``plotly.graph_objects.Figure`` with hover tooltips
        showing wavelength, gf, classification and detection status
        for every transition.

        Args:
            layout: ``"compact"``, ``"crossing"``, or ``"columns"``.
            color_by: ``"wavelength"`` or ``"classification"``.
            min_gap_eV: Minimum energy gap between terms in same column.
            col_spacing: Horizontal spacing between columns.
            bar_hw: Half-width of energy level bars.
            pin_left: Terms for exclusive leftmost columns.
            pin_right: Terms for exclusive rightmost columns.
            detected_waves: Vacuum wavelengths (Å) of detected lines.
            forbidden_waves: Vacuum wavelengths (Å) of detected forbidden
                lines only.
            detected_tol: Wavelength tolerance in Å for matching.
            title: Plot title.
            color_overrides: Custom colours for ``(lower, upper)`` pairs.
            height: Figure height in pixels.

        Returns:
            ``plotly.graph_objects.Figure``
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for interactive diagrams. "
                "Install with: pip install plotly") from None

        term_energies = self._term_mean_energies()
        term_par = self._term_parity()

        # Layout
        if layout == "compact":
            x_pos = self._layout_compact(
                term_energies, min_gap_eV=min_gap_eV,
                col_spacing=col_spacing,
                pin_left=pin_left, pin_right=pin_right)
        elif layout == "crossing":
            x_pos = self._layout_crossing(
                term_energies, min_gap_eV=min_gap_eV,
                col_spacing=col_spacing,
                pin_left=pin_left, pin_right=pin_right)
        else:
            x_pos = self._layout_columns(term_energies, term_par)

        # Match detected wavelengths
        highlighted: dict[tuple[str, str], float] | None = None
        extra_transitions: list[Transition] = []
        if detected_waves is not None:
            highlighted, extra_transitions = self._match_detected(
                detected_waves, detected_tol,
                forbidden_waves=forbidden_waves)

        # Group transitions by term pair (aggregated)
        pair_data: dict[tuple[str, str], list[Transition]] = defaultdict(list)
        for tr in list(self.transitions) + extra_transitions:
            if tr.lower.term in x_pos and tr.upper.term in x_pos:
                pair_data[(tr.lower.term, tr.upper.term)].append(tr)

        fig = go.Figure()

        # --- Transitions first (level bars added after to render on top) ---
        for pair, trs in pair_data.items():
            lo_term, up_term = pair
            strongest = max(trs, key=lambda t: t.gf)
            is_forbidden = all(
                t.classification == "forbidden" for t in trs)
            is_detected = (highlighted is not None
                           and pair in highlighted)

            x_lo, x_up = x_pos[lo_term], x_pos[up_term]
            y_lo, y_up = term_energies[lo_term], term_energies[up_term]

            wave = highlighted[pair] if is_detected else strongest.wave_vac
            regime = _wave_regime(wave)

            # Colour
            if color_overrides and pair in color_overrides:
                color = color_overrides[pair]
            elif color_by == "wavelength":
                color = _wave_color(wave)
            else:
                color = "#2060B0"

            # Style
            if is_detected:
                lw, opacity = 2.5, 0.9
            else:
                lw, opacity = 1.0, 0.25
            dash = "dash" if is_forbidden else "solid"

            # Classification label
            cls = strongest.classification
            n_j = len(trs)
            det_mark = "\u2714 Detected" if is_detected else "Not detected"

            hover = (
                f"<b>{lo_term} \u2192 {up_term}</b><br>"
                f"\u03bb = {wave:.1f} \u00c5 ({regime})<br>"
                f"gf = {strongest.gf:.4f}<br>"
                f"{cls} &middot; {n_j} J\u2192J\u2032<br>"
                f"<b>{det_mark}</b>"
                f"<extra></extra>"
            )

            # 5 points along the line so hover can distinguish the
            # middle (single-transition highlight) from endpoints
            # (level-hover highlight).
            xs = np.linspace(x_lo, x_up, 5).tolist()
            ys = np.linspace(y_lo, y_up, 5).tolist()

            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color=color, width=lw, dash=dash),
                opacity=opacity,
                hovertemplate=hover,
                showlegend=False,
                meta={"type": "transition", "lo": lo_term,
                      "up": up_term, "wave": f"{wave:.0f}"},
            ))

        # --- Energy level bars (as traces for hover interactivity) ---
        for term, x in x_pos.items():
            e = term_energies[term]
            fig.add_trace(go.Scatter(
                x=[x - bar_hw, x + bar_hw],
                y=[e, e],
                mode="lines",
                line=dict(color="black", width=2.5),
                hovertemplate=(
                    f"<b>{term}</b><br>"
                    f"{e:.3f} eV<extra></extra>"),
                showlegend=False,
                meta={"type": "level", "term": term},
            ))
            fig.add_annotation(
                x=x, y=e, text=f"<b>{term}</b>",
                showarrow=False, font=dict(size=9, color="#333"),
                yshift=10, xanchor="center",
            )

        # --- Layout ---
        fig.update_layout(
            title=dict(text=title or f"{self.ion} Grotrian Diagram",
                       font=dict(size=16)),
            yaxis=dict(title="Excitation energy (eV)", gridcolor="#eee"),
            xaxis=dict(showticklabels=False, title="",
                       showgrid=False, zeroline=False),
            hovermode="closest",
            plot_bgcolor="white",
            height=height,
            margin=dict(l=60, r=20, t=50, b=30),
        )

        return fig
