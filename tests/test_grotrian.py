"""Tests for the Grotrian diagram module."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from dotfit.grotrian import (
    EnergyLevel,
    GrotrianDiagram,
    Transition,
    _get_parity,
    _j_to_str,
    _term_to_latex,
)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestGetParity:
    def test_even(self):
        assert _get_parity("(5D)4s") == "even"

    def test_odd(self):
        assert _get_parity("(5D)4p") == "odd"

    def test_d7(self):
        # d^7 => l=2*7=14 => even
        assert _get_parity("d7") == "even"

    def test_empty(self):
        assert _get_parity("") == "even"


class TestTermToLatex:
    def test_prefixed(self):
        result = _term_to_latex("a6D")
        assert r"^{6}" in result
        assert "D" in result
        assert result.startswith("$")

    def test_odd_parity_star(self):
        result = _term_to_latex("z4F*")
        assert r"^\circ" in result

    def test_no_prefix(self):
        result = _term_to_latex("4F")
        assert result.startswith("$")
        assert "F" in result


class TestJToStr:
    def test_integer(self):
        assert _j_to_str(2.0) == "2"

    def test_half_integer(self):
        assert _j_to_str(4.5) == "9/2"

    def test_zero(self):
        assert _j_to_str(0.0) == "0"


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestEnergyLevel:
    def test_key(self):
        lv = EnergyLevel("a6D", 4.5, 0.0, "even", "(5D)4s")
        assert lv.key == ("a6D", 4.5, 0.0)


# ---------------------------------------------------------------------------
# GrotrianDiagram tests
# ---------------------------------------------------------------------------


def _make_test_table() -> Table:
    """Build a minimal table mimicking the Kurucz CSV format."""
    return Table(
        rows=[
            {
                "ion": "Fe II",
                "wave_vac": 5169.0,
                "Ei": 2.891,
                "Ek": 5.290,
                "gf": 0.045,
                "configuration": "(5D)4s-(5D)4p",
                "terms": "a6S-z6P",
                "Ji-Jk": "5/2-7/2",
                "classification": "permitted",
                "multiplet": 42,
            },
            {
                "ion": "Fe II",
                "wave_vac": 4924.0,
                "Ei": 2.891,
                "Ek": 5.408,
                "gf": 0.032,
                "configuration": "(5D)4s-(5D)4p",
                "terms": "a6S-z6P",
                "Ji-Jk": "5/2-5/2",
                "classification": "permitted",
                "multiplet": 42,
            },
            {
                "ion": "Fe II",
                "wave_vac": 4583.0,
                "Ei": 2.807,
                "Ek": 5.511,
                "gf": 0.080,
                "configuration": "(5D)4s-(5D)4p",
                "terms": "a4F-z4D",
                "Ji-Jk": "9/2-7/2",
                "classification": "permitted",
                "multiplet": 38,
            },
            {
                "ion": "Ti II",
                "wave_vac": 3349.0,
                "Ei": 0.049,
                "Ek": 3.751,
                "gf": 0.10,
                "configuration": "3d2(3F)4s-3d2(3F)4p",
                "terms": "a4F-z4G",
                "Ji-Jk": "3/2-5/2",
                "classification": "permitted",
                "multiplet": 1,
            },
        ]
    )


class TestFromTable:
    def test_loads_fe_ii(self):
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Fe II")
        assert gd.ion == "Fe II"
        assert len(gd.transitions) == 3
        assert len(gd.levels) >= 4  # at least 4 distinct levels

    def test_loads_ti_ii(self):
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Ti II")
        assert gd.ion == "Ti II"
        assert len(gd.transitions) == 1

    def test_empty_ion(self):
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Ca II")
        assert len(gd.transitions) == 0


class TestSelect:
    def test_wave_range(self):
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Fe II")
        sub = gd.select(wave_range=(4500, 5000), gf_min=0.0)
        waves = [tr.wave_vac for tr in sub.transitions]
        assert all(4500 <= w <= 5000 for w in waves)

    def test_gf_min(self):
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Fe II")
        sub = gd.select(gf_min=0.05)
        assert all(tr.gf >= 0.05 for tr in sub.transitions)

    def test_explicit_terms(self):
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Fe II")
        sub = gd.select(terms=["a6S", "z6P"], gf_min=0.0)
        terms_used = {tr.lower.term for tr in sub.transitions} | {
            tr.upper.term for tr in sub.transitions
        }
        assert terms_used <= {"a6S", "z6P"}

    def test_preset_unknown(self):
        gd = GrotrianDiagram(ion="Fe II")
        with pytest.raises(ValueError, match="Unknown preset"):
            gd.select(preset="xray")


class TestFromKurucz:
    def test_loads(self):
        gd = GrotrianDiagram.from_kurucz(ion="Fe II")
        assert len(gd.transitions) > 1000
        assert len(gd.levels) > 100

    def test_select_optical(self):
        gd = GrotrianDiagram.from_kurucz(ion="Fe II")
        sub = gd.select(preset="optical", gf_min=0.01, max_terms=10)
        assert 0 < len(sub.transitions) < len(gd.transitions)


class TestSummary:
    def test_returns_table(self):
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Fe II")
        s = gd.summary()
        assert "term" in s.colnames
        assert len(s) > 0


class TestPlot:
    def test_smoke(self, tmp_path):
        """Plot should run without error and produce a figure."""
        import matplotlib
        matplotlib.use("Agg")
        tab = _make_test_table()
        gd = GrotrianDiagram.from_table(tab, ion="Fe II")
        fig = gd.plot()
        assert fig is not None
        fig.savefig(tmp_path / "grotrian_test.png", dpi=72)

    def test_empty(self):
        """Empty diagram should still produce a figure."""
        import matplotlib
        matplotlib.use("Agg")
        gd = GrotrianDiagram(ion="Fe II")
        fig = gd.plot()
        assert fig is not None

    def test_kurucz_optical(self, tmp_path):
        """Full Kurucz optical diagram should render."""
        import matplotlib
        matplotlib.use("Agg")
        gd = GrotrianDiagram.from_kurucz(ion="Fe II")
        sub = gd.select(preset="optical", gf_min=0.01, max_terms=8)
        fig = sub.plot()
        assert fig is not None
        fig.savefig(tmp_path / "grotrian_optical.png", dpi=100)
