import numpy as np
from astropy.table import Table

from dotfit.emission_lines import EmissionLines, assign_multiplets, calculate_multiplet_emissivities


def test_emission_lines_list_has_default_file():
    files = EmissionLines.list()
    assert "emission_lines.csv" in files


def test_get_table_hi_contains_lya():
    el = EmissionLines()
    tab = el.get_table("H I")
    assert "LyA" in tab["key"]
    row = tab[tab["key"] == "LyA"][0]
    assert np.isclose(float(row["wave_vac"]), 1215.6701, rtol=0, atol=1e-4)


def test_get_table_group_oiii_keys():
    el = EmissionLines()
    tab = el.get_table("[OIII]")
    keys = set(tab["key"].tolist())
    assert keys == {"[OIII]-4364", "[OIII]-4960", "[OIII]-5008"}


def test_get_multiplet_feii_9217():
    el = EmissionLines()
    tab = el.get_multiplet("[FeII]-9217")
    assert tab is not None
    assert len(tab) == 3
    assert set(tab["key"].tolist()) == {"[FeII]-9217", "[FeII]-9439", "[FeII]-9484"}
    assert np.all(tab["multiplet"] == 1)


def test_remove_key_removes_matching_rows():
    el = EmissionLines()
    n_before = len(el.table)
    el.remove_key("LyA")
    assert "LyA" not in el.table["key"]
    assert len(el.table) < n_before


def test_assign_multiplets_basic_grouping():
    tab = Table(
        {
            "ion": ["[Fe II]", "[Fe II]", "[Fe II]"],
            "wave_vac": [9217.183, 9439.216, 9548.588],
            "configuration": ["3d6(3F2)4s", "3d6(3F2)4s", "3d7"],
            "terms": ["a2F-c2D", "a2F-c2D", "a4F-a4P"],
        }
    )

    out = assign_multiplets(tab)
    assert "multiplet" in out.colnames
    assert "multiplet_term" in out.colnames
    assert out["multiplet"][0] == out["multiplet"][1]
    assert out["multiplet"][2] == 0
    assert out["multiplet_term"][0] == "a2F-c2D"


def test_assign_multiplets_classification_rules():
    tab = Table(
        {
            "ion": ["X I", "X I"],
            "wave_vac": [5000.0, 5100.0],
            "configuration": ["a-b", "a-b"],
            "terms": ["x-y", "x-y"],
            "classification": ["forbidden", "permitted"],
        }
    )

    out = assign_multiplets(tab)
    terms = out["multiplet_term"].tolist()
    assert terms == ["x-y", "x"]


def test_calculate_multiplet_emissivities_formula():
    tab = Table(
        {
            "ion": ["X I", "X I"],
            "wave_vac": [5000.0, 6000.0],
            "Ek": [2.0, 2.2],
            "gigk": ["2-4", "2-2"],
            "gf": [1.0, 0.5],
            "multiplet": [1, 1],
        }
    )

    out = calculate_multiplet_emissivities(tab, Te=10_000)
    lr = np.array(out["line_ratio"], dtype=float)

    KB_EV = 8.617333262e-5
    lam = np.array([5000.0, 6000.0])
    Eu = np.array([2.0, 2.2])
    gi = np.array([2.0, 2.0])
    gu = np.array([4.0, 2.0])
    f = np.array([1.0, 0.5]) / gi
    delta_E = Eu - Eu[0]
    i_weight = (lam[0] / lam) ** 3 * (f / f[0]) * (gu / gu[0]) * np.exp(-delta_E / (KB_EV * 10_000))
    expected = i_weight / np.sum(i_weight)

    assert np.allclose(lr, expected, rtol=0, atol=1e-12)
