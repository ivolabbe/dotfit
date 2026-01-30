"""Tests for multiplet assignment and ratio calculation behavior."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from dotfit.emission_lines import (
    EmissionLines,
    assign_multiplets,
    multiplet_ratios,
    calculate_multiplet_emissivities,
)


def test_assign_multiplets_groups_by_ion_and_terms():
    """Lines with same ion+configuration+terms get same multiplet ID."""
    tab = Table(
        {
            "ion": ["[Fe II]", "[Fe II]", "[Fe II]", "[Fe II]"],
            "wave_vac": [9217.183, 9439.216, 9484.0, 9548.588],
            "configuration": ["3d6(3F2)4s", "3d6(3F2)4s", "3d6(3F2)4s", "3d7"],
            "terms": ["a2F-c2D", "a2F-c2D", "a2F-c2D", "a4F-a4P"],
        }
    )

    out = assign_multiplets(tab)

    # First three lines should be in same multiplet
    assert out["multiplet"][0] == out["multiplet"][1] == out["multiplet"][2]
    assert out["multiplet"][0] > 0  # Non-zero multiplet ID

    # Fourth line should be single (multiplet=0)
    assert out["multiplet"][3] == 0


def test_assign_multiplets_single_lines_get_zero():
    """Lines without partners get multiplet=0."""
    tab = Table(
        {
            "ion": ["X I", "Y II", "Z III"],
            "wave_vac": [5000.0, 6000.0, 7000.0],
            "configuration": ["a", "b", "c"],
            "terms": ["x", "y", "z"],
        }
    )

    out = assign_multiplets(tab)

    # All should have multiplet=0 (singles)
    assert np.all(out["multiplet"] == 0)


def test_assign_multiplets_distinguishes_ions():
    """Multiplet IDs are assigned per-ion but may overlap across ions."""
    tab = Table(
        {
            "ion": ["[O III]", "[O III]", "[N II]", "[N II]"],
            "wave_vac": [4960.0, 5008.0, 6550.0, 6585.0],
            "configuration": ["a", "a", "b", "b"],
            "terms": ["x-y", "x-y", "p-q", "p-q"],
        }
    )

    out = assign_multiplets(tab)

    # [O III] lines should share a multiplet
    assert out["multiplet"][0] == out["multiplet"][1]
    assert out["multiplet"][0] > 0

    # [N II] lines should share a multiplet
    assert out["multiplet"][2] == out["multiplet"][3]
    assert out["multiplet"][2] > 0

    # Multiplet IDs are unique per ion, so we check that lines within
    # the same ion+multiplet are grouped correctly
    oiii_mask = out["ion"] == "[O III]"
    nii_mask = out["ion"] == "[N II]"
    assert len(np.unique(out["multiplet"][oiii_mask & (out["multiplet"] > 0)])) == 1
    assert len(np.unique(out["multiplet"][nii_mask & (out["multiplet"] > 0)])) == 1


def test_compute_line_ratios_uses_pyneb_for_forbidden():
    """[O III] uses PyNeb emissivities."""
    el = EmissionLines()
    tab = el.get_table("[OIII]")

    # Assign multiplets first
    tab = assign_multiplets(tab)

    # Compute ratios using current function (will be replaced)
    tab = multiplet_ratios(tab, Te=10_000, Ne=100)

    # Check that line_ratio column exists and has non-trivial values
    assert "line_ratio" in tab.colnames
    ratios = np.array(tab["line_ratio"], dtype=float)

    # Should have different ratios (not all 1.0 or all 0.0)
    assert not np.allclose(ratios, 1.0)
    assert not np.allclose(ratios, 0.0)

    # Max should be normalized to 1.0 within multiplets
    for m in np.unique(tab["multiplet"]):
        if m == 0:
            continue
        mask = tab["multiplet"] == m
        ratios_m = ratios[mask]
        assert np.max(ratios_m) <= 1.0 + 1e-10  # Allow small numerical error


def test_compute_line_ratios_uses_boltzmann_for_permitted():
    """Fe II uses Boltzmann formula."""
    el = EmissionLines()
    # Get some Fe II lines that form multiplets
    tab = el.get_table("Fe II", wave=[9000, 9500])

    # Filter to lines with required atomic data
    has_ek = (np.array(tab["Ek"], dtype=float) > 0).tolist()
    has_gigk = [g != "" and "-" in str(g) for g in tab["gigk"]]
    has_data = [ek and gigk for ek, gigk in zip(has_ek, has_gigk)]
    tab = tab[has_data]

    if len(tab) < 2:
        pytest.skip("Not enough Fe II lines with atomic data")

    # Assign multiplets and compute ratios
    tab = assign_multiplets(tab, lower_only=True)
    tab = calculate_multiplet_emissivities(tab, Te=10_000)

    # Check that line_ratio column exists and has non-trivial values
    assert "line_ratio" in tab.colnames
    ratios = np.array(tab["line_ratio"], dtype=float)

    # Should have different ratios for multiplet members
    for m in np.unique(tab["multiplet"]):
        if m == 0:
            continue
        mask = tab["multiplet"] == m
        if np.sum(mask) > 1:
            ratios_m = ratios[mask]
            # At least some variation expected
            assert not np.allclose(ratios_m, ratios_m[0])
            break


def test_compute_line_ratios_fallback_to_uniform():
    """Missing atomic data falls back to uniform weights (new behavior)."""
    # Create table with missing atomic data
    tab = Table(
        {
            "ion": ["X I", "X I"],
            "wave_vac": [5000.0, 6000.0],
            "configuration": ["a", "a"],
            "terms": ["x-y", "x-y"],
            "Ek": [0.0, 0.0],  # No energy data
            "gigk": ["", ""],  # No g-values
            "multiplet": [1, 1],
        }
    )

    # New behavior: falls back to uniform weights when data is missing
    out = calculate_multiplet_emissivities(tab, Te=10_000)

    assert "line_ratio" in out.colnames
    ratios = np.array(out["line_ratio"], dtype=float)

    # New refactored behavior: uniform weights normalized to max=1
    # Both lines get 1/2, then normalized to max=1, so both are 1.0
    expected = np.ones(2)
    assert np.allclose(ratios, expected, rtol=1e-6)


def test_ratios_normalized_per_multiplet():
    """Ratios within each multiplet sum to 1 (or max=1)."""
    el = EmissionLines()
    tab = el.get_table("[OIII]")

    tab = assign_multiplets(tab)
    tab = multiplet_ratios(tab, Te=10_000, Ne=100)

    ratios = np.array(tab["line_ratio"], dtype=float)

    for m in np.unique(tab["multiplet"]):
        if m == 0:
            continue
        mask = tab["multiplet"] == m
        ratios_m = ratios[mask]

        # Check max normalization (current behavior)
        assert np.isclose(np.max(ratios_m), 1.0, rtol=1e-6)


def test_hydrogen_lines_use_pyneb():
    """H I lines can use PyNeb recombination emissivities when grouped."""
    el = EmissionLines()
    tab = el.get_table("H I", wave=[4000, 7000])  # Get Balmer lines

    if len(tab) < 2:
        pytest.skip("Not enough H I lines in range")

    # H I lines in the emission line table may not have configuration/terms
    # to form multiplets via assign_multiplets. Test that multiplet_ratios
    # can handle H I when multiplet IDs are set.

    # Manually set all H I lines to same multiplet for testing
    tab = assign_multiplets(tab)
    tab["multiplet"] = 1  # Force them all into one multiplet

    # Compute ratios using PyNeb
    tab = multiplet_ratios(tab, Te=10_000, Ne=100)

    assert "line_ratio" in tab.colnames
    ratios = np.array(tab["line_ratio"], dtype=float)

    # H I should have non-zero, non-uniform ratios from PyNeb
    assert not np.allclose(ratios, 0.0), "H I ratios should be non-zero"
    assert not np.allclose(ratios, ratios[0]), "H I ratios should vary (not all equal)"
    assert np.max(ratios) > 0, "Maximum ratio should be positive"


def test_emission_lines_get_table_with_multiplet_flag():
    """EmissionLines.get_table(multiplet=True) should compute ratios."""
    el = EmissionLines()
    tab = el.get_table("[OIII]", multiplet=True, Te=10_000, Ne=100)

    # Should have multiplet and line_ratio columns
    assert "multiplet" in tab.colnames
    assert "line_ratio" in tab.colnames

    # Should have non-trivial values
    assert np.any(tab["multiplet"] > 0)
    ratios = np.array(tab["line_ratio"], dtype=float)
    assert not np.allclose(ratios, 1.0)
