"""Tests for the LineExplorer interactive app."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table

from dotfit.line_explorer import ELEMENT_COLORS, LineExplorer


@pytest.fixture
def mock_spectrum():
    """Create a mock spectrum for testing."""
    wave = np.linspace(3000, 7000, 1000)
    flux = np.ones_like(wave) + 0.1 * np.random.randn(len(wave))
    return {'test_spectrum': {'wave': wave, 'flux': flux}}


@pytest.fixture
def mock_emission_lines():
    """Create a mock emission line table for testing."""
    lines = Table({
        'key': ['Ha', 'Hb', '[OIII]-5008', '[OIII]-4960', '[OII]-3727'],
        'ion': ['HI', 'HI', '[OIII]', '[OIII]', '[OII]'],
        'wave_vac': [6564.61, 4862.68, 5008.24, 4960.30, 3727.42],
        'gf': [0.64, 0.12, 0.001, 0.001, 0.001],
        'Aki': [0.0, 0.0, 2.03e7, 6.76e6, 1.82e7],
        'Ei': [10.0, 10.0, 10.0, 10.0, 10.0],
        'line_ratio': [1.0, 0.5, 1.0, 0.5, 1.0],
    })
    return lines


def test_line_explorer_init(mock_spectrum, mock_emission_lines):
    """Test LineExplorer initialization."""
    explorer = LineExplorer(
        spectrum=mock_spectrum,
        emission_lines=mock_emission_lines,
        redshift=0.5,
        object_name="Test Object"
    )

    assert explorer.object_name == "Test Object"
    assert explorer.redshift == 0.5
    assert explorer.current_spectrum == 'test_spectrum'
    assert len(explorer.visible_ions) > 0
    assert explorer.selected_lines == []


def test_get_element_color():
    """Test element color extraction."""
    explorer = LineExplorer({}, Table(), redshift=0)

    # Test various ion formats
    assert explorer._get_element_color('[O III]') == ELEMENT_COLORS['O']
    assert explorer._get_element_color('[He I]') == ELEMENT_COLORS['He']  # Bracket format
    assert explorer._get_element_color('Ha') == ELEMENT_COLORS['H']
    assert explorer._get_element_color('UnknownIon') == ELEMENT_COLORS['default']


def test_filter_lines_in_range(mock_spectrum, mock_emission_lines):
    """Test filtering lines by wavelength range."""
    explorer = LineExplorer(
        spectrum=mock_spectrum,
        emission_lines=mock_emission_lines,
        redshift=0
    )

    # Filter to narrow range
    filtered = explorer._filter_lines_in_range(4800, 5100)

    assert len(filtered) == 3  # Should get Hb and both [OIII] lines
    assert 'Hb' in filtered['key']
    assert '[OIII]-5008' in filtered['key']
    assert '[OIII]-4960' in filtered['key']


def test_add_remove_line(mock_spectrum, mock_emission_lines):
    """Test adding and removing lines from selection."""
    explorer = LineExplorer(
        spectrum=mock_spectrum,
        emission_lines=mock_emission_lines,
        redshift=0
    )

    # Initially no lines selected
    assert len(explorer.selected_lines) == 0

    # Add a line
    explorer.add_line('Ha')
    assert len(explorer.selected_lines) == 1
    assert explorer.selected_lines[0]['key'] == 'Ha'

    # Try adding the same line again (should not duplicate)
    explorer.add_line('Ha')
    assert len(explorer.selected_lines) == 1

    # Add another line
    explorer.add_line('[OIII]-5008')
    assert len(explorer.selected_lines) == 2

    # Remove a line
    explorer.remove_line('Ha')
    assert len(explorer.selected_lines) == 1
    assert explorer.selected_lines[0]['key'] == '[OIII]-5008'


def test_compute_visible_labels(mock_spectrum, mock_emission_lines):
    """Test label overlap computation."""
    explorer = LineExplorer(
        spectrum=mock_spectrum,
        emission_lines=mock_emission_lines,
        redshift=0
    )

    # Get lines in a range
    lines_in_range = explorer._filter_lines_in_range(4800, 5100)

    # Compute visible labels with large spacing (all should be visible)
    visible_idx = explorer._compute_visible_labels(
        lines_in_range,
        x_range=(4800, 5100),
        approx_char_width=1.0
    )

    # With reasonable spacing, should show strongest lines
    assert len(visible_idx) > 0
    assert len(visible_idx) <= len(lines_in_range)


def test_panel_creation(mock_spectrum, mock_emission_lines):
    """Test Panel layout creation."""
    explorer = LineExplorer(
        spectrum=mock_spectrum,
        emission_lines=mock_emission_lines,
        redshift=0,
        object_name="Test"
    )

    # Create panel layout
    layout = explorer.panel()

    # Check that layout is created
    assert layout is not None


def test_empty_spectrum():
    """Test LineExplorer with empty spectrum dictionary."""
    explorer = LineExplorer(
        spectrum={},
        emission_lines=Table(),
        redshift=0
    )

    assert explorer.current_spectrum is None
    assert len(explorer.unique_ions) == 0


def test_spectrum_change(mock_spectrum, mock_emission_lines):
    """Test changing the active spectrum."""
    # Add a second spectrum
    multi_spectrum = {
        'spectrum1': {'wave': np.linspace(3000, 5000, 500), 'flux': np.ones(500)},
        'spectrum2': {'wave': np.linspace(5000, 7000, 500), 'flux': np.ones(500)},
    }

    explorer = LineExplorer(
        spectrum=multi_spectrum,
        emission_lines=mock_emission_lines,
        redshift=0
    )

    # Initially should be first spectrum
    assert explorer.current_spectrum == 'spectrum1'

    # Simulate spectrum change
    class MockEvent:
        def __init__(self, new_value):
            self.new = new_value

    explorer._on_spectrum_change(MockEvent('spectrum2'))
    assert explorer.current_spectrum == 'spectrum2'


def test_extract_unique_ions(mock_emission_lines):
    """Test extraction of unique ions from table."""
    explorer = LineExplorer(
        spectrum={},
        emission_lines=mock_emission_lines,
        redshift=0
    )

    # Should have 3 unique ions: HI, [OIII], [OII]
    assert len(explorer.unique_ions) == 3
    assert 'HI' in explorer.unique_ions
    assert '[OIII]' in explorer.unique_ions
    assert '[OII]' in explorer.unique_ions
