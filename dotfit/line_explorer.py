"""Interactive emission line identification app using Panel + HoloViews.

This module provides an interactive tool for exploring spectra, identifying emission lines,
and building a selected line list for analysis.

TODO: f_lambda/f_nu flux unit selector feature (parked for later debugging)
    - Add flux_unit_selector widget (Select: 'f_lambda', 'f_nu')
    - f_lambda mode: normalized display (median = 0.5)
    - f_nu mode: convert from f_λ (10^-20 erg/s/cm²/Å) to f_ν (µJy)
        Formula: f_ν [µJy] = f_λ × λ² / c × 10^9 (c = 2.99792458e18 Å/s)
        Use absolute scaling (no normalization) with autoscale to min/max
        Y-axis label: "f_ν [µJy]"
    - Needs debugging: verify µJy conversion is correct and zooming works properly
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import panel as pn
import holoviews as hv
from holoviews import streams
from astropy.table import Table
from astropy.utils.data import download_file
from astropy import units as u
from bokeh.models import HoverTool

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Color scheme for elements (based on common astronomical notation)
ELEMENT_COLORS = {
    'H': '#3b82f6',   # blue
    'He': '#ec4899',  # pink
    'O': '#22c55e',   # green
    'N': '#a855f7',   # purple
    'S': '#f59e0b',   # amber
    'Fe': '#ef4444',  # red
    'Ne': '#06b6d4',  # cyan
    'C': '#8b5cf6',   # violet
    'Si': '#14b8a6',  # teal
    'Mg': '#84cc16',  # lime
    'Ca': '#f97316',  # orange
    'Ar': '#06b6d4',  # cyan
    'default': '#6b7280',  # gray
}

# DJA base URL
DJA_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/'


def load_dja_spectra(filenames: str | list[str], redshift: float, base_url: str = DJA_URL, cache: bool = True) -> dict:
    """Load NIRSpec spectra from DJA using astropy.

    Parameters
    ----------
    filenames : str or list of str
        FITS filename(s) for each grating. Can be a single filename or list.
    redshift : float
        Source redshift for converting to rest-frame
    base_url : str, optional
        Base URL for data access
    cache : bool, optional
        Whether to cache downloaded files

    Returns
    -------
    dict
        Dictionary with spectrum data for each grating, plus 'z' key with redshift

    Examples
    --------
    >>> filenames = ['abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits']
    >>> spectra = load_dja_spectra(filenames, redshift=4.465)
    >>> print(spectra.keys())
    dict_keys(['z', 'g235m'])
    """
    # Handle single filename
    if isinstance(filenames, str):
        filenames = [filenames]

    # Units for conversion
    λ_unit = u.Angstrom
    fλ_unit = u.erg / u.s / u.cm**2 / u.Angstrom

    root = filenames[0].split('_')[0]
    monster = {'z': redshift}

    for fname in filenames:
        grating = fname.split('_')[1].split('-')[0]
        url = f"{base_url}{root}/{fname}"
        logger.info(f"Downloading {grating} from {url}")

        try:
            # Download file using astropy
            spec_file = download_file(url, cache=cache, show_progress=False)

            # Load the spectrum from SPEC1D extension
            spec = Table.read(spec_file, 'SPEC1D')

            # Unpack relevant columns and convert units
            wave = spec['wave'].to(λ_unit)
            flux = spec['flux'].to(fλ_unit, equivalencies=u.spectral_density(wave)).value
            err = spec['err'].to(fλ_unit, equivalencies=u.spectral_density(wave)).value
            wave = wave.value

            # Valid rows are where 'err' is not masked (if column is masked)
            # For non-masked columns (e.g., mock spectra), all values are valid
            if hasattr(spec['err'], 'mask'):
                valid = ~spec['err'].mask
            else:
                valid = np.ones(len(spec['err']), dtype=bool)

            # Convert to rest-frame wavelength
            wave_rest = wave[valid] / (1 + redshift)
            flux_rest = flux[valid]
            err_rest = err[valid]

            monster[grating] = {
                'url': url,
                'grating': grating,
                'wave': wave_rest,
                'flux': flux_rest,
                'err': err_rest,
            }

            logger.info(f"Loaded {grating}: {len(wave_rest)} valid pixels")

        except Exception as e:
            logger.error(f"Error loading {grating}: {e}")
            continue

    return monster


class LineExplorer:
    """Interactive emission line identification app.

    This class provides an interactive interface for exploring spectra,
    identifying emission lines, and building a selected line list.

    Parameters
    ----------
    spectrum : dict or str or list of str
        Either:
        - Dictionary mapping spectrum names to spectrum data (each with 'wave' and 'flux' keys)
        - Single filename to load from DJA
        - List of filenames to load from DJA
    emission_lines : astropy.table.Table or str or list[Table] or list[str]
        Either:
        - Emission line catalog as astropy Table (from EmissionLines.table)
        - Path to CSV file containing emission line catalog
        - List of Tables (multiple catalogs)
        - List of CSV file paths (multiple catalogs)
    redshift : float, optional
        Redshift for converting rest-frame to observed wavelengths
    object_name : str, optional
        Name of the object being analyzed
    debug : bool, optional
        If True, print debug messages when events are triggered and callbacks respond
    max_H_series : int, optional
        Maximum hydrogen transition to include for ALL series (Balmer, Paschen, Lyman, etc.)
        (e.g., 15 means H15 is max, filters out H16, H17, etc. from all series)
    selected_lines_csv : str, optional
        Path to CSV file with previously selected lines (from Save to CSV button).
        If provided, these lines will be automatically added to the selection on startup.

    Examples
    --------
    >>> from dotfit.line_explorer import LineExplorer
    >>> from dotfit import EmissionLines
    >>> el = EmissionLines()
    >>>
    >>> # Option 1: Pass spectrum dict and Table
    >>> spectrum = {'g235m': {'wave': np.array([...]), 'flux': np.array([...])}}
    >>> app = LineExplorer(spectrum, el.table, redshift=4.465, object_name="test")
    >>>
    >>> # Option 2: Pass filename(s) to load from DJA
    >>> filenames = ['abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits']
    >>> app = LineExplorer(filenames, el.table, redshift=4.465, object_name="test")
    >>>
    >>> # Option 3: Load emission lines from CSV
    >>> app = LineExplorer(filenames, 'my_lines.csv', redshift=4.465, object_name="test")
    >>>
    >>> app.serve()  # Launch standalone server
    """

    def __init__(
        self,
        spectrum: dict | str | list[str],
        emission_lines: Table | str | list[Table] | list[str],
        redshift: float = 0,
        object_name: str = "",
        debug: bool = False,
        max_H_series: int = 15,
        selected_lines_csv: str | None = None,
    ):
        """Initialize the LineExplorer app."""
        # Initialize Panel and HoloViews extensions
        pn.extension('tabulator')
        hv.extension('bokeh')

        # Store redshift first (needed for loading)
        self.redshift = redshift
        self.object_name = object_name
        self.debug = debug
        self.max_H_series = max_H_series
        self.selected_lines_csv_path = selected_lines_csv  # Remember where we loaded from

        # Handle different spectrum input types
        if isinstance(spectrum, dict):
            # Direct dictionary input
            self.spectrum_dict = spectrum
        elif isinstance(spectrum, (str, list)):
            # Load from DJA
            logger.info(f"Loading spectra from DJA...")
            monster = load_dja_spectra(spectrum, redshift=redshift)

            # Convert to spectrum_dict format
            self.spectrum_dict = {}
            for key in monster:
                if key != 'z':
                    self.spectrum_dict[key] = {
                        'wave': monster[key]['wave'],
                        'flux': monster[key]['flux'],
                        'err': monster[key]['err'],  # Include error data
                    }
            logger.info(f"Loaded {len(self.spectrum_dict)} spectra: {list(self.spectrum_dict.keys())}")
        else:
            raise TypeError(f"spectrum must be dict, str, or list, not {type(spectrum)}")

        # Handle different emission_lines input types
        # Support single catalog or list of catalogs
        self.emission_catalogs = {}  # Dict mapping catalog name to Table

        if isinstance(emission_lines, Table):
            # Direct Table input (single catalog)
            self.emission_catalogs['default'] = emission_lines
        elif isinstance(emission_lines, str):
            # Load from CSV file (single catalog)
            logger.info(f"Loading emission lines from {emission_lines}")
            table = Table.read(emission_lines, format='csv')
            catalog_name = Path(emission_lines).stem  # Use filename without extension as catalog name
            self.emission_catalogs[catalog_name] = table
            logger.info(f"Loaded {len(table)} emission lines from {catalog_name}")
        elif isinstance(emission_lines, list):
            # List of Tables or list of CSV paths (multiple catalogs)
            for i, item in enumerate(emission_lines):
                if isinstance(item, Table):
                    catalog_name = f'catalog_{i+1}'
                    self.emission_catalogs[catalog_name] = item
                    logger.info(f"Added catalog {catalog_name} with {len(item)} lines")
                elif isinstance(item, str):
                    logger.info(f"Loading emission lines from {item}")
                    table = Table.read(item, format='csv')
                    catalog_name = Path(item).stem  # Use filename without extension
                    self.emission_catalogs[catalog_name] = table
                    logger.info(f"Loaded {len(table)} emission lines from {catalog_name}")

                    # Debug: show first few ion values
                    if 'ion' in table.colnames and len(table) > 0:
                        sample_ions = [str(table['ion'][i]) for i in range(min(3, len(table)))]
                        logger.info(f"  Sample ions from {catalog_name}: {sample_ions}")
                else:
                    raise TypeError(f"List items must be Table or str, not {type(item)}")
            if len(self.emission_catalogs) == 0:
                raise ValueError("emission_lines list is empty")
        else:
            raise TypeError(f"emission_lines must be Table, str, or list, not {type(emission_lines)}")

        # Set initial catalog (first one in the dict)
        self.current_catalog_name = list(self.emission_catalogs.keys())[0]
        self.emission_lines = self.emission_catalogs[self.current_catalog_name]
        logger.info(f"Initial catalog: {self.current_catalog_name} ({len(self.emission_lines)} lines)")

        # Apply hydrogen series filtering to initial catalog
        self.emission_lines = self._filter_hydrogen_series(self.emission_lines)

        # Apply spectrum scaling ONCE at initialization
        # This modifies spectrum_dict in-place, scaling all spectra to reference
        # Reference: first PRISM if available, otherwise first spectrum
        self._apply_spectrum_scaling()

        # Initialize state
        self.visible_ions: set[str] = set()
        self.selected_lines: list[dict] = []
        self.current_spectrum: str = list(self.spectrum_dict.keys())[0] if self.spectrum_dict else None
        self.x_range: tuple[float, float] | None = None
        self._auto_scaling_mode: bool = True  # Stay in auto-scaling until user explicitly zooms

        # Extract unique ions from table
        self._extract_unique_ions()

        # Initialize all ions as visible (ensure H I is always shown)
        self.visible_ions = set(self.unique_ions)
        # Explicitly ensure H I is visible by default
        if 'H I' in self.unique_ions:
            self.visible_ions.add('H I')

        # Create widgets and components
        self._create_widgets()
        self._create_plot_components()

        # Load previously selected lines from CSV if provided
        if selected_lines_csv is not None:
            self._load_selected_lines_from_csv(selected_lines_csv)

    def _extract_unique_ions(self) -> None:
        """Extract unique ion names from the emission line table."""
        if 'ion' not in self.emission_lines.colnames:
            logger.warning("No 'ion' column found in emission lines table")
            self.unique_ions = []
            return

        ions = []
        for row in self.emission_lines:
            # Get ion value and convert to native Python string
            # Handle both dict-like access and direct attribute access
            try:
                if hasattr(row, 'get'):
                    ion = row.get('ion', '')
                else:
                    ion = row['ion']

                # Convert to native Python string to avoid numpy string issues
                ion_str = str(ion).strip() if ion else ''

                if ion_str and ion_str not in ions:
                    ions.append(ion_str)
            except (KeyError, IndexError, AttributeError) as e:
                logger.warning(f"Error extracting ion from row: {e}")
                continue

        self.unique_ions = sorted(ions)
        logger.info(f"Found {len(self.unique_ions)} unique ions: {self.unique_ions[:5]}...")

    def _get_element_color(self, ion: str) -> str:
        """Get color for an element/ion.

        Parameters
        ----------
        ion : str
            Ion name (e.g., '[O III]', 'HeI', 'Ha')

        Returns
        -------
        str
            Hex color code
        """
        # Extract element from ion name
        # Examples: '[O III]' -> 'O', '[He I]' -> 'He', 'HeI' -> 'He', 'Ha' -> 'H'
        ion_clean = ion.strip('[]').strip()

        # Try to match element symbols (check longest first for 2-letter elements)
        # Sort by length descending to match 'He' before 'H'
        elements = sorted([e for e in ELEMENT_COLORS if e != 'default'], key=len, reverse=True)

        for element in elements:
            if ion_clean.startswith(element):
                return ELEMENT_COLORS[element]

        return ELEMENT_COLORS['default']

    def _filter_lines_in_range(self, wave_min: float, wave_max: float) -> Table:
        """Filter emission lines to visible wavelength range.

        Parameters
        ----------
        wave_min : float
            Minimum wavelength in Angstroms (rest-frame)
        wave_max : float
            Maximum wavelength in Angstroms (rest-frame)

        Returns
        -------
        astropy.table.Table
            Filtered emission line table
        """
        # Handle empty table
        if len(self.emission_lines) == 0 or 'wave_vac' not in self.emission_lines.colnames:
            return self.emission_lines[0:0]  # Return empty table with same structure

        # Filter by wavelength range
        mask = (self.emission_lines['wave_vac'] >= wave_min) & \
               (self.emission_lines['wave_vac'] <= wave_max)

        # Filter by visible ions
        if self.visible_ions and 'ion' in self.emission_lines.colnames:
            ion_mask = np.zeros(len(self.emission_lines), dtype=bool)
            for ion in self.visible_ions:
                ion_mask |= (self.emission_lines['ion'] == ion)
            mask &= ion_mask

        # Filter by line_ratio slider (lines with missing ratio are exempt)
        if hasattr(self, 'line_ratio_slider') and 'line_ratio' in self.emission_lines.colnames:
            lr_min = self.line_ratio_slider.value
            lr_vals = np.asarray(self.emission_lines['line_ratio'], dtype=float)
            mask &= (~np.isfinite(lr_vals)) | (lr_vals >= lr_min)

        # Filter by Ei cutoff slider (lines with missing Ei are exempt)
        if hasattr(self, 'ei_cutoff_slider') and 'Ei' in self.emission_lines.colnames:
            ei_max = self.ei_cutoff_slider.value
            ei_vals = np.asarray(self.emission_lines['Ei'], dtype=float)
            mask &= (~np.isfinite(ei_vals)) | (ei_vals <= ei_max)

        return self.emission_lines[mask]

    def _filter_lines_by_strength(
        self,
        lines: Table,
        x_range: tuple[float, float],
    ) -> Table:
        """Filter lines by strength based on zoom level.

        When zoomed out, only show strongest lines for performance.
        When zoomed in, show more lines.

        Parameters
        ----------
        lines : Table
            Emission lines to filter
        x_range : tuple[float, float]
            Current plot range (x_min, x_max)

        Returns
        -------
        Table
            Filtered lines (only strongest based on zoom level)
        """
        if len(lines) == 0:
            return lines

        # Compute line strength (use gf for permitted, Aki for forbidden)
        gf_vals = np.asarray(lines['gf']) if 'gf' in lines.colnames else np.zeros(len(lines))
        aki_vals = np.asarray(lines['Aki']) if 'Aki' in lines.colnames else np.zeros(len(lines))

        # Replace NaN with 0
        gf_vals = np.nan_to_num(gf_vals, nan=0.0)
        aki_vals = np.nan_to_num(aki_vals, nan=0.0)

        strength = np.maximum(gf_vals, aki_vals)

        # Determine strength threshold based on zoom level
        data_range = x_range[1] - x_range[0]

        if data_range > 3000:
            # Very zoomed out - only top 5% strongest lines
            percentile = 95
        elif data_range > 2000:
            # Zoomed out - top 10% strongest lines
            percentile = 90
        elif data_range > 1000:
            # Medium zoom - top 30% strongest lines
            percentile = 70
        elif data_range > 500:
            # Closer zoom - top 50% strongest lines
            percentile = 50
        else:
            # Zoomed in - show all lines
            return lines

        # Compute threshold
        if np.any(strength > 0):
            strength_threshold = np.percentile(strength[strength > 0], percentile)
            mask = strength >= strength_threshold

            # Always include selected lines regardless of strength
            selected_keys = {line['key'] for line in self.selected_lines}
            for i, row in enumerate(lines):
                if row['key'] in selected_keys:
                    mask[i] = True

            # Always include ALL Hydrogen and Oxygen lines at any zoom level
            for i, row in enumerate(lines):
                ion = row['ion']
                # Check for Hydrogen (H I) and any Oxygen species (O I, O II, O III, [O II], [O III], etc.)
                if ion.startswith('H ') or ion.startswith('[H') or \
                   ion.startswith('O ') or ion.startswith('[O'):
                    mask[i] = True

            return lines[mask]
        else:
            return lines

    def _compute_visible_labels(
        self,
        lines_in_range: Table,
        x_range: tuple[float, float],
        approx_char_width: float = 10.0,
    ) -> list[int]:
        """Compute which labels should be visible to avoid overlap.

        Uses a greedy algorithm to place non-overlapping labels, prioritizing
        stronger lines. Shows fewer labels when zoomed out (wide range).

        Parameters
        ----------
        lines_in_range : Table
            Emission lines within the current view
        x_range : tuple[float, float]
            Current plot range (x_min, x_max)
        approx_char_width : float
            Approximate width of one character in data units (Angstroms)

        Returns
        -------
        list[int]
            Indices of lines that should have visible labels
        """
        if len(lines_in_range) == 0:
            return []

        # Compute line strength (use gf for permitted, Aki for forbidden)
        # Use np.asarray to handle both masked and regular arrays
        gf_vals = np.asarray(lines_in_range['gf']) if 'gf' in lines_in_range.colnames else np.zeros(len(lines_in_range))
        aki_vals = np.asarray(lines_in_range['Aki']) if 'Aki' in lines_in_range.colnames else np.zeros(len(lines_in_range))

        # Replace NaN with 0
        gf_vals = np.nan_to_num(gf_vals, nan=0.0)
        aki_vals = np.nan_to_num(aki_vals, nan=0.0)

        strength = np.maximum(gf_vals, aki_vals)

        # When zoomed out (wide range), only show strongest lines
        # When zoomed in (narrow range), show more lines
        data_range = x_range[1] - x_range[0]

        # Determine strength threshold based on zoom level
        # Be VERY aggressive when zoomed out to keep performance good
        if data_range > 3000:
            # Very zoomed out - only top 5% strongest lines
            strength_threshold = np.percentile(strength[strength > 0], 95) if np.any(strength > 0) else 0
        elif data_range > 2000:
            # Zoomed out - top 10% strongest lines
            strength_threshold = np.percentile(strength[strength > 0], 90) if np.any(strength > 0) else 0
        elif data_range > 1000:
            # Medium zoom - top 30% strongest lines
            strength_threshold = np.percentile(strength[strength > 0], 70) if np.any(strength > 0) else 0
        elif data_range > 500:
            # Closer zoom - top 50% strongest lines
            strength_threshold = np.percentile(strength[strength > 0], 50) if np.any(strength > 0) else 0
        else:
            # Zoomed in - show all lines
            strength_threshold = 0

        # Filter to only consider strong enough lines
        strong_mask = strength >= strength_threshold
        strong_idx = np.where(strong_mask)[0]

        if len(strong_idx) == 0:
            return []

        # Sort strong lines by strength (descending)
        sorted_idx = strong_idx[np.argsort(-strength[strong_idx])]

        visible_labels = []
        occupied_ranges = []  # List of (x_min, x_max) occupied

        # Label spacing depends on zoom level
        padding_factor = 0.03  # Minimal spacing for labels

        for idx in sorted_idx:
            wave = lines_in_range['wave_vac'][idx]
            key = lines_in_range['key'][idx]
            label_width = len(key) * approx_char_width

            label_min = wave - label_width * padding_factor
            label_max = wave + label_width * padding_factor

            # Check overlap with existing labels
            overlaps = any(
                not (label_max < occ[0] or label_min > occ[1])
                for occ in occupied_ranges
            )

            if not overlaps:
                visible_labels.append(idx)
                occupied_ranges.append((label_min, label_max))

        return visible_labels

    def _filter_hydrogen_series(self, table: Table) -> Table:
        """Filter out high hydrogen transitions from emission line table.

        Parameters
        ----------
        table : astropy.table.Table
            Emission line table to filter

        Returns
        -------
        astropy.table.Table
            Filtered table with high H transitions removed
        """
        # Filter out high hydrogen transitions (H16, H17, etc.) for ALL series
        # This applies to Balmer, Paschen, Lyman, Brackett, Pfund, and any other H series
        if self.max_H_series is None or 'key' not in table.colnames:
            return table

        mask = np.ones(len(table), dtype=bool)
        for i, key in enumerate(table['key']):
            h_num = None

            # Try different hydrogen line key formats:
            # 1. "H16", "H17", "H16-Balmer", "H16-Paschen", etc.
            if key.startswith('H') and len(key) > 1:
                try:
                    num_str = key[1:].split('-')[0].split('_')[0]
                    if num_str.isdigit():
                        h_num = int(num_str)
                except (ValueError, IndexError):
                    pass

            # 2. "Pa16", "Pa17" (Paschen)
            elif key.startswith('Pa') and len(key) > 2:
                try:
                    num_str = key[2:].split('-')[0].split('_')[0]
                    if num_str.isdigit():
                        h_num = int(num_str)
                except (ValueError, IndexError):
                    pass

            # 3. "Ly16", "Ly17" (Lyman)
            elif key.startswith('Ly') and len(key) > 2:
                try:
                    num_str = key[2:].split('-')[0].split('_')[0]
                    if num_str.isdigit():
                        h_num = int(num_str)
                except (ValueError, IndexError):
                    pass

            # 4. "Br16", "Br17" (Brackett)
            elif key.startswith('Br') and len(key) > 2:
                try:
                    num_str = key[2:].split('-')[0].split('_')[0]
                    if num_str.isdigit():
                        h_num = int(num_str)
                except (ValueError, IndexError):
                    pass

            # 5. "Pf16", "Pf17" (Pfund)
            elif key.startswith('Pf') and len(key) > 2:
                try:
                    num_str = key[2:].split('-')[0].split('_')[0]
                    if num_str.isdigit():
                        h_num = int(num_str)
                except (ValueError, IndexError):
                    pass

            # Filter out if transition number exceeds maximum
            if h_num is not None and h_num > self.max_H_series:
                mask[i] = False

        original_len = len(table)
        filtered_table = table[mask]
        filtered_count = original_len - len(filtered_table)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} hydrogen transitions (all series) beyond H{self.max_H_series}")

        return filtered_table

    def _apply_spectrum_scaling(self) -> None:
        """Apply spectrum scaling ONCE at initialization.

        Scales all spectra in spectrum_dict IN-PLACE to match the reference spectrum.
        Reference: First PRISM in spectrum_dict if available, otherwise first spectrum.
        All spectra are scaled to match the reference's median f_λ in the
        wavelength overlap region between each spectrum and the reference.

        This modifies spectrum_dict['flux'] and spectrum_dict['err'] directly.
        After this, all spectra are already scaled and no further scaling is needed.
        """
        if not self.spectrum_dict:
            return

        # Get all spectrum keys
        all_keys = list(self.spectrum_dict.keys())

        if len(all_keys) == 1:
            logger.info("Single spectrum loaded, no scaling needed")
            return

        # Find reference spectrum following the priority rule:
        # 1. First PRISM in the input list (if any exists)
        # 2. If no PRISM, use first spectrum in the input list
        ref_key = None
        prism_found = False
        for key in all_keys:
            if 'prism' in key.lower():
                ref_key = key
                prism_found = True
                break
        if ref_key is None:
            ref_key = all_keys[0]

        # Store reference key for later use
        self.reference_spectrum_key = ref_key

        logger.info("=" * 60)
        if prism_found:
            logger.info(f"SCALING REFERENCE: '{ref_key}' (first PRISM)")
        else:
            logger.info(f"SCALING REFERENCE: '{ref_key}' (first spectrum, no PRISM found)")

        # Scale each spectrum to match the reference using their overlap region
        ref_spec = self.spectrum_dict[ref_key]
        ref_wave = ref_spec['wave']  # rest-frame Angstroms
        ref_flux = ref_spec['flux']

        # Scale all spectra to match reference in overlap region
        for spectrum_key in all_keys:
            if spectrum_key == ref_key:
                logger.info(f"  {spectrum_key}: REFERENCE (scale=1.000)")
                continue

            spec = self.spectrum_dict[spectrum_key]
            wave = spec['wave']
            flux = spec['flux']
            err = spec.get('err', None)

            # Find overlap wavelength range
            overlap_min = max(ref_wave.min(), wave.min())
            overlap_max = min(ref_wave.max(), wave.max())

            if overlap_min >= overlap_max:
                logger.warning(f"  '{spectrum_key}': no wavelength overlap with reference, cannot scale")
                continue

            ref_mask = (ref_wave >= overlap_min) & (ref_wave <= overlap_max)
            spec_mask = (wave >= overlap_min) & (wave <= overlap_max)

            ref_median = np.nanmedian(ref_flux[ref_mask])
            spec_median = np.nanmedian(flux[spec_mask])

            logger.info(f"  {spectrum_key}: overlap {overlap_min:.0f}-{overlap_max:.0f} Å, "
                        f"ref median={ref_median:.3e}, spec median={spec_median:.3e}")

            if ref_median > 0 and spec_median > 0:
                scale_factor = ref_median / spec_median
                self.spectrum_dict[spectrum_key]['flux'] = flux * scale_factor
                if err is not None:
                    self.spectrum_dict[spectrum_key]['err'] = err * scale_factor
                logger.info(f"  {spectrum_key}: SCALED by {scale_factor:.3f}")
            else:
                logger.warning(f"  {spectrum_key}: Cannot scale (invalid median)")

        # Verify scaling
        logger.info("")
        logger.info("VERIFICATION (median f_λ in overlap after scaling):")
        for spectrum_key in all_keys:
            if spectrum_key == ref_key:
                continue
            spec = self.spectrum_dict[spectrum_key]
            wave = spec['wave']
            flux = spec['flux']
            overlap_min = max(ref_wave.min(), wave.min())
            overlap_max = min(ref_wave.max(), wave.max())
            if overlap_min < overlap_max:
                mask = (wave >= overlap_min) & (wave <= overlap_max)
                median_scaled = np.nanmedian(flux[mask])
                logger.info(f"  {spectrum_key}: {median_scaled:.3e}")

        logger.info("=" * 60)

    def _create_spectrum_curve(self) -> hv.Overlay:
        """Create HoloViews plot of spectrum as steps with error bands.

        Handles multiple selected spectra with different colors.
        All spectra are already scaled at initialization (see _apply_spectrum_scaling).
        No scaling is performed here - only normalization for display.

        Returns
        -------
        hv.Overlay
            Spectrum as steps + error area overlay
        """
        # Colors for spectra: 1st=black (g235m), 2nd=blue (g395m), 3rd=orange (prism), then extras
        spectrum_colors = ['black', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Get list of selected spectra
        selected = getattr(self, 'selected_spectra', [])
        if not selected:
            # Fallback to current_spectrum for backward compatibility
            if self.current_spectrum and self.current_spectrum in self.spectrum_dict:
                selected = [self.current_spectrum]
            else:
                return hv.Curve([])

        overlays = []

        for idx, spectrum_key in enumerate(selected):
            if spectrum_key not in self.spectrum_dict:
                continue

            spec = self.spectrum_dict[spectrum_key]
            wave = spec['wave']
            flux = spec['flux']  # Already scaled at initialization - use directly for display

            # Get error if available (also already scaled)
            err = spec.get('err', None)

            # Use flux directly without additional normalization
            # All spectra were already scaled to reference at initialization
            flux_display = flux
            err_display = err

            # Get color for this spectrum
            color = spectrum_colors[idx % len(spectrum_colors)]

            # Calculate additional values for hover tooltip
            z = self.redshift

            # Create explicit step coordinates for better hover detection
            # This ensures data points exist at all step edges
            wave_step = []
            flux_display_step = []
            flux_step = []  # Already-scaled flux for tooltips

            for i in range(len(wave)):
                # Calculate step edges
                if i == 0:
                    wave_left = wave[i] - (wave[1] - wave[0]) / 2 if len(wave) > 1 else wave[i]
                else:
                    wave_left = (wave[i-1] + wave[i]) / 2

                if i == len(wave) - 1:
                    wave_right = wave[i] + (wave[i] - wave[i-1]) / 2 if i > 0 else wave[i]
                else:
                    wave_right = (wave[i] + wave[i+1]) / 2

                # Add two points for step (left and right edge)
                wave_step.extend([wave_left, wave_right])
                flux_display_step.extend([flux_display[i], flux_display[i]])
                flux_step.extend([flux[i], flux[i]])  # Already-scaled flux

            wave_step = np.array(wave_step)
            flux_display_step = np.array(flux_display_step)
            flux_step = np.array(flux_step)  # Already-scaled flux

            # Calculate hover values for step data
            wave_obs_micron = wave_step * (1 + z) / 10000.0  # Observed wavelength in microns

            # Observed-frame wavelength in Å (for unit conversion)
            wave_obs_angstrom = wave_step * (1 + z)

            # f_nu in µJy: f_nu [erg/s/cm²/Hz] = f_lambda [erg/s/cm²/Å] × λ²[Å] / c[Å/s]
            # 1 µJy = 1e-29 erg/s/cm²/Hz, so f_nu [µJy] = f_nu [cgs] × 1e29
            c_angstrom_per_s = 2.99792458e18
            with np.errstate(divide='ignore', invalid='ignore'):
                flux_nu_ujy = flux_step * wave_obs_angstrom**2 / c_angstrom_per_s * 1e29

                # AB magnitude: AB = -2.5 * log10(f_nu [µJy] / 3631e6)
                # (3631 Jy = 3631e6 µJy is the AB zeropoint)
                ab_mag = np.where(
                    flux_nu_ujy > 0,
                    -2.5 * np.log10(flux_nu_ujy / 3631e6),
                    np.nan,
                )

            # Tooltip columns in human-readable units
            flux_lambda_tooltip = flux_display_step * 1e20  # raw cgs → units of 10⁻²⁰

            # Create spectrum curve with explicit step data
            # 'flux_raw' is the y-axis (raw cgs), tooltip vdims are scaled
            curve_data = {
                'Wavelength [Å]': wave_step,
                'flux_raw': flux_display_step,
                'f_λ (10⁻²⁰)': flux_lambda_tooltip,
                'λ_obs (µm)': wave_obs_micron,
                'λ_rest (Å)': wave_step,
                'f_ν (µJy)': flux_nu_ujy,
                'AB mag': ab_mag,
            }

            curve = hv.Curve(
                curve_data,
                kdims='Wavelength [Å]',
                vdims=['flux_raw', 'f_λ (10⁻²⁰)', 'λ_obs (µm)', 'λ_rest (Å)', 'f_ν (µJy)', 'AB mag']
            )
            curve = curve.relabel(spectrum_key)  # Set label for legend
            curve = curve.opts(
                color=color,
                alpha=0.8,
                line_width=2.0,
                line_join='miter',  # Sharp corners for steps
                tools=['hover'],
                hover_tooltips=[
                    ('λ_obs (µm)', '@{λ_obs (µm)}{0.000}'),
                    ('λ_rest (Å)', '@{λ_rest (Å)}{0.0}'),
                    ('f_λ (10⁻²⁰)', '@{f_λ (10⁻²⁰)}{0.000}'),
                    ('f_ν (µJy)', '@{f_ν (µJy)}{0.000}'),
                    ('AB mag', '@{AB mag}{0.00}'),
                ],
            )

            # Add error bands if available - make them step-like too!
            if err_display is not None and len(err_display) == len(flux_display):
                # Create step-like coordinates for error bands
                flux_upper = flux_display + err_display
                flux_lower = flux_display - err_display

                # Create step coordinates
                wave_step = []
                upper_step = []
                lower_step = []

                for i in range(len(wave)):
                    if i == 0:
                        wave_left = wave[i] - (wave[1] - wave[0]) / 2 if len(wave) > 1 else wave[i]
                    else:
                        wave_left = (wave[i-1] + wave[i]) / 2

                    if i == len(wave) - 1:
                        wave_right = wave[i] + (wave[i] - wave[i-1]) / 2 if i > 0 else wave[i]
                    else:
                        wave_right = (wave[i] + wave[i+1]) / 2

                    # Add two points for step
                    wave_step.extend([wave_left, wave_right])
                    upper_step.extend([flux_upper[i], flux_upper[i]])
                    lower_step.extend([flux_lower[i], flux_lower[i]])

                # Create area with step coordinates
                error_area = hv.Area((wave_step, lower_step, upper_step),
                                    kdims=['Wavelength [Å]'],
                                    vdims=['y', 'y2'])
                error_area = error_area.opts(
                    color=color,
                    alpha=0.2,
                    line_width=0,
                )

                # Append error area and curve separately
                overlays.append(error_area)
                overlays.append(curve)
            else:
                overlays.append(curve)

        # Return all spectra overlaid
        return hv.Overlay(overlays) if overlays else hv.Curve([])

    def _create_line_markers(self, x_range: tuple[float, float] | None = None) -> hv.Overlay:
        """Create vertical line segments for emission lines with hover tooltips.

        Parameters
        ----------
        x_range : tuple[float, float], optional
            Current x-axis range for filtering lines

        Returns
        -------
        hv.Overlay
            Overlay of line markers with hover info
        """
        # Check if lines should be shown
        if not self.show_lines_toggle.value:
            return hv.Overlay([])

        if x_range is None:
            # Use full spectrum range
            if self.current_spectrum and self.current_spectrum in self.spectrum_dict:
                spec = self.spectrum_dict[self.current_spectrum]
                x_range = (spec['wave'].min(), spec['wave'].max())
            else:
                x_range = (0, 10000)

        lines_in_range = self._filter_lines_in_range(x_range[0], x_range[1])

        if len(lines_in_range) == 0:
            return hv.Overlay([])

        # Filter by strength based on zoom level (fewer lines when zoomed out)
        lines_in_range = self._filter_lines_by_strength(lines_in_range, x_range)

        if len(lines_in_range) == 0:
            return hv.Overlay([])

        # Get set of selected line keys for faster lookup
        selected_keys = {line['key'] for line in self.selected_lines}

        # Compute y midpoint from visible flux for hover point placement
        selected = getattr(self, 'selected_spectra', [])
        all_medians = []
        for spec_key in selected:
            if spec_key in self.spectrum_dict:
                flux = self.spectrum_dict[spec_key]['flux']
                wave = self.spectrum_dict[spec_key]['wave']
                if self.x_range is not None:
                    mask = (wave >= self.x_range[0]) & (wave <= self.x_range[1])
                    flux = flux[mask]
                valid = flux[np.isfinite(flux)]
                if len(valid) > 0:
                    all_medians.append(float(np.nanmedian(valid)))
        y_mid = float(np.mean(all_medians)) if all_medians else 0.5

        # Create vertical line segments for each line
        overlays = []

        # Collect data for hover-enabled scatter points with all info
        hover_data = []

        for row in lines_in_range:
            wave = row['wave_vac']
            ion = row['ion']
            key = row['key']
            color = self._get_element_color(ion)

            # Make selected lines thicker, solid, and more transparent
            # Non-selected lines are dashed
            is_selected = key in selected_keys
            line_width = 5 if is_selected else 2  # Selected lines thicker
            alpha = 0.6 if is_selected else 0.7  # Selected lines slightly more transparent
            line_dash = 'solid' if is_selected else 'dashed'  # Non-selected lines dashed

            # Create a vertical line from y=0 to y=1 (in normalized units)
            vline = hv.VLine(wave).opts(
                color=color,
                line_width=line_width,
                alpha=alpha,
                line_dash=line_dash,
            )
            overlays.append(vline)

            # Collect line info for hover tooltip
            tooltip_dict = {
                'wave': wave,
                'Line': key,
                'Ion': ion,
                'Wave': f"{wave:.2f}",
            }

            # Add other columns if available
            if 'gf' in row.colnames and not np.isnan(row['gf']):
                tooltip_dict['gf'] = f"{row['gf']:.2e}"
            else:
                tooltip_dict['gf'] = 'N/A'

            if 'Aki' in row.colnames and not np.isnan(row['Aki']):
                tooltip_dict['Aki'] = f"{row['Aki']:.2e}"
            else:
                tooltip_dict['Aki'] = 'N/A'

            if 'Ei' in row.colnames and not np.isnan(row['Ei']):
                tooltip_dict['Ei'] = f"{row['Ei']:.2f}"
            else:
                tooltip_dict['Ei'] = 'N/A'

            # Try both 'Ek' and 'El' (upper level energy)
            if 'Ek' in row.colnames and not np.isnan(row['Ek']):
                tooltip_dict['Ek'] = f"{row['Ek']:.2f}"
            elif 'El' in row.colnames and not np.isnan(row['El']):
                tooltip_dict['Ek'] = f"{row['El']:.2f}"
            else:
                tooltip_dict['Ek'] = 'N/A'

            # Parse lower/upper level terms from 'terms' column (e.g. "a6D-u6D*")
            if 'terms' in row.colnames and row['terms'] and str(row['terms']).strip():
                terms_str = str(row['terms']).strip()
                if '-' in terms_str:
                    parts = terms_str.split('-', 1)
                    tooltip_dict['Lower'] = parts[0]
                    tooltip_dict['Upper'] = parts[1]
                else:
                    tooltip_dict['Lower'] = terms_str
                    tooltip_dict['Upper'] = 'N/A'
            else:
                tooltip_dict['Lower'] = 'N/A'
                tooltip_dict['Upper'] = 'N/A'

            hover_data.append(tooltip_dict)

        # Create hover-detectable points at each line position
        # Using Points with mode='vline' so hover triggers at the line's x regardless of y
        if hover_data:
            from bokeh.models import HoverTool

            points_dict = {
                'x': [d['wave'] for d in hover_data],
                'y': [y_mid for _ in hover_data],
                'Line': [d['Line'] for d in hover_data],
                'Ion': [d['Ion'] for d in hover_data],
                'Wave': [d['Wave'] for d in hover_data],
                'gf': [d['gf'] for d in hover_data],
                'Aki': [d['Aki'] for d in hover_data],
                'Ei': [d['Ei'] for d in hover_data],
                'Ek': [d['Ek'] for d in hover_data],
                'Lower': [d['Lower'] for d in hover_data],
                'Upper': [d['Upper'] for d in hover_data],
            }

            hover_tool = HoverTool(
                mode='vline',
                point_policy='snap_to_data',
                attachment='vertical',
                tooltips=[
                    ('Line', '@Line'),
                    ('Ion', '@Ion'),
                    ('λ (Å)', '@Wave'),
                    ('Lower', '@Lower'),
                    ('Upper', '@Upper'),
                    ('gf', '@gf'),
                    ('Aki', '@Aki'),
                    ('Ei (eV)', '@Ei'),
                    ('Ek (eV)', '@Ek'),
                ],
            )

            hover_points = hv.Points(
                points_dict,
                kdims=['x', 'y'],
                vdims=['Line', 'Ion', 'Wave', 'gf', 'Aki', 'Ei', 'Ek', 'Lower', 'Upper'],
            ).opts(
                size=1,
                alpha=0.01,
                tools=[hover_tool],
            )
            overlays.append(hover_points)

        return hv.Overlay(overlays)

    def _create_line_labels(self, x_range: tuple[float, float] | None = None):
        """Create text labels for emission lines with smart positioning.

        Labels dynamically update based on zoom level - more labels when zoomed in.

        Parameters
        ----------
        x_range : tuple[float, float], optional
            Current x-axis range for filtering and overlap detection

        Returns
        -------
        hv.Labels
            Text labels for lines
        """
        # Check if lines should be shown
        if not self.show_lines_toggle.value:
            return hv.Labels([])

        if x_range is None:
            # Use full spectrum range
            if self.current_spectrum and self.current_spectrum in self.spectrum_dict:
                spec = self.spectrum_dict[self.current_spectrum]
                x_range = (spec['wave'].min(), spec['wave'].max())
            else:
                x_range = (0, 10000)

        lines_in_range = self._filter_lines_in_range(x_range[0], x_range[1])

        if len(lines_in_range) == 0:
            return hv.Labels([])

        # Get current spectrum to determine flux values at line positions
        spec = None
        if self.current_spectrum and self.current_spectrum in self.spectrum_dict:
            spec = self.spectrum_dict[self.current_spectrum]

        # Compute approximate character width in data units
        # This depends on current zoom level!
        data_range = x_range[1] - x_range[0]
        approx_char_width = data_range / 1000 * 8  # ~8 pixels per char

        # Get visible label indices (dynamically computed based on zoom)
        visible_idx = self._compute_visible_labels(lines_in_range, x_range, approx_char_width)

        if len(visible_idx) == 0:
            return hv.Labels([])

        # Get normalized flux values (same as displayed spectrum)
        # Need to match the normalization done in _create_spectrum_curve
        flux_display = None
        wave_spec = None
        flux_in_range = None

        if spec is not None:
            # Use flux directly (already scaled at initialization, no further normalization)
            flux_display = spec['flux']
            wave_spec = spec['wave']

            # Get flux values in the current x_range for determining y limits
            in_range_mask = (wave_spec >= x_range[0]) & (wave_spec <= x_range[1])
            flux_in_range = flux_display[in_range_mask]

        # Position each label individually just above the spectrum at that wavelength
        label_data = []
        # Calculate label offset as percentage of flux value (10% above)
        label_offset_pct = 0.10

        # Determine maximum label position based on visible data
        if flux_in_range is not None and len(flux_in_range) > 0:
            flux_max_in_view = np.nanmax(flux_in_range)
            max_label_y = flux_max_in_view
        else:
            max_label_y = 1.0  # Fallback

        for idx in visible_idx:
            row = lines_in_range[idx]
            wave = row['wave_vac']
            key = row['key']

            # Get flux value at this specific wavelength
            if flux_display is not None and wave_spec is not None:
                # Find closest wavelength in spectrum
                wave_idx = np.argmin(np.abs(wave_spec - wave))
                if wave_idx < len(flux_display):
                    flux_at_line = flux_display[wave_idx]
                    # Position label above spectrum at this wavelength (10% higher)
                    label_y = flux_at_line * (1.0 + label_offset_pct)
                else:
                    # Fallback if wavelength not found
                    label_y = max_label_y * 0.75
            else:
                # Fallback if no spectrum data
                label_y = max_label_y * 0.75

            # When zoomed in too much, clamp to max visible flux
            if label_y > max_label_y:
                label_y = max_label_y * 0.95

            label_data.append((wave, label_y, key))

        # Create labels - these are now clickable via the tap stream!
        labels = hv.Labels(label_data, kdims=['x', 'y'], vdims='text').opts(
            text_font_size='10pt',  # Larger font
            text_alpha=0.9,  # More opaque
            text_baseline='bottom',
            text_align='center',
            angle=90,  # Vertical labels
            text_font_style='bold',  # Bolder text
        )

        return labels

    def _create_ion_checkboxes(self) -> None:
        """Create ion checkboxes for ALL unique ions in two columns.

        Note: Checkboxes are created once and stay visible. Unchecking just
        makes lines invisible, doesn't remove the checkbox.
        """
        # Only create once
        if hasattr(self, '_checkboxes_created') and self._checkboxes_created:
            return

        # Clear existing checkboxes
        self.ion_checkboxes.clear()
        self.ion_checkbox_widgets = {}

        # Show ALL unique ions (not filtered by current view)
        ions_sorted = sorted(self.unique_ions)

        # Split into two columns
        mid = (len(ions_sorted) + 1) // 2
        col1_ions = ions_sorted[:mid]
        col2_ions = ions_sorted[mid:]

        col1 = pn.Column()
        col2 = pn.Column()

        # Create checkboxes for first column
        for ion in col1_ions:
            checkbox = pn.widgets.Checkbox(
                name=ion,
                value=ion in self.visible_ions,
            )
            # Store reference
            self.ion_checkbox_widgets[ion] = checkbox
            # Add callback
            checkbox.param.watch(lambda event, i=ion: self._on_ion_toggle(event, i), 'value')
            col1.append(checkbox)

        # Create checkboxes for second column
        for ion in col2_ions:
            checkbox = pn.widgets.Checkbox(
                name=ion,
                value=ion in self.visible_ions,
            )
            # Store reference
            self.ion_checkbox_widgets[ion] = checkbox
            # Add callback
            checkbox.param.watch(lambda event, i=ion: self._on_ion_toggle(event, i), 'value')
            col2.append(checkbox)

        # Add columns side by side in a fixed-height scrollable container
        checkbox_row = pn.Row(col1, col2)
        self.ion_checkboxes.append(checkbox_row)

        self._checkboxes_created = True

    def _create_widgets(self) -> None:
        """Create all UI widgets."""
        # Object name display
        self.object_name_text = pn.pane.Markdown(
            f"## {self.object_name}" if self.object_name else "## Line Explorer",
            sizing_mode='stretch_width',
        )

        # Observed wavelength range display
        self.obs_wave_text = pn.pane.Markdown(
            "",
            sizing_mode='stretch_width',
            styles={'font-size': '10pt', 'color': '#666'},
        )
        self._update_obs_wave_display()

        # Redshift input (editable)
        self.redshift_input = pn.widgets.FloatInput(
            name='Redshift',
            value=self.redshift,
            step=0.001,
            start=0,
            end=20,
            width=150,
        )
        self.redshift_input.param.watch(self._on_redshift_change, 'value')

        # Y-axis scaling selector
        self.yscale_selector = pn.widgets.Select(
            name='Y-axis scale',
            options=['linear', 'log', 'symlog'],
            value='linear',
            width=150,
        )
        self.yscale_selector.param.watch(self._on_yscale_change, 'value')

        # Ion checkboxes in fixed-height scrollable container
        self.ion_checkboxes = pn.Column(
            height=300,  # Fixed height
            scroll=True,  # Enable scrolling
        )
        self.ion_checkbox_widgets = {}
        self._create_ion_checkboxes()

        # Spectrum selector - multi-select (can show multiple spectra)
        spectrum_options = list(self.spectrum_dict.keys())
        # Start with first spectrum selected
        if self.current_spectrum and self.current_spectrum in spectrum_options:
            initial_selection = [self.current_spectrum]
        elif spectrum_options:
            initial_selection = [spectrum_options[0]]
            self.current_spectrum = spectrum_options[0]
        else:
            initial_selection = []

        self.spectrum_selector = pn.widgets.CheckBoxGroup(
            name='Spectrum',
            options=spectrum_options,
            value=initial_selection,
        )
        self.selected_spectra = initial_selection
        self.spectrum_selector.param.watch(self._on_spectrum_change, 'value')

        # Show lines toggle (above catalog selector)
        self.show_lines_toggle = pn.widgets.Checkbox(
            name='Show Emission Lines',
            value=True,  # Lines shown by default
        )
        self.show_lines_toggle.param.watch(self._on_show_lines_toggle, 'value')

        # Catalog selector (only shown if multiple catalogs)
        if len(self.emission_catalogs) > 1:
            catalog_options = list(self.emission_catalogs.keys())
            self.catalog_selector = pn.widgets.RadioBoxGroup(
                name='Emission Line Catalog',
                options=catalog_options,
                value=self.current_catalog_name,
                inline=False,  # Vertical layout
            )
            self.catalog_selector.param.watch(self._on_catalog_change, 'value')
        else:
            self.catalog_selector = None

        # Line filter sliders
        # Compute median line_ratio and Ei across all loaded catalogs
        all_lr, all_ei = [], []
        for cat in self.emission_catalogs.values():
            if 'line_ratio' in cat.colnames:
                lr = np.asarray(cat['line_ratio'], dtype=float)
                lr = lr[np.isfinite(lr)]
                all_lr.append(lr)
            if 'Ei' in cat.colnames:
                ei = np.asarray(cat['Ei'], dtype=float)
                ei = ei[np.isfinite(ei)]
                all_ei.append(ei)

        # line_ratio not used for initial value — always start at 0.5
        _ = all_lr  # computed above but not needed for default

        if all_ei:
            combined_ei = np.concatenate(all_ei)
            ei_max = float(np.ceil(np.max(combined_ei)))
            ei_median = float(np.round(np.median(combined_ei), 1))
        else:
            ei_max, ei_median = 20.0, 5.0

        self.line_ratio_slider = pn.widgets.FloatSlider(
            name='Line ratio min',
            start=0.0,
            end=1.0,
            step=0.01,
            value=0.5,
            width=160,
        )
        self.line_ratio_slider.param.watch(self._on_filter_slider_change, 'value_throttled')

        self.ei_cutoff_slider = pn.widgets.FloatSlider(
            name='Ei (eV) max',
            start=0.0,
            end=ei_max,
            step=0.1,
            value=ei_median,
            width=160,
        )
        self.ei_cutoff_slider.param.watch(self._on_filter_slider_change, 'value_throttled')

        # Invert ion selection button
        self.invert_ion_selection_button = pn.widgets.Button(
            name='Invert Ion Selection',
            button_type='default',
            width=180,
        )
        self.invert_ion_selection_button.on_click(self._on_invert_ion_selection)

        # Selected lines counter
        self.selected_counter = pn.widgets.Button(
            name='0 lines selected',
            button_type='primary',
        )
        self.selected_counter.on_click(self._on_counter_click)

        # Selected lines table in fixed-height container
        self.selected_table = pn.widgets.Tabulator(
            value=self._get_selected_table_data(),
            titles={'key': 'Line', 'ion': 'Ion', 'wave_vac': 'λ (Å)'},
            show_index=False,
            disabled=True,
            height=200,  # Fixed height
        )

        # Save button - show filename if loaded from CSV
        if self.selected_lines_csv_path:
            # Show just the filename (not full path) in button
            csv_filename = Path(self.selected_lines_csv_path).name
            button_label = f'Save to {csv_filename}'
        else:
            button_label = 'Save to CSV'

        self.save_button = pn.widgets.Button(
            name=button_label,
            button_type='success',
        )
        self.save_button.on_click(self._on_save_csv)

        # Load selected lines button
        self.load_selected_button = pn.widgets.FileInput(
            accept='.csv',
            name='Load Selected Lines CSV',
            sizing_mode='stretch_width',
        )
        self.load_selected_button.param.watch(self._on_load_selected_csv, 'value')

        # Load emission lines catalog button
        self.load_catalog_button = pn.widgets.FileInput(
            accept='.csv',
            name='Load Emission Lines Catalog',
            sizing_mode='stretch_width',
        )
        self.load_catalog_button.param.watch(self._on_load_catalog_csv, 'value')

    def _create_plot_components(self) -> None:
        """Create plot components using DynamicMap with streams."""
        # Create streams FIRST
        self.range_stream = streams.RangeXY()
        self.tap_stream = streams.Tap(x=None, y=None)

        # Create callback with EXACT parameter names matching streams
        def make_plot_callback(x_range, y_range, x, y):
            """DynamicMap callback - parameters match stream names EXACTLY."""
            if self.debug:
                print(f"🎨 DynamicMap callback: x_range={x_range}, x={x}, y={y}")

            # Update stored range
            if x_range is not None:
                self.x_range = x_range

            # Create plot
            return self._make_plot_static()

        # Create DynamicMap with streams
        # CRITICAL: streams parameter tells DynamicMap to call callback when streams fire
        dmap = hv.DynamicMap(make_plot_callback, streams=[self.range_stream, self.tap_stream])

        # Subscribe to stream events for side effects (UI updates, etc)
        self.range_stream.add_subscriber(self._on_range_update)
        self.tap_stream.add_subscriber(self._on_tap)

        # Wrap in Panel pane - Panel will handle DynamicMap updates
        self.plot_pane = pn.pane.HoloViews(dmap, sizing_mode='stretch_both')

    def _make_plot_static(self) -> hv.Overlay:
        """Create plot overlay with current state.

        Returns
        -------
        hv.Overlay
            Complete plot overlay with styling
        """
        # Create plot components
        spectrum_curve = self._create_spectrum_curve()
        line_markers = self._create_line_markers(self.x_range)
        line_labels = self._create_line_labels(self.x_range)

        # Add horizontal line at zero
        zero_line = hv.HLine(0).opts(
            color='gray',
            line_width=1,
            alpha=0.5,
            line_dash='dashed',
        )

        # Get y-axis scale
        yscale = self.yscale_selector.value if hasattr(self, 'yscale_selector') else 'linear'

        # Combine into overlay
        opts = dict(
            responsive=True,
            min_height=400,
            xlabel='Rest-frame Wavelength [Å]',
            ylabel='f_λ [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]',
            show_grid=True,
            tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'tap'],
            active_tools=['pan', 'wheel_zoom'],
            fontscale=1.3,  # Scale all fonts by 30%
        )

        # Apply y-axis scaling
        hooks = []

        # Pre-compute flux stats from visible spectra within the current window
        selected = getattr(self, 'selected_spectra', [])
        all_flux = []
        all_err = []
        for spec_key in selected:
            if spec_key in self.spectrum_dict:
                spec = self.spectrum_dict[spec_key]
                wave = spec['wave']
                flux = spec['flux']
                err = spec.get('err', None)
                # Filter to visible x-range window
                if self.x_range is not None:
                    mask = (wave >= self.x_range[0]) & (wave <= self.x_range[1])
                    all_flux.append(flux[mask])
                    if err is not None:
                        all_err.append(err[mask])
                else:
                    all_flux.append(flux)
                    if err is not None:
                        all_err.append(err)
        if all_flux:
            combined = np.concatenate(all_flux)
            valid = combined[np.isfinite(combined)]
        else:
            valid = np.array([])
        if all_err:
            combined_err = np.concatenate(all_err)
            valid_err = combined_err[np.isfinite(combined_err)]
        else:
            valid_err = np.array([])

        if len(valid) > 0:
            flux_min = float(np.nanmin(valid))
            flux_median = float(np.nanmedian(valid))
            flux_max = float(np.nanmax(valid))
            # Smallest positive value for log scales
            pos = valid[valid > 0]
            flux_pos_min = float(np.nanmin(pos)) if len(pos) > 0 else 1e-4
        else:
            flux_min, flux_median, flux_max, flux_pos_min = 0.0, 0.5, 1.0, 1e-4

        if len(valid_err) > 0:
            err_median = float(np.nanmedian(valid_err))
        else:
            err_median = 0.0

        if yscale == 'log':
            opts['logy'] = True

            # Sensible log range: smallest positive value / 10 to 2*median
            y_log_min = flux_pos_min / 10.0
            y_log_max = max(2.0 * flux_median, flux_pos_min * 10.0)

            def apply_log(plot, element):
                """Set sensible y-range for log scale on initialization."""
                from bokeh.models import LogScale, Range1d

                fig = plot.state
                is_initializing = not isinstance(fig.y_scale, LogScale)
                if is_initializing:
                    fig.y_range = Range1d(start=y_log_min, end=y_log_max)

            hooks.append(apply_log)

        elif yscale == 'symlog':
            # linthresh = median flux (transition point from linear to log)
            linthresh = flux_median if flux_median > 0 else 0.1

            # Sensible symlog range
            y_symlog_min = flux_pos_min / 10.0
            y_symlog_max = max(2.0 * flux_median, flux_pos_min * 10.0)

            def apply_symlog(plot, element):
                """Apply log scaling with sensible range derived from data."""
                from bokeh.models import LogScale, Range1d
                from bokeh.models.tickers import FixedTicker
                import numpy as np

                fig = plot.state
                is_initializing = not isinstance(fig.y_scale, LogScale)
                fig.y_scale = LogScale()

                if is_initializing:
                    fig.y_range = Range1d(start=y_symlog_min, end=y_symlog_max)

                # Custom ticks spanning the data range
                current_y_max = fig.y_range.end if hasattr(fig.y_range, 'end') else y_symlog_max
                current_y_min = fig.y_range.start if hasattr(fig.y_range, 'start') else y_symlog_min

                min_exp = int(np.floor(np.log10(max(current_y_min, 1e-30))))
                max_exp = int(np.ceil(np.log10(max(current_y_max, 1e-30))))

                tick_values = [10**e for e in range(min_exp, max_exp + 1)]
                if linthresh not in tick_values:
                    tick_values.append(linthresh)
                    tick_values.sort()

                fig.yaxis.ticker = FixedTicker(ticks=tick_values)

            hooks.append(apply_symlog)

        else:
            # Linear scale
            opts['logy'] = False

            y_lin_min = -2.0 * err_median if err_median > 0 else flux_min
            y_lin_max = 2.0 * flux_median if flux_median > 0 else flux_max

            def apply_linear(plot, element):
                """Apply linear scaling with sensible y-range from data."""
                from bokeh.models import LinearScale, BasicTicker, Range1d, LogScale
                from bokeh.models.formatters import BasicTickFormatter

                fig = plot.state
                transitioning_from_log = isinstance(fig.y_scale, LogScale)

                fig.y_scale = LinearScale()
                fig.yaxis.ticker = BasicTicker()
                fig.yaxis.formatter = BasicTickFormatter()

                # Always reset y-range when switching from log/symlog,
                # or on first render / auto-scaling
                if transitioning_from_log or self._auto_scaling_mode:
                    fig.y_range = Range1d(start=y_lin_min, end=y_lin_max)

            hooks.append(apply_linear)

        # Add hook to enable Shift + drag for box zoom
        def configure_box_zoom(plot, element):
            """Configure BoxZoomTool to activate with Shift held.

            Attaches document-level keydown/keyup listeners (once) that
            toggle between pan and box-zoom based on the Shift key.
            """
            from bokeh.models import CustomJS

            fig = plot.state

            pan_tool = None
            box_zoom_tool = None
            for tool in fig.toolbar.tools:
                if tool.__class__.__name__ == 'PanTool':
                    pan_tool = tool
                elif tool.__class__.__name__ == 'BoxZoomTool':
                    box_zoom_tool = tool

            if not pan_tool or not box_zoom_tool:
                return

            # Guard ensures listeners are added only once per document,
            # and references are updated on each redraw so they always
            # point to the current figure's tools.
            js = CustomJS(args=dict(pan=pan_tool, zoom=box_zoom_tool), code="""
                // Store current tool refs on window so listeners use latest
                window._le_pan = pan;
                window._le_zoom = zoom;

                if (window._le_shift_listeners) return;  // already attached
                window._le_shift_listeners = true;

                document.addEventListener('keydown', (e) => {
                    if (e.key === 'Shift' && window._le_pan && window._le_zoom) {
                        try {
                            window._le_pan.active = false;
                            window._le_zoom.active = true;
                        } catch(err) {}
                    }
                });
                document.addEventListener('keyup', (e) => {
                    if (e.key === 'Shift' && window._le_pan && window._le_zoom) {
                        try {
                            window._le_zoom.active = false;
                            window._le_pan.active = true;
                        } catch(err) {}
                    }
                });
            """)
            # rangesupdate is a PlotEvent — fires on initial render and
            # on every DynamicMap redraw when the new figure computes ranges.
            # (document_ready is a DocumentEvent and does NOT fire on figures.)
            fig.js_on_event('rangesupdate', js)

        hooks.append(configure_box_zoom)

        # Add hook to create secondary x-axis for observed wavelength
        def add_observed_axis(plot, element):
            """Add secondary x-axis showing observed wavelength in microns (twin axis)."""
            from bokeh.models import LinearAxis, CustomJSTickFormatter, FixedTicker
            import numpy as np

            # Get the Bokeh figure
            fig = plot.state

            # Remove any existing secondary axes from 'above' position
            # to prevent duplicates on redraw
            existing_above = [item for item in fig.above if isinstance(item, LinearAxis)]
            for axis in existing_above:
                fig.above.remove(axis)

            z = self.redshift

            # Get the x-axis range to determine nice tick positions
            x_range = fig.x_range
            rest_min = x_range.start if hasattr(x_range, 'start') else 3000
            rest_max = x_range.end if hasattr(x_range, 'end') else 7000

            # Convert to observed wavelength in microns
            obs_min_micron = rest_min * (1 + z) / 10000.0
            obs_max_micron = rest_max * (1 + z) / 10000.0

            # Determine nice tick interval based on range
            range_micron = obs_max_micron - obs_min_micron
            if range_micron > 5:
                tick_interval = 1.0  # 1 µm intervals for large ranges
            elif range_micron > 2:
                tick_interval = 0.5  # 0.5 µm intervals for medium ranges
            else:
                tick_interval = 0.2  # 0.2 µm intervals for small ranges

            # Generate nice tick positions in microns
            first_tick = np.ceil(obs_min_micron / tick_interval) * tick_interval
            last_tick = np.floor(obs_max_micron / tick_interval) * tick_interval
            tick_positions_micron = np.arange(first_tick, last_tick + tick_interval/2, tick_interval)

            # Convert back to rest-frame wavelength for positioning
            tick_positions_rest = tick_positions_micron * 10000.0 / (1 + z)

            # Create formatter that converts rest-frame to observed in microns
            formatter = CustomJSTickFormatter(code=f"""
                const obs_micron = tick * {1 + z} / 10000.0;
                return obs_micron.toFixed(1);
            """)

            # Create ticker with fixed positions
            ticker = FixedTicker(ticks=tick_positions_rest.tolist())

            # Create the twin axis
            obs_axis = LinearAxis(
                axis_label=f'Observed Wavelength [µm] (z={z:.3f})',
                axis_label_text_font_size='11pt',
                formatter=formatter,
                ticker=ticker,
            )

            # Add axis to top of plot
            fig.add_layout(obs_axis, 'above')

        hooks.append(add_observed_axis)

        # Add hook to configure legend and ensure it's created
        def configure_legend(plot, element):
            """Configure legend appearance and position."""
            from bokeh.models import Legend, LegendItem
            from bokeh.models.glyphs import Line

            fig = plot.state

            # More aggressive cleanup: find and remove ALL Legend objects
            # Collect all legends first to avoid modifying list during iteration
            all_legends = []

            # Check all possible locations where legends might be stored
            for attr in ['renderers', 'above', 'below', 'left', 'right']:
                if hasattr(fig, attr):
                    container = getattr(fig, attr)
                    for item in list(container):  # Use list() to avoid modification during iteration
                        if isinstance(item, Legend):
                            all_legends.append((container, item))

            # Check center grid if it exists
            if hasattr(fig, 'center') and len(fig.center) > 0:
                if hasattr(fig.center[0], 'renderers'):
                    for item in list(fig.center[0].renderers):
                        if isinstance(item, Legend):
                            all_legends.append((fig.center[0].renderers, item))

            # Remove all found legends
            for container, legend in all_legends:
                try:
                    container.remove(legend)
                except (ValueError, AttributeError):
                    pass  # Already removed or invalid container

            # Get selected spectra for legend
            selected = getattr(self, 'selected_spectra', [])

            # Always create legend if any spectra are selected
            if len(selected) >= 1:
                # Find line renderers (spectrum curves) - they use Line glyph
                line_renderers = []
                for renderer in fig.renderers:
                    if hasattr(renderer, 'glyph') and isinstance(renderer.glyph, Line):
                        # Check if this is a spectrum line (not a VLine for emission lines)
                        # Spectrum lines have data_source with 'Wavelength [Å]' column
                        if hasattr(renderer, 'data_source'):
                            columns = renderer.data_source.column_names
                            if 'Wavelength [Å]' in columns and 'Normalized Flux' in columns:
                                line_renderers.append(renderer)

                # Create legend items - match as many as we can
                legend_items = []
                num_items = min(len(line_renderers), len(selected))
                for i in range(num_items):
                    legend_items.append(LegendItem(label=selected[i], renderers=[line_renderers[i]]))

                if legend_items:
                    # Create new legend INSIDE the plot area (top right)
                    legend = Legend(
                        items=legend_items,
                        location='top_right',
                        click_policy='hide',
                        label_text_font_size='10pt',
                        background_fill_alpha=0.8,  # Semi-transparent background
                        border_line_color='gray',
                        border_line_width=1,
                    )
                    # Add to center[0] for inside plot positioning
                    fig.add_layout(legend)

        hooks.append(configure_legend)
        opts['hooks'] = hooks

        plot = (zero_line * spectrum_curve * line_markers * line_labels).opts(**opts)

        return plot

    def _get_selected_table_data(self):
        """Get selected lines as a DataFrame for Tabulator widget.

        Returns
        -------
        pandas.DataFrame
            DataFrame with 'key', 'ion', 'wave_vac' columns
        """
        import pandas as pd

        if not self.selected_lines:
            return pd.DataFrame({'key': [], 'ion': [], 'wave_vac': []})

        keys = [line['key'] for line in self.selected_lines]
        ions = [line['ion'] for line in self.selected_lines]
        waves = [line['wave_vac'] for line in self.selected_lines]

        return pd.DataFrame({'key': keys, 'ion': ions, 'wave_vac': waves})

    def _update_obs_wave_display(self) -> None:
        """Update the observed wavelength range display."""
        if self.current_spectrum and self.current_spectrum in self.spectrum_dict:
            spec = self.spectrum_dict[self.current_spectrum]
            wave_rest = spec['wave']

            # Get current view range or full range
            if self.x_range:
                wave_min, wave_max = self.x_range
            else:
                wave_min, wave_max = wave_rest.min(), wave_rest.max()

            # Convert to observed frame
            wave_obs_min = wave_min * (1 + self.redshift)
            wave_obs_max = wave_max * (1 + self.redshift)

            self.obs_wave_text.object = (
                f"**Observed wavelength range:** {wave_obs_min:.1f} - {wave_obs_max:.1f} Å "
                f"(z = {self.redshift:.3f})"
            )
        else:
            self.obs_wave_text.object = ""

    def _on_range_update(self, x_range=None, y_range=None) -> None:
        """Callback for plot range changes (zoom/pan).

        This updates the stored range. DynamicMap will automatically redraw.

        Parameters
        ----------
        x_range : tuple, optional
            New x-axis range
        y_range : tuple, optional
            New y-axis range
        """
        if x_range is None:
            return

        # If in auto-scaling mode, detect if user is zooming/panning
        if self._auto_scaling_mode:
            # Get full data range to compare against
            full_range = None
            if self.current_spectrum and self.current_spectrum in self.spectrum_dict:
                wave = self.spectrum_dict[self.current_spectrum]['wave']
                full_range = (wave.min(), wave.max())

            if full_range is not None:
                # Check if the new range is significantly different from full range
                new_range_size = x_range[1] - x_range[0]
                full_range_size = full_range[1] - full_range[0]

                # If range is smaller than 95% of full range, user has zoomed in
                if new_range_size < 0.95 * full_range_size:
                    if self.debug:
                        print(f"🔍 User zoom detected (range {new_range_size:.0f} < 95% of {full_range_size:.0f}), exiting auto-scaling")
                    self._auto_scaling_mode = False
                    self.x_range = x_range
                    self._update_obs_wave_display()
                    return

                # If range center has shifted significantly, user has panned
                new_center = (x_range[0] + x_range[1]) / 2
                full_center = (full_range[0] + full_range[1]) / 2
                center_shift = abs(new_center - full_center)

                if center_shift > 0.05 * full_range_size:  # 5% shift = pan
                    if self.debug:
                        print(f"🔍 User pan detected (shift {center_shift:.0f}), exiting auto-scaling")
                    self._auto_scaling_mode = False
                    self.x_range = x_range
                    self._update_obs_wave_display()
                    return

            # Still in auto-scaling mode, ignore this range update
            if self.debug:
                print(f"🔍 Range update ignored (staying in auto-scaling mode)")
            return

        # Not in auto-scaling mode, track the range normally
        if self.debug:
            print(f"🔍 Range update: x={x_range}")

        self.x_range = x_range
        self._update_obs_wave_display()

    def _on_tap(self, x=None, y=None) -> None:
        """Callback for tap/click on plot.

        Finds the nearest line to click position and toggles selection.

        Parameters
        ----------
        x : float, optional
            Click x-coordinate (wavelength)
        y : float, optional
            Click y-coordinate (flux)
        """
        if self.debug:
            print(f"👆 Tap event: x={x}, y={y}")

        if x is None or self.x_range is None:
            return

        # Get lines in current range
        lines_in_range = self._filter_lines_in_range(self.x_range[0], self.x_range[1])

        if len(lines_in_range) == 0:
            return

        # Find nearest line to click
        waves = np.array(lines_in_range['wave_vac'])
        distances = np.abs(waves - x)
        closest_idx = np.argmin(distances)

        # Check if close enough (within 1% of visible range)
        tolerance = (self.x_range[1] - self.x_range[0]) * 0.01
        if distances[closest_idx] > tolerance:
            if self.debug:
                print(f"  ⚠️  Click too far from line: {distances[closest_idx]:.2f} > {tolerance:.2f}")
            return

        # Get the line key
        line_key = lines_in_range['key'][closest_idx]
        if self.debug:
            print(f"  ✓ Clicked on line: {line_key}")
        logger.info(f"Clicked on line: {line_key}")

        # Toggle selection (add_line/remove_line will call _update_plot_simple)
        if any(line['key'] == line_key for line in self.selected_lines):
            self.remove_line(line_key)
        else:
            self.add_line(line_key)

    def _update_plot_simple(self) -> None:
        """Update the plot by triggering stream event or plot recreation."""
        if not self._auto_scaling_mode and self.x_range is not None:
            # If we have a specific range (user has zoomed/panned), preserve it
            if hasattr(self, 'range_stream'):
                original_range = self.x_range
                # Trigger with slightly different value
                epsilon = 0.001
                self.range_stream.event(
                    x_range=(self.x_range[0] - epsilon, self.x_range[1] + epsilon),
                    y_range=None
                )
                # Immediately trigger with original range
                self.range_stream.event(x_range=original_range, y_range=None)
        else:
            # In auto-scaling mode or x_range is None, trigger update without setting range
            if hasattr(self, 'range_stream'):
                # Trigger range stream with None to force plot update
                # Auto-scaling will be preserved because _on_range_update will ignore updates
                self.range_stream.event(x_range=None, y_range=None)

        # Update observed wavelength display
        self._update_obs_wave_display()

    def _on_ion_toggle(self, event, ion: str) -> None:
        """Callback for ion checkbox toggle.

        Parameters
        ----------
        event : param.Event
            Parameter change event
        ion : str
            Ion name being toggled
        """
        # Skip if suppressed (e.g., during invert selection batch update)
        if getattr(self, '_suppress_ion_callbacks', False):
            return

        if self.debug:
            print(f"☑️  Ion toggle: {ion} = {event.new}")

        if event.new:
            self.visible_ions.add(ion)
        else:
            self.visible_ions.discard(ion)

        # Update plot
        self._update_plot_simple()

    def _on_invert_ion_selection(self, event) -> None:
        """Callback for invert ion selection button.

        Inverts the state of all ion checkboxes.
        Efficiently updates all checkboxes at once without triggering individual callbacks.
        """
        if self.debug:
            print("🔄 Inverting ion selection")

        # Get all unique ions
        all_ions = set(self.unique_ions)

        # Invert: visible becomes invisible, invisible becomes visible
        new_visible_ions = all_ions - self.visible_ions

        # Update visible_ions first
        self.visible_ions = new_visible_ions

        # Suppress individual checkbox callbacks during batch update
        self._suppress_ion_callbacks = True
        try:
            for ion, checkbox in self.ion_checkbox_widgets.items():
                checkbox.value = ion in self.visible_ions
        finally:
            self._suppress_ion_callbacks = False

        # Update plot ONCE after all checkboxes are updated
        logger.info(f"Inverted ion selection: {len(self.visible_ions)} ions now visible")
        self._update_plot_simple()

    def _on_spectrum_change(self, event) -> None:
        """Callback for spectrum selector change (multi-select).

        Parameters
        ----------
        event : param.Event
            Parameter change event
        """
        # event.new is a list of selected spectra
        if event.new == self.selected_spectra:
            return  # No change

        if self.debug:
            print(f"📡 Spectrum change: {self.selected_spectra} -> {event.new}")

        logger.info(f"Spectrum change: {self.selected_spectra} -> {event.new}")
        self.selected_spectra = event.new if event.new else []

        # Set current_spectrum to first selected (for compatibility with label positioning)
        if self.selected_spectra:
            self.current_spectrum = self.selected_spectra[0]
        else:
            self.current_spectrum = None

        # Don't reset x_range - preserve current zoom/pan state for auto-scaling
        # If x_range is None (initial state), it will remain None and plot will auto-scale
        # If x_range is set (user has zoomed/panned), it will be preserved

        # Trigger plot update
        self._update_plot_simple()

    def _on_redshift_change(self, event) -> None:
        """Callback for redshift input change.

        Re-computes rest-frame wavelengths for all spectra with new redshift.

        Parameters
        ----------
        event : param.Event
            Parameter change event
        """
        if event.new != self.redshift:
            if self.debug:
                print(f"🔴 Redshift change: {self.redshift} -> {event.new}")

            logger.info(f"Redshift change: {self.redshift} -> {event.new}")

            old_z = self.redshift
            new_z = event.new
            self.redshift = new_z

            # Recompute rest-frame wavelengths for all spectra
            # The observed wavelengths are fixed, so we just need to re-scale
            for spec_key in self.spectrum_dict:
                spec = self.spectrum_dict[spec_key]
                # Convert current rest-frame back to observed, then to new rest-frame
                wave_obs = spec['wave'] * (1 + old_z)
                wave_rest_new = wave_obs / (1 + new_z)
                spec['wave'] = wave_rest_new

            # Update observed wavelength display
            self._update_obs_wave_display()

            # Trigger plot update to show shifted spectrum
            self._update_plot_simple()

    def _on_catalog_change(self, event) -> None:
        """Callback for catalog selector change.

        Parameters
        ----------
        event : param.Event
            Parameter change event
        """
        new_catalog = event.new
        if new_catalog == self.current_catalog_name:
            return  # No change

        if self.debug:
            print(f"📚 Catalog change: {self.current_catalog_name} -> {new_catalog}")

        logger.info(f"Catalog change: {self.current_catalog_name} -> {new_catalog}")

        old_catalog_name = self.current_catalog_name
        self.current_catalog_name = new_catalog

        # Switch to new catalog (apply hydrogen filtering)
        logger.info(f"Switching from catalog '{old_catalog_name}' to '{new_catalog}'")
        self.emission_lines = self._filter_hydrogen_series(
            self.emission_catalogs[new_catalog].copy()
        )
        logger.info(f"New catalog has {len(self.emission_lines)} lines after filtering")

        # Re-extract unique ions from the new catalog
        self._extract_unique_ions()

        # Reset visible ions to include all ions from new catalog
        self.visible_ions = set(self.unique_ions)
        if 'H I' in self.unique_ions:
            self.visible_ions.add('H I')

        # Update filter sliders to new catalog's medians
        self._update_filter_sliders()

        # Update ion checkboxes (different catalog may have different ions)
        # Must reset the creation flag to allow recreation
        self._checkboxes_created = False
        self._create_ion_checkboxes()

        # Trigger plot update
        logger.info(f"Triggering plot update for catalog '{new_catalog}'")
        self._update_plot_simple()

    def _on_show_lines_toggle(self, event) -> None:
        """Callback for show lines toggle.

        Parameters
        ----------
        event : param.Event
            Parameter change event
        """
        show_lines = event.new
        if self.debug:
            print(f"👁️ Show lines toggle: {show_lines}")

        logger.info(f"Show lines: {show_lines}")
        # Trigger plot update
        self._update_plot_simple()

    def _on_yscale_change(self, event) -> None:
        """Callback for y-axis scale change.

        Parameters
        ----------
        event : param.Event
            Parameter change event
        """
        if self.debug:
            print(f"📊 Y-scale change: {event.new}")

        logger.info(f"Y-scale change: {event.new}")

        # Trigger plot update
        self._update_plot_simple()

    def _update_filter_sliders(self) -> None:
        """Update filter slider ranges for the current catalog.

        Only adjusts the slider *range* (end) to fit the new catalog.
        Slider *values* are preserved so filters persist across catalog switches.
        """
        cat = self.emission_lines

        if 'Ei' in cat.colnames:
            ei = np.asarray(cat['Ei'], dtype=float)
            ei_valid = ei[np.isfinite(ei)]
            if len(ei_valid) > 0:
                new_max = float(np.ceil(np.max(ei_valid)))
                self.ei_cutoff_slider.end = new_max
                # Clamp value if it exceeds the new range
                if self.ei_cutoff_slider.value > new_max:
                    self.ei_cutoff_slider.value = new_max

    def _on_filter_slider_change(self, event) -> None:
        """Callback for gf or Ei cutoff slider change."""
        if self.debug:
            print(f"🔧 Filter slider change: {event.obj.name} = {event.new}")
        self._update_plot_simple()

    def _on_counter_click(self, event) -> None:
        """Callback for selected lines counter click.

        Parameters
        ----------
        event : param.Event
            Button click event
        """
        # Show/hide selected lines panel
        logger.info(f"Selected {len(self.selected_lines)} lines")

    def _on_save_csv(self, event) -> None:
        """Callback for save to CSV button.

        If loaded from CSV, saves back to the same file.
        Otherwise, saves to <object_name>-<date>.csv format.

        Parameters
        ----------
        event : param.Event
            Button click event
        """
        if not self.selected_lines:
            logger.warning("No lines selected to save")
            return

        # Convert to astropy table and save
        from astropy.table import Table
        import datetime

        # Create table from selected lines
        keys = [line['key'] for line in self.selected_lines]
        ions = [line['ion'] for line in self.selected_lines]
        waves = [line['wave_vac'] for line in self.selected_lines]

        table = Table([keys, ions, waves], names=['key', 'ion', 'wave_vac'])

        # Determine filename
        if self.selected_lines_csv_path:
            # Save back to the same file we loaded from
            filename = self.selected_lines_csv_path
            logger.info(f"Saving back to original file: {filename}")
        else:
            # Generate new filename: <object_name>-<date>.csv
            date_str = datetime.datetime.now().strftime('%Y%m%d')
            # Clean object name (replace spaces/special chars with underscores)
            object_name_clean = self.object_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
            if not object_name_clean:
                object_name_clean = 'selected_lines'
            filename = f'{object_name_clean}-{date_str}.csv'
            logger.info(f"Saving to new file: {filename}")

        # Save
        table.write(filename, format='csv', overwrite=True)
        logger.info(f"✓ Saved {len(self.selected_lines)} lines to {filename}")

    def _load_selected_lines_from_csv(self, csv_path: str) -> None:
        """Load previously selected lines from a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to CSV file containing selected lines (from previous Save to CSV)
        """
        try:
            # Load CSV
            saved_table = Table.read(csv_path, format='csv')
            logger.info(f"Loading {len(saved_table)} lines from {csv_path}...")

            # Add each line from the CSV WITHOUT updating UI (batch mode)
            if 'key' in saved_table.colnames:
                loaded_count = 0
                for row in saved_table:
                    line_key = row['key']
                    try:
                        # Use update_ui=False to skip UI updates during batch loading
                        self.add_line(line_key, update_ui=False)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Could not add line {line_key}: {e}")

                # Update display ONCE after all lines are added
                self._update_selected_counter()
                self.selected_table.value = self._get_selected_table_data()
                # Update plot to show selected lines
                self._update_plot_simple()

                logger.info(f"✓ Loaded {loaded_count}/{len(saved_table)} lines from CSV")
            else:
                logger.error(f"CSV file {csv_path} does not have 'key' column")

        except Exception as e:
            logger.error(f"Error loading selected lines from {csv_path}: {e}")

    def _on_load_selected_csv(self, event) -> None:
        """Callback for load selected lines CSV file input.

        Parameters
        ----------
        event : param.Event
            FileInput change event
        """
        if event.new is None:
            return

        try:
            # FileInput.value contains the file contents as bytes
            import io
            from astropy.table import Table

            # Convert bytes to string and load as CSV
            csv_data = io.BytesIO(event.new)
            saved_table = Table.read(csv_data, format='csv')

            logger.info(f"Loading {len(saved_table)} lines from uploaded CSV...")

            # Clear current selection first
            self.selected_lines = []

            # Add each line from the CSV WITHOUT updating UI (batch mode)
            if 'key' in saved_table.colnames:
                loaded_count = 0
                for row in saved_table:
                    line_key = row['key']
                    try:
                        # Use update_ui=False to skip UI updates during batch loading
                        self.add_line(line_key, update_ui=False)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Could not add line {line_key}: {e}")

                # Update display ONCE after all lines are added
                self._update_selected_counter()
                self.selected_table.value = self._get_selected_table_data()
                self._update_plot_simple()

                logger.info(f"✓ Loaded {loaded_count}/{len(saved_table)} lines from CSV")
            else:
                logger.error("CSV file does not have 'key' column")

        except Exception as e:
            logger.error(f"Error loading selected lines from CSV: {e}")

    def _on_load_catalog_csv(self, event) -> None:
        """Callback for load emission lines catalog CSV file input.

        Parameters
        ----------
        event : param.Event
            FileInput change event
        """
        if event.new is None:
            return

        try:
            # FileInput.value contains the file contents as bytes
            import io
            from astropy.table import Table

            # Convert bytes to string and load as CSV
            csv_data = io.BytesIO(event.new)
            new_catalog = Table.read(csv_data, format='csv')

            logger.info(f"Loading {len(new_catalog)} emission lines from uploaded CSV")

            # Replace emission lines catalog
            self.emission_lines = new_catalog

            # Re-extract unique ions
            self._extract_unique_ions()

            # Recreate ion checkboxes with new catalog
            self._checkboxes_created = False
            self._create_ion_checkboxes()

            # Trigger plot update
            self._update_plot_simple()

            logger.info(f"Loaded {len(self.emission_lines)} emission lines from CSV")

        except Exception as e:
            logger.error(f"Error loading emission lines catalog from CSV: {e}")

    def _update_selected_counter(self) -> None:
        """Update the selected lines counter display."""
        n = len(self.selected_lines)
        self.selected_counter.name = f'{n} line{"s" if n != 1 else ""} selected'

    def add_line(self, line_key: str, update_ui: bool = True) -> None:
        """Add a line to the selection.

        Parameters
        ----------
        line_key : str
            Line key to add
        update_ui : bool, optional
            If True, update UI and plot after adding. Set to False for batch operations.
            Default is True.
        """
        # Check if already selected
        if any(line['key'] == line_key for line in self.selected_lines):
            logger.debug(f"Line {line_key} already selected")
            return

        # Find line in current catalog first
        mask = self.emission_lines['key'] == line_key
        if any(mask):
            # Line found in current catalog
            row = self.emission_lines[mask][0]
            # Convert to native Python types to avoid NaN display issues
            line_dict = {
                'key': str(row['key']),
                'ion': str(row['ion']),
                'wave_vac': float(row['wave_vac'])
            }
            self.selected_lines.append(line_dict)

            # Only update UI if requested (for batch operations, skip updates)
            if update_ui:
                self._update_selected_counter()
                self.selected_table.value = self._get_selected_table_data()
                self._update_plot_simple()
                logger.info(f"Added line {line_key} to selection")
            else:
                logger.debug(f"Added line {line_key} to selection (no UI update)")
        else:
            # Line not in current catalog, search other catalogs
            found_in_other_catalog = False
            for catalog_name, catalog_table in self.emission_catalogs.items():
                if catalog_name == self.current_catalog_name:
                    continue  # Already checked this one

                # Apply hydrogen filtering to the catalog before searching
                filtered_catalog = self._filter_hydrogen_series(catalog_table.copy())
                mask = filtered_catalog['key'] == line_key
                if any(mask):
                    row = filtered_catalog[mask][0]
                    line_dict = {
                        'key': str(row['key']),
                        'ion': str(row['ion']),
                        'wave_vac': float(row['wave_vac'])
                    }
                    self.selected_lines.append(line_dict)
                    logger.info(f"Added line {line_key} from catalog '{catalog_name}' (not in current catalog)")
                    found_in_other_catalog = True
                    break

            if not found_in_other_catalog:
                logger.warning(f"Line {line_key} not found in any catalog")
                return

            # Update UI if requested (even if line not in current catalog)
            if update_ui:
                self._update_selected_counter()
                self.selected_table.value = self._get_selected_table_data()
                # Don't update plot since line won't be visible in current catalog anyway

    def remove_line(self, line_key: str) -> None:
        """Remove a line from the selection.

        Parameters
        ----------
        line_key : str
            Line key to remove
        """
        # Remove from selected lines
        self.selected_lines = [line for line in self.selected_lines if line['key'] != line_key]

        # Update UI
        self._update_selected_counter()
        self.selected_table.value = self._get_selected_table_data()

        # Update plot to show normal line width
        self._update_plot_simple()

        logger.info(f"Removed line {line_key} from selection")

    def panel(self) -> pn.Row:
        """Return Panel layout for embedding in notebooks.

        Returns
        -------
        pn.Row
            Panel layout with 3 columns: plot, controls, selected lines
        """
        # Column 1: Main plot
        col1_plot = pn.Column(
            self.object_name_text,
            self.plot_pane,
            sizing_mode='stretch_both',
        )

        # Column 2: Controls (redshift, scale, spectrum, catalog, ion filters)
        col2_widgets = [
            pn.pane.Markdown("### Settings"),
            self.redshift_input,
            self.yscale_selector,
            pn.pane.Markdown("### Spectrum"),
            self.spectrum_selector,
        ]

        # Add show lines toggle
        col2_widgets.append(self.show_lines_toggle)

        # Add catalog selector if multiple catalogs available
        if self.catalog_selector is not None:
            col2_widgets.extend([
                pn.pane.Markdown("### Line Catalog"),
                self.catalog_selector,
            ])

        col2_widgets.extend([
            pn.pane.Markdown("### Line Filters"),
            self.line_ratio_slider,
            self.ei_cutoff_slider,
            pn.pane.Markdown("### Ion Filters"),
            self.invert_ion_selection_button,
            self.ion_checkboxes,
        ])

        col2_controls = pn.Column(
            *col2_widgets,
            width=180,  # 40% narrower (was 300)
        )

        # Column 3: Selected lines and CSV loaders
        col3_selected = pn.Column(
            self.selected_counter,
            pn.pane.Markdown("### Selected Lines"),
            self.selected_table,
            self.save_button,
            self.load_selected_button,
            pn.pane.Markdown("### Load Data"),
            self.load_catalog_button,
            width=300,
        )

        layout = pn.Row(
            col1_plot,
            col2_controls,
            col3_selected,
            sizing_mode='stretch_both',
        )
        return layout

    def serve(self, port: int = 5006, show: bool = True) -> None:
        """Launch standalone Panel server.

        Parameters
        ----------
        port : int, optional
            Port number for server
        show : bool, optional
            Whether to open browser automatically
        """
        logger.info(f"Starting LineExplorer server on port {port}")
        pn.serve(self.panel(), port=port, show=show)
