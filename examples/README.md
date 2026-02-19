# LineExplorer Examples

This directory contains examples demonstrating how to use the `LineExplorer` interactive app for emission line identification.

## What is LineExplorer?

`LineExplorer` is an interactive Panel + HoloViews application for exploring spectra, identifying emission lines, and building selected line lists for analysis. It provides:

- Interactive pan and zoom on spectrum plots
- Spectrum display with rest-frame wavelength
- Line markers colored by element
- Smart text labels that avoid overlap
- Ion filter checkboxes to toggle line visibility
- Spectrum selector for multi-grating data
- Selected lines table with CSV export

## Examples

### 1. Real JWST Data Demo (`line_explorer_real_data.py`) ⭐ **RECOMMENDED**

**Uses real JWST NIRSpec data from DJA (DAWN JWST Archive).**

Command line usage:
```bash
poetry run python examples/line_explorer_real_data.py

git clone https://github.com/ivolabbe/dotfit.git ; cd dotfit
poetry install ; cd examples
poetry run python line_explorer_monster.py --emission-lines-csv '../dotfit/data/emission_lines/*csv' --selected-lines-csv monster_v2.csv  --redshift 4.464 --port 8080     
```


This will:
1. Download real NIRSpec spectra for abell2744-8204-45924 (z=4.465)
2. Load emission line catalog
3. Launch interactive app at http://localhost:5006
4. Falls back to demo data if download fails

**Features:**
- Real JWST NIRSpec G235M, G395M, and PRISM data
- Uses `astropy.utils.data.download_file` (lightweight!)
- Multi-grating support
- Automatic line selection
- Works from command line
- No heavy dependencies needed!

**Implementation:**
- Uses standard astropy for FITS reading
- Proper unit handling with astropy.units
- Handles masked arrays correctly
- Downloads cached for fast re-runs

### 2. Full JWST Notebook (`line_explorer_jwst_demo.py`)

**Comprehensive notebook with real JWST data and detailed documentation.**

Jupyter notebook usage:
```python
# In notebook, run cells one by one
%run examples/line_explorer_jwst_demo.py
```

Or run as script (launches server):
```bash
poetry run python examples/line_explorer_jwst_demo.py
```

**Features demonstrated:**
- Real JWST data loading with `load_monster_spectra()` using astropy
- Emission line catalog filtering
- Multi-grating spectrum support (PRISM, G235M, G395M)
- Interactive app in notebooks vs. standalone
- Programmatic line selection
- CSV export
- Fallback demo data if download fails
- Full documentation in cells
- Unit conversion examples
- Masked array handling

**File structure:**
- Notebook-style with `# %%` cell markers
- Can be converted to .ipynb with jupytext
- Works both as script and in notebook
- No msaexp dependency required!

### 3. Simple Demo (`line_explorer_demo.py`)

**Quick standalone demo with synthetic spectrum.**

```bash
poetry run python examples/line_explorer_demo.py
```

Creates synthetic spectrum and launches browser app.

### 4. Basic Usage (`line_explorer_notebook_example.py`)

**Minimal usage pattern.**

```python
from dotfit import EmissionLines, LineExplorer

el = EmissionLines()
tab = el.table[el.table['Ei'] < 6]

spectrum_dict = {
    'g235m': {'wave': wave_array, 'flux': flux_array},
}

app = LineExplorer(
    spectrum_dict=spectrum_dict,
    emission_lines=tab,
    redshift=4.465,
    object_name="My Object",
)

app.panel()  # notebook
# app.serve(port=5006)  # standalone
```

## API Reference

### LineExplorer Class

```python
LineExplorer(
    spectrum_dict: dict,
    emission_lines: astropy.table.Table,
    redshift: float = 0,
    object_name: str = "",
)
```

**Parameters:**
- `spectrum_dict`: Dictionary mapping spectrum names to `{'wave': array, 'flux': array}` dicts
- `emission_lines`: Emission line catalog from `EmissionLines.table`
- `redshift`: Redshift for converting rest-frame to observed wavelengths
- `object_name`: Name of the object being analyzed

**Methods:**
- `panel()`: Returns Panel layout for embedding in notebooks
- `serve(port=5006, show=True)`: Launches standalone server
- `add_line(line_key)`: Programmatically add a line to selection
- `remove_line(line_key)`: Remove a line from selection

## Requirements

The LineExplorer requires the following packages (automatically installed with `poetry install`):
- panel >= 1.3
- holoviews >= 1.18
- bokeh >= 3.3

## Tips

1. **Filter the line catalog**: Use `el.table[el.table['Ei'] < 6]` to show only lower energy lines for cleaner displays

2. **Multiple spectra**: Pass multiple spectra in `spectrum_dict` to allow switching between gratings

3. **Large datasets**: For very large line catalogs, consider pre-filtering by wavelength range

4. **Export selections**: Use the "Save to CSV" button to export your selected lines for further analysis

5. **Integration**: The selected lines can be accessed programmatically via `app.selected_lines` for integration with fitting pipelines
