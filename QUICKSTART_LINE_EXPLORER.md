# LineExplorer Quick Start Guide

## Installation

The LineExplorer is included in the dotfit package. Make sure all dependencies are installed:

```bash
poetry install
```

This will install Panel, HoloViews, and Bokeh along with other dependencies.

## Quick Start (3 steps)

### 1. Load Emission Lines

```python
from dotfit import EmissionLines

el = EmissionLines()
tab = el.table[el.table['Ei'] < 6]  # Filter to lower energy lines
```

### 2. Prepare Your Spectrum

```python
# Your spectrum should be a dictionary with this structure:
spectrum_dict = {
    'G235M': {
        'wave': wave_array,  # Rest-frame wavelength in Angstroms
        'flux': flux_array,  # Flux (arbitrary units, will be normalized)
    },
    'G395M': {
        'wave': wave_array2,
        'flux': flux_array2,
    },
}
```

### 3. Launch the App

```python
from dotfit import LineExplorer

app = LineExplorer(
    spectrum_dict=spectrum_dict,
    emission_lines=tab,
    redshift=4.465,  # Your object's redshift
    object_name="abell2744-8204-45924",
)

# In a Jupyter notebook:
app.panel()

# Or as standalone app:
app.serve(port=5006)
```

## Examples

### Try the Interactive Notebook

```bash
poetry run python examples/line_explorer_interactive.py
```

This demonstrates:
- Creating synthetic spectra
- Multi-grating support
- Interactive exploration
- Line selection and export

### Run the Standalone Demo

```bash
poetry run python examples/line_explorer_demo.py
```

This will open a browser window with the interactive app.

## Working with Real JWST Data

If you have JWST spectra, you can load them and create the spectrum dictionary:

```python
import numpy as np
from astropy.io import fits
from astropy import units as u

# Example: Load a JWST spectrum
with fits.open('your_spectrum.fits') as hdul:
    wave = hdul[1].data['WAVELENGTH']  # Observed wavelength
    flux = hdul[1].data['FLUX']

# Convert observed to rest-frame
redshift = 4.465
wave_rest = wave / (1 + redshift)

# Create spectrum dict
spectrum_dict = {
    'JWST': {'wave': wave_rest, 'flux': flux}
}

# Launch explorer
from dotfit import EmissionLines, LineExplorer

el = EmissionLines()
app = LineExplorer(
    spectrum_dict=spectrum_dict,
    emission_lines=el.table,
    redshift=redshift,
    object_name="Your Object",
)

app.panel()  # Display in notebook
```

## Features

### Interactive Controls

- **Pan**: Click and drag on the spectrum
- **Zoom**: Use mouse wheel or box zoom tool
- **Reset**: Click the reset button to restore original view

### Ion Filters

- Check/uncheck ions to show/hide their lines
- Lines are color-coded by element:
  - H (blue), He (pink), O (green), N (purple), S (amber)
  - Fe (red), Ne (cyan), C (violet), and more

### Spectrum Selector

- Switch between different gratings/spectra
- Each spectrum can have different wavelength coverage

### Line Selection

```python
# Add lines programmatically
app.add_line('[OIII]-5008')
app.add_line('Ha')

# Remove lines
app.remove_line('Ha')

# Access selected lines
for line in app.selected_lines:
    print(line['key'], line['ion'], line['wave_vac'])

# Export to CSV
# Use the "Save to CSV" button in the app
```

## Customization

### Filter the Line Catalog

```python
# Show only forbidden lines
forbidden = el.table[np.char.startswith(el.table['ion'].astype(str), '[')]

# Show only specific wavelength range
uv_lines = el.table[(el.table['wave_vac'] > 1000) &
                     (el.table['wave_vac'] < 2000)]

# Show only specific ions
oiii_only = el.table[el.table['ion'] == '[O III]']

# Use filtered catalog
app = LineExplorer(spectrum_dict, forbidden, redshift=4.465)
```

### Normalize Your Spectrum

The app will normalize flux for display, but you can pre-normalize:

```python
# Normalize by median
flux_norm = flux / np.nanmedian(flux)

# Or normalize to a specific range
flux_norm = (flux - np.min(flux)) / (np.max(flux) - np.min(flux))

spectrum_dict = {'G235M': {'wave': wave, 'flux': flux_norm}}
```

## Troubleshooting

### "Line not found in catalog"

This happens when you try to add a line that's not in the filtered catalog.

**Solution**: Check which lines are available:
```python
print(app.emission_lines['key'][:20])  # Show first 20 line keys
```

### App doesn't display in notebook

**Solution**: Make sure Panel extension is loaded:
```python
import panel as pn
pn.extension('tabulator')
```

### Labels are too crowded

**Solution**:
1. Zoom in on specific regions
2. Filter to fewer ions using checkboxes
3. Use a more restrictive Ei filter: `tab = el.table[el.table['Ei'] < 3]`

## Next Steps

- See `examples/README.md` for detailed documentation
- Check `LINE_EXPLORER_SUMMARY.md` for implementation details
- Run tests: `poetry run pytest tests/test_line_explorer.py -v`

## Need Help?

If you encounter issues:
1. Check the examples in `examples/`
2. Run the tests to verify installation
3. Review the docstrings: `help(LineExplorer)`
