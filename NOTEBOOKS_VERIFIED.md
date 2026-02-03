# LineExplorer Notebooks - Verified Implementation

## Summary

Both LineExplorer notebooks have been updated to use **astropy** for data loading instead of the heavy **msaexp** dependency. The implementation has been tested and verified working.

## Updated Files

### 1. `examples/line_explorer_real_data.py` ✅

**Purpose**: Command-line ready demo with real JWST data

**Key Features**:
- Uses `astropy.utils.data.download_file` for FITS files
- Reads SPEC1D extension with `Table.read()`
- Proper unit handling with astropy.units
- Handles masked arrays correctly
- Falls back to demo data if download fails
- Launches server automatically

**Usage**:
```bash
poetry run python examples/line_explorer_real_data.py
```

**Data Loading Implementation**:
```python
def load_monster_spectra(filenames, redshift, base_url=DJA_URL, cache=True):
    """Load NIRSpec spectra from DJA using astropy."""
    λ_unit = u.Angstrom
    fλ_unit = u.erg / u.s / u.cm**2 / u.Angstrom

    for fname in filenames:
        # Download file using astropy
        spec_file = download_file(url, cache=cache, show_progress=False)

        # Load the spectrum from SPEC1D extension
        spec = Table.read(spec_file, 'SPEC1D')

        # Unpack relevant columns and convert units
        wave = spec['wave'].to(λ_unit)
        flux = spec['flux'].to(fλ_unit, equivalencies=u.spectral_density(wave)).value
        err = spec['err'].to(fλ_unit, equivalencies=u.spectral_density(wave)).value
        wave = wave.value

        # Handle masked arrays
        if hasattr(spec['err'], 'mask'):
            valid = ~spec['err'].mask
        else:
            valid = np.ones(len(spec['err']), dtype=bool)

        # Convert to rest-frame wavelength
        wave_rest = wave[valid] / (1 + redshift)
        flux_rest = flux[valid]
        err_rest = err[valid]
```

### 2. `examples/line_explorer_jwst_demo.py` ✅

**Purpose**: Full notebook-style demo with comprehensive documentation

**Key Features**:
- Same astropy-based data loading
- Notebook-style with `# %%` cell markers
- Detects notebook vs. script execution
- Comprehensive documentation in cells
- Shows data access and export patterns
- Works in both Jupyter and command line

**Usage in Jupyter**:
```python
%run examples/line_explorer_jwst_demo.py
```

**Usage from command line**:
```bash
poetry run python examples/line_explorer_jwst_demo.py
```

## Test Results

### Data Loading Test ✅

```
Testing data loading...
  Downloading g235m...
    ✓ g235m: 2138 valid pixels
    ✓ λ range: 3120.2-7067.8 Å

✓ Successfully loaded data
✓ Monster keys: ['z', 'g235m']
✓ Created spectrum_dict: ['g235m']
✓ LineExplorer created successfully
✓ 52 unique ions

✅ All tests passed!
```

### Real Data Specifications

**Object**: abell2744-8204-45924
**Redshift**: 4.465
**Gratings**: G235M, G395M, PRISM

**G235M Spectrum**:
- 2138 valid pixels
- Rest-frame λ range: 3120.2 - 7067.8 Å
- Data format: FITS SPEC1D extension
- Units: erg/s/cm²/Å

## Dependencies

### Required (already in pyproject.toml)
- astropy
- numpy
- panel >= 1.3
- holoviews >= 1.18
- bokeh >= 3.3

### Removed
- ~~msaexp~~ (no longer needed!)

## Implementation Details

### Unit Handling

Proper astropy units conversion:
```python
λ_unit = u.Angstrom
fλ_unit = u.erg / u.s / u.cm**2 / u.Angstrom

wave = spec['wave'].to(λ_unit)
flux = spec['flux'].to(fλ_unit, equivalencies=u.spectral_density(wave))
```

### Masked Array Handling

```python
# Valid rows are where 'err' is not masked
if hasattr(spec['err'], 'mask'):
    valid = ~spec['err'].mask
else:
    valid = np.ones(len(spec['err']), dtype=bool)
```

### Error Handling

Both notebooks include:
- Try/except blocks for data loading
- Fallback to demo data if download fails
- Clear error messages
- Graceful degradation

## Files Overview

| File | Size | Lines | Purpose | Tested |
|------|------|-------|---------|---------|
| `line_explorer_real_data.py` | 4.4 KB | 225 | Command-line demo | ✅ |
| `line_explorer_jwst_demo.py` | 11.1 KB | 355 | Full notebook | ✅ |
| `line_explorer_demo.py` | 2.9 KB | 107 | Synthetic demo | ✅ |
| `line_explorer_notebook_example.py` | 1.3 KB | 43 | Minimal example | ✅ |
| `line_explorer_interactive.py` | 8.6 KB | 302 | Synthetic notebook | ✅ |

## Usage Examples

### Quick Start (Command Line)

```bash
cd /Users/ivo/Astro/PROJECTS/UNCOVER/sci/deep/dotfit
poetry run python examples/line_explorer_real_data.py
```

Opens browser at http://localhost:5006

### Jupyter Notebook

```python
# In a Jupyter cell
%run examples/line_explorer_jwst_demo.py
```

Or copy cells from the notebook file into Jupyter.

### Programmatic Usage

```python
from astropy.utils.data import download_file
from astropy.table import Table
from astropy import units as u
from dotfit import EmissionLines, LineExplorer

# Download and load JWST data
url = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/...'
spec_file = download_file(url, cache=True)
spec = Table.read(spec_file, 'SPEC1D')

# Convert units
wave = spec['wave'].to(u.Angstrom).value
flux = spec['flux'].to(u.erg/u.s/u.cm**2/u.Angstrom,
                       equivalencies=u.spectral_density(spec['wave'])).value

# Convert to rest-frame
redshift = 4.465
wave_rest = wave / (1 + redshift)

# Create LineExplorer
el = EmissionLines()
app = LineExplorer(
    spectrum_dict={'g235m': {'wave': wave_rest, 'flux': flux}},
    emission_lines=el.table,
    redshift=redshift,
)

# Launch
app.serve(port=5006)
```

## Advantages of Astropy Implementation

1. **No heavy dependencies**: Removed msaexp requirement
2. **Standard tools**: Uses well-documented astropy
3. **Better unit handling**: Explicit unit conversions
4. **Cached downloads**: Astropy cache system
5. **Robust error handling**: Better error messages
6. **Masked array support**: Handles invalid pixels correctly

## Next Steps

Both notebooks are ready for use:

1. ✅ Data loading with astropy
2. ✅ Unit conversion
3. ✅ Masked array handling
4. ✅ Rest-frame conversion
5. ✅ LineExplorer integration
6. ✅ Error handling and fallbacks
7. ✅ Tested with real data

Users can now run:
```bash
poetry run python examples/line_explorer_real_data.py
```

And get a fully functional LineExplorer with real JWST data!
