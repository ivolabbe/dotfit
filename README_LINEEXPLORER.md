# LineExplorer - Interactive Emission Line Analysis

## Quick Start

```python
from dotfit import EmissionLines, LineExplorer

# Load emission lines
el = EmissionLines()
tab = el.table

# Load JWST spectra
filenames = [
    'abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits',
    'abell2744-greene-v4_g395m-f290lp_8204_45924.spec.fits',
]

# Create app
app = LineExplorer(
    spectrum=filenames,
    emission_lines=tab,
    redshift=4.465,
    object_name="abell2744-8204-45924",
    debug=False,  # Set to True to see callback events in console
)

# Display in Jupyter notebook
app.panel()
```

## Features

### Interactive Plot
- **Zoom/Pan**: Use mouse wheel or box zoom - labels automatically update
- **Click to select**: Click on or near any line to select/deselect it
- **Selected lines**: Appear 3x thicker with higher opacity
- **Spectrum display**: Steps with gray error bands (±1σ)
- **Label positioning**: Fixed offset above local spectrum value

### Controls (Right Panel)
- **Ion checkboxes**: Toggle visibility, displayed in 2 columns, filtered to current view
- **Spectrum selector**: Switch between gratings (G235M, G395M, PRISM)
- **Selected lines table**: Shows all selected lines
- **Save to CSV**: Exports as `<object_name>-<date>.csv`

### Dynamic Behavior
- **Labels**: More labels appear when zoomed in, fewer when zoomed out
- **Ion checkboxes**: Only show ions present in current wavelength range
- **Observed wavelength**: Display updates to show current view range

## Usage

### Basic Example

```python
# Simple spectrum dict
spectrum = {
    'test': {
        'wave': np.linspace(3000, 7000, 1000),
        'flux': np.ones(1000),
    }
}

app = LineExplorer(spectrum, el.table, redshift=0, object_name="Test")
app.panel()
```

### With Error Bars

```python
# Spectrum with uncertainties
spectrum = {
    'g235m': {
        'wave': wave_array,
        'flux': flux_array,
        'err': error_array,  # Will be shown as gray bands
    }
}

app = LineExplorer(spectrum, el.table, redshift=4.5, object_name="Galaxy")
app.panel()
```

### Programmatic Line Selection

```python
# Add lines programmatically
app.add_line('[OIII]-5008')
app.add_line('[OIII]-4960')
app.add_line('[OII]-3727')

# Remove lines
app.remove_line('[OII]-3727')

# View selected
print(app.selected_lines)
```

### Resume from Saved CSV

```python
# Save your selected lines to CSV
# (Uses Save to CSV button or programmatically saves to <object_name>-<date>.csv)

# Later, resume your work by loading the CSV
app = LineExplorer(
    spectrum=filenames,
    emission_lines=tab,
    redshift=4.465,
    object_name="Galaxy",
    selected_lines_csv='Galaxy-20260202.csv',  # Load previous selections
)
# Your previously selected lines are automatically loaded!
```

### Debug Mode

```python
# Enable debug output to see callbacks fire
app = LineExplorer(
    spectrum=filenames,
    emission_lines=tab,
    redshift=4.465,
    debug=True,  # Shows event messages in console
)
app.panel()
```

Debug output example:
```
🔍 Range update: x=(4000, 5000)
🎨 DynamicMap callback: x_range=(4000, 5000), x=None, y=None
👆 Tap event: x=5008.24, y=0.75
  ✓ Clicked on line: [OIII]-5008
☑️  Ion toggle: [OIII] = False
📡 Spectrum change: g235m -> g395m
```

## API Reference

### LineExplorer

```python
LineExplorer(
    spectrum,                 # dict, str, or list[str]
    emission_lines,           # Table or CSV path
    redshift=0,               # float
    object_name="",           # str
    debug=False,              # bool
    max_H_series=15,          # int
    selected_lines_csv=None,  # str | None
)
```

**Parameters**:
- `spectrum`: Dictionary of spectra, single filename, or list of filenames
- `emission_lines`: Astropy Table or path to CSV file
- `redshift`: Source redshift for rest-frame conversion
- `object_name`: Object name (used in CSV filename)
- `debug`: Enable debug console output
- `max_H_series`: Maximum Balmer series line to show (default 15, filters out H16+)
- `selected_lines_csv`: Path to previously saved CSV to resume line identification

**Methods**:
- `panel()`: Return Panel layout for notebook display
- `serve(port=5006)`: Launch standalone Panel server
- `add_line(line_key)`: Programmatically add line to selection
- `remove_line(line_key)`: Programmatically remove line from selection

**Attributes**:
- `selected_lines`: List of dictionaries with selected line info
- `visible_ions`: Set of currently visible ion names
- `current_spectrum`: Name of currently displayed spectrum
- `x_range`: Current wavelength range `(min, max)`

## Implementation Details

### Stream-Based Reactivity

Uses HoloViews DynamicMap with streams for reactive updates:
- **RangeXY stream**: Detects zoom/pan, updates labels dynamically
- **Tap stream**: Detects clicks, toggles line selection
- **Widget callbacks**: Trigger stream events to update plot

### Plot Components

- **Spectrum**: `hv.Curve` with `interpolation='steps-mid'`
- **Error bands**: `hv.Area` showing ±1σ in gray
- **Line markers**: `hv.VLine` colored by element
- **Labels**: `hv.Labels` positioned above local spectrum value

### Performance

- Label overlap algorithm: O(n²) but fast for typical catalogs (<1000 lines in view)
- Plot updates: Only recomputes when state changes
- Smooth 60 FPS interaction

## Files

- `dotfit/line_explorer.py`: Main implementation (~900 lines)
- `examples/line_explorer_real_data.py`: Example with JWST data
- `tests/test_line_explorer.py`: Test suite (9 tests, all passing)

## Requirements

- Python >=3.12
- Panel >=1.3
- HoloViews >=1.18
- Bokeh >=3.3
- Astropy
- NumPy

Install with: `poetry install`

## Testing

```bash
# Run tests
poetry run pytest tests/test_line_explorer.py -v

# Run example
poetry run python examples/line_explorer_real_data.py
```

## Troubleshooting

**Labels not updating on zoom**:
- Make sure you're running in Jupyter notebook, not standalone script
- DynamicMap callbacks only fire when plot is actively rendered

**Clicks not selecting lines**:
- Click tolerance is 1% of visible range
- Try clicking directly on the vertical line marker
- Enable `debug=True` to see tap events

**Widgets not updating plot**:
- Check that you're calling `app.panel()` to display
- Enable `debug=True` to verify callbacks fire

## Changelog

- **v1.0**: Initial release with all interactive features
  - Dynamic label updates on zoom
  - Click-to-select functionality
  - Spectrum switching
  - Ion filtering with 2-column layout
  - Step plots with error bands
  - Smart label positioning
  - CSV export with date
  - Debug mode
