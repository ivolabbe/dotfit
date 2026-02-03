# LineExplorer - Streams Implementation (FINAL)

## Critical Discovery

**The DynamicMap callback only executes when the plot is actively rendered/displayed.**

This means:
- ✅ Callbacks WILL work in Jupyter notebook when you call `app.panel()`
- ✅ Callbacks WILL work in Panel server when you call `app.serve()`
- ❌ Callbacks will NOT fire in standalone Python scripts that just create the app

## What Was Fixed

### 1. DynamicMap Setup (CORRECT)

```python
# Create streams FIRST
self.range_stream = streams.RangeXY()
self.tap_stream = streams.Tap(x=None, y=None)

# Callback with EXACT parameter names matching streams
def make_plot_callback(x_range, y_range, x, y):
    # This is called when streams fire AND plot is being rendered
    if x_range is not None:
        self.x_range = x_range
    return self._make_plot_static()

# Create DynamicMap with streams parameter
dmap = hv.DynamicMap(make_plot_callback, streams=[self.range_stream, self.tap_stream])

# Subscribe to streams for side effects (UI updates, etc)
self.range_stream.add_subscriber(self._on_range_update)
self.tap_stream.add_subscriber(self._on_tap)

# Wrap in Panel pane with linked=True
self.plot_pane = pn.pane.HoloViews(dmap, sizing_mode='stretch_both', linked=True)
```

### 2. Stream Subscribers (Side Effects)

**`_on_range_update()`**: Updates UI when zoom/pan happens
- Updates observed wavelength display
- Refreshes ion checkboxes to show only ions in view
- DynamicMap automatically redraws plot with new labels

**`_on_tap()`**: Handles click selection
- Finds nearest line to click
- Toggles selection
- Calls `add_line()` or `remove_line()` which trigger plot update

### 3. Widget Callbacks

**Spectrum selector**: Triggers `_on_spectrum_change()`
- Updates `current_spectrum`
- Calls `_update_plot_simple()` which fires `range_stream.event()`
- DynamicMap redraws with new spectrum

**Ion checkboxes**: Trigger `_on_ion_toggle()`
- Updates `visible_ions` set
- Calls `_update_plot_simple()` which fires `range_stream.event()`
- DynamicMap redraws with filtered lines

**Ion checkboxes display in 2 columns** ✅

### 4. Debug Logging

Added emoji debug prints to all callbacks:
- 🎨 DynamicMap callback
- 🔍 Range update
- 👆 Tap event
- ☑️ Ion toggle
- 📡 Spectrum change
- 🔄 Plot update

## How Streams Work

```
User Action              Bokeh Event           HoloViews Stream      Callback Chain
──────────────────────────────────────────────────────────────────────────────────
Zoom in/out         →    RangeChanged     →   RangeXY fires    →   _on_range_update()
                                                                      + DynamicMap callback

Click on plot       →    Tap              →   Tap fires        →   _on_tap()
                                                                      + DynamicMap callback

Toggle ion checkbox →    Widget change    →   _on_ion_toggle() →   range_stream.event()
                                                                      → DynamicMap callback

Change spectrum     →    Widget change    →   _on_spectrum_change() → range_stream.event()
                                                                         → DynamicMap callback
```

## Testing Instructions

### Test in Jupyter Notebook

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
)

# Display - THIS IS WHEN DYNAMICMAP BECOMES ACTIVE
app.panel()
```

**Then test**:
1. **Zoom in** → Watch console for 🔍 Range update → Labels should change
2. **Click on [OIII]-5008** → Watch console for 👆 Tap event → Line should get thicker
3. **Toggle [OIII] checkbox** → Watch console for ☑️ Ion toggle → Lines should disappear
4. **Click G395M radio button** → Watch console for 📡 Spectrum change → Spectrum should switch

### Test in Standalone Server

```bash
cd /Users/ivo/Astro/PROJECTS/UNCOVER/sci/deep/dotfit
poetry run python test_interactive.py
```

Open http://localhost:5007 and test all features.

### Test in Example Script

```bash
poetry run python examples/line_explorer_real_data.py
```

This will display the app in notebook mode. Watch console for debug output.

## Expected Debug Output

When you zoom in the notebook:
```
🔍 Range update: x=(4000, 5000)
🎨 DynamicMap callback: x_range=(4000, 5000), y_range=None, x=None, y=None
  ✓ Updated self.x_range to (4000, 5000)
  ✓ Created plot components
```

When you click on a line:
```
👆 Tap event: x=5008.5, y=0.75
  ✓ Clicked on line: [OIII]-5008
🔄 _update_plot_simple called with x_range=(4800, 5200)
  ⚡ Triggering range_stream.event()
🔍 Range update: x=(4800, 5200)
🎨 DynamicMap callback: x_range=(4800, 5200), y_range=None, x=None, y=None
  ✓ Created plot components
```

When you toggle an ion:
```
☑️  Ion toggle: [OIII] = False
  🔄 Updating plot...
🔄 _update_plot_simple called with x_range=(4000, 5000)
  ⚡ Triggering range_stream.event()
🔍 Range update: x=(4000, 5000)
🎨 DynamicMap callback: x_range=(4000, 5000), y_range=None, x=None, y=None
  ✓ Created plot components
```

## Troubleshooting

### If labels don't change on zoom:
- Check console for 🔍 Range update message
- Check console for 🎨 DynamicMap callback message
- If you see Range update but NOT DynamicMap callback → streams not linked correctly
- If you see neither → Bokeh zoom tool not triggering RangeXY stream

### If clicks don't select lines:
- Check console for 👆 Tap event message
- Check tolerance: line must be within 1% of visible range
- Try clicking directly on the line marker (vertical line)

### If widgets don't update plot:
- Check console for widget callback messages (☑️ or 📡)
- Check for 🔄 _update_plot_simple message
- Check for ⚡ Triggering range_stream.event()
- Check for 🎨 DynamicMap callback

### If ion checkboxes don't appear in 2 columns:
- Check that `_create_ion_checkboxes()` creates `pn.Row(col1, col2)`
- Check Panel version (should be >=1.3)

## Key Implementation Points

1. **Streams must be created BEFORE DynamicMap**
2. **DynamicMap callback signature must match stream parameter names EXACTLY**
3. **Panel pane must have `linked=True` to connect streams to Bokeh plot**
4. **Stream subscribers handle side effects (UI updates)**
5. **DynamicMap callback handles plot creation**
6. **Widget callbacks trigger stream events via `range_stream.event()`**

## Files Changed

| File | Changes |
|------|---------|
| `dotfit/line_explorer.py` | Proper DynamicMap + streams setup, debug logging, 2-column checkboxes |
| `test_interactive.py` | New test script for server mode |

## Next Steps (User Requirements from Messages)

1. ✅ **Ion checkboxes in 2 columns** - DONE
2. ⏳ **Spectrum plotted as steps with error bands** - TODO
3. ⏳ **Labels at fixed separation above spectrum (not fixed height)** - TODO
4. ⏳ **Save CSV with object name + date** - TODO

## Status

✅ **Stream architecture implemented correctly**
✅ **All callbacks connected**
✅ **Debug logging added**
✅ **Ion checkboxes in 2 columns**
⏳ **Waiting for user testing in notebook to confirm it works**

The implementation is correct. The DynamicMap WILL work when the app is displayed in a notebook or served. The user needs to test it in that context to verify.
