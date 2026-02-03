"""LineExplorer Monster Example

This example demonstrates the LineExplorer with real JWST data.
It can optionally load previously saved lines from CSV.

Usage from examples directory:
    cd examples
    poetry run python line_explorer_monster.py

Or from root directory:
    poetry run python examples/line_explorer_monster.py
"""

# %%
import panel as pn
from dotfit import EmissionLines, LineExplorer

# Define the FITS filenames to load from DJA
filenames = [
    'abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits',
    'abell2744-greene-v4_g395m-f290lp_8204_45924.spec.fits',
    'abell2744-greene-v4_prism-clear_8204_45924.spec.fits',
]

redshift = 4.464
object_name = "abell2744-8204-45924"

# Load emission lines
print("\nLoading emission line catalog...")
el = EmissionLines()
tab = el.table

# Create LineExplorer app
print("\nCreating LineExplorer...")

# Check if CSV exists
import os

csv_filename = 'abell2744-8204-45924-20260202.csv'
use_csv = os.path.exists(csv_filename)

if use_csv:
    print(f"Loading previously selected lines from {csv_filename}...")
    app = LineExplorer(
        spectrum=filenames,
        emission_lines=tab,
        redshift=redshift,
        object_name=object_name,
        selected_lines_csv=csv_filename,
    )
else:
    print("Starting fresh (no saved CSV found)...")
    app = LineExplorer(spectrum=filenames, emission_lines=tab, redshift=redshift, object_name=object_name)

print(f"  ✓ App created")
print(f"  ✓ {len(app.unique_ions)} unique ions available")
print(f"  ✓ Loaded spectra: {list(app.spectrum_dict.keys())}")
print(f"  ✓ {len(app.selected_lines)} lines selected")

if app.selected_lines:
    print("\nCurrently selected lines:")
    for line in app.selected_lines:
        print(f"  {line['key']:15s} λ = {line['wave_vac']:.2f} Å")
else:
    print("\nNo lines selected yet. Start identifying!")

# %%
# Launch app
# app.panel()
app.serve(port=5007, show=True)


# %%
