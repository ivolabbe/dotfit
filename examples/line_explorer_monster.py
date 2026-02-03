"""LineExplorer Monster Example

This example demonstrates the LineExplorer with real JWST data.
It can optionally load previously saved lines from CSV.

Usage from examples directory:
    cd examples
    poetry run python line_explorer_monster.py [options]

Or from root directory:
    poetry run python examples/line_explorer_monster.py [options]

Options:
    --emission-lines-csv PATH    Custom emission line catalog CSV file(s)
                                 Supports glob patterns like '/path/to/*.csv' to load multiple catalogs
                                 When multiple catalogs are loaded, you can switch between them in the UI
    --selected-lines-csv PATH    Previously selected lines to load
    --redshift FLOAT             Redshift of the object (default: 4.464)
    --object-name NAME           Name of the object (default: abell2744-8204-45924)
    --port INT                   Port for web server (default: 5007)

Examples:
    # Load single custom catalog
    poetry run python examples/line_explorer_monster.py --emission-lines-csv my_lines.csv

    # Load multiple catalogs using glob pattern
    poetry run python examples/line_explorer_monster.py --emission-lines-csv '/path/to/*_lines.csv'
"""

# %%
import argparse
import glob
import os
import panel as pn
from dotfit import EmissionLines, LineExplorer

# Parse command line arguments
parser = argparse.ArgumentParser(description='Launch LineExplorer with JWST spectra')
parser.add_argument('--emission-lines-csv', type=str, default=None,
                    help='Path to custom emission line catalog CSV file (supports glob patterns like /path/to/*.csv)')
parser.add_argument('--selected-lines-csv', type=str, default='abell2744-8204-45924-20260203.csv',
                    help='Path to previously selected lines CSV file')
parser.add_argument('--redshift', type=float, default=4.464,
                    help='Redshift of the object')
parser.add_argument('--object-name', type=str, default='abell2744-8204-45924',
                    help='Name of the object')
parser.add_argument('--port', type=int, default=5007,
                    help='Port for web server')

args = parser.parse_args()

# Define the FITS filenames to load from DJA
filenames = [
    'abell2744-greene-v4_g235m-f170lp_8204_45924.spec.fits',
    'abell2744-greene-v4_g395m-f290lp_8204_45924.spec.fits',
    'abell2744-greene-v4_prism-clear_8204_45924.spec.fits',
]

# Load emission lines (support glob patterns for multiple catalogs)
print("\nLoading emission line catalog(s)...")
if args.emission_lines_csv:
    # Expand glob pattern (e.g., /path/to/*.csv)
    catalog_files = sorted(glob.glob(args.emission_lines_csv))

    if len(catalog_files) == 0:
        print(f"  Warning: No files found matching pattern: {args.emission_lines_csv}")
        print(f"  Using default catalog")
        el = EmissionLines()
        emission_lines_input = el.table
    elif len(catalog_files) == 1:
        print(f"  Using custom catalog: {catalog_files[0]}")
        el = EmissionLines(filename=catalog_files[0])
        emission_lines_input = el.table
    else:
        print(f"  Found {len(catalog_files)} catalogs:")
        for f in catalog_files:
            print(f"    - {f}")
        # Pass list of CSV paths to LineExplorer
        emission_lines_input = catalog_files
else:
    print(f"  Using default catalog")
    el = EmissionLines()
    emission_lines_input = el.table

# Create LineExplorer app
print("\nCreating LineExplorer...")

# Check if selected lines CSV exists
use_csv = os.path.exists(args.selected_lines_csv)

if use_csv:
    print(f"Loading previously selected lines from {args.selected_lines_csv}...")
    app = LineExplorer(
        spectrum=filenames,
        emission_lines=emission_lines_input,
        redshift=args.redshift,
        object_name=args.object_name,
        selected_lines_csv=args.selected_lines_csv,
    )
else:
    print("Starting fresh (no saved CSV found)...")
    app = LineExplorer(
        spectrum=filenames,
        emission_lines=emission_lines_input,
        redshift=args.redshift,
        object_name=args.object_name
    )

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
print(f"\nLaunching app on port {args.port}...")
app.serve(port=args.port, show=True)


# %%
