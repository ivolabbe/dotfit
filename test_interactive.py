"""Test LineExplorer interactivity in a served Panel app."""

import numpy as np
import panel as pn
from dotfit import EmissionLines, LineExplorer

# Create simple test spectrum
spectrum = {
    'test': {
        'wave': np.linspace(3000, 7000, 1000),
        'flux': np.ones(1000) + 0.2 * np.sin(np.linspace(0, 10, 1000)),
    }
}

# Load emission lines
el = EmissionLines()
tab = el.table

# Create app
print('Creating LineExplorer...')
app = LineExplorer(spectrum, tab, redshift=0, object_name='Interactive Test')
print('✓ App created')

# Get panel layout
layout = app.panel()

print('\n' + '=' * 60)
print('Starting Panel server...')
print('Open http://localhost:5007 in your browser')
print('Then test:')
print('  1. Zoom in/out - do labels change?')
print('  2. Click on a line - does it get thicker?')
print('  3. Toggle ion checkboxes - do lines appear/disappear?')
print('  4. Watch console for debug output')
print('=' * 60)
print()

# Serve the app
layout.servable()
pn.serve(layout, port=5007, show=True)
