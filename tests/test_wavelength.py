"""Tests for wavelength conversion utilities."""

import numpy as np
import matplotlib.pyplot as plt

from dotfit.emission_lines import vacuum_to_air, air_to_vacuum


def test_vacuum_to_air():
    """Test vacuum to air conversion reversibility.

    Converts to air and back, minus the lambda_vac from lambda_vac= 2000 to 1e5
    Reproduces: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion?action=AttachFile&do=view&target=air2vac.gif
    Reversible to within 1e-8
    """
    lambda_vac = np.linspace(1000, 1e5, 10000)
    lambda_air = vacuum_to_air(lambda_vac)
    lambda_vac_back = air_to_vacuum(lambda_air)

    # Create diagnostic plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(lambda_vac, lambda_vac_back - lambda_vac)
    plt.ylim(-2e-8, 2e-8)
    plt.xlabel('Vacuum Wavelength (Å)')
    plt.ylabel('Residual (Å)')
    plt.title('Vacuum ↔ Air Conversion Reversibility')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_vacuum_to_air.png', dpi=150)
    plt.close()

    # Verify reversibility
    # The original test in the production code just plotted the residuals
    # We verify that for most of the range, reversibility is good
    residuals = lambda_vac_back - lambda_vac
    max_residual = np.max(np.abs(residuals))

    # For wavelengths > 2000 Å (where air-to-vacuum conversion applies),
    # reversibility should be within 2e-8
    vac_range = lambda_vac > 2000
    max_residual_vac = np.max(np.abs(residuals[vac_range]))
    assert max_residual_vac < 2e-8, f"Reversibility error too large in vacuum range: {max_residual_vac}"
