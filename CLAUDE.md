# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dotfit** is a scientific Python package for emission-line and continuum fitting of "little red dots" (compact, high-redshift galaxies). It provides tools for spectroscopic analysis using Bayesian inference with NumPyro/JAX backends.

## Environment and Dependencies

- **Python**: >=3.12
- **Package manager**: Poetry (use `poetry install` to set up the environment)
- **Key dependencies**: astropy, numpy, pandas, matplotlib, astroquery, pyneb, ipython
- **Testing**: pytest
- **JAX/NumPyro**: Used for Bayesian inference (commented out in pyproject.toml but used in code)

### Common Commands

```bash
# Install dependencies
poetry install

# Run all tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_emission_lines_legacy_helpers.py

# Run with verbose output
poetry run pytest -v

# Activate shell with environment
poetry shell
```

## Code Architecture

### Module Structure

```
dotfit/
├── __init__.py              # Package exports (EmissionLines)
├── emission_lines.py        # Core emission line analysis and multiplet handling
├── unite_simple.py          # UNITE-style spectral fitting engine (JAX/NumPyro)
├── broken_modified_bb.py    # Modified blackbody continuum models
├── models.py                # Placeholder for legacy line wavelength models (not yet ported)
└── data/
    └── emission_lines/      # CSV tables for emission line definitions
        ├── emission_lines.csv
        ├── feii_sigut03.csv
        └── feti_kurucz.csv
```

### Key Components

**emission_lines.py**:
- Main class: `EmissionLines` - loads and manages emission line catalogs
- Handles multiplet relationships and line ratio calculations
- Uses PyNEB for forbidden/semi-forbidden line physics
- Supports absorption line classification using Boltzmann statistics
- Line grouping system via `LINE_GROUPS` dict (e.g., '[OII]', '[OIII]', 'HI')

**unite_simple.py**:
- `LineDefinition` dataclass: defines individual spectral lines with wavelength, ion, multiplet info
- `LineGroupSpec` dataclass: groups lines sharing kinematic parameters
- `FitConfig` dataclass: full fitting configuration (lines + continuum model)
- Uses JAX for fast evaluation and NumPyro for MCMC sampling
- Supports single or broken modified blackbody continuum models

**broken_modified_bb.py**:
- Implements modified blackbody continuum models
- `single_modified_blackbody`: single-component continuum
- `broken_modified_blackbody`: two-component broken power law

### Data Flow

1. Emission line definitions loaded from CSV files in `data/emission_lines/`
2. `EmissionLines` class processes lines, assigns multiplets, calculates ratios
3. Line groups defined for fitting using `LineDefinition` and `LineGroupSpec`
4. Full model assembled in `FitConfig` combining emission lines + continuum
5. NumPyro MCMC samples posterior distributions over model parameters

## Development Workflow

### Before Writing Code

1. Read `GUIDE.md` for implementation guidelines
2. Check `CHECKLISTS.md` for current project status
3. Understand the modular structure - add code only inside `dotfit/` package

### Code Standards

- **Type hints required**: Use modern Python typing (e.g., `list[float]` not `List[float]`)
- **Docstrings**: Google-style format
- **Logging**: Use `logging` module, never `print`
- **Units**: Use `astropy.units` for physical quantities
- **Data structures**: Prefer `@dataclass` for structured configs
- **Imports**: `from __future__ import annotations` for forward references

### Testing Requirements

- All new functionality requires pytest unit tests in `tests/`
- Tests should produce diagnostic images/figures
- Use temporary file mocks for FITS I/O
- Avoid network access or external resources in tests
- Run full test suite before committing: `poetry run pytest`

### Project Constraints

- **Avoid over-engineering**: Keep solutions simple and focused
- **No project-wide refactoring** unless explicitly requested
- **Use existing patterns**: Follow the dataclass-based configuration approach
- **Prefer standard libraries**: Use numpy, scipy, astropy, photutils over reimplementation

### When Adding Features

1. Implement inside `dotfit/` package
2. Add corresponding tests in `tests/`
3. Update `CHECKLISTS.md` if behavior changes
4. Keep `pyproject.toml` current (never edit poetry.lock directly)
5. Consider adding example notebooks in `examples/` (currently no examples directory)

## Physics Context

**Multiplets**: Groups of spectral lines from the same ion/transition with fixed intensity ratios determined by atomic physics (e.g., [O III] 4959/5007 doublet).

**Line Classifications**:
- **Forbidden lines**: Low-density emission (e.g., [O II], [O III])
- **Semi-forbidden lines**: Intermediate case
- **Recombination lines**: Hydrogen Balmer/Paschen series
- **Absorption lines**: Stellar/ISM features (use Boltzmann statistics for multiplets)

**PyNEB Integration**: Used for temperature/density-dependent line ratio calculations for forbidden lines.

## Important Notes

- `models.py` contains placeholder implementations - line wavelength models not yet ported
- JAX/NumPyro dependencies are commented out in pyproject.toml but required by `unite_simple.py`
- Modified blackbody fitting may be under development (see `broken_modified_bb.py`)
- Check git status before major changes - emission_lines.py has uncommitted modifications
