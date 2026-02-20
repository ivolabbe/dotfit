from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
from astropy.table import MaskedColumn, Table, vstack
import astropy.units as u

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data" / "emission_lines"
DEFAULT_EMISSION_LINES_FILE = 'emission_lines.csv'

LINE_GROUPS = {
    # Common doublets/multiplets
    '[OII]': ['[OII]-3727', '[OII]-3730'],
    '[NII]': ['[NII]-5756', '[NII]-6550', '[NII]-6585'],
    '[SII]': ['[SII]-6718', '[SII]-6733'],
    '[OIII]': ['[OIII]-4364', '[OIII]-4960', '[OIII]-5008'],
    '[NeIII]': ['[NeIII]-3870'],
    '[OI]': ['[OI]-5579', '[OI]-6302', '[OI]-6366'],
    '[SIII]': ['[SIII]-9071', '[SIII]-9533'],
    # Helium Lines
    'HeI': ['HeI-10833', 'HeI-7067', 'HeI-5877'],
    'HeI_T': ['HeI-10833', 'HeI-7067', 'HeI-5877', 'HeI-3890'],
    'HeI_S': ['HeI-6680', 'HeI-5017', 'HeI-4923', 'HeI-4473', 'HeI-4122', 'HeI-3873'],
    # Broad Lines (typically used for BLR)
    'HI': ['Ha', 'Hb', 'Hg', 'Hd'],
    'OI': ['OI-1304', 'OI-8449'],
    'Pa': ['PaA', 'PaB', 'PaG', 'PaE', 'Pa9'],
    # Absorption Lines (typically stellar/ISM)
}

_MULTIPLET_KWARGS = {'Te', 'Ne', 'tolerance', 'verbose'}


def _filter_multiplet_kwargs(kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _MULTIPLET_KWARGS}
    if 'ne' in kwargs and 'Ne' not in filtered:
        filtered['Ne'] = kwargs['ne']
    return filtered


import operator as _op

_OPS = {'>=': _op.ge, '<=': _op.le, '!=': _op.ne, '>': _op.gt, '<': _op.lt, '==': _op.eq}


def filter_table(table, expr):
    """Filter an astropy Table using simple comparison expressions.

    Parses a string of comma-separated conditions and returns rows
    where all conditions are satisfied (AND logic).

    Args:
        table: Astropy Table to filter.
        expr: One or more comma-separated ``'column operator value'``
            expressions. Spaces around operators are optional.

            Supported operators: ``>``, ``<``, ``>=``, ``<=``, ``==``, ``!=``.

            Examples::

                filter_table(tab, 'line_ratio>0.1')
                filter_table(tab, 'Aki<1e-3, Ei<=4.0')
                filter_table(tab, 'gf>=0.01, wave_vac>3000, wave_vac<7000')

    Returns:
        Filtered copy of the input Table.

    Raises:
        ValueError: If an expression cannot be parsed.
        KeyError: If a column name does not exist in the table.
    """
    mask = np.ones(len(table), dtype=bool)
    for clause in expr.split(','):
        clause = clause.strip()
        if not clause:
            continue
        # Match longest operator first (>=, <= before >, <)
        matched = False
        for op_str, op_func in _OPS.items():
            if op_str in clause:
                col, val = clause.split(op_str, 1)
                col, val = col.strip(), val.strip()
                mask &= op_func(np.asarray(table[col], dtype=float), float(val))
                matched = True
                break
        if not matched:
            raise ValueError(f"Cannot parse filter expression: '{clause}'")
    return table[mask]


def _ensure_multiplet_ratios(tab, kwargs):
    has_Te = 'Te' in kwargs
    has_ne = 'ne' in kwargs or 'Ne' in kwargs
    if not (has_Te and has_ne):
        return tab

    has_multiplet = 'multiplet' in tab.colnames and np.any(tab['multiplet'] != 0)
    has_ratios = 'line_ratio' in tab.colnames and np.any(tab['line_ratio'] != 1.0)
    if not (has_multiplet and has_ratios):
        # Use new API
        filtered = _filter_multiplet_kwargs(kwargs)
        Te = filtered.get('Te', 1e4)
        Ne = filtered.get('Ne', 1e2)
        return compute_line_ratios(assign_multiplets(tab), Te=Te, Ne=Ne)
    return tab


def _build_multiplet_term(configuration, terms, lower_only):
    if isinstance(terms, np.ma.core.MaskedConstant):
        return configuration
    if lower_only:
        # Group by lower term only (extract before the '-')
        return (
            configuration.split('-')[0].strip() + '_' + terms.split('-')[0].strip()
            if '-' in terms
            else terms
        )
    # Group by full configuration and term (both lower and upper)
    return configuration + '_' + terms.strip()


def _parse_gigk(gigk_str):
    """Parse gigk string like '10-8' to get (g_i, g_k)."""
    if not gigk_str or gigk_str == '':
        return np.nan, np.nan
    try:
        parts = str(gigk_str).split('-')
        gi = float(parts[0])
        gk = float(parts[1])
        return gi, gk
    except (ValueError, IndexError):
        return np.nan, np.nan


def _should_use_pyneb(ion, group):
    """Check if PyNeb should be used for this multiplet.

    PyNeb is only used when:
    - Ion is H I, He I, or He II (recombination lines)
    - Multiple lines AND all are forbidden/semi-forbidden

    For mixed multiplets (permitted + forbidden), Boltzmann is used
    since PyNeb doesn't have most permitted lines and they share a
    common lower level anyway.
    """
    if ion in {'H I', 'He I', 'He II'}:
        return True
    if 'classification' not in group.colnames:
        # No classification info - try Boltzmann first
        return False
    classes = []
    for c in group['classification']:
        if isinstance(c, np.ma.core.MaskedConstant):
            continue
        classes.append(str(c).strip().lower())
    # Only use PyNeb if ALL lines are forbidden/semi-forbidden
    # (avoid trying PyNeb for permitted lines in mixed multiplets)
    return all(c in {'forbidden', 'semi-forbidden'} for c in classes)


def _compute_boltzmann_ratios(wave_vac, Ek, gu, f, Te):
    """
    Compute Boltzmann intensity ratios for permitted lines.

    For permitted lines sharing the same lower level:
        I ∝ (g_u * f_lu / λ³) * exp(-E_u / kT)

    Line ratios within a multiplet (same lower level):
        I₁/I₂ = (λ₂/λ₁)³ * (f₁/f₂) * (g_u1/g_u2) * exp[-(E_u1 - E_u2) / kT]

    Parameters
    ----------
    wave_vac : array
        Vacuum wavelengths [Å]
    Ek : array
        Upper level energies [eV]
    gu : array
        Upper level statistical weights
    f : array
        Oscillator strengths
    Te : float
        Excitation temperature [K]

    Returns
    -------
    ratios : array
        Normalized intensity ratios (sum=1)
    """
    KB_EV = 8.617333262e-5  # Boltzmann constant in eV/K

    wave_vac = np.asarray(wave_vac, dtype=float)
    Ek = np.asarray(Ek, dtype=float)
    gu = np.asarray(gu, dtype=float)
    f = np.asarray(f, dtype=float)

    # Use first line as reference
    delta_E = Ek - Ek[0]

    weights = (wave_vac[0] / wave_vac) ** 3 * (f / f[0]) * (gu / gu[0]) * np.exp(-delta_E / (KB_EV * Te))

    # Normalize
    if np.sum(weights) > 0:
        return weights / np.sum(weights)
    else:
        # Fallback to uniform if computation fails
        return np.ones_like(weights) / len(weights)


def _compute_pyneb_ratios(ion, wave_vac, Te, Ne, tolerance=0.1, verbose=False):
    """
    Compute emissivity ratios from PyNeb for forbidden/recombination lines.

    Parameters
    ----------
    ion : str
        Ion name (e.g., 'O III', '[O III]', 'H I')
    wave_vac : array
        Vacuum wavelengths [Å]
    Te : float
        Electron temperature [K]
    Ne : float
        Electron density [cm^-3]
    tolerance : float
        Wavelength matching tolerance [Å]
    verbose : bool
        Print diagnostic information

    Returns
    -------
    ratios : array
        Normalized intensity ratios (max=1), or None if PyNeb fails
    """
    try:
        # Extract atom and level
        ion_clean = ion.replace('[', '').replace(']', '')
        atom, level = ion_clean.split(' ')
        level_int = roman_to_int(level)

        # Use existing emissivity_ratios function
        ratios = emissivity_ratios(
            atom,
            level_int,
            np.asarray(wave_vac),
            Te=Te,
            Ne=Ne,
            relative=True,  # Normalize to max=1
            tolerance=tolerance,
            verbose=verbose,
        )

        # Check if valid
        if np.all(np.isnan(ratios)) or np.all(ratios == 0):
            return None

        return ratios

    except Exception as e:
        if verbose:
            print(f"PyNeb failed for {ion}: {e}")
        return None


def _has_atomic_data(group):
    """Check if group has sufficient atomic data for Boltzmann calculation."""
    if 'Ek' not in group.colnames or 'wave_vac' not in group.colnames:
        return False

    # Check for valid Ek values
    Ek = np.array(group['Ek'], dtype=float)
    if not np.any(Ek > 0):
        return False

    # Check for g-values (from gigk)
    if 'gigk' not in group.colnames:
        return False

    has_gigk = False
    for g in group['gigk']:
        if g and str(g).strip() and '-' in str(g):
            has_gigk = True
            break

    if not has_gigk:
        return False

    # Check for oscillator strengths
    has_f = False
    if 'gf' in group.colnames and np.any(np.array(group['gf'], dtype=float) > 0):
        has_f = True
    elif 'fik' in group.colnames:
        has_f = True

    return has_f


def _extract_atomic_params(group):
    """Extract atomic parameters for Boltzmann calculation."""
    wave_vac = np.array(group['wave_vac'], dtype=float)
    Ek = np.array(group['Ek'], dtype=float)

    # Extract g_u from gigk
    gu = []
    for g in group['gigk']:
        gi, gu_val = _parse_gigk(g)
        gu.append(gu_val)
    gu = np.array(gu)

    # Extract oscillator strengths
    if 'gf' in group.colnames and np.any(np.array(group['gf'], dtype=float) > 0):
        gi = []
        for g in group['gigk']:
            gi_val, _ = _parse_gigk(g)
            gi.append(gi_val)
        gi = np.array(gi)
        f = np.array(group['gf'], dtype=float) / gi
    elif 'fik' in group.colnames:
        f = np.array(group['fik'], dtype=float)
    else:
        f = np.ones(len(group))

    return wave_vac, Ek, gu, f


def _compute_group_ratios(group, Te, Ne, verbose=False):
    """
    Compute ratios for one multiplet group using fallback chain.

    Fallback order:
    1. PyNeb (for H I, He I/II, and forbidden/semi-forbidden lines)
    2. Boltzmann (if atomic data available)
    3. Uniform (last resort)

    Parameters
    ----------
    group : Table
        Lines in the multiplet group
    Te : float
        Electron temperature [K]
    Ne : float
        Electron density [cm^-3]
    verbose : bool
        Print diagnostic information

    Returns
    -------
    ratios : array
        Intensity ratios normalized to max=1
    method : str
        Method used: 'single', 'pyneb', 'boltzmann', or 'uniform'
    """
    if len(group) <= 1:
        return np.array([1.0]), 'single'

    ion = group['ion'][0]

    # Try PyNeb for supported ions
    if _should_use_pyneb(ion, group):
        ratios = _compute_pyneb_ratios(ion, group['wave_vac'], Te, Ne, tolerance=0.1, verbose=verbose)
        if ratios is not None and np.any(ratios > 0):
            # Normalize to max=1
            return ratios / ratios.max(), 'pyneb'

    # Try Boltzmann for permitted lines with atomic data
    if _has_atomic_data(group):
        try:
            wave_vac, Ek, gu, f = _extract_atomic_params(group)
            ratios = _compute_boltzmann_ratios(wave_vac, Ek, gu, f, Te)
            if np.any(ratios > 0):
                # Normalize to max=1
                return ratios / ratios.max(), 'boltzmann'
        except Exception as e:
            if verbose:
                print(f"Boltzmann calculation failed for {ion}: {e}")

    # Fallback to uniform
    n = len(group)
    if verbose:
        print(f"Using uniform ratios for {ion} multiplet (n={n})")
    return np.ones(n) / n, 'uniform'


def plot_lines(
    tab: Table,
    wave=None,
    fwhm_kms=300.0,
    per_multiplet_normalize=False,
    area_normalized=False,
    all_normalize=False,
    ngrid=4000,
    Te=5_000,
    offset=0.0,
    lower_only=False,
    inverse=False,
    legend=False,
    linestyle='-',
):
    """
    Plot each multiplet as a sum of Gaussian lines, one color per multiplet.

    Parameters
    ----------
    tab : astropy.table.Table
        Must contain columns: 'wave_vac' [Å], 'multiplet' [int], 'line_ratio' [float].
    fwhm_kms : float
        Gaussian FWHM in km/s (applies per line at its wavelength).
    per_multiplet_normalize : bool
        If True, scale each multiplet profile to unit peak after summation.
    area_normalized : bool
        If True, each Gaussian integrates to line_ratio (area-normalized).
        If False, each Gaussian peaks at line_ratio (peak-normalized).
    all_normalize : float or False
        If set, normalize all line ratios to this peak value.
    ngrid : int
        Number of wavelength grid points.
    offset : float
        Vertical offset for the profiles.
    inverse : bool
        If True, plot profiles downward from offset instead of upward.
    legend : bool
        If True, show legend with term labels.
    Te : float
        Temperature in K. If provided, apply a Boltzmann factor based on the
        average lower-level energy per term (and per ion when multiple ions are
        present). The factor is 1 when $E_i = kT$.
    """
    import matplotlib.pyplot as plt

    linetab = tab.copy()
    if wave is not None:
        iwave = (linetab['wave_vac'] >= wave[0]) & (linetab['wave_vac'] <= wave[1])
        linetab = linetab[iwave]

    required = {"wave_vac"}
    missing = required - set(linetab.colnames)
    if missing:
        raise ValueError(f"Missing columns in table: {missing}")

    # always recalculate multiplets and line ratios if missing
    if "multiplet" not in linetab.colnames:
        linetab = assign_multiplets(linetab, lower_only=lower_only)
        linetab = compute_line_ratios(linetab, Te=Te, Ne=1e2)

    if "line_ratio" not in linetab.colnames:
        linetab = compute_line_ratios(linetab, Te=Te, Ne=1e2)

    # Scalar constants
    c_kms = 299_792.458
    wave = linetab["wave_vac"].astype(float)

    # Build a common wavelength grid with padding based on max sigma
    wmin, wmax = np.min(wave), np.max(wave)
    sigma_max = (fwhm_kms / 2.354820045) * (wmax / c_kms)  # Å
    pad = 5 * sigma_max + 2.0
    lam = np.linspace(wmin - pad, wmax + pad, ngrid)

    ions = np.unique(linetab["ion"].astype(str)) if "ion" in linetab.colnames else np.array([])
    multi_ion = len(ions) > 1

    # Group by ion + multiplet when multiple ions are present
    group_keys = ["ion", "multiplet_term"] if multi_ion else ["multiplet_term"]
    gtab = linetab.group_by(group_keys)
    groups = gtab.groups

    # Color cycle (by ion or by multiplet)
    if multi_ion:
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(ions), 3)))
        color_map = {ion: colors[i % len(colors)] for i, ion in enumerate(ions)}
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(groups), 3)))
        color_map = None

    if all_normalize:
        lr_all = linetab["line_ratio"].astype(float)
        lr_all_norm = all_normalize / lr_all.max()

    seen_labels = set()

    k_B_eV = 8.617333262e-5
    kT_eV = k_B_eV * Te if Te is not None else None

    for i, grp in enumerate(groups):
        mu = grp["wave_vac"].astype(float)  # centers (Å)
        lr = grp["line_ratio"].astype(float)
        sig = (fwhm_kms / 2.354820045) * (mu / c_kms)  # σ_λ per line (Å)

        prof = np.zeros_like(lam)
        for mu_i, s_i, a_i in zip(mu, sig, lr):
            if s_i <= 0:
                continue
            g = np.exp(-0.5 * ((lam - mu_i) / s_i) ** 2)
            if area_normalized:
                # Area(g) = sqrt(2π)*σ; scale so ∫ = a_i
                g *= a_i / (np.sqrt(2.0 * np.pi) * s_i)
            else:
                # Peak-normalized to a_i
                g *= a_i
            prof += g

        if kT_eV is not None and kT_eV > 0 and "Ei" in grp.colnames:
            avg_ei = np.nanmean(grp["Ei"].astype(float))
            if np.isfinite(avg_ei):
                boltz = np.exp(-(avg_ei - kT_eV) / kT_eV)
                prof = prof * boltz

        if per_multiplet_normalize and prof.max() > 0:
            prof = prof / prof.max() * per_multiplet_normalize

        if all_normalize:
            prof = prof * lr_all_norm

        # Apply inverse if requested
        if inverse:
            plot_prof = offset - prof
            baseline = offset
            idx = np.where(plot_prof < baseline)[0]
        else:
            plot_prof = prof + offset
            baseline = offset
            idx = np.where(plot_prof > baseline)[0]

        if multi_ion:
            grp_ion = str(grp["ion"][0])
            color = color_map[grp_ion]
            label = grp_ion
        else:
            color = colors[i % len(colors)]
            label = f"{grp['multiplet_term'][0]}"

        plot_label = label if (legend and label not in seen_labels) else None
        if plot_label is not None:
            seen_labels.add(label)

        if idx.size == 0:
            continue
        plt.plot(lam[idx], plot_prof[idx], color=color, lw=1.5, ls=linestyle, label=plot_label, alpha=0.7)

        # Optional vertical markers at line centers
        ymark = np.interp(mu, lam, prof, left=0.0, right=0.0)
        for x, y in zip(mu, ymark):
            if inverse:
                plt.vlines(x, baseline, baseline - y, color=color, alpha=0.5, lw=2, ls=linestyle)
            else:
                plt.vlines(x, baseline, baseline + y, color=color, alpha=0.5, lw=2, ls=linestyle)

    plt.xlabel(r"Wavelength $\lambda$ (Å)")
    plt.ylabel("Relative intensity")
    if legend:
        plt.legend(frameon=False, ncol=3)


def plot_ion_models(
    ion='H I',
    wave_range=None,
    tab=None,
    sigut_table=None,
    spectrum_dict=None,
    Te=5_000,
    fwhm_permitted=400.0,
    fwhm_forbidden=100.0,
    norm_permitted=0.3,
    norm_forbidden=0.35,
    norm_semiforbidden=0.35,
    gf_threshold=0.0001,
    Aki_threshold=0.001,
    figsize=(7, 4.5),
    spectrum_key='g235m',
    spectrum_scale=50,
    plot_forbidden=True,
    plot_semiforbidden=True,
    plot_permitted=True,
    merge_semiforbidden=True,
    verbose=False,
    legend=True,
):
    """
    Plot emission line models (permitted, semi-forbidden, forbidden) for any ion.

    Parameters
    ----------
    ion : str
        Ion name (e.g., 'Fe II', 'O III', 'He I')
    wave_range : list of two floats
        Wavelength range [min, max] in Å (rest-frame)
    sigut_table : Table, optional
        Sigut+03 Fe II template table (only used if provided)
    spectrum_dict : dict, optional
        Spectrum dictionary with keys like 'g235m' containing 'wave' and 'flux'
    Te : float
        Excitation temperature [K] for multiplet emissivities
    fwhm_permitted : float
        FWHM in km/s for permitted lines
    fwhm_forbidden : float
        FWHM in km/s for forbidden lines
    norm_permitted : float
        Normalization height for permitted lines
    norm_forbidden : float
        Normalization height for forbidden lines
    norm_semiforbidden : float
        Normalization height for semi-forbidden lines
    gf_threshold : float
        Minimum gf value for permitted/semi-forbidden lines
    Aki_threshold : float
        Minimum Aki value for forbidden lines
    figsize : tuple
        Figure size
    spectrum_key : str
        Key for spectrum in spectrum_dict to overlay
    spectrum_scale : float
        Scale factor for spectrum flux
    plot_semiforbidden : bool
        Whether to plot semi-forbidden lines
    verbose : bool
        Print diagnostic info

    Returns
    -------
    dict
        Dictionary with keys 'permitted', 'forbidden', 'semi-forbidden' containing the tables
    """
    import matplotlib.pyplot as plt

    results = {}
    linetab = tab.copy() if tab is not None else None
    if linetab is None:
        # Fetch lines from NIST
        forbidden = get_line_nist(
            ion=ion,
            wave=wave_range,
            tolerance=1,
            sortkey='Aki',
            threshold=f'Aki>{Aki_threshold}',
            verbose=verbose,
            classification='forbidden',
            multiplet_lower_only=False,
        )

        permitted = get_line_nist(
            ion=ion,
            wave=wave_range,
            tolerance=1,
            sortkey='gf',
            threshold=f'gf>{gf_threshold}',
            verbose=verbose,
            classification='permitted',
            multiplet_lower_only=True,
        )

        semiforbidden = get_line_nist(
            ion=ion,
            wave=wave_range,
            tolerance=1,
            sortkey='gf',
            threshold=f'gf>{gf_threshold}',
            verbose=verbose,
            classification='semi-forbidden',
            multiplet_lower_only=True,
        )
    else:
        permitted = linetab[[']' not in str(n) for n in linetab['ion']]]
        forbidden = linetab[['[' in str(n) for n in linetab['ion']]]
        semiforbidden = linetab[[']' in str(n) and '[' not in str(n) for n in linetab['ion']]]

    if merge_semiforbidden and semiforbidden is not None and len(semiforbidden) > 0:
        permitted = vstack([permitted, semiforbidden])
        permitted = assign_multiplets(permitted, lower_only=True)
        permitted = compute_line_ratios(permitted, Te=Te, Ne=1e2)
        semiforbidden = None

    plt.figure(figsize=figsize)

    # Permitted lines (bottom)
    if permitted is not None and len(permitted) > 0 and plot_permitted:
        plot_lines(
            permitted, fwhm_kms=fwhm_permitted, all_normalize=norm_permitted, offset=0.0, linestyle='-'
        )

    # Forbidden lines (top, inverted)
    if forbidden is not None and len(forbidden) > 0 and plot_forbidden:
        plot_lines(
            forbidden,
            fwhm_kms=fwhm_forbidden,
            all_normalize=norm_forbidden,
            offset=1.0,
            inverse=True,
            legend=legend,
            linestyle=':',
        )

    # Semi-forbidden (optional)
    if plot_semiforbidden and semiforbidden is not None and len(semiforbidden) > 0:
        plot_lines(
            semiforbidden,
            fwhm_kms=fwhm_forbidden,
            all_normalize=norm_semiforbidden,
            offset=0.9,
            inverse=True,
            legend=legend,
            linestyle='-.',
        )

    # Overlay spectrum if provided
    spectrum_ymax = 0
    if spectrum_dict is not None and spectrum_key in spectrum_dict:
        spec = spectrum_dict[spectrum_key]

        # Filter spectrum to wavelength range
        in_range = (spec['wave'] >= wave_range[0]) & (spec['wave'] <= wave_range[1])

        if np.any(in_range):
            # Normalize to median = 0.5 in the displayed window
            flux_in_range = spec['flux'][in_range]
            median_flux = np.nanmedian(flux_in_range)
            flux_norm = spec['flux'] / median_flux * 0.5

            plt.plot(spec['wave'], flux_norm, color='black', alpha=0.5, label='Observed', zorder=1)
            spectrum_ymax = np.nanmax(flux_norm[in_range])

    # Add vertical tick marks at line positions (just above spectrum at each wavelength)
    tick_height = 0.03  # Height of tick marks (fraction of plot)

    # Permitted lines (blue ticks)
    if permitted is not None and len(permitted) > 0 and plot_permitted:
        for wave in permitted['wave_vac']:
            # Get spectrum value at this wavelength
            if spectrum_ymax > 0 and np.any(in_range):
                spec_y = np.interp(wave, spec['wave'], flux_norm, left=0, right=0)
                tick_y = spec_y + 0.05
            else:
                tick_y = 0.05
            plt.plot(
                [wave, wave], [tick_y, tick_y + tick_height], color='blue', alpha=0.6, lw=1.0, zorder=10
            )

    # Forbidden lines (dashed ticks)
    if forbidden is not None and len(forbidden) > 0 and plot_forbidden:
        for wave in forbidden['wave_vac']:
            if spectrum_ymax > 0 and np.any(in_range):
                spec_y = np.interp(wave, spec['wave'], flux_norm, left=0, right=0)
                tick_y = spec_y + 0.05
            else:
                tick_y = 0.05
            plt.plot(
                [wave, wave],
                [tick_y, tick_y + tick_height],
                ls='--',
                color='blue',
                alpha=0.6,
                lw=1.0,
                zorder=10,
            )

    # Semi-forbidden lines (orange ticks)
    if plot_semiforbidden and semiforbidden is not None and len(semiforbidden) > 0:
        for wave in semiforbidden['wave_vac']:
            if spectrum_ymax > 0 and np.any(in_range):
                spec_y = np.interp(wave, spec['wave'], flux_norm, left=0, right=0)
                tick_y = spec_y + 0.05
            else:
                tick_y = 0.05
            plt.plot(
                [wave, wave],
                [tick_y, tick_y + tick_height],
                ls='-.',
                color='blue',
                alpha=0.6,
                lw=1.0,
                zorder=10,
            )

    if wave_range is None:
        if spectrum_dict is not None and spectrum_key in spectrum_dict:
            spec = spectrum_dict[spectrum_key]
            wave_range = [min(spec['wave']), max(spec['wave'])]
        else:
            wave_range = [4000, 7000]

    plt.xlim(*wave_range)
    plt.ylim(0, 1)
    plt.xlabel('Rest wavelength [Å]')
    plt.ylabel('Normalized intensity')

    plt.tight_layout()

    return results


class EmissionLines:
    def __init__(
        self,
        filename: str | Path | None = DEFAULT_EMISSION_LINES_FILE,
        datadir: str | Path | None = DATA_DIR,
    ):
        if not Path(filename).exists():
            filename = Path(datadir) / filename

        self.datadir = Path(filename).parent

        self.filename = Path(filename)
        self.table = Table.read(self.filename)
        self.lines = {l['key']: dict(l) for l in self.table}

        # Define common line groups
        self.groups = LINE_GROUPS

    # static
    @staticmethod
    def search_line_nist(ion, **kwargs):
        return get_line_nist(ion=ion, **kwargs)

    @staticmethod
    def list():
        """List contents of the emission-lines data directory."""
        datadir = Path(DATA_DIR)
        if not datadir.exists():
            return []
        return sorted(p.name for p in datadir.iterdir())

    def remove_key(self, key):
        ix_remove = np.where(self.table['key'] == key)[0]
        if len(ix_remove) > 0:
            self.table.remove_rows(ix_remove)
            self.lines = {l['key']: dict(l) for l in self.table}

    def get_table(self, search_key=None, wave=None, filter=None, multiplet=False, **kwargs):
        """Return a subset of the emission line table.

        Args:
            search_key: Ion name (``'Fe II'``), line key (``'Ha'``),
                or group alias (``'[OIII]'``).  ``None`` returns all lines.
            wave: Two-element list ``[lo, hi]`` restricting vacuum
                wavelength range in Angstroms.
            filter: Column filter expression passed to :func:`filter_table`.
                Comma-separated conditions, e.g. ``'line_ratio>1e-2,Ei<4'``.
            multiplet: If ``True``, ensure multiplet ratios are computed.
            **kwargs: Forwarded to multiplet ratio calculation
                (``Te``, ``Ne``, etc.).

        Returns:
            Filtered astropy Table.

        Examples::

            el.get_table('Fe II', wave=[3700, 7100])
            el.get_table(wave=[3700, 7100], filter='line_ratio>1e-2')
            el.get_table('[OIII]', filter='Aki>1e3')
        """
        if wave is not None:
            iw = (self.table['wave_vac'] >= wave[0]) & (self.table['wave_vac'] <= wave[1])
        else:
            iw = self.table['wave_vac'] > 0

        if search_key is None:
            tab = self.table[iw]
        elif search_key in self.groups:
            # Create a mask for all keys in the alias list
            mask = np.zeros(len(self.table), dtype=bool)
            for key in self.groups[search_key]:
                mask |= self.table['key'] == key
            tab = self.table[mask & iw]
        elif search_key in self.table['key']:
            # Exact key match
            tab = self.table[(self.table['key'] == search_key) & iw]
        elif ' ' in search_key or ('[' in search_key and ']' in search_key):  # ions always have spaces
            # Try to format as "Element Roman" (e.g. "FeII" -> "Fe II")
            formatted_key = re.sub(r'(?<!\s)([IVX])', r' \1', search_key.replace(' ', ''), count=1)
            # Check if it matches 'ion' column directly or the formatted version
            ik = (self.table['ion'] == search_key) | (self.table['ion'] == formatted_key)
            tab = self.table[ik & iw]
        else:
            tab = self.table[(self.table['key'] == search_key) & iw]

        if filter is not None:
            tab = filter_table(tab, filter)

        if multiplet:
            # Only recalculate if multiplet/line_ratio columns are missing or trivial
            tab = _ensure_multiplet_ratios(tab, kwargs)

        return tab

    def get_multiplet(self, key):
        ix = self.table['key'] == key
        if np.sum(ix) == 0:
            return None

        ion = self.table['ion'][ix][0]

        # Get all lines for this ion to correctly assign multiplets
        ion_tab = self.table[self.table['ion'] == ion]
        # Assign multiplets (this returns a copy/subset with 'multiplet' column populated)
        # should have already been populated, dont redo implicitly
        # ion_tab = assign_multiplets(ion_tab)

        # Find the multiplet ID for the requested key
        match = ion_tab['key'] == key
        # this should be true by definition
        #        if np.sum(match) == 0:
        #            return None

        m_id = ion_tab['multiplet'][match][0]

        if m_id > 0:
            return ion_tab[ion_tab['multiplet'] == m_id]
        else:
            return ion_tab[match]

    def plot_lines(self, search_key=None, **kwargs):
        return plot_lines(self.get_table(search_key=search_key, **kwargs), **kwargs)

    def plot_ion_models(self, **kwargs):
        return plot_ion_models(**kwargs)

    def find_duplicates(self, keys=['ion', 'wave_vac'], remove=False):
        """
        Find duplicate entries in the table based on specified keys.

        Parameters:
            keys (list): List of column names to check for duplicates.
                         Default is ['ion', 'wave_vac'].
            remove (bool): If True, remove duplicates from the table, keeping only the first occurrence.

        Returns:
            Table: A table containing the duplicate entries.
        """

        from astropy.table import unique

        # Get groups of duplicates
        t = self.table.group_by(keys)

        # Find indices of groups with size > 1
        indices = t.groups.indices
        duplicates = []

        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            if end - start > 1:
                # This group has duplicates
                duplicates.append(t[start:end])

        if duplicates:
            dup_table = vstack(duplicates)

            if remove:
                n_before = len(self.table)
                self.table = unique(self.table, keys=keys)
                n_after = len(self.table)

                # Update the lines dictionary
                self.lines = {l['key']: dict(l) for l in self.table}
                print(f"Removed {n_before - n_after} duplicate rows.")

            return dup_table
        else:
            return None

    def add_lines(self, new_table, sort=True):
        """
        Add a table of new lines to the existing table.

        Parameters:
            new_table (Table): Table containing new lines to add.
            sort (bool): If True, sort the table by 'wave_vac' after adding.
        """
        self.table = vstack([self.table, new_table])

        if sort:
            self.table.sort('wave_vac')

        # Update the lines dictionary
        self.lines = {l['key']: dict(l) for l in self.table}

    def remove_lines(self, lines_to_remove):
        """
        Remove rows from the table that match the rows in lines_to_remove.

        Parameters:
            lines_to_remove (Table): Table containing lines to remove.
        """
        # We need to identify rows to remove.
        # We'll match on 'key', 'ion', and 'wave_vac' to be safe
        indices_to_remove = []

        for row in lines_to_remove:
            # Find matching rows in self.table
            mask = (
                (self.table['key'] == row['key'])
                & (self.table['ion'] == row['ion'])
                & (np.isclose(self.table['wave_vac'], row['wave_vac'], atol=1e-5))
            )

            indices = np.where(mask)[0]
            indices_to_remove.extend(indices)

        if indices_to_remove:
            indices_to_remove = np.unique(indices_to_remove)
            self.table.remove_rows(indices_to_remove)

            # Update the lines dictionary
            self.lines = {l['key']: dict(l) for l in self.table}

            print(f"Removed {len(indices_to_remove)} rows.")
        else:
            print("No matching rows found to remove.")

    def get_line_wavelengths(self, multiplet=True):
        lw = {row['key']: [row['wave_vac']] for row in self.table}
        lr = {row['key']: [row['line_ratio']] for row in self.table}
        if multiplet:
            multi = np.unique(self.table['multiplet'])
            for m in multi:
                ix = self.table['multiplet'] == m
                if self.table['ion'][ix][0] == 'H I':
                    k = 'Hydrogen'
                else:
                    k = (
                        self.table['key'][ix][0].split('-')[0]
                        + '-'
                        + ','.join([f'{w:.0f}' for w in self.table['wave_vac'][ix]])
                    )

                lw[k] = list(self.table['wave_vac'][ix])
                lr[k] = list(self.table['line_ratio'][ix])
                print(k, lw[k], lr[k])

        return lw, lr

    def save(self, filename=None):
        if filename is None:
            filename = self.filename
        destination = Path(filename)
        if destination == DEFAULT_EMISSION_LINES_FILE:
            print(
                'Cannot overwrite default file emission_lines.csv, please specify new filename save(filename)'
            )

        self.table.write(destination, overwrite=True)

    def add_nist_columns(self, columns=['fik'], tolerance=1.0, verbose=False):
        """
        Add columns from NIST database to the emission lines table.

        Parameters:
            columns (list): List of column names to add from NIST (e.g., ['fik', 'Aki'])
            tolerance (float): Wavelength tolerance in Angstroms for matching lines
            verbose (bool): Print progress information

        Returns:
            Table: Updated table with new columns
        """
        from astroquery.nist import Nist
        import astropy.units as u

        # Make a copy of the table
        updated_table = self.table.copy()

        # Add columns if they don't exist
        for col in columns:
            if col not in updated_table.colnames:
                updated_table.add_column(Column(np.zeros(len(updated_table)), name=col))
                if verbose:
                    print(f"Added column '{col}' to table")

        # Process each row
        for i, row in enumerate(updated_table):
            ion = row['ion'].replace('[', '').replace(']', '')
            wave_vac = row['wave_vac']

            if verbose:
                print(f"Processing {i+1}/{len(updated_table)}: {ion} {wave_vac:.3f} Å")

            try:
                # Query NIST for the closest line
                nist_result = get_line_nist(
                    ion=ion,
                    wave=wave_vac,
                    tolerance=tolerance,
                    single=True,
                    clear_cache=False,
                    verbose=False,
                )

                if nist_result is not None and len(nist_result) > 0:
                    # Update requested columns
                    for col in columns:
                        if col in nist_result.colnames:
                            updated_table[col][i] = nist_result[col][0]
                            if verbose:
                                print(f"  Found {col} = {nist_result[col][0]}")
                        else:
                            if verbose:
                                print(f"  Warning: Column '{col}' not found in NIST result")
                else:
                    if verbose:
                        print(f"  No NIST match found within {tolerance} Å")

            except Exception as e:
                if verbose:
                    print(f"  Error querying NIST: {e}")
                continue

        return updated_table

    def regenerate_table(self, tolerance=1.0, verbose=False, skip_on_error=True):
        """
        Regenerate the emission lines table by querying NIST for each entry.

        This method creates a new table by searching NIST for the nearest line
        to each entry in the current table based on ion and wavelength.

        Parameters:
            tolerance (float): Wavelength tolerance in Angstroms for matching lines
            verbose (bool): Print progress information
            skip_on_error (bool): Skip entries that fail instead of raising an error

        Returns:
            Table: New table with updated entries from NIST
        """
        from astropy.table import vstack

        new_tables = []
        skipped = []

        for i, row in enumerate(self.table):
            ion = row['ion'].replace('[', '').replace(']', '')
            wave_vac = row['wave_vac']

            if verbose:
                print(f"Processing {i+1}/{len(self.table)}: {ion} {wave_vac:.3f} Å")

            try:
                # Query NIST for the closest line
                nist_result = get_line_nist(
                    ion=ion,
                    wave=wave_vac,
                    sortkey='Aki',
                    tolerance=tolerance,
                    single=True,
                    clear_cache=False,
                    verbose=False,
                )

                if nist_result is not None and len(nist_result) > 0:
                    new_tables.append(nist_result)
                    if verbose:
                        print(f"  Found: {nist_result['key'][0]} at {nist_result['wave_vac'][0]:.3f} Å")
                else:
                    skipped.append((ion, wave_vac, "No NIST match found"))
                    if verbose:
                        print(f"  No NIST match found within {tolerance} Å")

            except Exception as e:
                skipped.append((ion, wave_vac, str(e)))
                if verbose:
                    print(f"  Error querying NIST: {e}")
                if not skip_on_error:
                    raise

        # Stack all results into a single table
        if new_tables:
            regenerated_table = vstack(new_tables)
            regenerated_table.sort('wave_vac')

            if verbose:
                print(f"\nRegenerated table with {len(regenerated_table)} entries")
                if skipped:
                    print(f"Skipped {len(skipped)} entries:")
                    for ion, wave, reason in skipped:
                        print(f"  {ion} {wave:.3f} Å: {reason}")

            return regenerated_table
        else:
            if verbose:
                print("No entries could be regenerated from NIST")
            return None

    def to_unite(self, groups, save=False, line_ratio_min=None):
        """
        Produces unite style jsons from the line catalogs of el.table

        Parameters:
            groups (list): List of dictionaries defining the groups.
                           e.g. [{'name': 'default'}, {'emission1': '[OII],[NII]'}, ...]
            save (bool): If True, dump to json file named "{name}.json"
            line_ratio_min (float, optional): If provided, only include lines with
                                             line_ratio > line_ratio_min in multiplets.
                                             Default is None (include all lines).

        Returns:
            dict: The unite style dictionary
        """
        import json
        import re
        from astropy.table import vstack

        unite_dict = {'Name': 'all', 'Unit': 'Angstrom', 'Groups': {}}
        # Add region if present

        for k, v in groups[0].items():
            print(k, v)
            unite_dict[k.capitalize()] = v

        for g in groups:
            # Handle the 'name' in the first element if present
            # Identify group name and line string
            group_name = None
            line_string = None
            kwargs = {}

            for k, v in g.items():
                if k in ['TieRedshift', 'TieDispersion', 'Te', 'Ne', 'ne', 'additional']:
                    kwargs[k] = v
                elif k == 'wave':
                    kwargs['wave'] = v
                elif isinstance(v, str):
                    group_name = k
                    line_string = v
                else:
                    # Assume other keys are kwargs too
                    kwargs[k] = v

            if 'ne' in kwargs and 'Ne' not in kwargs:
                kwargs['Ne'] = kwargs['ne']
            elif 'Ne' in kwargs and 'ne' not in kwargs:
                kwargs['ne'] = kwargs['Ne']

            if group_name is None:
                continue

            # Auto-increment group name if no digits provided (e.g. 'emission' -> 'emission1')
            if not re.search(r'\d+$', group_name):
                base_name = group_name
                i = 1
                while True:
                    candidate = f"{base_name}{i}"
                    if candidate not in unite_dict['Groups']:
                        group_name = candidate
                        break
                    i += 1

            # Determine LineType (remove trailing digits)
            line_type = re.sub(r'\d+$', '', group_name)

            # Prepare Unite Group Config
            unite_group = {
                'TieRedshift': kwargs.pop('TieRedshift', True),
                'TieDispersion': kwargs.pop('TieDispersion', True),
                'Species': [],
            }

            # Prepare get_table kwargs
            additional = kwargs.pop('additional', None)
            should_recalc = 'Te' in kwargs and ('ne' in kwargs or 'Ne' in kwargs)

            # Fetch tables for each alias
            tables = []
            aliases = [x.strip() for x in line_string.split(',')]
            for alias in aliases:
                try:
                    multiplet = alias.endswith('*')
                    alias = alias[:-1].strip() if multiplet else alias
                    t = None
                    # If multiplet is requested, check if alias is a specific line key that belongs to a multiplet
                    if multiplet and alias in self.table['key']:
                        t = self.get_multiplet(alias)

                    if t is None:
                        # Pass multiplet and kwargs (Te, Ne, etc.)
                        t = self.get_table(alias, multiplet=multiplet, **kwargs)
                    elif multiplet and should_recalc:
                        # If we got a table from get_multiplet, ensure ratios are calculated if missing
                        t = _ensure_multiplet_ratios(t, kwargs)

                    if t is not None and 'use_multiplet' not in t.colnames:
                        t = t.copy()
                        t['use_multiplet'] = np.full(len(t), multiplet, dtype=bool)

                    if len(t) > 0:
                        tables.append(t)
                except Exception as e:
                    print(f"Warning: Could not get table for alias '{alias}': {e}")

            if not tables:
                continue

            full_table = vstack(tables)

            # Group by Ion/Species
            ions = np.unique(full_table['ion'])

            for ion in ions:
                mask = full_table['ion'] == ion
                ion_table = full_table[mask]

                # Determine subgroups based on multiplets
                subgroups = []
                if 'use_multiplet' in ion_table.colnames:
                    multi_mask = ion_table['use_multiplet'] & (ion_table['multiplet'] > 0)
                    if np.any(multi_mask):
                        for m in np.unique(ion_table['multiplet'][multi_mask]):
                            subgroups.append(ion_table[multi_mask & (ion_table['multiplet'] == m)])
                    non_multi = ion_table[~multi_mask]
                    if len(non_multi) > 0:
                        subgroups.append(non_multi)
                else:
                    subgroups.append(ion_table)

                for species_table in subgroups:
                    # Normalize ratios per multiplet (weakest to 1.0)
                    if 'use_multiplet' in species_table.colnames:
                        use_multi = bool(np.any(species_table['use_multiplet']))
                    else:
                        use_multi = False

                    if (
                        use_multi
                        and 'multiplet' in species_table.colnames
                        and 'line_ratio' in species_table.colnames
                    ):
                        # Since we split by multiplet, we can just normalize the whole table if m > 0
                        m_vals = species_table['multiplet']
                        if len(m_vals) > 0 and m_vals[0] > 0:
                            ratios = species_table['line_ratio']
                            # Handle masked/nan
                            if hasattr(ratios, 'mask'):
                                valid_mask = ~ratios.mask & ~np.isnan(ratios)
                            else:
                                valid_mask = ~np.isnan(ratios)
                            # this may not be necessary but just in case
                            if np.any(valid_mask):
                                max_r = np.max(ratios[valid_mask])
                                if max_r > 0:
                                    species_table['line_ratio'] = ratios / max_r

                    # Format Species Name: 'H I' -> 'HI', '[O III]' -> '[OIII]'
                    species_name = ion.replace(' ', '')

                    # Append multiplet ID to species name if applicable
                    if use_multi and 'multiplet' in species_table.colnames:
                        m_vals = species_table['multiplet']
                        if len(m_vals) > 0 and m_vals[0] > 0:
                            species_name = f"{species_name}m{m_vals[0]}"

                    lines = []
                    for row in species_table:
                        wave = float(row['wave_vac'])

                        # Handle RelStrength
                        rel_strength = None
                        # Only use ratio if multiplet calculation was enabled
                        if use_multi and 'line_ratio' in row.colnames and row['multiplet'] > 0:
                            val = row['line_ratio']
                            if not np.ma.is_masked(val) and not np.isnan(val):
                                rel_strength = float(val)

                        # Filter by line_ratio_min if specified
                        if line_ratio_min is not None and rel_strength is not None:
                            if rel_strength <= line_ratio_min:
                                continue  # Skip weak lines

                        lines.append({'Wavelength': wave, 'RelStrength': rel_strength})

                    species_entry = {'Name': species_name, 'LineType': line_type}
                    if additional:
                        if isinstance(additional, list):
                            species_entry['AdditionalComponents'] = {k: group_name for k in additional}
                        else:
                            species_entry['AdditionalComponents'] = {str(additional): group_name}
                    species_entry['Lines'] = lines
                    unite_group['Species'].append(species_entry)

            unite_dict['Groups'][group_name] = unite_group

        if save:
            filename = f"{unite_dict['Name']}.json"
            with open(filename, 'w') as f:
                json.dump(unite_dict, f, indent=4)

        return unite_dict

    @staticmethod
    def to_lines(linetype: str, ion: str, wavelengths: list[float]) -> dict:
        """Build a unite-style line dict from explicit wavelengths.

        Negative wavelengths are treated as multiplet lines (appends ``*``).

        Args:
            linetype: Line type key (e.g. ``'emission'``).
            ion: Ion label (e.g. ``'FeII'``).
            wavelengths: Wavelengths in Angstrom. Use negative values to
                flag multiplet lines.

        Returns:
            Dict of ``{linetype: 'ion-w1,ion-w2*,...'}`` ready for
            :meth:`to_unite` group lists.
        """
        parts = []
        ion_clean = ion.rstrip('*')
        ion_star = '*' if ion.endswith('*') else ''
        for w in wavelengths:
            suffix = '*' if w < 0 else ion_star
            parts.append(f"{ion_clean}-{abs(w)}{suffix}")
        return {linetype: ','.join(parts)}


def read_kurucz_table(
    path: str | Path,
    ion_override: str | None = None,
    ndigits: int = 3,
    assign_multiplet: bool = True,
    emissivities: bool = True,
    Te: float = 10_000,
):
    """
    Read a Kurucz fixed-width line list file into the dotfit emission line table format.

    The Kurucz format uses fixed-width columns with the following layout:
        Wl_vac      Wl_air   log_gf   A-Value   Elem. Element E_lower_lev.   J   Config.    E_upper_lev.   J   Config.    Ref.

    Parameters
    ----------
    path : str or Path
        Path to the Kurucz file.
    ion_override : str, optional
        If provided, use this ion string for all rows (e.g. "Fe II").
    ndigits : int
        Rounding digits for wavelengths in Angstrom.
    assign_multiplet : bool
        If True, assign multiplets using configuration/terms.
    emissivities : bool
        If True, compute multiplet emissivity ratios (line_ratio) when possible.
    Te : float
        Electron temperature [K] for emissivity ratios.

    Returns
    -------
    astropy.table.Table
        Table with columns compatible with dotfit emission line tables.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # Verify header is present
    header_found = False
    for line in lines:
        if "Wl_vac" in line and "Wl_air" in line and "log_gf" in line:
            header_found = True
            break

    if not header_found:
        raise ValueError("Kurucz header line not found; expected 'Wl_vac' and 'log_gf'.")

    # Fixed-width column specifications for the Kurucz format
    # These are determined empirically from the data layout, NOT from header positions
    # Column positions: (start, end) - 0-indexed, end is exclusive
    colspecs = [
        (0, 12),  # Wl_vac (nm)
        (12, 24),  # Wl_air (nm, may be blank for vacuum UV)
        (24, 31),  # log_gf
        (31, 42),  # A-Value (1/s)
        (42, 48),  # Elem. code (e.g., 26.01)
        (48, 60),  # Element name (e.g., "Fe II")
        (60, 72),  # E_lower_lev. (eV)
        (72, 76),  # J_lower
        (76, 88),  # Config. lower (orbital config + term)
        (88, 100),  # E_upper_lev. (eV)
        (100, 106),  # J_upper
        (106, 117),  # Config. upper (orbital config + term)
        (117, None),  # Ref.
    ]

    def _slice(line: str, start: int, end: int | None) -> str:
        return line[start:end].strip() if end is not None else line[start:].strip()

    def _j_to_fraction(j_val: float) -> str:
        """Convert J value like 4.5 to '9/2' fractional format."""
        if np.isnan(j_val):
            return ""
        # If J is integer, return as integer string
        if j_val == int(j_val):
            return str(int(j_val))
        # Otherwise, convert to fraction: 4.5 -> 9/2
        numerator = int(2 * j_val)
        return f"{numerator}/2"

    def _parse_config_term(cfg_str: str) -> tuple[str, str]:
        """Parse Kurucz config string into configuration and term parts.

        Examples:
            "(5D)4s a6D" -> config="(5D)4s", term="a6D"
            "d7 a4F" -> config="d7", term="a4F"
            "(3F2)5p 4F" -> config="(3F2)5p", term="4F"
        """
        cfg_str = cfg_str.strip()
        if not cfg_str:
            return "", ""

        parts = cfg_str.split()
        if len(parts) == 1:
            # Check if it matches term pattern: [a-z prefix]digit+L[*]
            if re.match(r'^[a-z]?\d+[SPDFGHIKLMN](\*)?$', parts[0]):
                return "", parts[0]
            return parts[0], ""

        # Multiple parts - last part is typically the term
        term = parts[-1]
        config = " ".join(parts[:-1])
        return config, term

    records = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        if "Wl_vac" in line or "/ nm" in line:
            continue

        fields = [_slice(line, s, e) for (s, e) in colspecs]
        if not fields[0]:
            continue

        try:
            wl_vac_nm = float(fields[0])
        except ValueError:
            continue

        wl_air_nm = None
        if fields[1]:
            try:
                wl_air_nm = float(fields[1])
            except ValueError:
                wl_air_nm = None

        try:
            log_gf = float(fields[2])
        except ValueError:
            log_gf = np.nan

        try:
            A_value = float(fields[3])
        except ValueError:
            A_value = np.nan

        elem_name = fields[5] or ""
        ion = ion_override or elem_name

        try:
            E_lower = float(fields[6])
        except ValueError:
            E_lower = np.nan

        try:
            J_lower = float(fields[7])
        except ValueError:
            J_lower = np.nan

        cfg_lower_full = fields[8] or ""

        try:
            E_upper = float(fields[9])
        except ValueError:
            E_upper = np.nan

        try:
            J_upper = float(fields[10])
        except ValueError:
            J_upper = np.nan

        cfg_upper_full = fields[11] or ""
        ref = fields[12] or ""

        # Convert wavelength from nm to Angstrom
        wl_vac = np.round(wl_vac_nm * 10.0, ndigits)
        if wl_air_nm is None:
            wl_air = np.round(vacuum_to_air(wl_vac), ndigits)
        else:
            wl_air = np.round(wl_air_nm * 10.0, ndigits)

        # Parse configuration and term from Kurucz format
        lower_config, lower_term = _parse_config_term(cfg_lower_full)
        upper_config, upper_term = _parse_config_term(cfg_upper_full)

        terms = f"{lower_term}-{upper_term}" if lower_term and upper_term else "--"

        # Build configuration string: use only electron configuration parts
        # Remove spaces inside each side so e.g. 'd5 4s2' -> 'd54s2'
        def _compact_cfg(cfg: str) -> str:
            return cfg.replace(" ", "") if cfg else ""

        lc = _compact_cfg(lower_config)
        uc = _compact_cfg(upper_config)

        if lc and uc:
            configuration = f"{lc}-{uc}"
        elif lc:
            configuration = lc
        elif uc:
            configuration = uc
        else:
            configuration = ""

        # Format J values as fractions (e.g., "9/2-9/2")
        j_lower_str = _j_to_fraction(J_lower)
        j_upper_str = _j_to_fraction(J_upper)
        ji_jk = f"{j_lower_str}-{j_upper_str}" if j_lower_str and j_upper_str else "--"

        # Calculate statistical weights (g = 2J + 1)
        if np.isnan(J_lower) or np.isnan(J_upper):
            gigk = ""
            gi = 0
        else:
            gi = int(round(2 * J_lower + 1))
            gk = int(round(2 * J_upper + 1))
            gigk = f"{gi}-{gk}"

        gf = 10 ** log_gf if np.isfinite(log_gf) else 0.0
        fik = gf / gi if (gi > 0) else 0.0

        try:
            classification = classify_transition(
                configuration, terms, ji_jk, nist_type="E1", Aki=A_value, fik=fik
            )
        except Exception:
            classification = "permitted"

        # Adjust key and ion depending on classification:
        # permitted: key 'FeII-XXXX', ion 'Fe II'
        # semi-forbidden: key 'FeII]-XXXX', ion 'Fe II]'
        # forbidden: key '[FeII]-XXXX', ion '[Fe II]'
        ion_plain = ion  # e.g. 'Fe II'
        ion_nospace = replace_greek(ion_plain, tex=False).replace(" ", "")  # 'FeII'

        if classification == "permitted":
            key = f"{ion_nospace}-{wl_vac:.0f}"
            ion_out = ion_plain
        elif classification == "semi-forbidden":
            # Use closing bracket style for semi-forbidden
            key = f"{ion_nospace}]-{wl_vac:.0f}"
            ion_out = ion_plain + "]"
        elif classification == "forbidden":
            key = f"[{ion_nospace}]-{wl_vac:.0f}"
            ion_out = f"[{ion_plain}]"
        else:
            key = f"{ion_nospace}-{wl_vac:.0f}"
            ion_out = ion_plain

        if classification == "semi-forbidden":
            ion_out = ion_out.replace("]", '')  # loose the ] in ion for semi-forbidden

        records.append(
            (
                key,
                ion_out,
                wl_vac,
                E_lower,
                E_upper,
                A_value,
                fik,
                gigk,
                configuration,
                terms,
                ji_jk,
                "E1",
                wl_air,
                classification,
                gf,
                0,
                1.0,
                "",
                ref,
                "--",
            )
        )

    tab = Table(
        rows=records,
        names=[
            "key",
            "ion",
            "wave_vac",
            "Ei",
            "Ek",
            "Aki",
            "fik",
            "gigk",
            "configuration",
            "terms",
            "Ji-Jk",
            "type",
            "wave_air",
            "classification",
            "gf",
            "multiplet",
            "line_ratio",
            "multiplet_term",
            "references",
            "note",
        ],
        dtype=[
            "U14",
            "U9",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "U8",
            "U37",
            "U14",
            "U9",
            "U2",
            "float64",
            "U27",
            "float64",
            "int64",
            "float64",
            "U32",
            "U14",
            "U10",
        ],
    )

    if assign_multiplet and len(tab) > 0:
        tab = assign_multiplets(tab)
        if emissivities:
            tab = compute_line_ratios(tab, Te=Te, Ne=1e2)

    return tab


def read_sigut_table(path: str | Path) -> Table:
    """
    Read Sigut et al. 2003 Fe II table into an Astropy Table.

    Parameters
    ----------
    path : str or Path
        Path to sigut_03.tab file

    Returns
    -------
    Table
        Astropy Table with columns similar to get_line_nist output
    """
    path = Path(path)

    waves = []
    upper_levels = []
    lower_levels = []
    lower_energies = []
    fluxes = []

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(line) < 52 or not line[:9].strip():
                continue
            try:
                wave = float(line[0:9])
            except ValueError:
                continue

            waves.append(wave)
            upper_levels.append(line[10:24].strip())
            lower_levels.append(line[25:39].strip())
            lower_energies.append(float(line[40:46]))
            fluxes.append(int(line[47:52]))

    def parse_sigut_level(level_str):
        """
        Parse Sigut level string like 'a 4F9/2' into (term, J).
        Returns ('', '') if parsing fails.
        """
        if not level_str:
            return '', ''
        parts = level_str.strip().replace('^', '').split('_')
        return parts[0].replace('o', '*').replace('e', ''), parts[1]

    # Parse levels
    terms_list = []
    jijk_list = []

    for lower, upper in zip(lower_levels, upper_levels):
        lower_term, lower_j = parse_sigut_level(lower)
        upper_term, upper_j = parse_sigut_level(upper)

        if lower_term and upper_term:
            terms = f"{lower_term}-{upper_term}"
            jijk = f"{lower_j}-{upper_j}"
        else:
            terms = ''
            jijk = ''

        terms_list.append(terms)
        jijk_list.append(jijk)

    # Create keys similar to NIST format
    keys = [f"FeII-{int(w)}" for w in waves]

    tab = Table(
        {
            'key': keys,
            'ion': ['Fe II'] * len(waves),
            'wave_vac': air_to_vacuum(waves),
            'Ei': lower_energies,
            'Ek': [0.0] * len(waves),
            'Aki': [0.0] * len(waves),
            'fik': [0.0] * len(waves),
            'gigk': [''] * len(waves),
            'configuration': [''] * len(waves),
            'terms': terms_list,
            'Ji-Jk': jijk_list,
            'type': ['E1'] * len(waves),
            'wave_air': waves,
            'classification': ['permitted'] * len(waves),
            'gf': [0.0] * len(waves),
            'multiplet': [0] * len(waves),
            'line_ratio': fluxes,
            'multiplet_term': [''] * len(waves),
            'references': ['Sigut+03'] * len(waves),
            'note': [''] * len(waves),
        }
    )

    return assign_multiplets(tab, lower_only=True)


# Backwards compatible alias with the legacy implementation name.
# Emission_Lines = EmissionLines


# https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion#
def vacuum_to_air(lambda_vac_in):
    lambda_vac = np.atleast_1d(lambda_vac_in)  # Ensure input is a numpy array for vectorized operations
    s2 = (1e4 / lambda_vac) ** 2

    # Compute the refractive index of air using the formula from Donald Morton (2000)
    n_air = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    lambda_air = lambda_vac / n_air
    lambda_air[lambda_vac < 2000] = lambda_vac[lambda_vac < 2000]

    if np.isscalar(lambda_vac_in):
        lambda_air = lambda_air.item()
    return lambda_air.tolist() if isinstance(lambda_vac_in, list) else lambda_air


# by N. Piskunov
def air_to_vacuum(lambda_air_in):
    # Ensure input is a numpy array for vectorized operations
    lambda_air = np.atleast_1d(lambda_air_in)
    s2 = (1e4 / lambda_air) ** 2

    # Compute the refractive index of air using the provided formula
    n_air = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s2)
        + 0.0001599740894897 / (38.92568793293 - s2)
    )
    lambda_vacuum = lambda_air * n_air
    lambda_vacuum[lambda_vacuum < 2000] = lambda_vacuum[lambda_vacuum < 2000]

    if np.isscalar(lambda_air_in):
        lambda_vacuum = lambda_vacuum.item()
    return lambda_vacuum.tolist() if isinstance(lambda_air_in, list) else lambda_vacuum

    # if np.iterable(lambda_vacuum):
    #     lambda_vacuum[lambda_vacuum < 2000] = lambda_air_arr[lambda_vacuum < 2000]
    #     return list(lambda_vacuum) if type(lambda_air) is list else lambda_vacuum
    # else:
    #     return lambda_air if lambda_vacuum < 2000 else lambda_vacuum


def roman_to_int(roman):
    roman_map = {
        'I': 1,
        'II': 2,
        'III': 3,
        'IV': 4,
        'V': 5,
        'VI': 6,
        'VII': 7,
        'VIII': 8,
        'IX': 9,
        'X': 10,
        'XI': 11,
        'XII': 12,
        'XIII': 13,
        'XIV': 14,
        'XV': 15,
    }
    return roman_map.get(roman, None)  # Return None if not a valid numeral


def replace_with_latex(text):
    replacements = {
        'A': r'$\alpha$',
        'a': r'$\alpha$',
        'B': r'$\beta$',
        'b': r'$\beta$',
        'G': r'$\gamma$',
        'g': r'$\gamma$',
        'D': r'$\delta$',
        'd': r'$\delta$',
        'E': r'$\epsilon$',
        'e': r'$\epsilon$',
    }
    return ''.join(replacements.get(c, c) for c in text)


def construct_hydrogen_key(config):
    """
    Construct the 'key' for hydrogen entries based on configuration rules.
    """
    parts = config.split('-')
    if len(parts) != 2:
        return None  # Invalid configuration
    try:
        start = int(parts[0][0])  # first digit of first level: for Ly it is 1s, for H it is 2, etc.
        end = int(parts[1])
    except ValueError:
        return None  # Non-integer configuration

    diff = end - start
    series_map = {1: 'Ly', 2: 'H', 3: 'Pa', 4: 'Br', 5: 'Pf'}

    # Determine series
    if start in series_map:
        series = series_map[start]
        if start == 2:  # Special case for H series (2-level)
            special_names = ['a', 'b', 'g', 'd']
            if diff <= 4:
                return f"{series}{special_names[diff - 1]}"
            else:
                return f"{series}{end}"
        else:  # Use A, B, G, D, E for the first 5 entries, then numbers
            special_names = ['A', 'B', 'G', 'D', 'E']
            if diff <= 5:
                return f"{series}{special_names[diff - 1]}"
            else:  # Use numbers for others
                return f"{series}{end}"
    return None


def classify_transition(config, terms, Ji_Jk, nist_type=None, Aki=None, fik=None, verbose=False):
    """
    Classify a transition as forbidden, semi-forbidden, or permitted.
    Uses NIST type classification when available, with numerical overrides.
    """
    # Handle empty terms or Ji_Jk as permitted transitions (e.g., for H I)
    if not terms or not Ji_Jk:
        return "permitted"

    # Split inputs
    lower_config, upper_config = config.split('-')
    lower_term, upper_term = terms.split('-')
    lower_J, upper_J = map(lambda x: eval(x.strip()), Ji_Jk.split('-'))

    def get_parity(config):
        """
        Calculate parity based on the orbital contributions in the configuration.
        """
        l_values = {'s': 0, 'p': 1, 'd': 2, 'f': 3, 'g': 4, 'h': 5, 'i': 6}
        total_l = 0

        # Extract orbitals and their counts using regex
        orbitals = re.findall(r'([0-9]*)([spdfghi])([0-9]*)', config)
        for count_str, orbital, exponent_str in orbitals:
            count = int(count_str) if count_str else 1
            exponent = int(exponent_str) if exponent_str else 1
            if orbital in l_values:
                l = l_values[orbital]
                total_l += l * exponent
            else:
                print(f"Warning: Unknown orbital '{orbital}' in config '{config}', skipping")

        parity = 'even' if total_l % 2 == 0 else 'odd'
        return parity

    # Get parities
    lower_parity = get_parity(lower_config)
    upper_parity = get_parity(upper_config)

    def parse_term(term):
        """
        Parse a term string to extract spin multiplicity and orbital angular momentum.
        Handles NIST notation with prefixes (a, b, c, x, y, z, etc.) and asterisks.
        """
        # Remove leading lowercase letters (a, b, c, x, y, z, w, etc.) and asterisks
        # These are NIST labels for distinguishing terms, not part of the term symbol
        clean_term = re.sub(r'^[a-z]+', '', term.strip())
        clean_term = clean_term.rstrip('*')

        # Match: multiplicity + L symbol
        match = re.match(r'(\d+)([SPDFGHIKLMN]|\[\d+/?\d*\])', clean_term)
        if match:
            multiplicity = int(match.group(1))
            L_symbol = match.group(2)
            L_values = {
                'S': 0,
                'P': 1,
                'D': 2,
                'F': 3,
                'G': 4,
                'H': 5,
                'I': 6,
                'K': 7,
                'L': 8,
                'M': 9,
                'N': 10,
            }
            if L_symbol in L_values:
                L = L_values[L_symbol]
            elif L_symbol.startswith('[') and L_symbol.endswith(']'):
                L = float(L_symbol.strip('[]'))
            else:
                raise ValueError(f"Unknown term symbol: {L_symbol}")
            S = (multiplicity - 1) / 2
            return S, L
        else:
            raise ValueError(f"Invalid term format: {term}")

    # Extract term information
    try:
        S_lower, L_lower = parse_term(lower_term)
    except Exception as e:
        print(f"Warning: Failed to parse lower term '{lower_term}': {e}. Using default S=0, L=0.")
        S_lower, L_lower = 0, 0
    try:
        S_upper, L_upper = parse_term(upper_term)
    except Exception as e:
        print(f"Warning: Failed to parse upper term '{upper_term}': {e}. Using default S=0, L=0.")
        S_upper, L_upper = 0, 0

    # Calculate selection rules
    delta_S = abs(S_upper - S_lower)
    delta_L = abs(L_upper - L_lower)
    delta_J = abs(upper_J - lower_J)

    # Normalize NIST type
    nist_type = (nist_type or "").strip()
    if nist_type == "--":
        nist_type = "E1"

    if verbose:
        print(
            f"Classifying transition: {config}, {terms}, {Ji_Jk}, NIST type: {nist_type}, Aki: {Aki}, fik: {fik}"
        )
        print(
            f"  Lower parity: {lower_parity}, Upper parity: {upper_parity}, ΔS: {delta_S}, ΔL: {delta_L}, ΔJ: {delta_J}"
        )

    # 0) If NIST explicitly labels a forbidden multipole, respect it
    if nist_type in {"M1", "E2", "M2", "E3"}:
        return "forbidden"

    # 1) If parity changes and NIST implies E1, treat as permitted/intercombination
    if lower_parity != upper_parity and nist_type in {"E1", ""}:
        return "semi-forbidden" if delta_S != 0 else "permitted"

    # 2) Numeric overrides: big A or f + parity change => effectively E1
    if lower_parity != upper_parity:
        if (Aki is not None and Aki >= 1e2) or (fik is not None and fik >= 1e-5):
            return "permitted"

    # 3) Classical fallbacks
    if lower_parity == upper_parity:
        return "forbidden"
    if delta_S != 0:
        return "semi-forbidden"
    if delta_J in {0, 1} and not (lower_J == 0 and upper_J == 0):
        return "permitted"
    return "forbidden"


def get_line_nist(
    ion='H I',
    wave=[4000, 6600],
    tolerance=1.0,
    single=False,
    sortkey=None,
    threshold=None,
    classification=None,
    clear_cache=False,
    multiplet=True,
    verbose=False,
    multiplet_lower_only=False,
    emissitivies=True,
    Te=10_000,
):
    from astroquery.nist import Nist
    import numpy.ma as ma

    if clear_cache:
        Nist.clear_cache()
    if np.iterable(wave):
        minwave, maxwave = wave
    else:
        minwave, maxwave = wave - tolerance, wave + tolerance

    ion = ion.replace(']', '').replace('[', '')
    print(f"Querying NIST for {ion} lines between {minwave} and {maxwave} Å")

    try:
        results = Nist.query(
            minwave << u.angstrom, maxwave << u.angstrom, linename=ion, wavelength_type='vacuum'
        )
    except Exception as e:
        print(f"Failed to query NIST: {e}")
        return None

    if results is None:
        print(f"No results found for {ion} in wavelength range {minwave}-{maxwave} Å")
        return None

    # Clean column names
    clean = lambda text: re.sub(r'\s+', '', text)
    results.rename_columns(results.colnames, [clean(c).replace('.', '') for c in results.colnames])

    # Filter hydrogen series
    if ion == 'H I':
        if verbose:
            print(results)
        ix = [str(r['Upperlevel']).split('|')[0].replace(' ', '').isdigit() for r in results]
        results = results[ix]

    # Helper function to check if value is masked/missing
    def is_masked_or_nan(val):
        """Check if a value is masked, None, empty string, or NaN."""
        if isinstance(val, ma.core.MaskedConstant):
            return True
        if val is None or val == '' or val == '--':
            return True
        try:
            return np.isnan(float(val))
        except (ValueError, TypeError):
            return False

    # Prepare output table
    output = Table(
        names=[
            'key',
            'ion',
            'wave_vac',
            'Ei',
            'Ek',
            'Aki',
            'fik',
            'gigk',
            'configuration',
            'terms',
            'Ji-Jk',
            'type',
            'wave_air',
            'classification',
            'multiplet',
            'line_ratio',
            'multiplet_term',
            'references',
            'note',
        ],
        dtype=[
            'U14',
            'U9',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'U8',
            'U33',
            'U14',
            'U8',
            'U2',
            'float64',
            'U27',
            'int64',
            'float64',
            'U32',
            'U14',
            'U10',
        ],
    )

    for row in results:
        for col in results.colnames:
            # only operate on string-like columns
            if results[col].dtype.kind in ('U', 'S', 'O'):
                if not ma.is_masked(row[col]):
                    row[col] = re.sub(r'[\s?]+', '', row[col])

        row = dict(row)
        if verbose:
            print(row)
        ion_tab = ion

        def parse_level(level):
            """Parse level string, handling multiplet J values (e.g., '0,1,2')."""
            parts = [part.strip() for part in level.split('|')]
            if len(parts) == 3:
                config = parts[0]
                term = parts[1]
                J_str = parts[2]

                # Handle multiplet J values (e.g., '0,1,2')
                if ',' in J_str:
                    # Take the first J value for multiplets
                    J_str = J_str.split(',')[0].strip()

                return config, term, J_str
            else:
                raise ValueError(f"Invalid level format: {level}")

        # Parse Ei and Ek (be defensive: values may have trailing letters like 'u' or flags)
        if is_masked_or_nan(row.get('EiEk', None)):
            continue
        # split on hyphen with optional spaces
        energy_levels = re.split(r"\s*-\s*", str(row['EiEk']))
        if len(energy_levels) < 2:
            continue

        def _extract_float(s: str) -> float:
            """Extract first floating-point number from string s, or return NaN."""
            if s is None:
                return np.nan
            s = str(s)
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if not m:
                return np.nan
            try:
                return float(m.group())
            except Exception:
                return np.nan

        Ei = _extract_float(energy_levels[0].strip().strip('[]').strip('()'))
        Ek = _extract_float(energy_levels[1].strip().strip('[]').strip('()'))
        if not np.isfinite(Ei) or not np.isfinite(Ek):
            # skip entries with non-numeric energy levels
            continue

        # Handle NIST type
        nist_type = 'E1' if str(row['Type']) == '--' else str(row['Type'])

        # Construct key
        if ion == 'H I':
            lower = str(row['Lowerlevel']).split('|')[0].replace(' ', '')
            upper = str(row['Upperlevel']).split('|')[0].replace(' ', '')
            config = f"{lower}-{upper}"
            terms = JiJk = ''
            cval = 'permitted'
            key = construct_hydrogen_key(config)
            if 'gigk' not in row.keys():
                row['gigk'] = None
            if key is None:
                continue
        else:
            # Parse lower and upper levels
            lower_config, lower_terms, lower_J_str = parse_level(row['Lowerlevel'])
            upper_config, upper_terms, upper_J_str = parse_level(row['Upperlevel'])

            config = f"{lower_config}-{upper_config}"
            terms = f"{lower_terms}-{upper_terms}"
            JiJk = f"{lower_J_str}-{upper_J_str}"

            if verbose:
                print(config, terms, JiJk)

            # Extract Aki and fik values, handling masked data
            Aki_val = None if is_masked_or_nan(row['Aki']) else float(row['Aki'])
            fik_val = None if 'fik' not in row.keys() or is_masked_or_nan(row['fik']) else float(row['fik'])

            # Classify transition with NIST type and numerical data
            cval = classify_transition(config, terms, JiJk, nist_type=nist_type, Aki=Aki_val, fik=fik_val)

            # Ion label with brackets
            if cval == 'forbidden':
                ion_tab = f'[{ion}]'
            elif cval == 'semi-forbidden':
                ion_tab = f'{ion}]'

            # get rid of 8224.6+ notation in Ritz
            if type(row['Ritz']) is np.str_:
                row['Ritz'] = float(re.sub(r'[^\d.]', '', row['Ritz']))

            key = ion_tab.replace(' ', '') + f"-{row['Ritz']:.0f}"

        # use of 'ion' is not entirely consistent here: ion is both ion + transition type
        # use same ion code for semi-forbidden as for permitted, for assign_multiplets etc
        if cval == 'semi-forbidden':
            ion_tab = ion_tab.replace(']', '')

        if verbose:
            print(f"Classification: {cval}")

        # Convert vacuum to air wavelength
        wave_air = vacuum_to_air(row['Ritz'])

        # Output type defaults to NIST type
        out_type = nist_type if nist_type else 'E1'

        # Add row to output table
        output.add_row(
            [
                key.replace(' ', ''),
                ion_tab,
                round(row['Ritz'], 3),
                Ei,
                Ek,
                row['Aki'] if not is_masked_or_nan(row['Aki']) else 0.0,
                row['fik'] if 'fik' in row.keys() and not is_masked_or_nan(row['fik']) else 0.0,
                row['gigk'] if row['gigk'] is not None and not is_masked_or_nan(row['gigk']) else '',
                config.replace('.', ''),
                terms,
                JiJk,
                out_type,
                wave_air,
                cval,
                0,
                1.0,
                '',
                'NIST',
                '',
            ]
        )

    # parse gigk like "8-10" into gi,gk and add gf = gi * fik
    if len(output) > 0:
        gf = np.full(len(output), np.nan)
        for i, (gi, fik) in enumerate(output['gigk', 'fik']):
            if ma.is_masked(gi) or ma.is_masked(fik) or gi == '':
                continue
            gf[i] = float(gi.split('-')[0]) * fik

        output['gf'] = gf

    # Filter by classification if requested
    if classification is not None and len(output) > 0 and ion != 'H I':
        output = output[output['classification'] == classification]

    if len(output) == 0:
        print(f"No matching lines found for {ion} in the specified range.")
        return None

    if isinstance(threshold, str):
        output = filter_table(output, threshold)

    if len(output) == 0:
        print(f"No matching lines found for {ion} in the specified range.")
        return None

    if sortkey is not None:
        if sortkey == 'wave_vac':
            dwave = np.abs(output['wave_vac'] - (wave if np.isscalar(wave) else np.mean(wave)))
            output['dwave'] = dwave
            output.sort('dwave')
            output.remove_column('dwave')
        else:
            print('sorting by ', sortkey)
            output.sort(sortkey)
            print(output['key', sortkey])
            if sortkey in ['Aki', 'fik', 'gf']:
                output.reverse()

    # e.g., Select highest Aki value
    if single:
        output = output[0:1]

    if multiplet:
        output = assign_multiplets(output, lower_only=multiplet_lower_only, verbose=verbose)

    if emissitivies:
        output = compute_line_ratios(output, Te=Te, Ne=1e2)

    return output


# calc emission line ratios for a given ion at a given temperature and density
def emissivity_ratios(atom, level, wave_vac, Te=1e4, Ne=1e2, relative=False, tolerance=0.1, verbose=False):
    """
    Compute emissivity for one or more lines, given rest-frame vacuum wavelengths.

    Parameters
    ----------
    atom : str
        Element symbol, e.g. 'H', 'O', 'Fe'.
    level : int
        Ionization stage in PyNeb convention (1=I, 2=II, ...).
    wave_vac : float or array
        Vacuum wavelengths [Å] of the lines you care about.
    Te : float
        Electron temperature [K].
    Ne : float
        Electron density [cm^-3].
    relative : bool
        If True, normalize to max(emissivity) = 1 (ignoring NaNs).
    tolerance : float
        Max allowed |w_req - w_PyNeb| [Å] to consider a match.

    Returns
    -------
    emissivity : ndarray
        Emissivities for each input line (np.nan if no match within tolerance).
    """
    import pyneb as pn

    # ensure array handling
    wave_vac = np.atleast_1d(wave_vac)
    wave_air = vacuum_to_air(wave_vac)

    # Instantiate PyNeb atom
    if atom == 'H' or atom == 'He':
        neb_atom = pn.RecAtom(atom, level)
        neb_atom.case = 'B'
    else:
        neb_atom = pn.Atom(atom, level)

    # Get available line wavelengths (air Å)
    if neb_atom.wave_Ang is None:
        em = neb_atom.getEmissivity(Te, Ne)
        line_waves = np.array(list(em.keys()), dtype='float')
    else:
        line_waves = np.array(neb_atom.wave_Ang, dtype=float)
        if line_waves.ndim == 2:
            line_waves = line_waves[line_waves > 0]  # 1D array of all transition wavelengths

    emissivity = np.full(wave_air.shape, 0.0, dtype=float)

    for i, w in enumerate(wave_air):
        # Nearest PyNeb line
        if verbose and i == 0:
            if line_waves.size > 0:
                print(
                    f"{atom} {level} PyNeb lines={len(line_waves)} "
                    f"range={line_waves.min():.2f}-{line_waves.max():.2f} Te={Te} Ne={Ne}"
                )
            else:
                print(f"{atom} {level} PyNeb lines=0 Te={Te} Ne={Ne}")
        idx = np.argmin(np.abs(line_waves - w))
        dw = abs(line_waves[idx] - w)

        if dw <= tolerance:
            w_neb = line_waves[idx]
            try:
                emissivity[i] = neb_atom.getEmissivity(Te, Ne, wave=w_neb)
            except Exception as e:
                if verbose:
                    print(f"Failed emissivity for {atom} {level} at {w_neb:.2f} Å: {e}")
        else:
            if verbose:
                print(
                    f"No PyNeb line for {atom} {level} within {tolerance:.2f} Å of {w:.2f} Å (closest {line_waves[idx]:.2f} Å, dw={dw:.2f} Å )"
                )

    # Normalize
    if relative:
        finite = emissivity > 0.0
        if finite.any():
            emissivity[finite] /= emissivity[finite].max()

    if verbose:
        print(f"Theoretical ratio {neb_atom.atom} n={len(emissivity)}")
    return emissivity


def hydrogen_ratios(intab, wave=[2000, 1e5], Te=1e4, Ne=1e2, tolerance=1.0):
    """Deprecated: Use compute_line_ratios() instead."""
    warnings.warn(
        "hydrogen_ratios() is deprecated and will be removed in a future version. "
        "Use compute_line_ratios() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    tab = intab.copy()
    ix = np.where((tab['ion'] == 'H I') & (tab['wave_vac'] > wave[0]) & (tab['wave_vac'] < wave[1]))
    if not 'multiplet' in tab.colnames:
        tab['multiplet'] = 0
        tab['line_ratio'] = 0.0

    er = emissivity_ratios('H', 1, np.asarray(tab['wave_vac'][ix]), Te=Te, Ne=Ne, tolerance=tolerance)
    tab['line_ratio'][ix] = er
    if all(tab['multiplet'][ix] == 0):
        tab['multiplet'][ix] = np.max(tab['multiplet']) + 1
    else:
        tab['multiplet'][ix] = np.min(tab['multiplet'][ix])

    return tab


def assign_multiplets(intab, verbose=False, lower_only=False):
    """
    Assign multiplet numbers to transitions with the same configuration and term multiplicity.
    Multiplet numbers are unique within each ion, not globally.

    Parameters:
        tab: Table with columns 'ion', 'configuration', 'terms'
        verbose: Print diagnostic information
        lower_only: If True, group by lower term only (ignoring upper term)

    Returns:
        Table with added 'multiplet' and 'multiplet_key' columns
    """
    tab = intab.copy()
    if 'configuration' not in tab.colnames:
        tab['configuration'] = [''] * len(tab)

    # Initialize multiplet column
    if 'multiplet' not in tab.colnames:
        tab['multiplet'] = 0

    if 'terms' not in tab.colnames:
        print("Table must hav 'terms' columns")
        return tab

    if lower_only:
        # Group by lower term only (extract before the '-')
        # @@@ do we need the full configuration here? or just the term? Sometimes terms are abbreviated eg 6P
        multiplet_term = [
            _build_multiplet_term(c, t, lower_only=True) for c, t in zip(tab['configuration'], tab['terms'])
        ]
    else:
        # Group by full configuration and term (both lower and upper) unless classification suggests otherwise
        # Forbidden: full lower+upper terms. Permitted/semi-forbidden: lower term only.
        if 'classification' in tab.colnames:
            multiplet_term = []
            for c, t, cls in zip(tab['configuration'], tab['terms'], tab['classification']):
                if isinstance(cls, np.ma.core.MaskedConstant):
                    cls_val = ''
                else:
                    cls_val = str(cls).strip().lower()
                use_lower_only = cls_val in {'permitted', 'semi-forbidden'}
                multiplet_term.append(_build_multiplet_term(c, t, lower_only=use_lower_only))
        else:
            multiplet_term = [
                _build_multiplet_term(c, t, lower_only=False)
                for c, t in zip(tab['configuration'], tab['terms'])
            ]

    tab['multiplet_term'] = multiplet_term
    if verbose:
        print('multiplet_term', ' -- ', tab['multiplet_term'])

    # Group by ion first, then assign multiplets within each ion
    ion_groups = tab.group_by('ion')

    for ion_group in ion_groups.groups:
        # Group by multiplet_term within this ion
        multiplet_groups = ion_group.group_by('multiplet_term')

        multiplet_index = 1
        for ig, group in enumerate(multiplet_groups.groups):
            if len(group) > 1:
                if verbose:
                    print(
                        f"Multiplet {multiplet_index}: ion={group['ion'][0]}, "
                        f"config={group['multiplet_term'][0]} n={len(group['wave_vac'])}"
                    )

                # Get indices in original table
                for row in group:
                    mask = (tab['ion'] == row['ion']) & (tab['wave_vac'] == row['wave_vac'])
                    tab['multiplet'][mask] = multiplet_index

                multiplet_index += 1
            else:
                if verbose:
                    print(f"Single line: {group['ion'][0]} {group['multiplet_term'][0]}")
    for t in tab:
        term_val = t['multiplet_term']
        if isinstance(term_val, np.ma.core.MaskedConstant):
            continue
        term_str = str(term_val)
        if '_' in term_str:
            t['multiplet_term'] = term_str.split('_', 1)[1]
    return tab


def compute_line_ratios(
    tab: Table, Te: float = 1e4, Ne: float = 1e2, normalize: str = 'max', verbose: bool = False
) -> Table:
    """
    Compute intensity ratios for all multiplets in table.

    This is the new simplified API for computing line ratios. It replaces
    the previous multiplet_ratios() and calculate_multiplet_emissivities() functions.

    Uses a clear fallback chain for each multiplet:
    1. PyNeb (for H I, He I/II, and forbidden/semi-forbidden lines)
    2. Boltzmann (for permitted lines with atomic data)
    3. Uniform (fallback when data is missing)

    Parameters
    ----------
    tab : Table
        Table with 'multiplet' column (from assign_multiplets)
        Must have columns: 'ion', 'wave_vac', 'multiplet'
    Te : float
        Electron temperature [K], default 10,000 K
    Ne : float
        Electron density [cm^-3], default 100 cm^-3
    normalize : str
        Normalization method: 'max' (strongest=1) or 'sum' (sum=1)
        Default is 'max' for consistency with previous behavior
    verbose : bool
        Print diagnostic information

    Returns
    -------
    Table
        Copy of input table with 'line_ratio' column filled

    Examples
    --------
    >>> from dotfit.emission_lines import EmissionLines, assign_multiplets, compute_line_ratios
    >>> el = EmissionLines()
    >>> tab = el.get_table('[OIII]')
    >>> tab = assign_multiplets(tab)
    >>> tab = compute_line_ratios(tab, Te=10000, Ne=100)
    >>> print(tab['key', 'wave_vac', 'multiplet', 'line_ratio'])
    """
    if 'multiplet' not in tab.colnames:
        raise ValueError("Table must have 'multiplet' column. Run assign_multiplets() first.")

    out = tab.copy()

    # Add line_ratio column if it doesn't exist
    if 'line_ratio' not in out.colnames:
        out['line_ratio'] = 1.0

    # Process each ion separately
    for ion in np.unique(out['ion']):
        ion_mask = out['ion'] == ion

        # Process each multiplet within this ion
        for m in np.unique(out['multiplet'][ion_mask]):
            if m == 0:
                # Single lines keep ratio=1.0
                continue

            mask = ion_mask & (out['multiplet'] == m)
            group = out[mask]

            # Compute ratios using fallback chain
            ratios, method = _compute_group_ratios(group, Te, Ne, verbose)

            if verbose:
                print(f"{ion} multiplet {m}: method={method}, " f"ratios={ratios}, n={len(ratios)}")

            # Apply normalization
            if normalize == 'sum' and np.sum(ratios) > 0:
                ratios = ratios / np.sum(ratios)
            elif normalize == 'max' and np.max(ratios) > 0:
                ratios = ratios / np.max(ratios)

            # Assign back to table
            out['line_ratio'][mask] = ratios

    return out


def calculate_multiplet_ratio(
    tab,
    ion,
    multiplet_number,
    Te=1e4,
    Ne=1e2,
    tolerance=0.1,
    default=1.0,
    relative=True,
    verbose=False,
    return_method=False,
):
    """
    Calculate line intensity ratios for a specific multiplet.

    Parameters:
        tab: Table with 'multiplet', 'ion', 'wave_vac' columns
        ion: Ion name (e.g., 'O III', '[O III]')
        multiplet_number: Multiplet identifier number
        Te: Electron temperature [K]
        Ne: Electron density [cm^-3]
        tolerance: Wavelength matching tolerance [Å]

    Returns:
        Array of line ratios for the multiplet transitions
    """
    # Select the multiplet
    mask = (tab['multiplet'] == multiplet_number) & (tab['ion'] == ion)
    group = tab[mask]

    if len(group) == 0:
        print(f"No multiplet found for ion={ion}, multiplet={multiplet_number}")
        return None

    # Extract atom and ionization level
    atom, level = ion.replace('[', '').replace(']', '').split(' ')

    # Calculate emissivity ratios
    try:
        if _should_use_pyneb(ion, group):
            ratios = emissivity_ratios(
                atom,
                roman_to_int(level),
                np.asarray(group['wave_vac']),
                Te=Te,
                Ne=Ne,
                tolerance=tolerance,
                relative=relative,
                verbose=False,
            )
            # Check if ratios are all NaN or zero (which might happen if PyNeb returns nothing useful)
            if np.all(np.isnan(ratios)) or np.all(ratios == 0):
                raise ValueError("PyNeb returned no valid emissivities")
            method = "pyneb"
        else:
            # Use Boltzmann statistics for permitted lines
            if _has_atomic_data(group):
                wave_vac, Ek, gu, f = _extract_atomic_params(group)
                ratios = _compute_boltzmann_ratios(wave_vac, Ek, gu, f, Te)
                method = "boltzmann"
            else:
                # No atomic data available, use uniform ratios
                ratios = np.full(len(group), default)
                method = "default"

    except Exception as e:
        if verbose:
            print(f"Emissivity calculation failed for {ion}: {e}. Falling back to flat ratios.")
        ratios = np.full(len(group), default)
        method = "default"

    if return_method:
        return ratios, method
    return ratios


def multiplet_ratios(tab, Te=1e4, Ne=1e2, tolerance=0.1, relative=True, verbose=False):
    """
    Calculate line intensity ratios for all multiplets in the table.

    .. deprecated:: 1.0
        Use :func:`compute_line_ratios` instead. This function is kept for
        backward compatibility and will be removed in a future version.

    Parameters
    ----------
    tab : Table
        Table with 'multiplet', 'ion', 'wave_vac' columns already assigned
    Te : float
        Electron temperature [K]
    Ne : float
        Electron density [cm^-3]
    tolerance : float
        Wavelength matching tolerance [Å] (ignored, kept for compatibility)
    relative : bool
        Normalize ratios (ignored, kept for compatibility)
    verbose : bool
        Print diagnostic information

    Returns
    -------
    Table
        Table with updated 'line_ratio' column
    """
    warnings.warn(
        "multiplet_ratios() is deprecated and will be removed in a future version. "
        "Use compute_line_ratios() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_line_ratios(tab, Te=Te, Ne=Ne, normalize='max', verbose=verbose)


def apply_multiplet_rules(
    tab,
    Te_emission=10_000,
    Te_absorption=5_000,
    Ne=1e4,
    tolerance=0.1,
    verbose=False,
    #    except_keys=['CaII-3935', 'CaII-8500'],
    except_keys=['CaII-8500'],
    normalize_multiplet=True,
):
    """
    Assign multiplets and line ratios using classification-aware rules.

    Forbidden lines use lower+upper terms; permitted/semi-forbidden use lower terms only.
    Line ratios use Te_emission for emission lines and Te_absorption for absorption lines.
    """
    out = assign_multiplets(tab)

    if 'line_ratio' not in out.colnames:
        out['line_ratio'] = 0.0

    abs_mask = np.ones(len(out), dtype=bool)

    if 'classification' in out.colnames:
        ltype = np.array(out['classification'])
        for i, val in enumerate(ltype):
            if isinstance(val, np.ma.core.MaskedConstant):
                continue
            if str(val).strip().lower() == 'forbidden':
                abs_mask[i] = False

    em_mask = ~abs_mask

    if np.any(em_mask):
        em_tab = out[em_mask]
        em_tab = compute_line_ratios(em_tab, Te=Te_emission, Ne=Ne, verbose=verbose)
        out['line_ratio'][em_mask] = em_tab['line_ratio']

    if np.any(abs_mask):
        abs_tab = out[abs_mask]
        abs_tab = compute_line_ratios(abs_tab, Te=Te_absorption, Ne=1e2, verbose=verbose)
        out['line_ratio'][abs_mask] = abs_tab['line_ratio']

    if normalize_multiplet and 'multiplet' in out.colnames and 'line_ratio' in out.colnames:
        for ion in np.unique(out['ion']):
            ion_mask = out['ion'] == ion
            for m in np.unique(out['multiplet'][ion_mask]):
                if m <= 0:
                    continue
                m_mask = ion_mask & (out['multiplet'] == m)
                ratios = out['line_ratio'][m_mask]
                if hasattr(ratios, 'mask'):
                    valid = ~ratios.mask & ~np.isnan(ratios)
                else:
                    valid = ~np.isnan(ratios)
                if np.any(valid):
                    max_r = np.max(ratios[valid])
                    if max_r > 0:
                        out['line_ratio'][m_mask] = ratios / max_r

    if 'key' in out.colnames:
        except_keys = set(except_keys)
        for key in out['key']:
            if key not in except_keys:
                continue
            key_mask = out['key'] == key
            if not np.any(key_mask):
                continue
            multiplet_id = out['multiplet'][key_mask][0] if 'multiplet' in out.colnames else 0
            if multiplet_id and multiplet_id != 0:
                out['line_ratio'][out['multiplet'] == multiplet_id] = 1.0
            else:
                out['line_ratio'][key_mask] = 1.0

    if 'references' in out.colnames:
        out.columns.move_to_end('references')
    if 'note' in out.colnames:
        out.columns.move_to_end('note')

    if verbose:
        has_key = 'key' in out.colnames
        has_term = 'multiplet_term' in out.colnames
        for row in out:
            name = row['key'] if has_key else row['ion']
            term = str(row['multiplet_term']) if has_term else ''
            ratio_val = row['line_ratio'] if 'line_ratio' in out.colnames else np.nan
            if np.ma.is_masked(ratio_val) or np.isnan(ratio_val):
                ratio_str = 'nan'
            else:
                ratio_str = f"{float(ratio_val):.6g}"
            print(
                f"{name}, {float(row['wave_vac']):.3f}, m={int(row['multiplet'])}, "
                f"term={term}, ratio={ratio_str}"
            )

    return out


def add_hydrogen_entries(emission_table, hydrogen_table, tolerance=0.27):
    """
    Replace existing hydrogen entries in the emission line table with entries from the hydrogen table.

    .. deprecated:: 1.0
        This function is deprecated and will be removed in a future version.

    Parameters:
        emission_table (Table): The original emission line table.
        hydrogen_table (Table): The table containing hydrogen line data.
        tolerance (float): The wavelength tolerance (in Å) for matching entries.

    Returns:
        Table: The updated emission line table.
    """
    warnings.warn(
        "add_hydrogen_entries() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Mask to mark rows for removal in the emission table
    remove_index = []
    updated_emission_table = emission_table.copy()

    # Process each row in the hydrogen table
    for h_row in hydrogen_table:
        h_wave = h_row['wave_vac']
        config = h_row['configuration']
        key = construct_hydrogen_key(config)

        # Check for a matching entry in the emission table
        for i, e_row in enumerate(emission_table):
            e_wave = e_row['wave_vac']
            if abs(h_wave - e_wave) <= tolerance:
                remove_index.append(i)
                break

        updated_emission_table.add_row(h_row)

    # Remove the marked rows
    updated_emission_table.remove_rows(remove_index)
    updated_emission_table.sort('wave_vac')

    return updated_emission_table


# missing are optical Ca lines
# ------------------------------------------------------------ first time generate table
from astropy.table import Column


def generate_line_table(
    tab_input='/Users/ivo/Desktop/current/agn/agn/data/emission_lines/emission_lines.csv', ndigits=3
):
    """Deprecated legacy generator retained for compatibility."""
    warnings.warn(
        "generate_line_table() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    tab = Table.read(tab_input)

    # add Aki column for transition probabilities
    tab.add_column(Column(np.zeros(len(tab)), name='Aki'), index=4)

    b_air = tab['wave_mix'] > 2000
    tab['wave_vac'] = tab['wave_mix']
    tab['wave_vac'][b_air] = np.round(air_to_vacuum(tab['wave_mix'])[b_air], ndigits)
    tab['wave_air'] = MaskedColumn(tab['wave_mix'], mask=~b_air)

    for t in [tab]:
        for c in t.colnames:
            if isinstance(t[c], MaskedColumn):
                if np.issubdtype(t[c].dtype, str):
                    t[c][t[c].mask] = ''
                if np.issubdtype(t[c].dtype, float):
                    t[c][t[c].mask] = 0.0

    tab['type'][tab['type'].mask] = 'E1'
    for c in ['configuration', 'terms', 'Ji-Jk']:
        tab[c] = [
            #            r if isinstance(r, np.ma.core.MaskedConstant) else r.encode(
            r.encode('ascii', 'ignore').decode('ascii').replace(' ', '')
            for r in tab[c]
        ]

    # add hydrogen lines (legacy placeholder; NIST-based replacement below)
    # htab = hydrogen_lines(ndigits=ndigits)

    # cross check with grizli table

    tab['key'] = [
        replace_greek(i, tex=False).replace(' ', '') + f'-{w:.0f}'
        for i, w in zip(tab['ion'], tab['wave_vac'])
    ]

    tab.remove_columns(['wave_mix', 'creationip'])
    for c in ['wave_vac', 'ion', 'key']:
        tab.columns.move_to_end(c, last=False)

    # add hydrogen lines
    htab = get_line_nist(ion='H I', wave=[900, 1e5])
    tab = add_hydrogen_entries(tab, htab, tolerance=0.27)

    # add grizli lines
    tab = add_grizli_lines(tab)

    # add multiplet ratios
    mtab = multiplet_ratios(tab)

    mtab.write(DEFAULT_EMISSION_LINES_FILE, overwrite=True)
    return mtab


# agn.emission_lines.get_line_nist('O I', agn.emission_lines.air_to_vacuum(7990), single=True)
# mtab = agn.emission_lines.multiplet_ratios(tab)


def add_grizli_lines(tab):
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "add_grizli_lines() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    # no match HeII-5412 H F [5412.5] nearest [FeVI]-5426 5425.728 # not in NIST
    # no match MgII M M [2799.117] nearest MgII-2796 2796.352 -> [MgII]-2803 already in table
    # no match SiIV+OIV-1398 S O [1398.0] nearest OIV]-1397 1397.232 -> [OIV]-1398 already in table
    # no match NI-5199 N F [5199.4] nearest FeII-5199 5199.024 -> [NI]-5202 already in table
    # no match NIII-1750 N N [1750.0] nearest NIII]-1749 1748.656 -> [NIII]-1750 already in table
    # no match NV-1240 N N [1240.81] nearest NV-1239 1238.821 -> [NV]-1240 already in table
    lines = {
        'O I': [5578, 7990, 11290.00, 13168.4, 7777.5],
        'He I': [6680, 10832],
        'Ne IV': [2425, 2422],
        'Na I': [5891, 5897],
        'Ca II': [3934.78, 3969.591],
    }

    # Loop through the dictionary and call the function
    ion_list = [
        get_line_nist(ion, w, tolerance=1, single=True)
        for ion, wavelengths in lines.items()
        for w in wavelengths
    ]

    new_tab = vstack([tab] + ion_list)
    #    new_tab.sort('wave_vac')
    return new_tab


def compare_grizli_entries(tab, tolerance=1.0):
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "compare_grizli_entries() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    from . import models

    lw, lr = models.get_line_wavelengths()
    # double check existence of each single length line entry in the grizli lw, lr dictionaries
    element = np.asarray([i.replace('[', '').replace(']', '')[0] for i in tab['key']])
    wave = np.asarray(tab['wave_vac'])

    for k in lw:
        if len(lw[k]) > 1:
            continue
        dw = lw[k][0] - wave
        imin = np.argmin(np.abs(dw))
        #        print(k, lw[k], element[imin], tab['ion'][imin], element[imin] == k[0], dw[imin], dw[imin] < tolerance)
        grizli_key = k.replace('[', '').replace(']', '')
        if (element[imin] == k[0]) and (dw[imin] < tolerance):
            print(f" match {k} {k[0]} {lw[k][0]} {tab['key'][imin]} {tab['wave_vac'][imin]}")
        else:
            print(
                'no match',
                k,
                k[0],
                element[imin],
                lw[k],
                f"nearest {tab['key'][imin]} {tab['wave_vac'][imin]}",
            )
    return element


# get nearest value in dict of lists.
# search among the first element of the list
# only consider lists with <= max_len
def find_nearest_key(lw_dict, value, min_len=1, max_len=3, **kwargs):
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "find_nearest_key() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    singledict = single_line_dict(lw_dict, min_len=1, max_len=3, atol=0.1)
    #    print(singledict['OI-6302'])
    #    singledict = {k: v[0] for k, v in dictionary.items() if len(v) >= min_len and len(v) <= max_len}
    # singledict = {k: v[0] for k, v in dictionary.items() if len(v) == min_len}
    return min(singledict.keys(), key=lambda k: abs(singledict[k][0] - value))


def single_line_dict(lw_dict, min_len=1, max_len=3, atol=0.1, verbose=False):
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "single_line_dict() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    singledict = {k: [v[0]] for k, v in lw_dict.items() if len(v) == 1}
    sdv = np.array(list(singledict.values()))
    for k, v in lw_dict.items():
        if len(v) > min_len and len(v) < max_len:
            for vv in v:
                close = np.isclose(sdv, vv, atol=atol)
                if not any(close):
                    if verbose:
                        print(k, vv, ' not found in single dict, adding ', k.split('-')[0] + f'-{vv:.0f}')
                    singledict[k.split('-')[0] + f'-{vv:.0f}'] = [vv]
    return singledict


def get_line_keys(lw, line_complex, **kwargs):
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "get_line_keys() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    return [find_nearest_key(lw, w, **kwargs) for w in lw[line_complex]]


def get_line_wavelengths(multiplet=True):
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "get_line_wavelengths() is deprecated and will be removed in a future version. "
        "Use EmissionLines().get_line_wavelengths() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return EmissionLines().get_line_wavelengths(multiplet=multiplet)


def get_line_list():
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "get_line_list() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    lw, lr = get_line_wavelengths()
    ln = {k: get_line_keys(lw, k) for k in lw}
    return lw, lr, ln


def unique_lines():
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "unique_lines() is deprecated and will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    lw, lr = get_line_wavelengths()
    uw = list(set([w for k in lw for w in lw[k]]))
    uw.sort()
    un = [find_nearest_key(lw, w) for w in uw]
    return {find_nearest_key(lw, w): w for w in uw}, uw, un


def cdf(wave, flux):
    """Deprecated legacy helper retained for compatibility."""
    warnings.warn(
        "cdf() is deprecated and will be removed in a future version.", DeprecationWarning, stacklevel=2
    )
    norm = np.trapezoid(flux, wave)
    if norm == 0:
        return np.zeros_like(wave)
    else:
        return np.cumsum(flux / np.trapezoid(flux, wave) * np.gradient(wave))


def calculate_multiplet_emissivities(tab, Te=10_000, default=1.0, verbose=False):
    """
    Calculate relative line emissivities for multiplets using optically-thin approximation.

    .. deprecated:: 1.0
        Use :func:`compute_line_ratios` instead. This function is kept for
        backward compatibility and will be removed in a future version.

    Parameters
    ----------
    tab : Table
        Must contain: 'ion', 'multiplet', 'wave_vac', 'Ek' (upper energy in eV),
        and either 'gf' or ('fik' and 'gigk')
    Te : float
        Excitation temperature [K]
    default : float
        Default ratio for single lines (ignored, kept for compatibility)
    verbose : bool
        Print diagnostic information

    Returns
    -------
    Table
        Input table with added 'line_ratio' column (normalized within each multiplet)
    """
    warnings.warn(
        "calculate_multiplet_emissivities() is deprecated and will be removed in a future version. "
        "Use compute_line_ratios() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_line_ratios(tab, Te=Te, Ne=1e2, normalize='max', verbose=verbose)


# adopted from pyqso
continuum_windows = [
    (1150.0, 1170.0),
    (1275.0, 1290.0),
    (1350.0, 1360.0),
    (1445.0, 1465.0),
    (1690.0, 1705.0),
    (1770.0, 1810.0),
    (1970.0, 2400.0),
    (2480.0, 2675.0),
    (2925.0, 3400.0),
    (3775.0, 3832.0),
    (4000.0, 4050.0),
    (4200.0, 4230.0),
    (4435.0, 4640.0),
    (5100.0, 5535.0),
    (6005.0, 6035.0),
    (6110.0, 6250.0),
    (6800.0, 7000.0),
    (7160.0, 7180.0),
    (7500.0, 7800.0),
    (8050.0, 8150.0),
]


def replace_greek(text, tex=True):
    # Dictionary mapping Unicode Greek letters to LaTeX representations
    greek_to_latex = {
        "α": r"$\alpha$",
        "β": r"$\beta$",
        "γ": r"$\gamma$",
        "δ": r"$\delta$",
        "ε": r"$\epsilon$",
        "ζ": r"$\zeta$",
        "η": r"$\eta$",
        "θ": r"$\theta$",
        "ι": r"$\iota$",
        "κ": r"$\kappa$",
        "λ": r"$\lambda$",
        "μ": r"$\mu$",
        "ν": r"$\nu$",
        "ξ": r"$\xi$",
        "ο": r"$o$",
        "π": r"$\pi$",
        "ρ": r"$\rho$",
        "σ": r"$\sigma$",
        "τ": r"$\tau$",
        "υ": r"$\upsilon$",
        "φ": r"$\phi$",
        "χ": r"$\chi$",
        "ψ": r"$\psi$",
        "ω": r"$\omega$",
        "Α": r"$\Alpha$",
        "Β": r"$\Beta$",
        "Γ": r"$\Gamma$",
        "Δ": r"$\Delta$",
        "Ε": r"$E$",
        "Ζ": r"$Z$",
        "Η": r"$H$",
        "Θ": r"$\Theta$",
        "Ι": r"$I$",
        "Κ": r"$K$",
        "Λ": r"$\Lambda$",
        "Μ": r"$M$",
        "Ν": r"$N$",
        "Ξ": r"$\Xi$",
        "Ο": r"$O$",
        "Π": r"$\Pi$",
        "Ρ": r"$P$",
        "Σ": r"$\Sigma$",
        "Τ": r"$T$",
        "Υ": r"$\Upsilon$",
        "Φ": r"$\Phi$",
        "Χ": r"$X$",
        "Ψ": r"$Ψ$",
        "Ω": r"$\Omega$",
    }

    # Replace each Greek letter in the text
    if tex:
        for greek, latex in greek_to_latex.items():
            text = text.replace(greek, latex)
    else:
        for greek, latex in greek_to_latex.items():
            text = text.replace(greek, '')

    return text


def replace_brackets_with_dollars(text):
    text = text.replace("[", r"$[$")
    text = text.replace("]", r"$]$")
    return text.replace('$$', '')
