"""Broken modified blackbody continuum fitting utilities.

This module provides JAX-accelerated helpers to model ultraviolet and optical
continua with a broken modified blackbody (greybody) representation. A single
break wavelength separates two sets of emissivity indices, temperatures and
amplitudes that are sampled with NumPyro.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import distributions as dist
from numpyro import infer, sample
from numpyro.infer import MCMC, Trace_ELBO, autoguide, init_to_median, init_to_value
from numpyro.optim import Adam
from scipy import optimize

logger = logging.getLogger(__name__)

# Default break wavelength (micron) separating UV and optical segments
DEFAULT_BREAK_MICRON = 0.3645

# Physical constants for Planck function (SI units)
_H = 6.62607015e-34
_C = 2.99792458e8
_KB = 1.380649e-23


@jax.jit
def _safe_log_expm1(x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute log(exp(x) - 1) with numerically stable gradients.

    For large x: log(exp(x) - 1) ≈ x
    For small x: log(exp(x) - 1) ≈ log(x)
    Uses smooth sigmoid transition to avoid discontinuous gradients.
    """
    # Wider transition zone: blend from x=5 to x=15
    alpha = jax.nn.sigmoid((x - 10.0) / 3.0)

    large_x = x  # Asymptotic form
    small_x = jnp.log(jnp.maximum(jnp.expm1(x), 1e-100))  # Floor to avoid -inf

    return alpha * large_x + (1 - alpha) * small_x


@jax.jit
def _log_planck_ratio(wavelength_m: jnp.ndarray, pivot_m: float, temperature_k: jnp.ndarray) -> jnp.ndarray:
    """
    Compute log(B_lambda(wavelength) / B_lambda(pivot)) with stable gradients.

    This avoids computing log(Planck) separately which causes gradient issues.
    Instead, we compute the ratio directly in log space.
    """
    # Clip temperature to safe range
    temperature_k = jnp.clip(temperature_k, 500.0, 1e6)

    # Exponents: hc/(λkT)
    x_wave = (_H * _C) / (wavelength_m * _KB * temperature_k)
    x_pivot = (_H * _C) / (pivot_m * _KB * temperature_k)

    # Clip to safe range (tighter bounds for stability)
    x_wave = jnp.clip(x_wave, 0.01, 200.0)
    x_pivot = jnp.clip(x_pivot, 0.01, 200.0)

    # log(B_λ1 / B_λ2) = -5*log(λ1/λ2) + log((exp(x2)-1)/(exp(x1)-1))
    #                   = -5*log(λ1/λ2) + log(exp(x2)-1) - log(exp(x1)-1)
    log_wavelength_ratio = -5.0 * jnp.log(wavelength_m / pivot_m)
    log_expm1_diff = _safe_log_expm1(x_pivot) - _safe_log_expm1(x_wave)

    # Clip final result to prevent overflow in exp later
    return jnp.clip(log_wavelength_ratio + log_expm1_diff, -100.0, 100.0)


@jax.jit
def modified_blackbody(
    wavelength_micron: jnp.ndarray,
    temperature_k: jnp.ndarray,
    emissivity_index: jnp.ndarray,
    pivot_micron: float,
) -> jnp.ndarray:
    r"""Compute a normalized modified blackbody spectrum with stable gradients."""
    # Clip inputs to safe ranges
    temperature_k = jnp.clip(temperature_k, 500.0, 1e6)
    emissivity_index = jnp.clip(emissivity_index, -10.0, 10.0)

    wavelength_m = wavelength_micron * 1e-6
    pivot_m = pivot_micron * 1e-6

    # Use gradient-safe log-ratio computation
    log_planck_ratio = _log_planck_ratio(wavelength_m, pivot_m, temperature_k)

    # Emissivity term in log space to avoid overflow
    log_emissivity = -emissivity_index * jnp.log(wavelength_m / pivot_m)
    log_emissivity = jnp.clip(log_emissivity, -100.0, 100.0)

    return jnp.exp(log_emissivity + log_planck_ratio)


@jax.jit
def single_modified_blackbody(
    wavelength_micron: jnp.ndarray,
    amplitude: jnp.ndarray,
    temperature_k: jnp.ndarray,
    emissivity_index: jnp.ndarray,
    pivot_micron: float = DEFAULT_BREAK_MICRON,
) -> jnp.ndarray:
    """Evaluate a single modified blackbody continuum.

    Args:
        wavelength_micron: Wavelength array in microns.
        amplitude: Overall scaling applied at the pivot wavelength.
        temperature_k: Blackbody temperature in Kelvin.
        emissivity_index: Emissivity power-law index (beta).
        pivot_micron: Pivot wavelength used for normalization.

    Returns:
        Flux density evaluated at ``wavelength_micron``.
    """

    return amplitude * modified_blackbody(wavelength_micron, temperature_k, emissivity_index, pivot_micron)


@jax.jit
def broken_modified_blackbody(
    wavelength_micron: jnp.ndarray,
    break_micron: float,
    blue_params: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    red_params: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """Evaluate a broken modified blackbody model.

    Args:
        wavelength_micron: Wavelengths in microns.
        break_micron: Break wavelength in microns separating the two segments.
        blue_params: Tuple of (amplitude, temperature, emissivity index) used
            for wavelengths shorter than ``break_micron``.
        red_params: Tuple of (amplitude, temperature, emissivity index) used for
            wavelengths longer than or equal to ``break_micron``.

    Returns:
        Model flux density evaluated at ``wavelength_micron``.
    """
    amp_blue, temp_blue, beta_blue = blue_params
    amp_red, temp_red, beta_red = red_params

    # Compute both branches
    blue = amp_blue * modified_blackbody(wavelength_micron, temp_blue, beta_blue, break_micron)
    red = amp_red * modified_blackbody(wavelength_micron, temp_red, beta_red, break_micron)

    # Use jnp.where - JAX will compute gradients correctly for each branch
    return jnp.where(wavelength_micron < break_micron, blue, red)


def _broken_modified_bb_model(
    wavelength_micron: jnp.ndarray, flux: jnp.ndarray, flux_uncertainty: jnp.ndarray, break_micron: float
) -> None:
    """NumPyro model definition for the broken modified blackbody."""

    # More conservative priors with higher minimum temperatures
    amp_blue = sample("amplitude_blue", dist.LogNormal(0.0, 1.0))
    amp_red = sample("amplitude_red", dist.LogNormal(0.0, 1.0))

    # Increase minimum temperature to avoid numerical issues at optical wavelengths
    temp_blue = sample("temperature_blue", dist.Uniform(5_000.0, 50_000.0))
    temp_red = sample("temperature_red", dist.Uniform(3_000.0, 20_000.0))

    beta_blue = sample("beta_blue", dist.Uniform(-1.0, 4.0))
    beta_red = sample("beta_red", dist.Uniform(-1.0, 4.0))

    model_flux = broken_modified_blackbody(
        wavelength_micron, break_micron, (amp_blue, temp_blue, beta_blue), (amp_red, temp_red, beta_red)
    )
    sample("obs", dist.Normal(model_flux, flux_uncertainty), obs=flux)


@dataclass
class BrokenModifiedBBResult:
    """Container for broken modified blackbody inference results."""

    wavelengths: jnp.ndarray
    flux: jnp.ndarray
    flux_uncertainty: jnp.ndarray
    flux_scale: float
    break_wavelength: float
    samples: Mapping[str, jnp.ndarray]
    mcmc: MCMC | None

    def median_parameters(self) -> Mapping[str, float]:
        """Return posterior medians for each sampled parameter."""

        medians = {name: float(jnp.median(values)) for name, values in self.samples.items()}
        for amp_key in ("amplitude_blue", "amplitude_red"):
            if amp_key in medians:
                medians[amp_key] *= self.flux_scale

        return medians


def fit_broken_modified_blackbody(
    wavelength_micron: jnp.ndarray,
    flux: jnp.ndarray,
    flux_uncertainty: jnp.ndarray,
    break_micron: float = DEFAULT_BREAK_MICRON,
    num_samples: int = 1500,
    num_warmup: int = 500,
    rng_key: jax.Array | None = None,
) -> BrokenModifiedBBResult:
    """Fit a broken modified blackbody continuum using NumPyro NUTS.

    Args:
        wavelength_micron: Rest-frame wavelengths in microns.
        flux: Observed flux density matching ``wavelength_micron``.
        flux_uncertainty: 1-sigma uncertainties on ``flux``.
        break_micron: Wavelength where the model transitions between blue and
            red components.
        num_samples: Number of posterior samples to draw after warmup.
        num_warmup: Number of warmup steps for NUTS.
        rng_key: Optional JAX PRNG key. If ``None`` a new key is created.

    Returns:
        :class:`BrokenModifiedBBResult` containing posterior samples and helper
        metadata.
    """

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    wavelengths = jnp.asarray(wavelength_micron)
    target_flux = jnp.asarray(flux)
    target_unc = jnp.asarray(flux_uncertainty)
    flux_scale = jnp.maximum(jnp.quantile(jnp.abs(target_flux), 0.9), 1.0)
    target_flux = target_flux / flux_scale
    target_unc = jnp.maximum(target_unc / flux_scale, 5e-2)

    logger.info("Starting broken modified blackbody fit with %d samples", num_samples)
    kernel = infer.NUTS(_broken_modified_bb_model, init_strategy=init_to_median(num_samples=10))
    mcmc = infer.MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, progress_bar=False)
    flux_scale_out = float(flux_scale)
    pivot_flux = jnp.interp(break_micron, wavelengths, target_flux)
    heuristic_params = {
        "amplitude_blue": jnp.maximum(0.05, pivot_flux),
        "amplitude_red": jnp.maximum(0.05, pivot_flux),
        "temperature_blue": jnp.array(10_000.0),
        "temperature_red": jnp.array(5_000.0),
        "beta_blue": jnp.array(1.0),
        "beta_red": jnp.array(1.0),
    }

    try:
        mcmc.run(
            rng_key,
            wavelength_micron=wavelengths,
            flux=target_flux,
            flux_uncertainty=target_unc,
            break_micron=break_micron,
        )
    except RuntimeError:
        logger.warning("Falling back to heuristic initialization for broken modified blackbody fit")
        fallback_kernel = infer.NUTS(
            _broken_modified_bb_model, init_strategy=init_to_value(values=heuristic_params)
        )
        mcmc = infer.MCMC(fallback_kernel, num_samples=num_samples, num_warmup=num_warmup, progress_bar=False)
        try:
            mcmc.run(
                rng_key,
                wavelength_micron=wavelengths,
                flux=target_flux,
                flux_uncertainty=target_unc,
                break_micron=break_micron,
                init_params=heuristic_params,
            )
        except RuntimeError:
            logger.warning("Switching to AutoNormal SVI fallback for broken modified blackbody fit")
            guide = autoguide.AutoNormal(
                lambda wavelength_micron, flux, flux_uncertainty, break_micron: _broken_modified_bb_model(
                    wavelength_micron, flux, flux_uncertainty, break_micron
                )
            )
            svi = infer.SVI(
                _broken_modified_bb_model,
                guide,
                Adam(5e-3),
                Trace_ELBO(),
                wavelength_micron=wavelengths,
                flux=target_flux,
                flux_uncertainty=target_unc,
                break_micron=break_micron,
            )
            try:
                svi_result = svi.run(rng_key, 600)
                predictive = infer.Predictive(
                    _broken_modified_bb_model, guide=guide, params=svi_result.params, num_samples=num_samples
                )
                samples = predictive(
                    rng_key,
                    wavelength_micron=wavelengths,
                    flux=target_flux,
                    flux_uncertainty=target_unc,
                    break_micron=break_micron,
                )
            except RuntimeError:
                logger.warning(
                    "Attempting deterministic least-squares fallback for broken modified blackbody fit"
                )

                def _residual(theta: np.ndarray) -> np.ndarray:
                    amp_b, temp_b, beta_b, amp_r, temp_r, beta_r = theta
                    model_val = broken_modified_blackbody(
                        wavelengths, break_micron, (amp_b, temp_b, beta_b), (amp_r, temp_r, beta_r)
                    )
                    return np.asarray((model_val - target_flux) / target_unc)

                theta0 = np.array(
                    [
                        float(heuristic_params["amplitude_blue"]),
                        float(heuristic_params["temperature_blue"]),
                        float(heuristic_params["beta_blue"]),
                        float(heuristic_params["amplitude_red"]),
                        float(heuristic_params["temperature_red"]),
                        float(heuristic_params["beta_red"]),
                    ]
                )
                bounds = (
                    [0.0, 2_000.0, -1.0, 0.0, 2_000.0, -1.0],
                    [100.0, 15_000.0, 4.0, 100.0, 15_000.0, 4.0],
                )

                try:
                    ls_result = optimize.least_squares(_residual, theta0, bounds=bounds, method="trf")
                    theta_best = ls_result.x
                    logger.info("Least-squares fallback converged with cost %.3f", float(ls_result.cost))
                    model_flux = broken_modified_blackbody(
                        wavelengths,
                        break_micron,
                        (theta_best[0], theta_best[1], theta_best[2]),
                        (theta_best[3], theta_best[4], theta_best[5]),
                    )
                    samples = {
                        "amplitude_blue": jnp.repeat(theta_best[0], num_samples),
                        "amplitude_red": jnp.repeat(theta_best[3], num_samples),
                        "temperature_blue": jnp.repeat(theta_best[1], num_samples),
                        "temperature_red": jnp.repeat(theta_best[4], num_samples),
                        "beta_blue": jnp.repeat(theta_best[2], num_samples),
                        "beta_red": jnp.repeat(theta_best[5], num_samples),
                        "obs": jnp.repeat(model_flux[None, :], num_samples, axis=0),
                    }
                    flux_scale_out = float(flux_scale)
                    return BrokenModifiedBBResult(
                        wavelengths=wavelengths,
                        flux=target_flux,
                        flux_uncertainty=target_unc,
                        flux_scale=flux_scale_out,
                        break_wavelength=break_micron,
                        samples=samples,
                        mcmc=None,
                    )
                except Exception:
                    logger.warning("Deterministic heuristic sampling used for broken modified blackbody fit")
                    model_flux = broken_modified_blackbody(
                        wavelengths,
                        break_micron,
                        (
                            heuristic_params["amplitude_blue"],
                            heuristic_params["temperature_blue"],
                            heuristic_params["beta_blue"],
                        ),
                        (
                            heuristic_params["amplitude_red"],
                            heuristic_params["temperature_red"],
                            heuristic_params["beta_red"],
                        ),
                    )
                    samples = {
                        "amplitude_blue": jnp.repeat(heuristic_params["amplitude_blue"], num_samples),
                        "amplitude_red": jnp.repeat(heuristic_params["amplitude_red"], num_samples),
                        "temperature_blue": jnp.repeat(heuristic_params["temperature_blue"], num_samples),
                        "temperature_red": jnp.repeat(heuristic_params["temperature_red"], num_samples),
                        "beta_blue": jnp.repeat(heuristic_params["beta_blue"], num_samples),
                        "beta_red": jnp.repeat(heuristic_params["beta_red"], num_samples),
                        "obs": jnp.repeat(model_flux[None, :], num_samples, axis=0),
                    }

            samples["amplitude_blue"] = samples["amplitude_blue"] * flux_scale
            samples["amplitude_red"] = samples["amplitude_red"] * flux_scale
            flux_scale_out = 1.0

            return BrokenModifiedBBResult(
                wavelengths=wavelengths,
                flux=target_flux,
                flux_uncertainty=target_unc,
                flux_scale=flux_scale_out,
                break_wavelength=break_micron,
                samples=samples,
                mcmc=None,
            )

    samples = mcmc.get_samples()
    return BrokenModifiedBBResult(
        wavelengths=wavelengths,
        flux=target_flux,
        flux_uncertainty=target_unc,
        flux_scale=flux_scale_out,
        break_wavelength=break_micron,
        samples=samples,
        mcmc=mcmc,
    )
