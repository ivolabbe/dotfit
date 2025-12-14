"""Streamlined spectral fitting utilities inspired by the UNITE engine.

This module implements a compact, readable interface for fitting emission-line
groups together with a modified blackbody continuum (single or broken). It is
intentionally minimal yet expressive enough to cover common near-IR applications such as
NIRSpec G235M spectra. The fitting backend uses NumPyro for robust sampling and
JAX for fast evaluation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpyro import distributions as dist
from numpyro import infer, sample
from numpyro.infer.initialization import init_to_feasible

from .broken_modified_bb import (
    DEFAULT_BREAK_MICRON,
    broken_modified_blackbody,
    single_modified_blackbody,
)

logger = logging.getLogger(__name__)

_C_KMS = 299_792.458


@dataclass
class LineDefinition:
    """Simple description of a single spectral line.

    Attributes:
        ion: Ion label (e.g. ``"[O III]"``).
        wavelength: Rest-frame vacuum wavelength in Angstrom.
        key: Optional unique name used in output parameters.
        line_ratio: Optional relative intensity used when ratios are fixed.
        multiplet: Optional multiplet identifier for keeping line ratios tied.
    """

    ion: str
    wavelength: float
    key: str | None = None
    line_ratio: float | None = None
    multiplet: str | None = None

    def to_dict(self) -> Mapping[str, str | float | None]:
        return {
            "ion": self.ion,
            "wavelength": float(self.wavelength),
            "key": self.key,
            "line_ratio": None if self.line_ratio is None else float(self.line_ratio),
            "multiplet": self.multiplet,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, str | float | None]) -> "LineDefinition":
        return cls(
            ion=str(data["ion"]),
            wavelength=float(data["wavelength"]),
            key=None if data.get("key") is None else str(data.get("key")),
            line_ratio=None if data.get("line_ratio") is None else float(data.get("line_ratio")),
            multiplet=None if data.get("multiplet") is None else str(data.get("multiplet")),
        )


@dataclass
class LineGroupSpec:
    """Configuration for a group of lines sharing kinematic parameters."""

    name: str
    lines: Sequence[LineDefinition]
    allow_negative: bool = False
    use_multiplet_ties: bool = True
    velocity_sigma: float = 400.0
    width_logmean: float = float(jnp.log(150.0))
    width_logstd: float = 0.35
    amplitude_scale: float = 5.0

    def to_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "lines": [ln.to_dict() for ln in self.lines],
            "allow_negative": self.allow_negative,
            "use_multiplet_ties": self.use_multiplet_ties,
            "velocity_sigma": self.velocity_sigma,
            "width_logmean": self.width_logmean,
            "width_logstd": self.width_logstd,
            "amplitude_scale": self.amplitude_scale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LineGroupSpec":
        return cls(
            name=str(data["name"]),
            lines=[LineDefinition.from_dict(d) for d in data["lines"]],
            allow_negative=bool(data.get("allow_negative", False)),
            use_multiplet_ties=bool(data.get("use_multiplet_ties", True)),
            velocity_sigma=float(data.get("velocity_sigma", 400.0)),
            width_logmean=float(data.get("width_logmean", float(jnp.log(150.0)))),
            width_logstd=float(data.get("width_logstd", 0.35)),
            amplitude_scale=float(data.get("amplitude_scale", 5.0)),
        )


@dataclass
class ContinuumConfig:
    """Settings for the continuum model used in simple UNITE fits."""

    enabled: bool = True
    model: str = "single"
    break_micron: float = DEFAULT_BREAK_MICRON
    amplitude_logmean: float = float(jnp.log(1.0))
    amplitude_logstd: float = 1.0
    beta_bounds: tuple[float, float] = (-2.0, 4.0)
    temperature_bounds: tuple[float, float] = (2_000.0, 15_000.0)

    def __post_init__(self) -> None:
        model = self.model.lower()
        if model not in {"single", "broken"}:
            raise ValueError("continuum.model must be 'single' or 'broken'")
        self.model = model

    def to_dict(self) -> Mapping[str, object]:
        return {
            "enabled": self.enabled,
            "model": self.model,
            "break_micron": self.break_micron,
            "amplitude_logmean": self.amplitude_logmean,
            "amplitude_logstd": self.amplitude_logstd,
            "beta_bounds": self.beta_bounds,
            "temperature_bounds": self.temperature_bounds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ContinuumConfig":
        return cls(
            enabled=bool(data.get("enabled", True)),
            model=str(data.get("model", "single")),
            break_micron=float(data.get("break_micron", DEFAULT_BREAK_MICRON)),
            amplitude_logmean=float(data.get("amplitude_logmean", float(jnp.log(1.0)))),
            amplitude_logstd=float(data.get("amplitude_logstd", 1.0)),
            beta_bounds=tuple(data.get("beta_bounds", (-2.0, 4.0))),
            temperature_bounds=tuple(data.get("temperature_bounds", (2_000.0, 15_000.0))),
        )


@dataclass
class FitConfig:
    """Complete configuration for a UNITE-style fit."""

    line_groups: Sequence[LineGroupSpec]
    continuum: ContinuumConfig = field(default_factory=ContinuumConfig)
    velocity_systemic_sigma: float = 600.0
    mask_regions: Sequence[tuple[float, float]] = field(default_factory=tuple)

    def to_json(self) -> str:
        return json.dumps(
            {
                "line_groups": [grp.to_dict() for grp in self.line_groups],
                "continuum": self.continuum.to_dict(),
                "velocity_systemic_sigma": self.velocity_systemic_sigma,
                "mask_regions": list(self.mask_regions),
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> "FitConfig":
        raw = json.loads(payload)
        return cls(
            line_groups=[LineGroupSpec.from_dict(g) for g in raw["line_groups"]],
            continuum=ContinuumConfig.from_dict(raw.get("continuum", {})),
            velocity_systemic_sigma=float(raw.get("velocity_systemic_sigma", 600.0)),
            mask_regions=[tuple(region) for region in raw.get("mask_regions", [])],
        )


@dataclass
class FitResult:
    """Container for posterior samples and helper predictions."""

    wavelengths: np.ndarray
    flux: np.ndarray
    flux_uncertainty: np.ndarray
    mask: np.ndarray
    samples: Mapping[str, np.ndarray]
    log_likelihood: np.ndarray | None
    config_json: str

    def save_npz(self, path: str | Path) -> None:
        payload = {
            "wavelengths": self.wavelengths,
            "flux": self.flux,
            "flux_uncertainty": self.flux_uncertainty,
        }
        payload.update({"sample_" + k: v for k, v in self.samples.items()})
        payload["mask"] = self.mask
        payload["log_likelihood"] = self.log_likelihood if self.log_likelihood is not None else np.array([])
        payload["config_json"] = np.array(self.config_json)
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> "FitResult":
        data = np.load(path, allow_pickle=True)
        sample_keys = [k for k in data.keys() if k.startswith("sample_")]
        samples = {k.replace("sample_", ""): data[k] for k in sample_keys}
        log_likelihood = data["log_likelihood"]
        if log_likelihood.size == 0:
            log_likelihood = None
        return cls(
            wavelengths=data["wavelengths"],
            flux=data["flux"],
            flux_uncertainty=data["flux_uncertainty"],
            mask=data["mask"].astype(bool),
            samples=samples,
            log_likelihood=log_likelihood,
            config_json=str(data["config_json"].item()),
        )

    def posterior_median(self) -> Mapping[str, float]:
        return {name: float(np.median(values)) for name, values in self.samples.items()}

    def config(self) -> FitConfig:
        return FitConfig.from_json(self.config_json)


def load_line_table(path: str | Path) -> list[LineDefinition]:
    """Load emission lines from a CSV file with UNITE-friendly columns."""

    df = pd.read_csv(path)
    if "ion" not in df or "wave_vac" not in df:
        raise ValueError("CSV table must contain 'ion' and 'wave_vac' columns")

    lines = []
    for idx, row in df.iterrows():
        key = str(row.get("key", f"line_{idx}"))
        lines.append(
            LineDefinition(
                ion=str(row["ion"]),
                wavelength=float(row["wave_vac"]),
                key=key,
                line_ratio=(
                    None
                    if "line_ratio" not in row or pd.isna(row["line_ratio"])
                    else float(row["line_ratio"])
                ),
                multiplet=(
                    None if "multiplet" not in row or pd.isna(row["multiplet"]) else str(row["multiplet"])
                ),
            )
        )

    return lines


def _apply_mask(wavelengths: jnp.ndarray, flux: jnp.ndarray, flux_uncertainty: jnp.ndarray, mask: np.ndarray):
    return wavelengths[mask], flux[mask], flux_uncertainty[mask]


def _gaussian_profile(wavelengths: jnp.ndarray, center: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-0.5 * ((wavelengths[:, None] - center[None, :]) / sigma[None, :]) ** 2)


def _sanitize_multiplet(label: str) -> str:
    return label.replace(" ", "_").replace("[", "").replace("]", "")


def _line_amplitudes(
    group: LineGroupSpec, parameter_prefix: str
) -> tuple[list[tuple[int, str]], dict[str, tuple[str, list[int]]]]:
    multiplet_map: dict[str, list[int]] = {}
    per_line_params: list[tuple[int, str]] = []
    if group.use_multiplet_ties:
        for idx, line in enumerate(group.lines):
            if line.line_ratio is not None and line.multiplet:
                multiplet_map.setdefault(line.multiplet, []).append(idx)

    tied = {mp: (f"{parameter_prefix}_{_sanitize_multiplet(mp)}", idxs) for mp, idxs in multiplet_map.items()}

    for idx, line in enumerate(group.lines):
        if line.multiplet in multiplet_map:
            continue
        per_line_params.append((idx, f"{parameter_prefix}_{idx}"))

    return per_line_params, tied


def _unite_model(wavelengths: jnp.ndarray, flux: jnp.ndarray, flux_uncertainty: jnp.ndarray, cfg: FitConfig):
    velocity_systemic = sample("velocity_systemic", dist.Normal(0.0, cfg.velocity_systemic_sigma))
    model = jnp.zeros_like(wavelengths)

    if cfg.continuum.enabled:
        beta_low, beta_high = cfg.continuum.beta_bounds
        temp_low, temp_high = cfg.continuum.temperature_bounds
        model_type = cfg.continuum.model.lower()
        if model_type == "broken":
            amp_blue = sample(
                "amplitude_blue", dist.LogNormal(cfg.continuum.amplitude_logmean, cfg.continuum.amplitude_logstd)
            )
            amp_red = sample(
                "amplitude_red", dist.LogNormal(cfg.continuum.amplitude_logmean, cfg.continuum.amplitude_logstd)
            )
            temp_blue = sample("temperature_blue", dist.Uniform(temp_low, temp_high))
            temp_red = sample("temperature_red", dist.Uniform(temp_low, temp_high))
            beta_blue = sample("beta_blue", dist.Uniform(beta_low, beta_high))
            beta_red = sample("beta_red", dist.Uniform(beta_low, beta_high))
            cont = broken_modified_blackbody(
                wavelengths / 10_000.0,
                cfg.continuum.break_micron,
                (amp_blue, temp_blue, beta_blue),
                (amp_red, temp_red, beta_red),
            )
        else:
            amp = sample(
                "amplitude_continuum",
                dist.LogNormal(cfg.continuum.amplitude_logmean, cfg.continuum.amplitude_logstd),
            )
            temp = sample("temperature_continuum", dist.Uniform(temp_low, temp_high))
            beta = sample("beta_continuum", dist.Uniform(beta_low, beta_high))
            cont = single_modified_blackbody(
                wavelengths / 10_000.0, amp, temp, beta, pivot_micron=cfg.continuum.break_micron
            )
        model = model + cont

    for group in cfg.line_groups:
        vel = sample(f"velocity_{group.name}", dist.Normal(0.0, group.velocity_sigma))
        width = sample(f"width_{group.name}", dist.LogNormal(group.width_logmean, group.width_logstd))

        amplitude_dist: dist.Distribution
        if group.allow_negative:
            amplitude_dist = dist.Normal(0.0, group.amplitude_scale)
        else:
            amplitude_dist = dist.LogNormal(0.0, 0.8)

        per_line_params, tied_params = _line_amplitudes(group, f"amplitude_{group.name}")
        amplitudes = [0.0 for _ in group.lines]
        for _, (param_name, indices) in tied_params.items():
            base = sample(param_name, amplitude_dist)
            for idx in indices:
                ratio = group.lines[idx].line_ratio if group.lines[idx].line_ratio is not None else 1.0
                amplitudes[idx] = base * ratio

        for idx, param_name in per_line_params:
            amplitudes[idx] = sample(param_name, amplitude_dist)

        centers = []
        sigmas = []
        for line_idx, line in enumerate(group.lines):
            total_velocity = velocity_systemic + vel
            center = line.wavelength * (1.0 + total_velocity / _C_KMS)
            sigma = jnp.maximum(line.wavelength * width / _C_KMS, 1e-5)
            centers.append(center)
            sigmas.append(sigma)

        centers_arr = jnp.stack(centers)
        sigmas_arr = jnp.stack(sigmas)
        profile = _gaussian_profile(wavelengths, centers_arr, sigmas_arr)
        amps = jnp.stack(amplitudes)
        model = model + jnp.sum(profile * amps[None, :], axis=1)

    sample("obs", dist.Normal(model, flux_uncertainty), obs=flux)


def fit_spectrum(
    wavelengths: np.ndarray,
    flux: np.ndarray,
    flux_uncertainty: np.ndarray,
    config: FitConfig,
    num_warmup: int = 600,
    num_samples: int = 1200,
    rng_key: jax.Array | None = None,
) -> FitResult:
    """Fit a spectrum with line groups plus optional continuum."""

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    mask = build_mask(wavelengths, config.mask_regions)
    wl_masked, flux_masked, unc_masked = _apply_mask(
        jnp.asarray(wavelengths), jnp.asarray(flux), jnp.asarray(flux_uncertainty), mask
    )

    kernel = infer.NUTS(lambda w, f, e: _unite_model(w, f, e, config), init_strategy=init_to_feasible())
    mcmc = infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    logger.info("Starting UNITE-style fit with %d samples", num_samples)
    mcmc.run(rng_key, wl_masked, flux_masked, unc_masked)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    log_like = infer.util.log_likelihood(
        lambda w, f, e: _unite_model(w, f, e, config), samples, wl_masked, flux_masked, unc_masked
    )["obs"]
    return FitResult(
        wavelengths=np.asarray(wavelengths),
        flux=np.asarray(flux),
        flux_uncertainty=np.asarray(flux_uncertainty),
        mask=mask,
        samples=samples,
        log_likelihood=np.asarray(log_like).sum(axis=1),
        config_json=config.to_json(),
    )


def build_mask(wavelengths: np.ndarray, mask_regions: Sequence[tuple[float, float]]) -> np.ndarray:
    """Construct a boolean mask that excludes specified wavelength regions."""

    mask = np.ones_like(wavelengths, dtype=bool)
    for lower, upper in mask_regions:
        mask &= ~((wavelengths >= lower) & (wavelengths <= upper))
    return mask


def evaluate_components(
    wavelengths: np.ndarray, params: Mapping[str, float], cfg: FitConfig
) -> Mapping[str, np.ndarray]:
    """Evaluate continuum and line components for a given parameter set."""

    wavelengths_jnp = jnp.asarray(wavelengths)
    components: dict[str, np.ndarray] = {}
    model = jnp.zeros_like(wavelengths_jnp)

    if cfg.continuum.enabled:
        model_type = cfg.continuum.model.lower()
        if model_type == "broken":
            cont = broken_modified_blackbody(
                wavelengths_jnp / 10_000.0,
                cfg.continuum.break_micron,
                (
                    params.get("amplitude_blue", 0.0),
                    params.get("temperature_blue", 10_000.0),
                    params.get("beta_blue", 0.0),
                ),
                (
                    params.get("amplitude_red", 0.0),
                    params.get("temperature_red", 10_000.0),
                    params.get("beta_red", 0.0),
                ),
            )
        else:
            cont = single_modified_blackbody(
                wavelengths_jnp / 10_000.0,
                params.get("amplitude_continuum", 0.0),
                params.get("temperature_continuum", 10_000.0),
                params.get("beta_continuum", 0.0),
                pivot_micron=cfg.continuum.break_micron,
            )
        components["continuum"] = np.asarray(cont)
        model = model + cont

    v_sys = params.get("velocity_systemic", 0.0)

    for group in cfg.line_groups:
        vel = params.get(f"velocity_{group.name}", 0.0)
        width = params.get(f"width_{group.name}", np.exp(group.width_logmean))
        per_line_params, tied_params = _line_amplitudes(group, f"amplitude_{group.name}")
        amplitudes = [0.0 for _ in group.lines]
        for _, (param_name, indices) in tied_params.items():
            base = params.get(param_name, 0.0)
            for idx in indices:
                ratio = group.lines[idx].line_ratio if group.lines[idx].line_ratio is not None else 1.0
                amplitudes[idx] = base * ratio

        for idx, name in per_line_params:
            amplitudes[idx] = params.get(name, 0.0)

        centers = []
        sigmas = []
        for line in group.lines:
            center = line.wavelength * (1.0 + (v_sys + vel) / _C_KMS)
            sigma = np.maximum(line.wavelength * width / _C_KMS, 1e-5)
            centers.append(center)
            sigmas.append(sigma)

        center_arr = np.asarray(centers)
        sigma_arr = np.asarray(sigmas)
        profile = np.exp(-0.5 * ((wavelengths[:, None] - center_arr[None, :]) / sigma_arr[None, :]) ** 2)
        amps = np.asarray(amplitudes)
        component = np.sum(profile * amps[None, :], axis=1)
        components[group.name] = component
        model = model + component

    components["model"] = np.asarray(model)
    return components


def plot_fit(
    result: FitResult,
    params: Mapping[str, float] | None = None,
    ax: plt.Axes | None = None,
    show_components: bool = True,
    show_residuals: bool = True,
) -> plt.Axes:
    """Plot data, best-fit model, individual components, and residuals."""

    if ax is None:
        fig, ax = plt.subplots(2 if show_residuals else 1, 1, figsize=(8, 5), sharex=True)
        if show_residuals:
            main_ax, res_ax = ax
        else:
            main_ax = ax
    else:
        main_ax = ax
        res_ax = None

    params = result.posterior_median() if params is None else params
    cfg = result.config()
    comps = evaluate_components(result.wavelengths, params, cfg)
    main_ax.plot(result.wavelengths, result.flux, color="0.5", lw=1.0, label="data")
    main_ax.fill_between(
        result.wavelengths,
        result.flux - result.flux_uncertainty,
        result.flux + result.flux_uncertainty,
        color="0.85",
        alpha=0.5,
        label="1σ",
    )
    main_ax.plot(result.wavelengths, comps["model"], color="k", lw=1.5, label="model")

    if show_components:
        for key, comp in comps.items():
            if key in {"model"}:
                continue
            main_ax.plot(result.wavelengths, comp, lw=1.0, label=key)

    main_ax.legend(loc="upper right", ncol=2, fontsize=9)
    main_ax.set_ylabel("Flux")

    if show_residuals and res_ax is not None:
        residuals = result.flux - comps["model"]
        res_ax.axhline(0.0, color="k", ls="--", lw=1)
        res_ax.plot(result.wavelengths, residuals, color="tab:red", lw=1)
        res_ax.set_ylabel("Residuals")
        res_ax.set_xlabel("Wavelength [Å]")
        return res_ax

    main_ax.set_xlabel("Wavelength [Å]")
    return main_ax
