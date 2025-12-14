"""Placeholder models module for dotfit package."""

from __future__ import annotations

from typing import Iterable, Mapping


def get_line_wavelengths() -> tuple[Mapping[str, list[float]], Mapping[str, list[float]]]:
    """Placeholder implementation for line wavelength lookup.

    The legacy emission line module expects functions in a ``models`` module.
    They are not yet implemented in this port, so calling this function raises
    a clear error message.
    """
    raise NotImplementedError(
        "Line wavelength models have not been ported to dotfit yet."
    )


def find_nearest_key(lines: Mapping[str, Iterable[float]], value: float, **_: object) -> str:
    """Placeholder implementation for compatibility with the legacy API."""
    raise NotImplementedError(
        "Nearest line lookup is not yet available in dotfit models."
    )
