"""Non-sinusoidal beat envelopes via harmonic carrier pairs.

US5213562A shapes the beat envelope by superimposing recorded EEG waveforms
onto the carriers, decomposing them into many carrier pairs by Fourier
analysis. That EEG dataset was never published, so this module supplies the
same mechanism with declared coefficients instead: carrier pairs at integer
multiples of the beat frequency, with configurable amplitudes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_HARMONICS: tuple[float, ...] = (1.0, 0.35, 0.15)


@dataclass(frozen=True)
class Harmonic:
    index: int
    """1-based multiplier of the beat frequency."""

    amplitude: float


def expand_harmonics(ratios: Sequence[float]) -> tuple[Harmonic, ...]:
    if not ratios:
        raise ValueError("at least one harmonic ratio is required")
    if any(r < 0.0 for r in ratios):
        raise ValueError("harmonic ratios must be non-negative")
    harmonics = tuple(
        Harmonic(index=i + 1, amplitude=float(r))
        for i, r in enumerate(ratios)
        if r != 0.0
    )
    if not harmonics:
        raise ValueError("at least one harmonic ratio must be non-zero")
    return harmonics


def normalize_harmonics(
    harmonics: Sequence[Harmonic],
) -> tuple[Harmonic, ...]:
    total = sum(h.amplitude for h in harmonics)
    if total <= 0.0:
        raise ValueError("harmonic amplitudes must sum to a positive value")
    return tuple(
        Harmonic(index=h.index, amplitude=h.amplitude / total)
        for h in harmonics
    )
