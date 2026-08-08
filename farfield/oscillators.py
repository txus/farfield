"""Phase-continuous sine generation.

Every tone in the system comes from here. Frequency is accepted as a
per-sample array so a tone can glide without discontinuity, and the final
phase is returned so consecutive renders of the same oscillator can be
chained seamlessly across segment boundaries.
"""

from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def frequency_ramp(start_hz: float, end_hz: float, n_samples: int) -> np.ndarray:
    """A linear frequency trajectory inclusive of both endpoints."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    return np.linspace(float(start_hz), float(end_hz), n_samples, dtype=np.float64)


def phase_track(
    freq_hz: float | np.ndarray,
    n_samples: int,
    sample_rate: int,
    initial_phase: float = 0.0,
    handoff_index: int | None = None,
) -> tuple[np.ndarray, float]:
    """The accumulated phase of an oscillator, plus its wrapped handoff phase.

    This is ``render_tone`` without the final sine: callers that need the
    phase itself (SAM's carrier, whose phase is summed with a modulator
    before the sine, and SAM's modulator, which is consumed as a function
    of its own phase) go through here, and ``render_tone`` is a thin
    wrapper so both share exactly one accumulator and one handoff rule.

    ``freq_hz`` may be a scalar or an array of length ``n_samples``.

    ``handoff_index`` selects which sample's phase is reported back for
    chaining into the next render: ``None`` (the default) reports the phase
    one sample past the end, i.e. at sample ``n_samples`` — today's
    behaviour. An explicit index ``h`` reports the phase the oscillator HAS
    at sample ``h`` (the true integral of the frequency trajectory up to
    that sample), which is what a continuing layer starting at that same
    absolute sample needs in order to pick up in phase. ``h == n_samples``
    reproduces the legacy value exactly.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    freq = np.broadcast_to(
        np.asarray(freq_hz, dtype=np.float64), (n_samples,)
    )
    increments = TWO_PI * freq / float(sample_rate)
    phase = initial_phase + np.cumsum(increments) - increments
    if handoff_index is None or handoff_index == n_samples:
        handoff_phase = float((initial_phase + increments.sum()) % TWO_PI)
    else:
        handoff_phase = float(
            (initial_phase + increments[:handoff_index].sum()) % TWO_PI
        )
    return phase, handoff_phase


def render_tone(
    freq_hz: float | np.ndarray,
    n_samples: int,
    sample_rate: int,
    amplitude: float = 1.0,
    initial_phase: float = 0.0,
    handoff_index: int | None = None,
) -> tuple[np.ndarray, float]:
    """Render a sine tone, returning the samples and the wrapped handoff phase.

    See ``phase_track`` for the accumulator and the ``handoff_index`` rule.
    """
    phase, handoff_phase = phase_track(
        freq_hz,
        n_samples,
        sample_rate,
        initial_phase=initial_phase,
        handoff_index=handoff_index,
    )
    return amplitude * np.sin(phase), handoff_phase


def gate_envelope(
    rate_hz: float | np.ndarray,
    n_samples: int,
    sample_rate: int,
    duty: float,
    edge_s: float,
    initial_phase: float = 0.0,
    handoff_index: int | None = None,
) -> tuple[np.ndarray, float]:
    """A hard on/off envelope in [0, 1] with raised-cosine edges.

    This is the isochronic shape: full on for ``duty`` of each cycle, off for
    the rest, with an ``edge_s`` ramp on each side of the on-time. It is a
    different animal from a sinusoidal tremolo — the pulse train's spectrum
    has energy at the rate AND its harmonics, which is the whole point (an
    isochronic tone drives the ear with discrete events, not a smooth swell).

    Phase is carried in **cycles** in [0, 1), not radians, because the shape
    is piecewise in cycle fraction. ``handoff_index`` matches render_tone's
    contract so a gated layer continues mid-pulse across a segment boundary
    instead of restarting.

    The edges are raised-cosine rather than linear: a linear ramp has a
    discontinuous slope at both ends of the ramp, and those corners put
    audible high-frequency splatter into a stimulus whose spectral purity is
    the thing being measured.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if not 0.0 < duty < 1.0:
        raise ValueError("duty must be in (0, 1)")

    rate = np.broadcast_to(
        np.asarray(rate_hz, dtype=np.float64), (n_samples,)
    )
    increments = rate / float(sample_rate)  # cycles per sample
    phase = initial_phase + np.cumsum(increments) - increments
    if handoff_index is None or handoff_index == n_samples:
        handoff_phase = float((initial_phase + increments.sum()) % 1.0)
    else:
        handoff_phase = float(
            (initial_phase + increments[:handoff_index].sum()) % 1.0
        )

    p = np.mod(phase, 1.0)
    # Edge width as a fraction of a cycle, per sample (a gliding rate makes
    # the ramp a different fraction of the cycle at each moment, which is the
    # correct behaviour: the ramp lasts edge_s of wall-clock time throughout).
    e = edge_s * rate
    envelope = np.zeros(n_samples, dtype=np.float64)

    # A degenerate edge (0 s, or one that has grown past half the on-time
    # because the rate glided upward) collapses to the plateau test alone;
    # session._parse_gate rejects that case at load, so this is defensive.
    safe_e = np.minimum(e, duty / 2.0)

    rising = p < safe_e
    falling = (p >= duty - safe_e) & (p < duty)
    plateau = (p >= safe_e) & (p < duty - safe_e)

    envelope[plateau] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_rise = np.where(safe_e > 0, p / np.where(safe_e > 0, safe_e, 1.0), 1.0)
        frac_fall = np.where(
            safe_e > 0,
            (duty - p) / np.where(safe_e > 0, safe_e, 1.0),
            1.0,
        )
    envelope[rising] = 0.5 * (1.0 - np.cos(np.pi * np.clip(frac_rise[rising], 0.0, 1.0)))
    envelope[falling] = 0.5 * (
        1.0 - np.cos(np.pi * np.clip(frac_fall[falling], 0.0, 1.0))
    )
    return envelope, handoff_phase
