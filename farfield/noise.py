"""Pink noise generation.

Two algorithms. ``fft`` shapes white noise to an exact 1/f power spectrum and
is the default. ``lfsr`` reproduces the 16-bit shift register described in
US5356368A, which repeats every 65535 samples; it is generated one period at
a time and tiled rather than stepped sample-by-sample.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

LFSR_PERIOD = 65535
"""Period of a maximal-length 16-bit shift register, per US5356368A."""

_LFSR_TAPS = (0, 2, 3, 5)
"""Fibonacci taps for x^16 + x^14 + x^13 + x^11 + 1."""

# Paul Kellet's three one-pole sections, applied in parallel to white noise.
_KELLET_SECTIONS = (
    (0.0990460, 0.99765),
    (0.2965164, 0.96300),
    (1.0526913, 0.57000),
)
_KELLET_DIRECT = 0.1848


def _peak_normalize(signal: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(signal)))
    if peak == 0.0:
        return signal
    return signal / peak


def shaped_fft(
    n_samples: int,
    rng: np.random.Generator,
    slope_db_per_decade: float,
    sample_rate: int = 48000,
    shape=None,
) -> np.ndarray:
    """White noise shaped so amplitude ∝ f**(slope_db_per_decade/20).

    When ``shape`` is given (a ``BedShape``-like object with ``peak_hz``,
    ``rise_db_per_decade``, ``fall_db_per_decade``), it replaces the single
    slope with a two-slope hump: rising below ``peak_hz``, falling above it,
    joined smoothly in the power domain so there's no corner at the peak.
    ``slope_db_per_decade`` is ignored in that case. When ``shape`` is
    ``None`` this function is byte-identical to the single-slope behaviour.

    The shaping is done with a single FFT over the whole ``n_samples``
    span, so for a steep slope the lowest few bins (frequencies on the
    order of 1/n_samples, i.e. a fraction of a Hz for a multi-minute bed)
    get an enormous amplitude boost relative to 20 Hz+ content — tape
    never recorded that range and no listener can hear it, but it still
    counts toward the signal's RMS. Since bed level_db is calibrated
    against RMS, that near-DC energy silently eats most of the level
    budget and leaves the audible band far quieter than intended (before
    this fix, matching docs/tape-analysis/bed-results.json's ratios needed
    level_db values tens of dB higher than expected). Zeroing below 10 Hz
    and ramping smoothly up to full amplitude by 20 Hz removes that
    inaudible reservoir so RMS-referenced leveling means what it says.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    padded = max(n_samples, 16)
    spectrum = np.fft.rfft(rng.standard_normal(padded))
    freqs = np.fft.rfftfreq(padded, d=1.0)
    scale = np.ones_like(freqs)
    if shape is not None:
        freqs_hz_full = freqs * sample_rate
        # Divisor 20, the same amplitude-domain convention the single-slope
        # path uses -- and "dB per decade" reads the same on a PSD fit as on
        # an amplitude fit, because power = amplitude^2 exactly cancels
        # 10*log10 against 20*log10. Verified: the single-slope path at
        # -20/-10/-26.3 measures -19.99/-9.99/-26.29 dB/decade on a Welch
        # PSD fit, so a shape's rise/fall must be read the same way, with no
        # factor of 2 anywhere.
        a = shape.rise_db_per_decade / 20.0
        b = shape.fall_db_per_decade / 20.0
        # The power-domain join peaks BELOW its reference frequency, by a
        # factor that depends on both slopes: with scale(x) =
        # 1/sqrt(x^-2a + x^-2b) for x = f/f_ref, d/dx = 0 at
        # x* = (-b/a)^(1/(2b-2a)) (0.697 for the measured 7.7/-26.3 pair).
        # Dividing the reference by x* puts the actual maximum exactly at
        # peak_hz, so the parameter means what it is named.
        x_star = (-b / a) ** (1.0 / (2.0 * b - 2.0 * a))
        reference_hz = shape.peak_hz / x_star
        r = (freqs_hz_full[1:] / reference_hz) ** a
        g = (freqs_hz_full[1:] / reference_hz) ** b
        # Power-domain join: each asymptote dominates on its own side and
        # the knee is smooth, with no corner at the peak.
        scale[1:] = 1.0 / np.sqrt(1.0 / r ** 2 + 1.0 / g ** 2)
    elif slope_db_per_decade == -10.0:
        # 1/sqrt(f) rather than f ** (-10.0/20.0): the two differ by ~1 ULP,
        # which used to survive peak normalisation and shift a pink-bedded
        # session's whole mix, so this exact expression was kept for
        # byte-identity with the legacy pink bed. The bed-realism high-pass
        # taper below now perturbs every bedded render anyway, superseding
        # that guarantee. The branch stays regardless, so the taper is the
        # only change to the pink shape above the 20 Hz corner — swapping
        # in the generic exponent here would be a second, gratuitous
        # perturbation for no benefit.
        scale[1:] = 1.0 / np.sqrt(freqs[1:])
    else:
        scale[1:] = freqs[1:] ** (slope_db_per_decade / 20.0)
    scale[0] = 0.0  # drop DC

    # High-pass at the bottom of hearing: zero below 10 Hz, half-cosine
    # ramp 10->20 Hz, full amplitude above 20 Hz.
    freqs_hz = freqs * sample_rate
    taper = np.ones_like(freqs_hz)
    taper[freqs_hz < 10.0] = 0.0
    ramp = (freqs_hz >= 10.0) & (freqs_hz < 20.0)
    taper[ramp] = 0.5 * (1.0 - np.cos(np.pi * (freqs_hz[ramp] - 10.0) / 10.0))
    scale *= taper

    shaped = np.fft.irfft(spectrum * scale, padded)[:n_samples]
    return _peak_normalize(shaped)


def pink_fft(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """White noise shaped to an exact 1/f power spectrum."""
    return shaped_fft(n_samples, rng, -10.0)


def lfsr_white(n_samples: int, seed: int = 0xACE1) -> np.ndarray:
    """Bipolar white noise from a maximal-length 16-bit shift register."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    state = seed & 0xFFFF
    if state == 0:
        raise ValueError("seed must be non-zero")
    period = np.empty(LFSR_PERIOD, dtype=np.float64)
    for i in range(LFSR_PERIOD):
        feedback = 0
        for tap in _LFSR_TAPS:
            feedback ^= state >> tap
        state = ((state >> 1) | ((feedback & 1) << 15)) & 0xFFFF
        period[i] = 1.0 if state & 1 else -1.0
    repeats = -(-n_samples // LFSR_PERIOD)
    return np.tile(period, repeats)[:n_samples]


def pink_lfsr(n_samples: int, seed: int = 0xACE1) -> np.ndarray:
    """Shift-register white noise through a 1/f shaping filter."""
    white = lfsr_white(n_samples, seed=seed)
    shaped = _KELLET_DIRECT * white
    for gain, pole in _KELLET_SECTIONS:
        shaped = shaped + lfilter([gain], [1.0, -pole], white)
    return _peak_normalize(shaped)


def generate_pink(
    n_samples: int, algorithm: str, rng: np.random.Generator
) -> np.ndarray:
    if algorithm == "fft":
        return pink_fft(n_samples, rng)
    if algorithm == "lfsr":
        return pink_lfsr(n_samples)
    raise ValueError(
        f"unknown pink noise algorithm {algorithm!r}; expected 'fft' or 'lfsr'"
    )


def swept_comb(
    signal: np.ndarray,
    sample_rate: int,
    base_delay_s: float = 0.020,
    depth_s: float = 0.010,
    sweep_hz: float = 0.125,
    mix: float = 0.5,
) -> np.ndarray:
    """Comb filter whose delay sweeps, per the 1/8 Hz rate in US5356368A.

    The moving notches supply the cyclic spectral movement the phased pink
    sound calls for, without a time-varying IIR.
    """
    n = len(signal)
    positions = np.arange(n, dtype=np.float64)
    t = positions / float(sample_rate)
    delay_s = base_delay_s + depth_s * np.sin(2.0 * np.pi * sweep_hz * t)
    read_at = np.clip(positions - delay_s * sample_rate, 0.0, n - 1.0)
    delayed = np.interp(read_at, positions, signal)
    return (1.0 - mix) * signal + mix * delayed


def phased_pan(
    mono: np.ndarray,
    sample_rate: int,
    pan_rate_hz: float = 0.05,
    pan_rate_mod_hz: float = 0.011,
    pan_rate_depth: float = 0.5,
    amp_mod_hz: float = 0.017,
    amp_depth: float = 0.3,
) -> np.ndarray:
    """Rotate noise between channels with cyclic amplitude and rate changes.

    Equal-power panning keeps the summed level steady through the rotation.
    """
    n = len(mono)
    t = np.arange(n, dtype=np.float64) / float(sample_rate)

    instantaneous_rate = pan_rate_hz * (
        1.0 + pan_rate_depth * np.sin(2.0 * np.pi * pan_rate_mod_hz * t)
    )
    pan_phase = 2.0 * np.pi * np.cumsum(instantaneous_rate) / float(sample_rate)
    pan = 0.5 * (1.0 + np.sin(pan_phase))

    envelope = 1.0 - amp_depth * 0.5 * (
        1.0 + np.sin(2.0 * np.pi * amp_mod_hz * t)
    )

    angle = pan * (np.pi / 2.0)
    shaped = mono * envelope
    return np.stack([shaped * np.cos(angle), shaped * np.sin(angle)], axis=1)


def surf_envelope(
    n_samples: int,
    sample_rate: int,
    rate_hz: float,
    depth: float,
    phase_deg: float = 0.0,
) -> np.ndarray:
    """Amplitude envelope that swells the noise: 1 − depth·(0.5 + 0.5·sin(2π·rate·t + φ)).

    Peak amplitude is exactly 1.0 when depth is reached.

    ``phase_deg`` defaults to 0.0, which is bit-for-bit the previous
    behaviour (``sin(x + 0.0) == sin(x)`` for the non-negative x this
    envelope produces), so the tape-era beds that call this without a
    phase are unchanged. The MSS beds need it: their swell is measured
    to sit in quadrature with the texture layer's pan, so its phase is
    not free — see the MSS presets' notes and
    docs/tape-analysis/mss-results.json's bed.surf.

    The AC part of this envelope is −(depth/2)·sin(2π·rate·t + φ) about a
    mean of (1 − depth/2), so the FRACTIONAL amplitude modulation the
    listener gets is (depth/2)/(1 − depth/2) and its phasor sits at
    φ + 180°. Both facts are needed to derive a depth or a phase from a
    measurement of the finished mix.
    """
    t = np.arange(n_samples, dtype=np.float64) / float(sample_rate)
    return 1.0 - depth * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * rate_hz * t + np.radians(phase_deg))
    )


def _equal_power_ild_gains(ild_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel gains (a, b) for a target ILD in dB, equal-power (a**2+b**2=1).

    20*log10(a/b) = ild_db(t). Solving for a, b: g = 10**(ild_db/20),
    a = g/sqrt(1+g**2), b = 1/sqrt(1+g**2). Shared by crossfade_stereo (bed
    stereo) and render_texture (texture pan) so the algebra lives in one
    place.
    """
    g = 10.0 ** (ild_db / 20.0)
    a = g / np.sqrt(1.0 + g ** 2)
    b = 1.0 / np.sqrt(1.0 + g ** 2)
    return a, b


def crossfade_stereo(
    s1: np.ndarray,
    s2: np.ndarray,
    sample_rate: int,
    lfo_period_s: float,
    depth_db: float,
    phase_deg: float = 0.0,
) -> np.ndarray:
    """Assign one independent stream per ear, gain-swept to a target ILD.

    One stream per channel (not a swap-matrix blend of both streams into
    both channels): left = a(t)*s1, right = b(t)*s2. Since the streams are
    already independent, left/right stay decorrelated regardless of a, b
    (unlike a crosswise a*s1+b*s2 / b*s1+a*s2 mix, whose swapped terms make
    each channel's *average* power a**2*P1 + b**2*P2 identical to the other
    channel's b**2*P1 + a**2*P2 whenever P1 ~= P2 — that mix can't produce a
    real power-domain ILD from equal-power streams, only a much weaker
    decorrelation effect, however aggressively a, b are driven).

    a(t), b(t) satisfy a(t)**2 + b(t)**2 = 1 (equal power, so total energy
    stays close to constant through the sweep, matching the tape's small
    residual swell) while their gain ratio in dB is exactly the target ILD:
    20*log10(a/b) = ild_db(t) = (depth_db/2)*sin(2*pi*t/period). Solving for
    a, b: g = a/b = 10**(ild_db(t)/20), a = g/sqrt(1+g**2), b = 1/sqrt(1+g**2).
    Because each channel now carries only its own stream's power (P_left =
    a**2*P1, P_right = b**2*P2, no cross term), the measured ILD is
    20*log10(a/b) plus the streams' own (roughly constant) power mismatch,
    so its peak-to-peak swing tracks depth_db exactly, as required by the
    tape measurement.

    ``phase_deg`` offsets the LFO. It defaults to 0.0, which is bit-for-bit
    the previous behaviour (``sin(x + 0.0) == sin(x)`` for the non-negative
    x this sweep produces), so the tape-era beds that call this without a
    phase are unchanged. The texture layer needs it because the MSS tape's
    LFO phase is measured per track.
    """
    n = len(s1)
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    ild_db = (depth_db / 2.0) * np.sin(
        2.0 * np.pi * t / lfo_period_s + np.radians(phase_deg)
    )
    a, b = _equal_power_ild_gains(ild_db)
    left = a * s1
    right = b * s2
    return np.stack([left, right], axis=1)


def static_stereo(
    mono: np.ndarray, sample_rate: int, interaural_delay_us: float
) -> np.ndarray:
    """Fix one channel a constant fractional-sample delay behind the other.

    Positive delay: left leads (left = mono, right = delayed copy).
    Negative delay: mirrored (right leads).
    """
    n = len(mono)
    positions = np.arange(n, dtype=np.float64)
    delay_samples = abs(interaural_delay_us) * 1e-6 * sample_rate
    delayed = np.interp(positions - delay_samples, positions, mono)
    if interaural_delay_us >= 0.0:
        return np.stack([mono, delayed], axis=1)
    return np.stack([delayed, mono], axis=1)


def render_bed(
    n_samples: int, sample_rate: int, spec, rng: np.random.Generator
) -> np.ndarray:
    """Render a noise bed with color, comb filter, surf modulation, and a
    stereo stage (pan / two-stream crossfade / static interaural delay).

    The spec object is duck-typed and should provide:
      - .algorithm: 'fft' or 'lfsr' string
      - .color: noise color name (e.g. 'pink', 'brown')
      - .resolved_slope(): method returning slope in dB/decade
      - .shape: BedShape | None, a two-slope hump overriding the single
        slope above (None uses the single-slope colour as today)
      - .comb_sweep_hz: sweep rate for comb filter
      - .comb_enabled: bool, False bypasses the comb stage entirely
      - .pan_rate_hz: panning rate (stereo_mode 'pan' only)
      - .surf_rate_hz: surf modulation rate (None to disable)
      - .surf_depth: surf modulation depth (0.0 to 1.0)
      - .surf_phase_deg: surf LFO phase offset in degrees (optional,
        default 0.0). Both crossfade streams get the SAME envelope, so
        the surf is common mode: it swells the total, it does not pan.
      - .stereo_mode: 'pan', 'crossfade' or 'static'
      - .lfo_period_s: crossfade LFO period in seconds ('crossfade' only)
      - .stereo_depth_db: crossfade peak-to-peak ILD ('crossfade' only)
      - .interaural_delay_us: fixed L/R lead in microseconds, positive =
        left leads ('static' only)

    Returns an RMS-normalized bed (per-channel power average = a full-scale sine's RMS),
    so level_db is a power ratio against the tonal reference.
    """
    if spec.algorithm == "lfsr" and spec.color != "pink":
        raise ValueError(
            "lfsr algorithm only supports pink noise; non-pink color with lfsr is not supported"
        )

    def _generate(generator_rng: np.random.Generator) -> np.ndarray:
        if spec.algorithm == "lfsr":
            stream = generate_pink(n_samples, "lfsr", generator_rng)
        else:
            slope = spec.resolved_slope()
            stream = shaped_fft(
                n_samples, generator_rng, slope, sample_rate,
                shape=getattr(spec, "shape", None),
            )
        if spec.comb_enabled:
            stream = swept_comb(stream, sample_rate, sweep_hz=spec.comb_sweep_hz)
        if spec.surf_rate_hz is not None:
            envelope = surf_envelope(
                n_samples, sample_rate, spec.surf_rate_hz, spec.surf_depth,
                getattr(spec, "surf_phase_deg", 0.0),
            )
            stream = stream * envelope
        return stream

    if spec.stereo_mode == "crossfade":
        # crossfade needs two independent streams so left/right decorrelate;
        # spawn() (numpy >= 1.25) derives two child generators from rng's
        # bit-generator state, falling back to reseeding two fresh
        # generators from rng if spawn is unavailable.
        if hasattr(rng, "spawn"):
            child1, child2 = rng.spawn(2)
        else:
            child1 = np.random.default_rng(rng.integers(2**63))
            child2 = np.random.default_rng(rng.integers(2**63))
        s1 = _generate(child1)
        s2 = _generate(child2)
        stereo = crossfade_stereo(
            s1, s2, sample_rate, spec.lfo_period_s, spec.stereo_depth_db
        )
    elif spec.stereo_mode == "static":
        mono = _generate(rng)
        stereo = static_stereo(mono, sample_rate, spec.interaural_delay_us)
    else:
        mono = _generate(rng)
        stereo = phased_pan(mono, sample_rate, pan_rate_hz=spec.pan_rate_hz)

    rms = np.sqrt(((stereo ** 2).sum(axis=1) / 2).mean())
    if rms > 0.0:
        stereo = stereo / rms * (1 / np.sqrt(2))
    return stereo


def render_texture(
    n_samples: int, sample_rate: int, spec, rng: np.random.Generator
) -> np.ndarray:
    """Render the panned band-limited texture layer: TWO independent
    band-limited noise streams -> crossfade_stereo, RMS-normalized like
    render_bed.

    Two streams, not one mono source panned. The MSS tape's moving band is
    substantially DECORRELATED between the ears (measured magnitude-squared
    coherence 0.24-0.38 across 2.8-5.6 kHz, against a 0.05-0.15 floor
    elsewhere in the bed); a panned mono source would read close to 1.0
    there. That difference is audible and is the whole point of this
    layer: with decorrelated streams the ear does not stitch the outgoing
    and incoming sound into one object crossing the middle, so the gesture
    reads as "leaves right, reappears left" -- a wrap -- rather than as a
    retrace back through the centre. The trajectory is a sinusoid either
    way (the tape's own pan waveform has a second-harmonic ratio of
    0.0001), so the wrap percept comes from the decorrelation, not from a
    sawtooth. See docs/tape-analysis/mss-results.json's
    panned_hf_element.two_stream_construction.

    The spec object is duck-typed and should provide:
      - .band_hz: (lo, hi) bandpass edges in Hz
      - .pan_period_s: pan LFO period in seconds
      - .pan_ild_amplitude_db: pan ILD sinusoid amplitude in dB (not
        peak-to-peak; crossfade_stereo's depth_db IS peak-to-peak, so this
        is passed as 2x, matching the tape-fitted ILD sinusoid directly)
      - .pan_phase_deg: pan LFO phase offset in degrees
      - .surf_rate_hz / .surf_depth / .surf_phase_deg: optional
        common-mode swell, applied to BOTH streams before the crossfade
        so it changes the pair's total loudness without touching its
        position. The texture needs its own because it, not the bed, is
        what occupies 1.5-8 kHz in the finished mix.

    Time base is layer-relative (t starts at 0 for this layer), like the bed
    stereo laws, not absolute like the rotation feature.
    """
    sos = butter(4, spec.band_hz, btype="bandpass", fs=sample_rate, output="sos")
    if hasattr(rng, "spawn"):
        child1, child2 = rng.spawn(2)
    else:  # pragma: no cover - numpy < 1.25
        child1 = np.random.default_rng(rng.integers(2 ** 63))
        child2 = np.random.default_rng(rng.integers(2 ** 63))
    s1 = sosfilt(sos, child1.standard_normal(n_samples))
    s2 = sosfilt(sos, child2.standard_normal(n_samples))

    if getattr(spec, "surf_rate_hz", None) is not None:
        envelope = surf_envelope(
            n_samples, sample_rate, spec.surf_rate_hz, spec.surf_depth,
            spec.surf_phase_deg,
        )
        s1 = s1 * envelope
        s2 = s2 * envelope

    stereo = crossfade_stereo(
        s1, s2, sample_rate, spec.pan_period_s,
        2.0 * spec.pan_ild_amplitude_db, spec.pan_phase_deg,
    )

    rms = np.sqrt(((stereo ** 2).sum(axis=1) / 2).mean())
    if rms > 0.0:
        stereo = stereo / rms * (1 / np.sqrt(2))
    return stereo


def render_pink(
    n_samples: int,
    sample_rate: int,
    algorithm: str,
    comb_sweep_hz: float,
    pan_rate_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """The full phased pink sound chain, peak-normalised to 1.0.

    Convenience path, distinct from ``render_bed``: it wires
    ``generate_pink`` -> ``swept_comb`` -> ``phased_pan`` in one call
    with no spec object, for callers (and tests) that want the plain
    patent-style phased pink sound. The engine's bed layers render
    through ``render_bed``, which adds color/shape/surf and the
    measured stereo modes.
    """
    mono = generate_pink(n_samples, algorithm, rng)
    mono = swept_comb(mono, sample_rate, sweep_hz=comb_sweep_hz)
    stereo = phased_pan(mono, sample_rate, pan_rate_hz=pan_rate_hz)
    peak = float(np.max(np.abs(stereo)))
    if peak > 0.0:
        stereo = stereo / peak
    return stereo
