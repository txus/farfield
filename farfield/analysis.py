"""Spectral analysis for the visualizer.

The browser cannot do this. Web Audio's AnalyserNode caps fftSize at 32768
and hardcodes a Blackman window; resolving carriers 4 Hz apart needs a window
of at least 3/4 s (36,000 samples at 48 kHz), above that cap. Decimating to
3 kHz first yields the same 1.365 s window from a 4096-point FFT.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import decimate

from farfield.timeline import Timeline
from farfield.voices import voice_frequencies

ANALYSIS_RATE = 3000
"""Nyquist of 1500 Hz, matching the binaural fusion ceiling."""

ANALYSIS_NFFT = 4096
"""1.365 s at ANALYSIS_RATE, giving 0.73 Hz bins."""


def decimate_to(
    signal: np.ndarray, sample_rate: int, target_rate: int
) -> tuple[np.ndarray, int]:
    factor = int(round(sample_rate / target_rate))
    if factor <= 1:
        return np.asarray(signal, dtype=np.float64), sample_rate
    # scipy recommends factors of at most 13 per stage.
    out = np.asarray(signal, dtype=np.float64)
    remaining = factor
    for stage in (4, 4, 2, 2, 3, 5, 7, 11, 13):
        while remaining % stage == 0 and remaining > 1:
            out = decimate(out, stage, ftype="fir", zero_phase=True)
            remaining //= stage
    if remaining > 1:
        out = decimate(out, remaining, ftype="fir", zero_phase=True)
    return out, int(round(sample_rate / factor))


def spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    nfft: int,
    hop_samples: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude STFT cropped to a frequency band."""
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) < nfft:
        signal = np.pad(signal, (0, nfft - len(signal)))
    window = np.hanning(nfft)
    all_freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    keep = (all_freqs >= fmin) & (all_freqs <= fmax)
    freqs = all_freqs[keep]

    n_frames = 1 + (len(signal) - nfft) // hop_samples
    mags = np.empty((n_frames, len(freqs)), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_samples
        block = signal[start : start + nfft] * window
        mags[i] = np.abs(np.fft.rfft(block))[keep]
    return freqs, mags


def carrier_frequencies(timeline: Timeline) -> list[float]:
    """Every distinct fundamental carrier endpoint in the session.

    Harmonic partials are excluded: these frequencies become the meters and
    the spectrogram's band, which should track the carriers themselves.
    """
    found: set[float] = set()
    for layer in timeline.layers:
        found.update(voice_frequencies(layer.group))
    return sorted(round(f, 6) for f in found)


def carrier_span(
    timeline: Timeline, pad_hz: float = 25.0
) -> tuple[float, float]:
    carriers = carrier_frequencies(timeline)
    if not carriers:
        return 0.0, float(ANALYSIS_RATE / 2)
    return max(0.0, carriers[0] - pad_hz), carriers[-1] + pad_hz


def goertzel_track(
    signal: np.ndarray,
    sample_rate: int,
    freq_hz: float,
    block: int,
    hop: int,
) -> np.ndarray:
    """Per-block magnitude at one exact frequency.

    With ``block = sample_rate / beat`` the neighbouring carriers in a stack
    land precisely on this filter's nulls, giving exact separation at a
    quarter of the equivalent FFT's window length.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n_frames = 1 + max(0, (len(signal) - block) // hop)
    k = 2.0 * np.pi * freq_hz / sample_rate
    reference = np.exp(-1j * k * np.arange(block))
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        out[i] = abs(np.dot(signal[start : start + block], reference)) / block
    return out


def demodulate_ipd(
    audio: np.ndarray,
    sample_rate: int,
    carrier_hz: float,
    trim_s: float = 0.05,
) -> np.ndarray:
    """The interaural phase difference trace, radians, by complex demodulation.

    Each channel is multiplied by exp(-i*2*pi*f_s*t) and brick-wall
    low-passed in the frequency domain, leaving a complex baseband whose
    argument is that channel's instantaneous phase relative to the carrier.
    The difference of the two arguments is IPD(t). For a SAM signal that is
    ``2*phi_p*m(t) + (phi_L - phi_R)``.

    The low-pass cutoff is 0.9*f_s: the demodulation image sits 2*f_s away,
    so 0.9*f_s rejects it with margin while still passing several harmonics
    of any plausible modulation rate (rate_hz is bounded below f_s/2 at
    parse time).

    ``trim_s`` drops that much from each end, where the circular
    convolution implied by an FFT low-pass wraps.
    """
    audio = np.asarray(audio, dtype=np.float64)
    baseband = [
        _baseband(audio[:, ch], sample_rate, carrier_hz, trim_s=0.0)
        for ch in (0, 1)
    ]
    ipd = np.unwrap(np.angle(baseband[0])) - np.unwrap(np.angle(baseband[1]))
    return _trim(ipd, sample_rate, trim_s)


def _trim(signal: np.ndarray, sample_rate: int, trim_s: float) -> np.ndarray:
    n = len(signal)
    trim = int(round(trim_s * sample_rate))
    return signal[trim : n - trim] if trim > 0 and n > 2 * trim else signal


def _baseband(
    channel: np.ndarray,
    sample_rate: int,
    carrier_hz: float,
    trim_s: float = 0.05,
) -> np.ndarray:
    """One channel shifted down by ``carrier_hz`` and brick-wall low-passed.

    ``|baseband|`` is the carrier's instantaneous amplitude and
    ``angle(baseband)`` its instantaneous phase, both without the
    narrowband assumption a Hilbert envelope makes — which matters here,
    because SAM's modulation rate can be a sizeable fraction of its
    carrier and a Hilbert envelope then leaks odd harmonics that the true
    envelope does not contain.
    """
    channel = np.asarray(channel, dtype=np.float64)
    n = len(channel)
    t = np.arange(n) / float(sample_rate)
    freqs = np.fft.fftfreq(n, 1.0 / sample_rate)
    keep = np.abs(freqs) <= 0.9 * carrier_hz
    spectrum = np.fft.fft(channel * np.exp(-2j * np.pi * carrier_hz * t))
    return _trim(np.fft.ifft(spectrum * keep), sample_rate, trim_s)


def _line_over_floor_db(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    target_hz: float,
    guard_hz: float = 3.0,
    floor_hz: float = 30.0,
) -> float:
    """Power at ``target_hz`` over the median of its neighbourhood, in dB.

    The neighbourhood is the band within ``floor_hz`` of the target minus a
    ``guard_hz`` exclusion around it, so a line cannot raise its own floor.
    """
    power = np.abs(spectrum) ** 2
    near = np.abs(freqs - target_hz)
    in_guard = near <= guard_hz
    peak = float(power[in_guard].max()) if in_guard.any() else 0.0
    surround = (near <= floor_hz) & (near > guard_hz)
    floor = float(np.median(power[surround])) if surround.any() else 0.0
    if peak <= 0.0:
        return float("-inf")
    if floor <= 0.0:
        return float("inf")
    return 10.0 * np.log10(peak / floor)


def sam_signature(
    audio: np.ndarray,
    sample_rate: int,
    rate_hz: float,
    carrier_hz: float,
) -> dict:
    """Measure the SAM signature of a stereo signal.

    This is the detector `docs/monroe-sound-science.md` derives,
    implemented once so the tests and any recording analysis share it:

    - ``ipd_amplitude_rad`` — amplitude of the ``rate_hz`` component of the
      demodulated interaural phase difference. For SAM this is ``2*phi_p``.
    - ``ipd_peak_rad`` — half the peak-to-peak IPD swing: the same quantity
      measured without assuming a waveform.
    - ``ipd_rate_hz`` — the dominant rate actually present in the IPD.
    - ``itd_peak_s`` — ``ipd_peak_rad / (2*pi*f_s)``, the implied interaural
      delay at a sweep extreme.
    - ``ipd_bias_rad`` — the static component, i.e. ``phi_L - phi_R``.
    - ``residual_beat_hz`` — the IPD's linear slope over 2*pi. SAM's mean
      frequency difference is ZERO by construction, so this is ~0 for SAM
      and equals the beat frequency for a conventional binaural pair. The
      oscillatory measures above are taken on the linearly DETRENDED IPD,
      because a conventional pair's IPD is an unbounded ramp whose DFT
      coefficient at any rate is non-zero — without detrending, a plain
      binaural beat scores a spurious ``ipd_amplitude_rad`` of about 2 rad.
    - ``mono_sum_2f_db`` — the mono-sum test. L+R collapses to
      ``2*sin(a)*cos(phi_p*m(t) + delta)``, whose envelope is EVEN in the
      modulator and therefore carries a line at TWICE the modulation rate.
      Reported as dB over the local envelope-spectrum floor.
    - ``mono_sum_1f_db`` — the same at ``rate_hz`` itself, which SAM raises
      only when the static offsets are asymmetric.
    - ``mono_sum_contrast_db`` — ``mono_sum_2f_db - mono_sum_1f_db``, and
      the statistic that actually discriminates. The research doc's shorter
      claim, that a true binaural beat "vanishes" under the mono-sum test,
      is too strong: a pair beating at R has a RECTIFIED envelope, so it
      carries harmonics at 2R, 3R, ... as well as its fundamental at R (the
      2R line sits about 14 dB down, nowhere near absent). What is
      distinctive about SAM is the ABSENCE of the fundamental — its
      envelope's lowest line is 2*f_m — so the discriminator is the
      contrast between the two, not the 2*f_m line alone. Symmetric SAM
      scores +100 dB or better here; a conventional pair scores negative.

    The envelope is taken from ``_baseband``, not from a Hilbert transform:
    at f_m = 40 Hz against f_s = 300 Hz the narrowband assumption behind an
    analytic envelope fails and leaks a spurious f_m line into paths whose
    true envelope has none.
    """
    audio = np.asarray(audio, dtype=np.float64)
    ipd = demodulate_ipd(audio, sample_rate, carrier_hz)
    n = len(ipd)
    t = np.arange(n) / float(sample_rate)

    slope, intercept = np.polyfit(t, ipd, 1)
    detrended = ipd - (slope * t + intercept)

    coefficient = 2.0 * np.sum(detrended * np.exp(-2j * np.pi * rate_hz * t)) / n
    ipd_spectrum = np.fft.rfft(detrended * np.hanning(n))
    ipd_freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    band = ipd_freqs > 0.05
    dominant = float(ipd_freqs[band][np.argmax(np.abs(ipd_spectrum[band]))])
    peak = 0.5 * float(detrended.max() - detrended.min())

    mono = audio[:, 0] + audio[:, 1]
    envelope = np.abs(_baseband(mono, sample_rate, carrier_hz))
    envelope = envelope - envelope.mean()
    env_spectrum = np.fft.rfft(envelope * np.hanning(len(envelope)))
    env_freqs = np.fft.rfftfreq(len(envelope), 1.0 / sample_rate)

    two_f_db = _line_over_floor_db(env_spectrum, env_freqs, 2.0 * rate_hz)
    one_f_db = _line_over_floor_db(env_spectrum, env_freqs, rate_hz)

    return {
        "ipd_amplitude_rad": float(np.abs(coefficient)),
        "ipd_peak_rad": peak,
        "ipd_rate_hz": dominant,
        "itd_peak_s": peak / (2.0 * np.pi * carrier_hz),
        "ipd_bias_rad": float(np.mean(ipd)),
        "residual_beat_hz": float(slope) / (2.0 * np.pi),
        "mono_sum_2f_db": two_f_db,
        "mono_sum_1f_db": one_f_db,
        "mono_sum_contrast_db": two_f_db - one_f_db,
    }
