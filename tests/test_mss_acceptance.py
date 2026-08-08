"""Render-and-remeasure acceptance against docs/tape-analysis/mss-results.json.

Estimator discipline is the project's standing one (see
tests/test_bed_acceptance.py's docstrings for the full rationale): Welch
for continua, coherent projection for single lines, never one estimator
for both. Carrier and rotation measurements read the TONES ONLY, via the
same tones/bed separation test_bed_acceptance.py uses -- the mix is
exactly linear, so a re-render with the bed and texture layers removed
gives a noiseless tonal signal and makes every line measurement here
deterministic rather than a phase lottery.

Two deliberate departures from the tape-era acceptance files, both
documented at their use site: `_split` keeps only the 600-900 s analysis
window of each render (four full-length stereo float64 renders of a
40-minute program would need ~17 GB resident, and every window used here
sits inside 700-820 s), and `test_rotation_preserves_power` measures the
rotating carriers themselves rather than the whole tonal mix.
"""
import functools
from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import (butter, coherence, fftconvolve, sosfilt, sosfiltfilt,
                          welch)

from tests.support import load_preset
from farfield.render import render_timeline
from farfield.timeline import resolve

FS = 48000
NAMES = ["focus-10-mss", "focus-12-mss", "focus-15-mss",
         "focus-21-mss"]

# The cached slice. Every measurement window below lies inside 700-820 s;
# the extra 100 s on each side keeps the slice from being mistaken for a
# window boundary and leaves room for a wider window later.
CACHE_START_S, CACHE_END_S = 600.0, 900.0


FULL_PEAK = {}
"""name -> peak of the whole un-normalized render, filled in by _split.

The output level test needs it and cannot afford its own render: peak
normalization is global, so a window's dBFS depends on the peak of the
entire 40-minute session. _split renders that session anyway, so the
peak is free there and ruinously expensive anywhere else.
"""


@functools.lru_cache(maxsize=4)
def _split(name):
    """(tones, bed_plus_texture) over CACHE_START_S..CACHE_END_S.

    Raw, un-normalized, exactly summing: the renderer's layer sum is
    linear, so `full - tones` is the bed and texture alone, bit for bit.
    Only the analysis window is retained -- a whole 40-minute stereo
    float64 render is ~2 GB and four presets' worth of (tones, bed) pairs
    would not fit in memory.
    """
    timeline = resolve(load_preset(name))
    lo, hi = int(CACHE_START_S * FS), int(CACHE_END_S * FS)
    tones = render_timeline(
        replace(timeline, pink_layers=(), texture_layers=()), seed=11)[lo:hi]
    tones = np.array(tones)
    whole = render_timeline(timeline, seed=11)
    FULL_PEAK[name] = float(np.abs(whole).max())
    full = whole[lo:hi]
    return tones, np.array(full) - tones


def _win(x, t0_s, dur_s):
    """Absolute-session-time window out of the cached slice."""
    start = int((t0_s - CACHE_START_S) * FS)
    return x[start:start + int(dur_s * FS)]


def _demod(x, f0, t0_s, dur_s, rate=FS):
    seg = _win(x, t0_s, dur_s)
    t = np.arange(len(seg)) / rate
    return (seg * np.exp(-2j * np.pi * f0 * t)).mean() * 2.0


def _block_mags(tones, f0, t0_s, dur_s, block_s=0.5, rate=FS):
    """Per-block coherent magnitude of f0 in each ear."""
    seg = _win(tones, t0_s, dur_s)
    t = np.arange(len(seg)) / rate
    block = int(block_s * rate)
    n = len(seg) // block
    mags = []
    for ch in (0, 1):
        z = seg[:, ch] * np.exp(-2j * np.pi * f0 * t)
        mags.append(np.abs(z[:n * block].reshape(n, block).mean(axis=1)))
    return mags[0], mags[1], block_s


def _ild_trace(tones, f0, t0_s, dur_s, block_s=0.5, rate=FS):
    left, right, dt = _block_mags(tones, f0, t0_s, dur_s, block_s, rate)
    return 20 * np.log10(left / right), dt


def _fit_sine(trace, dt, period_s):
    tt = np.arange(len(trace)) * dt
    w = 2 * np.pi / period_s
    A = np.column_stack([np.sin(w * tt), np.cos(w * tt), np.ones_like(tt)])
    c, *_ = np.linalg.lstsq(A, trace, rcond=None)
    return float(np.hypot(c[0], c[1])), float(np.arctan2(c[1], c[0]))


# (preset, window_start_s, window_dur_s, pair_low_hz, pair_high_hz)
# Windows are inside each track's own layer_schedule_s for the theta layer
# and are whole multiples of neither 30 s nor 7.5 s -- a window that is an
# exact multiple of the rotation period averages the pan to zero and the
# tones read as dead mono (the trap the tape analysis nearly fell into).
ROTATION_CASES = [
    ("focus-10-mss", 700, 100, 193.9976, 197.9985),
    ("focus-12-mss", 700, 100, 193.9976, 197.9985),
    ("focus-15-mss", 700, 100, 193.9976, 197.9985),
    ("focus-21-mss", 700, 100, 193.9976, 197.9985),
]


@pytest.mark.parametrize("name,t0,dur,f_lo,f_hi", ROTATION_CASES)
def test_rotation_period_depth_and_counter_phase(name, t0, dur, f_lo, f_hi):
    tones, _ = _split(name)
    hi, dt = _ild_trace(tones, f_hi, t0, dur)
    lo, _ = _ild_trace(tones, f_lo, t0, dur)
    amp_hi, ph_hi = _fit_sine(hi, dt, 30.0)
    amp_lo, ph_lo = _fit_sine(lo, dt, 30.0)
    # the fit at the true period must beat neighbouring candidates
    for p in (20.0, 25.0, 40.0, 60.0):
        assert _fit_sine(hi, dt, p)[0] < amp_hi
    # 12.5-16.5 dB, not the 15-22 dB a linear-pan reading of the tape's
    # eps = 0.80 would predict: the recordings' summed power moves 0.12 dB
    # at the LFO rate against the 2.15 dB a linear pan of that index would
    # swing, so the law is equal-power. Re-measured on the tape with THIS
    # estimator (0.5 s blocks, 30 s least-squares sinusoid fit) the ILD
    # sinusoid fits 14.44 dB -- see mss-results.json's rotation block and
    # the presets' notes; depth 0.98 renders to ~14.3 dB against it.
    assert 12.5 < amp_hi < 16.5, f"{name}: rotation ILD {amp_hi:.1f} dB"
    d = abs(np.degrees(ph_hi - ph_lo)) % 360.0
    assert abs(d - 180.0) < 6.0, f"{name}: members {d:.1f} deg apart"


@pytest.mark.parametrize("name", NAMES)
def test_rotation_preserves_power(name):
    """rotation.power_preserving: L^2 + R^2 constant to under 0.5 dB.

    Measured on the rotating CARRIERS (per-block coherent magnitude in
    each ear, summed in power), which is the source analysis's own
    methodology -- it reports the figure per tone, from whole-window
    complex projections, not from the broadband mix. The mix itself is
    not a fair test of the pan law here: the ground pair's measured
    per-ear tremolo (index 0.83 at 0.250 and 0.500 Hz) swings the total
    tonal power by more than a dB all by itself, and that modulation is
    measured tape content, not a leak from the rotation.
    """
    tones, _ = _split(name)
    for f0 in (193.9976, 197.9982):
        left, right, _ = _block_mags(tones, f0, 700, 100)
        p = left ** 2 + right ** 2
        swing = 10 * np.log10(p.max() / p.min())
        assert swing < 0.6, f"{name}: {f0} Hz power swings {swing:.2f} dB"


@pytest.mark.parametrize("name", NAMES)
def test_texture_pan_period_and_band_isolation(name):
    _, bed = _split(name)
    seg = _win(bed, 700, 120)

    def band_trace(lo, hi):
        sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
        l, r = sosfilt(sos, seg[:, 0]), sosfilt(sos, seg[:, 1])
        n, hop = FS, FS // 4
        return np.array([
            10 * np.log10((l[s:s+n] ** 2).mean() / (r[s:s+n] ** 2).mean())
            for s in range(0, len(l) - n, hop)]), hop / FS

    # 4467-5623 Hz is the tape's own peak band for this gesture:
    # mss-results.json (panned_hf_element.ild_amplitude_by_band_db_v2) fits a 7.80 dB ILD
    # sinusoid there over nine windows on four tracks, sd 0.02 dB,
    # against a 0.84 dB off-period null.
    hi_tr, dt = band_trace(4467, 5623)
    amp, _ = _fit_sine(hi_tr, dt, 7.5)
    for p in (5.0, 9.0, 15.0):
        assert _fit_sine(hi_tr, dt, p)[0] < amp
    # The texture is now a two-stream crossfade, so it can carry the
    # tape's depth AND the tape's coherence at once -- the old "half the
    # pan depth" cap was an artifact of the mono model and is retired.
    # Bound set on the tape's 7.80 dB; the render measures 7.3-7.4.
    assert abs(amp - 7.80) < 1.5, f"{name}: texture pan {amp:.2f} dB"
    # Below the moving band the tape reads 0.11-0.12 dB, i.e. its own
    # estimator floor.
    lo_tr, dt2 = band_trace(400, 700)
    assert _fit_sine(lo_tr, dt2, 7.5)[0] < 1.0


@pytest.mark.parametrize("name", NAMES)
def test_moving_band_is_decorrelated(name):
    """The wrap percept: the moving band must NOT read as a panned mono.

    mss-results.json panned_hf_element.two_stream_construction measures
    magnitude-squared coherence of 0.24-0.38 across 2.8-5.6 kHz on the
    tape, where one mono source panned between the ears would read close
    to 1.0. That decorrelation is what stops the ear fusing the outgoing
    and incoming sound into a single object crossing the middle, and it
    is the difference between "leaves right, reappears left" and a
    retrace back through the centre. Our two-stream texture reads
    0.001-0.002 -- fully decorrelated, i.e. past the tape rather than
    short of it. The bound guards the direction that matters: anything
    approaching a mono pan fails.
    """
    _, bed = _split(name)
    seg = _win(bed, 700, 120)
    f, cxy = coherence(seg[:, 0], seg[:, 1], fs=FS, nperseg=2 ** 14)
    m = (f >= 2818) & (f <= 5623)
    msc = float(np.mean(cxy[m]))
    assert msc < 0.4, f"{name}: moving band coherence {msc:.3f}"


@pytest.mark.parametrize("name", NAMES)
def test_bed_hump_peak_and_slopes(name):
    _, bed = _split(name)
    seg = _win(bed, 700, 120)
    f, pl = welch(seg[:, 0], fs=FS, nperseg=2 ** 17)
    _, pr = welch(seg[:, 1], fs=FS, nperseg=2 ** 17)
    p = (pl + pr) / 2
    band = (f > 25) & (f < 400)
    # The tape's own argmax under this estimator is 88.1 +- 1.5 Hz over
    # nine windows (mss-results.json bed.spectrum_shape_v2).
    assert abs(f[band][np.argmax(p[band])] - 88.1) < 20.0

    def slope(lo, hi):
        m = (f >= lo) & (f <= hi) & (p > 0)
        # No /2: dB-per-decade reads the same on a PSD fit as on an
        # amplitude fit (power = amplitude^2 cancels 10*log10 vs 20*log10).
        # Ground truth: shaped_fft's single-slope path at -20 measures
        # -19.99 on this estimator.
        return float(np.polyfit(np.log10(f[m]),
                                10 * np.log10(p[m]), 1)[0])

    # Both bounds are the TAPE's own figures under exactly this fit,
    # measured over nine voice-free windows on four tracks
    # (mss-results.json bed.spectrum_shape_v2.
    # straight_line_fits_under_the_acceptance_estimator). They are not
    # the +7.7 / -26.3 a median-times-bandwidth estimator over wider fit
    # bands gives; this test fits the tape's continuum directly.
    assert abs(slope(25, 150) - 1.84) < 3.0        # tape 1.84 +- 0.46
    # 150-8000 on a mix that also contains the texture, both sides.
    # Narrowing this band to dodge our texture was tried and rejected in
    # review -- it hid a texture calibration that was ~14 dB too loud.
    assert abs(slope(150, 8000) - (-27.30)) < 2.5  # tape -27.30 +- 0.06


@pytest.mark.parametrize("name", NAMES)
def test_bed_level_against_the_theta_carrier(name):
    """mss-results.json bed.bed_vs_carrier_db_v2 = +15.4 dB.

    NOT the old bed_vs_carrier_db.vs_194_hz_theta = -8.6. That figure is a
    median-PSD-times-bandwidth reading of the 150-8000 Hz band, which on a
    band sloping this steeply sits ~18 dB under the same band's integral;
    this test then compared it against a 20-20000 Hz INTEGRAL, adding
    another ~7 dB. The two errors left the rendered bed about 21 dB too
    quiet. +15.4 dB (range 15.03-16.09 over nine windows on four tracks)
    is the tape's own 20-20000 Hz integral against the same coherent
    projection, and is what the bed level_db is now calibrated to.

    The carrier reference is |A|^2/2, a POWER, not the |A|^2 this test
    used to divide by -- a factor of two that used to make the comparison
    3 dB optimistic in a direction nobody had noticed.
    """
    tones, bed = _split(name)
    seg = _win(bed, 700, 120)
    f, bl = welch(seg[:, 0], fs=FS, nperseg=2 ** 17)
    _, br = welch(seg[:, 1], fs=FS, nperseg=2 ** 17)
    bp = (bl + br) / 2
    audible = (f >= 20.0) & (f <= 20000.0)
    bed_power = float(bp[audible].sum() * (f[1] - f[0]))
    # carrier: coherent projection on the noiseless tonal render, both ears
    c = sum(abs(_demod(tones[:, ch], 193.9976, 700, 120)) ** 2
            for ch in (0, 1)) / 4.0
    ratio_db = 10 * np.log10(bed_power / c)
    assert abs(ratio_db - 15.4) < 2.0, f"{name}: bed/carrier {ratio_db:.2f} dB"


# --------------------------------------------------------------------------
# The two checks below close a structural blind spot: every other check
# in this file is either NARROWBAND (a coherent projection at one
# carrier, a third-octave ILD trace) or GAIN-INVARIANT (a ratio, a
# slope, a phase difference, a coherence). Nothing else looks at how
# loud the output is or at the broadband envelope, so a preset could be
# 10 dB hot and a bed could have its total loudness pinned flat by
# construction, and the suite would pass every time. These two are
# deliberately the opposite: one absolute level, one broadband envelope.
# --------------------------------------------------------------------------

# Tape RMS over each track's own voice-free 120 s windows from
# mss-results.json's tracks[*].voice_free_windows_used_s, per-channel
# power average, the same convention render_bed levels to. The two
# windows that abut a music span (Freeflow 12 at 300 s, Freeflow 21 at
# 250 s) are excluded: they read 0.6-3.4 dB hot with music these presets
# do not render.
TAPE_RMS_DBFS = {
    "focus-10-mss": -27.31,   # 6 windows, sd 0.05
    "focus-12-mss": -27.18,   # 5 windows, sd 0.06
    "focus-15-mss": -27.52,   # 5 windows, sd 0.08
    "focus-21-mss": -28.09,   # 4 windows, sd 0.03
}


@pytest.mark.parametrize("name", NAMES)
def test_output_level_matches_the_tape(name):
    """The render must be as LOUD as the recording, not 10 dB hotter.

    peak_dbfs used to be -3.0 on all four presets while the sources sit
    near -27 dBFS RMS, so every render came out 9.2-10.1 dB hot and every
    A/B against the original was decided by level. peak_dbfs is now
    solved per preset for the tape's own RMS (RMS and not peak, because
    the crest factors differ: the tape's windows run 13.3-16.4 dB crest
    against our 14.5, and peak-matching would have missed the loudness by
    0.2-1.9 dB).

    Bound: 0.5 dB, which covers the tape's own window-to-window scatter
    (<= 0.08 dB) and the fact that this test's 700-820 s window is not
    one of the calibration windows and sits at a different point in each
    track's layer schedule (worst case 0.17 dB, on Free Flow 15). It is
    twenty times tighter than the fault it exists to catch.
    """
    tones, bed = _split(name)
    seg = _win(tones, 700, 120) + _win(bed, 700, 120)
    rms = float(np.sqrt((seg ** 2).sum(axis=1).mean() / 2))
    peak_dbfs = load_preset(name).peak_dbfs
    level = 20 * np.log10(rms / FULL_PEAK[name]) + peak_dbfs
    assert abs(level - TAPE_RMS_DBFS[name]) < 0.5, (
        f"{name}: renders at {level:.2f} dBFS RMS against the tape's "
        f"{TAPE_RMS_DBFS[name]:.2f}")
    # and it must not clip, which peak normalization guarantees but this
    # states out loud since the level is now the point.
    assert peak_dbfs < 0.0


def _swell_phasor(x, lo, hi, period_s=7.5, smooth_s=0.30, dec=20):
    """(S, P) at ``period_s`` on the smoothed per-ear amplitude envelopes.

    S = (L + R)/2 is the COMMON MODE -- both ears swelling together, i.e.
    what "the total loudness breathes" means to a headphone listener.
    P = (L - R)/2 is the pan. An ideal equal-power crossfade gives S = 0
    identically, which is exactly the fault this measures.

    The 0.30 s smoothing is the whole trick. The original analysis called
    the tape's surf "none" from an UNSMOOTHED full-bandwidth Hilbert
    envelope of the mono sum (53.1% against 52.3% for stationary Gaussian
    noise); that statistic is dominated by the noise's own fast
    fluctuation and cannot see a 7.5 s swell at any depth.
    """
    sos = butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")
    w = np.hanning(int(smooth_s * FS))
    w /= w.sum()
    out = []
    for ch in (0, 1):
        p = sosfiltfilt(sos, x[:, ch]) ** 2
        e = np.sqrt(fftconvolve(p, w, mode="valid")[::FS // dec])
        t = np.arange(len(e)) / dec
        omega = 2 * np.pi / period_s
        A = np.column_stack([np.sin(omega * t), np.cos(omega * t),
                             np.ones_like(t)])
        c, *_ = np.linalg.lstsq(A, e, rcond=None)
        out.append(complex(c[0], c[1]) / c[2])
    return (out[0] + out[1]) / 2, (out[0] - out[1]) / 2


@pytest.mark.parametrize("name", NAMES)
def test_total_loudness_breathes_with_the_pan(name):
    """mss-results.json bed.surf_v2: the bed swells, in quadrature.

    This SUPERSEDES bed.surf = "none", which was an estimator failure --
    see _swell_phasor's docstring. Re-measured with 0.30 s smoothing over
    22 voice-free windows on four tracks, the tape's broadband
    (20 Hz-20 kHz) amplitude AM at the 7.5000 s pan rate is 2.6-4.6%,
    grand mean 3.4%, where an equal-power crossfade gives 0 by
    construction and the pre-correction render measured 0.1-1.0%.

    The quadrature is the mechanism, not a coincidence. Unequal stream
    levels -- the other way to break a2+b2 = 1 -- would put the swell at
    0 or 180 deg to the pan and leave a DC ILD offset; the tape shows
    +72 to +103 deg and a mean ILD of -0.04 to -0.19 dB. That is the
    signature of two per-ear gain LFOs whose phase offset departs from
    180 deg, whose sum and difference are 90 deg apart for any offset.
    So the test checks BOTH numbers: a swell that appeared at the right
    depth but the wrong phase would not be this gesture.
    """
    tones, bed = _split(name)
    seg = _win(tones, 700, 120) + _win(bed, 700, 120)

    # depth: the whole mix, broadband. Bound spans the tape's own
    # 2.6-4.6% range with a little room for the render's carrier
    # dilution; the fault it replaces measured 0.1-1.0%.
    S, _ = _swell_phasor(seg, 20, 20000)
    depth = abs(S)
    assert 0.020 < depth < 0.060, f"{name}: broadband swell {100*depth:.2f}%"

    # ...and in the band the gesture is actually HEARD in, which is a
    # separate assertion because it is carried by a separate element. A
    # bed-only swell passes the broadband check and fails here: measured
    # on a rendered preset the texture is 62% of 1500-3000 Hz and 92% of
    # 4500-6000 Hz, so a swell applied only to the bed leaves 4500-6000
    # at 0.2-0.6% -- which is exactly what shipped in the first pass at
    # this fault and what a listener reported as still missing.
    #
    # 4500-6000 Hz is the calibration band: it sits inside the texture's
    # own peak (the tape's ILD peaks at 4467-5623 Hz), it is clear of the
    # encoder bit-allocation noise that makes everything above 6.3 kHz
    # unusable, and the tape reads 5.4-6.2% there over 22 voice-free
    # windows on four tracks (mss-results.json bed.surf_v2). The render
    # reads 5.9-6.3%.
    Shi, _ = _swell_phasor(seg, 4500, 6000)
    assert 0.045 < abs(Shi) < 0.080, (
        f"{name}: 4.5-6 kHz swell {100*abs(Shi):.2f}% against the tape's "
        f"5.4-6.2%")

    # it has to be a line at the pan rate, not broadband envelope wander:
    # off-period fits of the same trace must be smaller.
    for p in (5.0, 11.0, 17.0):
        off, _ = _swell_phasor(seg, 20, 20000, period_s=p)
        assert abs(off) < depth, f"{name}: {p}s fit beats the 7.5 s one"

    # phase: the swell leads the texture's pan by a quarter cycle. P is
    # read off the bed+texture alone in the moving band, where the pan
    # lives; the carriers only add their 30 s rotation's harmonics there.
    _, P = _swell_phasor(_win(bed, 700, 120), 1500, 8000)
    lead = (np.degrees(np.angle(S) - np.angle(P)) + 180) % 360 - 180
    assert abs(lead - 90.0) < 35.0, f"{name}: swell leads pan by {lead:.1f} deg"


@pytest.mark.parametrize("name", NAMES)
def test_carrier_grid_is_exact(name):
    tones, _ = _split(name)
    # the theta pair is present in every track's body window
    for f0 in (193.9976, 197.9985):
        z1 = _demod(tones[:, 0] + tones[:, 1], f0, 700, 60)
        z2 = _demod(tones[:, 0] + tones[:, 1], f0, 760, 60)
        drift_hz = np.angle(z2 / z1) / (2 * np.pi * 60.0)
        assert abs(drift_hz) < 0.005, (
            f"{name}: {f0} Hz off by {drift_hz*1000:.1f} mHz")
