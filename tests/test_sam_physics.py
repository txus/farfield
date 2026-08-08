"""Prove the SAM physics, not that it renders.

Every number here is predicted from the patents' equations first and then
measured off the rendered audio with farfield.analysis.sam_signature, the
same estimator the research doc derives for finding SAM in recordings.
"""

import math

import numpy as np
import pytest
from scipy.special import j1

from farfield.analysis import demodulate_ipd, sam_signature
from tests.support import load_preset
from farfield.render import render_timeline, sam_modulator
from farfield.session import (
    SAM_HEAD_SEPARATION_M,
    SAM_SPEED_OF_SOUND_MS,
    load_session_dict,
    sam_depth_from_arc,
)
from farfield.timeline import resolve

RATE = 48000
CARRIER = 300.0
MOD_RATE = 40.0
ARC = 180.0
PHI_P = sam_depth_from_arc(ARC, CARRIER)  # 0.4809 rad


def _sam_audio(seconds=8, **overrides):
    sam = {"carrier_hz": CARRIER, "rate_hz": MOD_RATE, "arc_deg": ARC}
    sam.update(overrides)
    sam = {k: v for k, v in sam.items() if v is not None}
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "patent", "sample_rate": RATE,
        "segments": [
            {"duration": seconds, "groups": [{"name": "s", "sam": sam}]}
        ],
    })
    return render_timeline(resolve(session))


def _binaural_audio(seconds=8, beat=MOD_RATE):
    """The negative control: a conventional pair, same centre, same rate."""
    session = load_session_dict({
        "name": "c", "title": "C", "fidelity": "patent", "sample_rate": RATE,
        "segments": [{"duration": seconds, "groups": [
            {"name": "b", "harmonics": [1.0],
             "pairs": [{"center": CARRIER, "beat": beat}]}
        ]}],
    })
    return render_timeline(resolve(session))


# --- the interaural phase difference ---------------------------------------


def test_ipd_traces_a_sinusoid_at_the_modulation_rate():
    sig = sam_signature(_sam_audio(), RATE, MOD_RATE, CARRIER)
    # S_L - S_R phase = 2*phi_p*sin(2*pi*f_m*t): peak magnitude 2*phi_p.
    assert sig["ipd_amplitude_rad"] == pytest.approx(2.0 * PHI_P, rel=0.02)
    assert sig["ipd_peak_rad"] == pytest.approx(2.0 * PHI_P, rel=0.02)
    assert sig["ipd_rate_hz"] == pytest.approx(MOD_RATE, rel=0.005)


def test_ipd_amplitude_scales_with_the_arc():
    for arc in (30.0, 90.0, 150.0):
        expected = 2.0 * sam_depth_from_arc(arc, CARRIER)
        sig = sam_signature(_sam_audio(arc_deg=arc), RATE, MOD_RATE, CARRIER)
        assert sig["ipd_amplitude_rad"] == pytest.approx(expected, rel=0.02)


def test_ipd_amplitude_follows_depth_rad_directly():
    sig = sam_signature(
        _sam_audio(arc_deg=None, depth_rad=0.25), RATE, MOD_RATE, CARRIER
    )
    assert sig["ipd_amplitude_rad"] == pytest.approx(0.5, rel=0.02)


def test_the_mean_frequency_difference_is_zero():
    # The whole reason SAM produces a usable 40 Hz beat where a binaural pair cannot:
    # instantaneous df = 2*phi_p*f_m*cos(2*pi*f_m*t), mean zero.
    sig = sam_signature(_sam_audio(), RATE, MOD_RATE, CARRIER)
    assert abs(sig["residual_beat_hz"]) < 0.01
    control = sam_signature(_binaural_audio(), RATE, MOD_RATE, CARRIER)
    assert abs(control["residual_beat_hz"]) == pytest.approx(MOD_RATE, rel=1e-3)


def test_static_offsets_bias_the_ipd_without_moving_the_sweep():
    sig = sam_signature(
        _sam_audio(offset_deg={"left": 15.0, "right": -15.0}),
        RATE, MOD_RATE, CARRIER,
    )
    assert sig["ipd_bias_rad"] == pytest.approx(math.radians(30.0), abs=0.01)
    assert sig["ipd_amplitude_rad"] == pytest.approx(2.0 * PHI_P, rel=0.02)


# --- the implied interaural delay ------------------------------------------


def test_itd_at_the_sweep_extremes_is_phi_p_over_pi_f_s():
    expected = PHI_P / (math.pi * CARRIER)
    sig = sam_signature(_sam_audio(), RATE, MOD_RATE, CARRIER)
    # 2% covers the demodulator's brick-wall edge behaviour; the measured
    # figure sits within 0.5% in practice.
    assert sig["itd_peak_s"] == pytest.approx(expected, rel=0.02)


def test_a_full_arc_traverses_the_heads_own_delay():
    # 0.175 m / 343 m/s = 510 us, the maximum physical ITD.
    expected = SAM_HEAD_SEPARATION_M / SAM_SPEED_OF_SOUND_MS
    sig = sam_signature(_sam_audio(arc_deg=180.0), RATE, MOD_RATE, CARRIER)
    assert sig["itd_peak_s"] == pytest.approx(expected, rel=0.02)


def test_the_itd_is_independent_of_carrier_at_a_fixed_arc():
    # The arc parameterization's point: an arc is a POSITION, so the same
    # arc gives the same delay whatever carrier renders it.
    for carrier in (200.0, 300.0, 440.0):
        audio = _sam_audio(carrier_hz=carrier, arc_deg=120.0)
        sig = sam_signature(audio, RATE, MOD_RATE, carrier)
        expected = (
            SAM_HEAD_SEPARATION_M / SAM_SPEED_OF_SOUND_MS
        ) * math.sin(math.radians(60.0))
        assert sig["itd_peak_s"] == pytest.approx(expected, rel=0.02)


# --- the mono-sum detector, and its negative control ------------------------


def test_mono_sum_carries_a_line_at_twice_the_modulation_rate():
    sig = sam_signature(_sam_audio(), RATE, MOD_RATE, CARRIER)
    assert sig["mono_sum_2f_db"] > 100.0
    # And nothing at f_m itself: the envelope is even in the modulator.
    assert sig["mono_sum_1f_db"] < 6.0
    assert sig["mono_sum_contrast_db"] > 100.0


def test_a_conventional_binaural_pair_shows_no_sam_signature():
    sig = sam_signature(_binaural_audio(), RATE, MOD_RATE, CARRIER)
    # It has an envelope line at its beat rate and rectification harmonics
    # above it, so the 2*f_m line alone is NOT the discriminator; the
    # contrast against f_m is, and it goes the other way.
    assert sig["mono_sum_contrast_db"] < 0.0
    # And the detrended IPD carries no oscillation at all: a pair's phase
    # difference is a ramp, not a sweep.
    assert sig["ipd_amplitude_rad"] < 0.01
    assert sig["ipd_peak_rad"] < 0.01


def test_the_detector_separates_sam_from_a_pair_by_orders_of_magnitude():
    sam = sam_signature(_sam_audio(), RATE, MOD_RATE, CARRIER)
    pair = sam_signature(_binaural_audio(), RATE, MOD_RATE, CARRIER)
    assert sam["mono_sum_contrast_db"] - pair["mono_sum_contrast_db"] > 100.0


# --- the three path types ---------------------------------------------------


def test_closed_path_is_the_patents_literal_sinusoid():
    phase = np.linspace(0.0, 4.0 * np.pi, 4001)
    spec = _sam_audio(seconds=1)  # only to keep the render path exercised
    assert spec is not None
    from farfield.session import SamSpec

    m = sam_modulator(phase, SamSpec(CARRIER, MOD_RATE, PHI_P, "closed"))
    assert np.allclose(m, np.sin(phase))


def test_open_path_reaches_the_same_extremes_but_a_different_trajectory():
    theta = math.asin(
        PHI_P / (math.pi * CARRIER) /
        (SAM_HEAD_SEPARATION_M / SAM_SPEED_OF_SOUND_MS)
    )
    sig = sam_signature(_sam_audio(path="open"), RATE, MOD_RATE, CARRIER)
    # Same peak deviation as closed...
    assert sig["ipd_peak_rad"] == pytest.approx(2.0 * PHI_P, rel=0.02)
    # ...but the waveform is sin(theta*sin(Phi))/sin(theta), whose
    # fundamental is 2*J1(theta)/sin(theta) times the peak. At a full arc
    # theta = pi/2 that is 1.1336 -- a fundamental LARGER than the peak, the
    # way a square wave's is, and a fact about this trajectory that a plain
    # "does it sweep" test would miss entirely.
    ratio = 2.0 * j1(theta) / math.sin(theta)
    assert ratio == pytest.approx(1.1336, abs=0.001)
    assert sig["ipd_amplitude_rad"] == pytest.approx(
        2.0 * PHI_P * ratio, rel=0.02
    )


def test_open_path_degenerates_to_closed_for_a_narrow_arc():
    narrow = sam_signature(
        _sam_audio(path="open", arc_deg=10.0), RATE, MOD_RATE, CARRIER
    )
    expected = 2.0 * sam_depth_from_arc(10.0, CARRIER)
    assert narrow["ipd_amplitude_rad"] == pytest.approx(expected, rel=0.02)
    assert narrow["ipd_rate_hz"] == pytest.approx(MOD_RATE, rel=0.005)


def test_discontinuous_path_holds_a_fixed_number_of_positions():
    from farfield.session import SamSpec

    steps = 6
    spec = SamSpec(CARRIER, MOD_RATE, PHI_P, "discontinuous", steps=steps)
    phase = np.linspace(0.1, 20.0 * np.pi + 0.1, 200001)
    m = sam_modulator(phase, spec)
    levels = np.sin(np.arange(steps) * 2.0 * np.pi / steps)
    # Every sample sits exactly on one of `steps` held positions -- not
    # near one: a sample-and-hold has no transit.
    assert np.abs(m[:, None] - levels[None, :]).min(axis=1).max() < 1e-12
    assert len(np.unique(np.round(m, 9))) == len(np.unique(np.round(levels, 9)))
    # This is NOT measured through demodulate_ipd: the hold rate is
    # steps*f_m = 240 Hz here and 320 Hz at the default 8 steps, at or above
    # the demodulator's 0.9*f_s low-pass, which smooths the steps away. The
    # rendered audio is checked against the analytic signal instead, below.


def test_discontinuous_path_is_still_detected_by_the_mono_sum_test():
    sig = sam_signature(
        _sam_audio(path="discontinuous", steps=8), RATE, MOD_RATE, CARRIER
    )
    # Lower contrast than the smooth paths: a sample-and-hold is wideband,
    # its sidebands run past the demodulator's low-pass and the negative
    # image folds a little energy back in. Still a decisive detection, and
    # still on the opposite side of zero from a conventional pair.
    assert sig["mono_sum_2f_db"] > 40.0
    assert sig["mono_sum_contrast_db"] > 15.0


@pytest.mark.parametrize("path", ["closed", "open", "discontinuous"])
def test_every_path_sweeps_at_the_modulation_rate(path):
    sig = sam_signature(_sam_audio(path=path), RATE, MOD_RATE, CARRIER)
    assert sig["ipd_rate_hz"] == pytest.approx(MOD_RATE, rel=0.005)
    assert sig["ipd_peak_rad"] == pytest.approx(2.0 * PHI_P, rel=0.03)


def test_every_modulator_stays_within_unit_amplitude():
    from farfield.session import SamSpec

    phase = np.linspace(0.0, 20.0 * np.pi, 200001)
    for path in ("closed", "open", "discontinuous"):
        m = sam_modulator(phase, SamSpec(CARRIER, MOD_RATE, PHI_P, path))
        assert np.abs(m).max() <= 1.0 + 1e-12


def test_every_modulator_is_two_pi_periodic_in_phase():
    # This is what makes the seam handoff correct: the handoff wraps the
    # modulator phase mod 2*pi, so any path that needed an unwrapped phase
    # would jump there.
    from farfield.session import SamSpec

    # Offset off the grid boundaries: at a phase landing exactly on a hold
    # edge, floor() is one ULP away from either neighbour, which is a
    # floating-point tie rather than a periodicity failure.
    phase = np.linspace(0.017, 2.0 * np.pi + 0.017, 5000, endpoint=False)
    for path in ("closed", "open", "discontinuous"):
        spec = SamSpec(CARRIER, MOD_RATE, PHI_P, path)
        assert np.allclose(
            sam_modulator(phase, spec),
            sam_modulator(phase + 6.0 * np.pi, spec),
        )


# --- seams ------------------------------------------------------------------


def _two_segment_sam(overlap=2, **overrides):
    sam = {"carrier_hz": CARRIER, "rate_hz": MOD_RATE, "arc_deg": ARC}
    sam.update(overrides)
    return load_session_dict({
        "name": "t", "title": "T", "fidelity": "patent", "sample_rate": RATE,
        "segments": [
            {"duration": 6, "overlap": overlap,
             "groups": [{"name": "s", "sam": dict(sam)}]},
            {"duration": 6, "groups": [{"name": "s", "sam": dict(sam)}]},
        ],
    })


def test_a_continuing_sam_group_is_judged_coherent():
    timeline = resolve(_two_segment_sam())
    assert timeline.layers[0].coherent_fade_out
    assert timeline.layers[1].coherent_fade_in


def test_a_changing_sam_group_is_not_judged_coherent():
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "patent", "sample_rate": RATE,
        "segments": [
            {"duration": 6, "overlap": 2, "groups": [{"name": "s", "sam": {
                "carrier_hz": 300.0, "rate_hz": 40.0, "arc_deg": 180.0}}]},
            {"duration": 6, "groups": [{"name": "s", "sam": {
                "carrier_hz": 440.0, "rate_hz": 40.0, "arc_deg": 180.0}}]},
        ],
    })
    timeline = resolve(session)
    assert not timeline.layers[0].coherent_fade_out
    assert not timeline.layers[1].coherent_fade_in


def test_a_sam_seam_does_not_dip_or_swell():
    audio = render_timeline(resolve(_two_segment_sam()))
    env = np.sqrt(
        np.convolve(audio[:, 0] ** 2, np.ones(4800) / 4800, "valid")
    )
    seam = env[int(3.5 * RATE): int(6.5 * RATE)]
    body = env[int(1.0 * RATE): int(3.0 * RATE)]
    # Both accumulators hand off, so the two copies are the SAME signal
    # through the overlap and the linear law sums them to exactly one.
    assert 20.0 * np.log10(seam.max() / body.mean()) < 0.2
    assert 20.0 * np.log10(seam.min() / body.mean()) > -0.2


def test_a_sam_seam_does_not_jump_the_modulator_phase():
    audio = render_timeline(resolve(_two_segment_sam()))
    ipd = demodulate_ipd(audio, RATE, CARRIER)
    n = len(ipd)
    t = np.arange(n) / RATE
    # One global sinusoid, fitted across the whole two-segment session. If
    # the modulator's phase jumped anywhere -- at the seam above all -- no
    # single sinusoid could fit both sides, and the residual would blow up.
    basis = np.column_stack([
        np.sin(2 * np.pi * MOD_RATE * t), np.cos(2 * np.pi * MOD_RATE * t),
        np.ones(n),
    ])
    coefficients, *_ = np.linalg.lstsq(basis, ipd, rcond=None)
    residual = ipd - basis @ coefficients
    amplitude = math.hypot(coefficients[0], coefficients[1])
    assert amplitude == pytest.approx(2.0 * PHI_P, rel=0.02)
    assert float(np.sqrt(np.mean(residual ** 2))) < 0.01 * amplitude


def _uninterrupted_sam(n, path="closed", steps=8):
    """S_L/S_R for ONE pair of oscillators run start to finish in one go.

    Both accumulators come from ``phase_track`` over the whole span, the
    same routine the renderer uses per layer -- so this is exactly the
    signal the two-segment render must reproduce if, and only if, its
    handoffs are correct.
    """
    from farfield.oscillators import phase_track
    from farfield.session import SamSpec

    spec = SamSpec(CARRIER, MOD_RATE, PHI_P, path, steps=steps)
    carrier, _ = phase_track(CARRIER, n, RATE)
    modulator, _ = phase_track(MOD_RATE, n, RATE)
    deviation = PHI_P * sam_modulator(modulator, spec)
    return np.column_stack(
        [np.sin(carrier + deviation), np.sin(carrier - deviation)]
    )


@pytest.mark.parametrize("path", ["closed", "open", "discontinuous"])
def test_a_sam_seam_reproduces_one_uninterrupted_oscillator_pair(path):
    # The strongest statement available: across a two-segment session with
    # a 2 s crossfade, the rendered samples equal what a SINGLE pair of
    # oscillators running from t=0 would have produced -- so neither
    # accumulator jumped, and the discontinuous path's hold grid (anchored
    # to modulator phase, not to sample index) did not slip across the
    # handoff's mod-2*pi wrap either.
    from farfield.oscillators import phase_track

    audio = render_timeline(resolve(_two_segment_sam(path=path)))
    n = audio.shape[0]
    predicted = _uninterrupted_sam(n, path=path)
    # Outside the session's own head/tail fade windows.
    window = slice(int(0.5 * RATE), n - int(0.5 * RATE))
    error = np.abs(audio[window] - predicted[window])

    if path != "discontinuous":
        # ~1e-7, not 0: splitting one 10 s cumsum into two chained ones
        # reassociates the additions, and the two accumulations separate by
        # about 9e-8 by the end. A phase jump would be O(1).
        assert error.max() < 1e-6
        return

    # A sample-and-hold cannot be compared sample-for-sample at its
    # transitions. At f_m = 40 Hz, 8 steps and 48 kHz a hold lasts exactly
    # 150 samples, so every boundary lands EXACTLY on a sample and floor()
    # is a perfect tie there: the ~1e-12 rad the chained accumulator differs
    # from the uninterrupted one decides which side of the tie that one
    # sample falls on, and it can differ for every boundary at once. What
    # must hold -- and what this asserts -- is that no other sample differs:
    # the hold grid keeps its positions and its timing to within the single
    # ambiguous transition sample, which is where the step is happening
    # anyway.
    modulator, _ = phase_track(MOD_RATE, n, RATE)
    from farfield.session import SamSpec

    held = sam_modulator(
        modulator, SamSpec(CARRIER, MOD_RATE, PHI_P, "discontinuous", steps=8)
    )[window]
    transition = np.nonzero(np.diff(held) != 0.0)[0]
    offending = np.nonzero(error.max(axis=1) > 1e-6)[0]
    assert len(offending) > 0  # otherwise this assertion proves nothing
    distance = np.abs(offending[:, None] - transition[None, :]).min(axis=1)
    assert distance.max() <= 1


def test_a_sam_seam_matches_the_closed_form_equation_across_the_join():
    # The same claim stated against the patents' formula in wall time,
    # rather than against the engine's own accumulator. Only the smooth
    # paths: a sample-and-hold's floor() sits one ULP from flipping a whole
    # step wherever a hold boundary falls between two samples, so wall-time
    # and cumsum phases (which differ by ~1e-7 rad after 12 s) disagree by a
    # full step at a handful of isolated samples -- a floating-point tie at
    # a boundary, not a phase jump.
    audio = render_timeline(resolve(_two_segment_sam()))
    t = np.arange(audio.shape[0]) / RATE
    deviation = PHI_P * np.sin(2 * np.pi * MOD_RATE * t)
    carrier = 2 * np.pi * CARRIER * t
    predicted = np.column_stack(
        [np.sin(carrier + deviation), np.sin(carrier - deviation)]
    )
    window = slice(int(0.5 * RATE), int(11.5 * RATE))
    assert np.abs(audio[window] - predicted[window]).max() < 1e-6


# --- the same claims, measured a second way ---------------------------------
#
# Everything above rests on complex demodulation. If that estimator were
# subtly wrong it would confirm itself. These two measurements share none
# of its machinery: one works in the time domain by cross-correlation (the
# other method the research doc names), the other in the frequency domain
# off the Bessel spectrum of a phase-modulated carrier.


def test_cross_correlation_recovers_the_same_itd_sweep():
    # Slowed to f_m = 1 Hz so a 20 ms correlation window -- long enough to
    # resolve lag at a 300 Hz carrier -- is still short against the
    # modulation. Nothing here touches demodulate_ipd.
    audio = _sam_audio(rate_hz=1.0, arc_deg=180.0)
    window, hop, max_lag = int(0.02 * RATE), int(0.01 * RATE), 60
    lags = []
    for start in range(0, len(audio) - window, hop):
        correlation = np.correlate(
            audio[start:start + window, 0], audio[start:start + window, 1],
            "full",
        )
        middle = len(correlation) // 2
        segment = correlation[middle - max_lag: middle + max_lag + 1]
        k = int(np.argmax(segment))
        if 0 < k < len(segment) - 1:
            y0, y1, y2 = segment[k - 1], segment[k], segment[k + 1]
            k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        lags.append((k - max_lag) / RATE)
    lags = np.array(lags)
    expected = PHI_P / (math.pi * CARRIER)

    peak = 0.5 * (lags.max() - lags.min())
    assert peak == pytest.approx(expected, rel=0.03)
    t = np.arange(len(lags)) * hop / RATE
    fitted = abs(2.0 * np.sum(lags * np.exp(-2j * np.pi * 1.0 * t)) / len(lags))
    assert fitted == pytest.approx(expected, rel=0.03)


def test_the_sideband_spectrum_is_the_bessel_series_of_phi_p():
    # A carrier phase-modulated at index phi_p has sidebands at
    # J_k(phi_p)/J_0(phi_p) relative to its own carrier line. This pins
    # phi_p from the magnitude spectrum alone, with no phase estimate
    # anywhere in it -- and it is a far tighter constraint than a peak
    # measurement, because it must hold for every k at once.
    from scipy.special import jv

    arc = 120.0
    phi = sam_depth_from_arc(arc, CARRIER)
    audio = _sam_audio(arc_deg=arc)
    n = len(audio)
    spectrum_l = np.fft.rfft(audio[:, 0])
    spectrum_r = np.fft.rfft(audio[:, 1])

    def line(freq):
        return int(round(freq * n / RATE))

    carrier_line = abs(spectrum_l[line(CARRIER)])
    for k in (1, 2, 3):
        measured = abs(spectrum_l[line(CARRIER + k * MOD_RATE)]) / carrier_line
        assert measured == pytest.approx(
            abs(jv(k, phi)) / abs(jv(0, phi)), rel=1e-4
        )

    # And the two channels' first sidebands are in ANTIPHASE -- the one
    # feature that separates SAM from an amplitude pan, which puts them in
    # phase. Measured: -180.000 deg.
    difference = np.angle(
        spectrum_l[line(CARRIER + MOD_RATE)]
        / spectrum_r[line(CARRIER + MOD_RATE)]
    )
    assert abs(abs(math.degrees(difference)) - 180.0) < 0.01


# --- the bundled preset -----------------------------------------------------


def test_the_bundled_preset_is_patent_tier_and_says_what_it_is():
    session = load_preset("sam-gamma")
    assert session.fidelity == "patent"
    notes = session.notes
    assert "ABANDONED" in notes
    assert "NO MEASURED RECORDING" in notes.upper()


def test_the_bundled_presets_first_block_measures_as_specified():
    session = load_preset("sam-gamma")
    group = next(g for g in session.segments[0].groups if g.sam is not None)
    spec = group.sam
    # Render only the opening minute rather than the whole 22.
    short = load_session_dict({
        "name": "x", "title": "X", "fidelity": "patent", "sample_rate": RATE,
        "segments": [{"duration": 20, "groups": [{
            "name": "sweep",
            "sam": {"carrier_hz": spec.carrier_hz, "rate_hz": spec.rate_hz,
                    "arc_deg": spec.arc_deg, "path": spec.path},
        }]}],
    })
    sig = sam_signature(
        render_timeline(resolve(short)), RATE, spec.rate_hz, spec.carrier_hz
    )
    assert sig["ipd_amplitude_rad"] == pytest.approx(
        2.0 * spec.depth_rad, rel=0.02
    )
    assert sig["itd_peak_s"] == pytest.approx(spec.peak_itd_s(), rel=0.02)
    assert sig["mono_sum_contrast_db"] > 100.0
