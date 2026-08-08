import numpy as np
import pytest

from farfield.render import render_session
from farfield.session import load_session_dict

RATE = 48000

PINNED_FIRST_8 = [0.0, 0.019082294059384757, 0.038150716046718845,
                   0.05719140400869658, 0.07619051622209468,
                   0.09513424129029975, 0.11400880821763494,
                   0.13280049645411032]


def _session(groups, seconds=4, **top):
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape",
        "sample_rate": RATE,
        "segments": [{"duration": seconds, "groups": groups}],
    }
    data.update(top)
    return load_session_dict(data)


def _peak_freqs(channel, n_peaks, lo=20.0, hi=1400.0):
    from scipy.signal import find_peaks
    spectrum = np.abs(np.fft.rfft(channel))
    freqs = np.fft.rfftfreq(len(channel), 1.0 / RATE)
    band = (freqs >= lo) & (freqs <= hi)
    peaks, props = find_peaks(spectrum[band], height=0.05 * spectrum[band].max())
    order = np.argsort(props["peak_heights"])[::-1][:n_peaks]
    return sorted(freqs[band][peaks[order]])


def test_byte_identity_of_stack_form_with_legacy_engine():
    # Back-compat guarantee: a stack-form session must render bit-for-bit
    # identical audio to the pre-voices engine. PINNED_FIRST_8 was captured
    # by rendering this exact session at base commit d6422ca, before any
    # voices-based render.py changes.
    # edge_fade_s: 0.0 opts out of the R2 mix-edge fade — the pin covers the
    # synthesis chain, not the (separately tested) new mix edge.
    session = _session([{"name": "A", "beat": 4.0, "carrier_base": 200.0,
                         "pairs": 3, "harmonics": [1.0, 0.35, 0.15]}],
                        output={"edge_fade_s": 0.0})
    audio = render_session(session, seed=3)
    expected = np.array(PINNED_FIRST_8, dtype=np.float64)
    assert np.array_equal(audio[:8, 0], expected)


# Pinned at head after the shaped_fft 1/sqrt(f) fix. This covers the two
# paths the first-8-samples stack pin misses: a GLIDING stack beat (whose
# per-voice ramps recompose the legacy float ops) and a PINK BED (whose
# spectral shaping feeds the mix's peak normalisation, so a 1-ULP change in
# the noise moves every sample of the output). The tolerance is 1e-12,
# roughly 1/100 of a 24-bit LSB.
# Re-pinned when bed levels became RMS-referenced: render_bed now returns
# RMS-normalized output (1/√2 target) instead of peak-normalized.
# Re-pinned again for the shaped_fft 10->20 Hz high-pass taper: steep-slope
# beds were putting most of their RMS below 20 Hz, silently eating the
# level_db budget on content tape never recorded and nobody can hear.
# shaped_fft now zeros below 10 Hz and ramps to full amplitude by 20 Hz,
# which changes every bedded render, this pink bed included.
GLIDE_BED_FIRST_8 = [0.0571199239296625, 0.005747073310990063,
                     0.05195228019050593, 0.055647812642166436,
                     0.06958415325298976, 0.08439328000200887,
                     0.07690855858208288, 0.11796483613917248]
GLIDE_BED_PEAK = 0.7079457843841379
GLIDE_BED_SCATTERED = {
    0: (0.0571199239296625, 0.05712051117336098),
    12345: (0.024211237397336556, -0.05096582713080645),
    99999: (-0.16261310344727925, 0.10399127132014677),
}


def test_gliding_stack_over_a_pink_bed_is_pinned():
    # edge_fade_s: 0.0 opts out of the R2 mix-edge fade — the pin covers the
    # synthesis chain, not the (separately tested) new mix edge.
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{
            "duration": 4,
            "groups": [{"name": "A", "beat": {"from": 10.0, "to": 4.0},
                        "carrier_base": 200.0, "pairs": 3,
                        "harmonics": [1.0, 0.35, 0.15]}],
            "pink": {"level_db": -20.0},
        }],
    }
    audio = render_session(load_session_dict(data), seed=3)
    assert np.allclose(audio[:8, 0], np.array(GLIDE_BED_FIRST_8),
                       rtol=0, atol=1e-12)
    assert abs(float(np.max(np.abs(audio))) - GLIDE_BED_PEAK) <= 1e-12
    for index, (left, right) in GLIDE_BED_SCATTERED.items():
        assert abs(audio[index, 0] - left) <= 1e-12
        assert abs(audio[index, 1] - right) <= 1e-12


def test_center_pair_polarity_swaps_ears():
    right_high = render_session(_session(
        [{"name": "A", "pairs": [{"center": 250.0, "beat": 4.0}],
          "harmonics": [1.0]}]))
    left_high = render_session(_session(
        [{"name": "A", "pairs": [{"center": 250.0, "beat": 4.0}],
          "high_ear": "left", "harmonics": [1.0]}]))
    assert _peak_freqs(right_high[:, 1], 1)[0] == pytest.approx(252.0, abs=0.5)
    assert _peak_freqs(left_high[:, 1], 1)[0] == pytest.approx(248.0, abs=0.5)
    assert _peak_freqs(left_high[:, 0], 1)[0] == pytest.approx(252.0, abs=0.5)


def test_explicit_pair_lands_measured_frequencies():
    audio = render_session(_session(
        [{"name": "deep", "pairs": [{"left": 304.8, "right": 300.0}],
          "harmonics": [1.0]}], seconds=8))
    assert _peak_freqs(audio[:, 0], 1)[0] == pytest.approx(304.8, abs=0.2)
    assert _peak_freqs(audio[:, 1], 1)[0] == pytest.approx(300.0, abs=0.2)


def test_mono_pair_is_phase_locked_across_ears():
    audio = render_session(_session(
        [{"name": "anchor", "pairs": [{"mono": 50.0}], "harmonics": [1.0]}]))
    assert np.array_equal(audio[:, 0], audio[:, 1])


def test_mono_anchor_beats_monaurally_against_a_center_pair():
    audio = render_session(_session(
        [{"name": "g", "pairs": [{"mono": 50.0},
                                 {"center": 50.125, "beat": 0.75}],
          "high_ear": "left", "harmonics": [1.0]}], seconds=30))
    # left ear carries 50.0 and 50.5 -> 0.5 Hz monaural beat
    left_env = np.abs(audio[:, 0])
    smoothed = np.convolve(left_env, np.ones(2400) / 2400, mode="valid")
    spectrum = np.abs(np.fft.rfft(smoothed - smoothed.mean()))
    freqs = np.fft.rfftfreq(len(smoothed), 1.0 / RATE)
    band = (freqs > 0.1) & (freqs < 2.0)
    assert freqs[band][np.argmax(spectrum[band])] == pytest.approx(0.5, abs=0.1)


def test_tremolo_modulates_both_ears_in_phase():
    audio = render_session(_session(
        [{"name": "B", "pairs": [{"center": 300.0, "beat": 4.0}],
          "tremolo": {"rate_hz": 0.5, "depth": 0.5}, "harmonics": [1.0]}],
        seconds=20))
    def env_peak(ch):
        env = np.abs(audio[:, ch])
        smoothed = np.convolve(env, np.ones(2400) / 2400, mode="valid")
        spectrum = np.abs(np.fft.rfft(smoothed - smoothed.mean()))
        freqs = np.fft.rfftfreq(len(smoothed), 1.0 / RATE)
        band = (freqs > 0.1) & (freqs < 2.0)
        return freqs[band][np.argmax(spectrum[band])]
    assert env_peak(0) == pytest.approx(0.5, abs=0.05)
    assert env_peak(1) == pytest.approx(0.5, abs=0.05)
    left_sm = np.convolve(np.abs(audio[:, 0]), np.ones(2400) / 2400, "valid")
    right_sm = np.convolve(np.abs(audio[:, 1]), np.ones(2400) / 2400, "valid")
    lag = np.argmax(np.correlate(left_sm - left_sm.mean(),
                                 right_sm - right_sm.mean(), "same"))
    assert abs(lag - len(left_sm) // 2) < RATE // 10  # in phase


def test_tremolo_phase_is_continuous_across_segments():
    # The tremolo phase is stored in the shared phases dict under
    # (group.name, "tremolo") specifically so a gliding/steady rate keeps
    # ramping across a segment boundary instead of restarting at phase 0.
    # A dropped writeback (or a per-segment key) would restart the sine at
    # the second segment's start and produce a visible kink in the envelope.
    # Segment durations are deliberately NOT a multiple of the 2 s tremolo
    # period (0.5 Hz): at a whole number of periods, restarting the phase
    # at 0 would coincidentally match the continued phase and mask a bug.
    # beat=0 keeps the carrier envelope flat so only the tremolo modulates
    # it -- with a binaural beat present, the beat's own envelope ripple
    # dominates the diffs and hides a dropped phase writeback.
    seg1_s = 6.3
    group = {"name": "B", "pairs": [{"center": 300.0, "beat": 0.0}],
             "tremolo": {"rate_hz": 0.5, "depth": 0.5}, "harmonics": [1.0]}
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": seg1_s, "groups": [group]},
            {"duration": 5.7, "groups": [group]},
        ],
    }
    audio = render_session(load_session_dict(data))
    env = np.abs(audio[:, 0])
    window = 2000
    smoothed = np.convolve(env, np.ones(window) / window, mode="valid")
    diffs = np.abs(np.diff(smoothed))

    seam = round(seg1_s * RATE) - window // 2  # seam position in diffs
    half = int(0.25 * RATE)
    lo, hi = max(0, seam - half), min(len(diffs), seam + half)
    seam_max = diffs[lo:hi].max()
    elsewhere = np.concatenate([diffs[:lo], diffs[hi:]])
    assert seam_max <= 3.0 * elsewhere.max()


def test_free_pair_beat_glide_demodulates_correctly():
    audio = render_session(_session(
        [{"name": "A", "pairs": [{"center": 102.0,
                                  "beat": {"from": 8.0, "to": 4.0}}],
          "harmonics": [1.0]}], seconds=20))
    mono = audio[:, 0] + audio[:, 1]
    def beat_at(seg):
        env = np.abs(mono[seg])
        env = env - env.mean()
        spectrum = np.abs(np.fft.rfft(env))
        freqs = np.fft.rfftfreq(len(env), 1.0 / RATE)
        band = (freqs > 1.0) & (freqs < 20.0)
        return freqs[band][np.argmax(spectrum[band])]
    assert beat_at(slice(0, 4 * RATE)) == pytest.approx(7.6, abs=0.6)
    assert beat_at(slice(-4 * RATE, None)) == pytest.approx(4.3, abs=0.6)


def test_ceiling_applies_to_pair_voices():
    session = _session([{"name": "x", "pairs": [{"center": 1490.0, "beat": 40.0}],
                         "harmonics": [1.0]}])
    with pytest.raises(ValueError, match="ceiling"):
        render_session(session)


def test_negative_frequency_is_rejected():
    session = _session([{"name": "x", "pairs": [{"center": 1.0, "beat": 40.0}],
                         "harmonics": [1.0]}])
    with pytest.raises(ValueError, match="negative"):
        render_session(session)


def test_brown_bed_does_not_trigger_the_pink_warning(recwarn):
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [{"duration": 4,
                      "groups": [{"name": "A", "beat": 4.0, "harmonics": [1.0]}],
                      "bed": {"level_db": -1.3, "color": "brown"}}],
    }
    render_session(load_session_dict(data))
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]
