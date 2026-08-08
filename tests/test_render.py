import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from farfield.render import (
    db_to_gain,
    fade_window,
    normalize,
    render_session,
    sidecar,
    write_wav,
)
from farfield.session import load_session_dict
from farfield.timeline import resolve

RATE = 48000


def _one_segment(**group) -> dict:
    base = {"name": "A", "beat": 4.0, "pairs": 1, "harmonics": [1.0]}
    base.update(group)
    return {
        "name": "t",
        "title": "T",
        "fidelity": "original",
        "sample_rate": RATE,
        "segments": [{"duration": 4, "groups": [base]}],
    }


def _peak_freq(channel: np.ndarray, rate: int = RATE) -> float:
    spectrum = np.abs(np.fft.rfft(channel))
    return float(np.fft.rfftfreq(len(channel), 1.0 / rate)[np.argmax(spectrum)])


def _band_energy(channel: np.ndarray, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(channel))
    freqs = np.fft.rfftfreq(len(channel), 1.0 / RATE)
    band = (freqs >= low) & (freqs <= high)
    return float(np.sum(spectrum[band] ** 2))


def test_db_to_gain_is_the_amplitude_convention():
    assert abs(db_to_gain(0.0) - 1.0) < 1e-12
    assert abs(db_to_gain(-20.0) - 0.1) < 1e-12
    assert abs(db_to_gain(-6.0) - 0.5) < 0.01


def test_fade_window_is_flat_without_fades():
    assert np.allclose(fade_window(100, 0, 0), 1.0)


def test_fade_window_ramps_from_zero_to_one():
    window = fade_window(100, 20, 20)
    assert window[0] < 1e-9
    assert abs(window[50] - 1.0) < 1e-9
    assert window[-1] < 0.2


def test_fade_window_is_equal_power():
    # Two complementary equal-power ramps sum to 1.0 in squared amplitude.
    rising = fade_window(100, 100, 0)
    falling = fade_window(100, 0, 100)
    assert np.allclose(rising**2 + falling**2, 1.0, atol=1e-6)


def test_fade_window_scales_down_when_fades_overlap_the_whole_window():
    # fade_in + fade_out > n_samples must not let the fade-out write clobber
    # the fade-in's tail (no dead flat-1.0 head, no discontinuity).
    window = fade_window(100, 100, 100)
    assert window[0] < 0.05
    deltas = np.abs(np.diff(window))
    assert deltas.max() < 0.05  # smooth ramp, no step


def test_fade_window_scaled_overlap_has_no_step():
    window = fade_window(100, 60, 60)
    deltas = np.diff(window)
    # Monotone rising then falling: no sign flip beyond the single peak,
    # and no single-sample jump larger than the local ramp slope.
    peak = int(np.argmax(window))
    assert np.all(deltas[:peak] >= -1e-12)
    assert np.all(deltas[peak:] <= 1e-12)
    assert np.abs(deltas).max() < 0.05


def test_single_pair_puts_the_carriers_where_expected():
    audio = render_session(load_session_dict(_one_segment(carrier_base=200.0)))
    assert abs(_peak_freq(audio[:, 0]) - 200.0) < 1.0
    assert abs(_peak_freq(audio[:, 1]) - 204.0) < 1.0


def test_septon_produces_all_six_carrier_peaks():
    data = _one_segment(carrier_base=200.0, pairs=3)
    audio = render_session(load_session_dict(data))
    left = np.abs(np.fft.rfft(audio[:, 0]))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / RATE)
    for expected in (200.0, 204.0, 208.0):
        near = np.abs(freqs - expected) < 0.5
        assert left[near].max() > 0.2 * left.max()


def test_harmonics_add_carrier_pairs_at_beat_multiples():
    data = _one_segment(carrier_base=200.0, harmonics=[1.0, 0.5])
    audio = render_session(load_session_dict(data))
    right = np.abs(np.fft.rfft(audio[:, 1]))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / RATE)
    # Fundamental gives 204 Hz; the second harmonic gives 208 Hz.
    assert right[np.abs(freqs - 204.0) < 0.5].max() > 0.2 * right.max()
    assert right[np.abs(freqs - 208.0) < 0.5].max() > 0.1 * right.max()


def test_a_high_harmonic_over_the_fusion_ceiling_is_rejected():
    # Fundamental stack tops out at 1400 + 3*20 = 1460 Hz (under the 1500 Hz
    # ceiling), but the 3rd harmonic reaches 1400 + 3*3*20 = 1580 Hz, over
    # it. Validation must look at every harmonic, not just the fundamental.
    data = _one_segment(
        carrier_base=1400.0,
        beat=20.0,
        pairs=3,
        harmonics=[1.0, 0.5, 0.25],
    )
    with pytest.raises(ValueError):
        render_session(load_session_dict(data))


def test_neither_channel_contains_the_beat():
    # The premise of the technique: no amplitude modulation in either ear.
    #
    # NOTE (diagnosed): with pairs=1 this channel is a single pure
    # 200 Hz sinusoid, so its raw rectified samples necessarily dip toward
    # zero at every zero-crossing (twice per cycle) -- that is a property of
    # abs(sin(...)) at any sample rate, not amplitude modulation. The
    # smallest nonzero rectified sample near a crossing is bounded below by
    # the sample-grid slope there (amplitude * 2*pi*freq/rate), which for
    # this fixture is ~0.0185 -- just over the 0.01 filter threshold below,
    # so a raw min/max ratio on unsmoothed samples is mathematically stuck
    # near 1.0 regardless of implementation correctness and was removed.
    # The running-average check below is the sound envelope estimator (it
    # averages over ~8 cycles) and is what actually verifies the absence of
    # beat-rate modulation.
    audio = render_session(load_session_dict(_one_segment(carrier_base=200.0)))
    envelope = np.abs(audio[RATE : 3 * RATE, 0])
    smoothed = np.convolve(envelope, np.ones(2000) / 2000, mode="valid")
    assert (np.max(smoothed) - np.min(smoothed)) / np.mean(smoothed) < 0.05


def test_the_mono_sum_does_contain_a_four_hz_envelope():
    audio = render_session(load_session_dict(_one_segment(carrier_base=200.0)))
    mono = audio[:, 0] + audio[:, 1]
    envelope = np.abs(mono)
    envelope = envelope - np.mean(envelope)
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), 1.0 / RATE)
    band = (freqs > 1.0) & (freqs < 20.0)
    assert abs(freqs[band][np.argmax(spectrum[band])] - 4.0) < 0.5


def test_group_level_offsets_are_honoured():
    data = _one_segment()
    data["segments"] = [
        {
            "duration": 4,
            "groups": [
                {"name": "A", "beat": 4.0, "carrier_base": 200.0, "pairs": 1,
                 "harmonics": [1.0], "level_db": 0.0},
                {"name": "B", "beat": 4.0, "carrier_base": 600.0, "pairs": 1,
                 "harmonics": [1.0], "level_db": -12.0},
            ],
        }
    ]
    audio = render_session(load_session_dict(data))
    loud = _band_energy(audio[:, 0], 195.0, 215.0)
    quiet = _band_energy(audio[:, 0], 595.0, 615.0)
    ratio_db = 10.0 * np.log10(quiet / loud)
    assert abs(ratio_db - (-12.0)) < 1.5


def test_pink_level_offset_is_honoured():
    data = _one_segment(carrier_base=200.0)
    data["segments"][0]["pink"] = {"level_db": -20.0}
    audio = render_session(load_session_dict(data))
    tones = _band_energy(audio[:, 0], 195.0, 215.0)
    noise = _band_energy(audio[:, 0], 2000.0, 6000.0)
    assert noise < tones


def test_loud_pink_triggers_the_patent_warning():
    data = _one_segment(carrier_base=200.0)
    data["segments"][0]["pink"] = {"level_db": -6.0}
    with pytest.warns(UserWarning, match="10 dB"):
        render_session(load_session_dict(data))


def test_quiet_pink_does_not_warn(recwarn):
    data = _one_segment(carrier_base=200.0)
    data["segments"][0]["pink"] = {"level_db": -15.0}
    render_session(load_session_dict(data))
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_butt_joined_segments_have_no_click():
    data = {
        "name": "t", "title": "T", "fidelity": "original",
        "sample_rate": RATE,
        "segments": [
            {"duration": 2, "groups": [{"name": "A", "beat": 10.0, "pairs": 1,
                                        "harmonics": [1.0]}]},
            {"duration": 2, "groups": [{"name": "A", "beat": 4.0, "pairs": 1,
                                        "harmonics": [1.0]}]},
        ],
    }
    audio = render_session(load_session_dict(data))
    deltas = np.abs(np.diff(audio[:, 0]))
    boundary = 2 * RATE
    local = deltas[boundary - 20 : boundary + 20].max()
    assert local <= np.percentile(deltas, 99.99) * 3.0


def test_glide_sweeps_the_carrier_over_the_segment():
    data = _one_segment(carrier_base=200.0, beat={"from": 12.0, "to": 4.0})
    audio = render_session(load_session_dict(data))
    head = _peak_freq(audio[: RATE // 2, 1], RATE)
    tail = _peak_freq(audio[-RATE // 2 :, 1], RATE)
    assert head > 209.0
    assert tail < 207.0


def test_output_is_stereo_float_and_the_right_length():
    audio = render_session(load_session_dict(_one_segment()))
    assert audio.shape == (4 * RATE, 2)
    assert audio.dtype == np.float64


def test_normalisation_hits_the_requested_peak():
    audio = normalize(np.ones((10, 2)) * 4.0, -3.0)
    assert abs(float(np.max(np.abs(audio))) - 10.0 ** (-3.0 / 20.0)) < 1e-9


def test_render_is_reproducible_for_a_given_seed():
    data = _one_segment()
    data["segments"][0]["pink"] = {"level_db": -20.0}
    session = load_session_dict(data)
    assert np.array_equal(render_session(session, seed=7), render_session(session, seed=7))


def test_write_wav_round_trips(tmp_path: Path):
    audio = render_session(load_session_dict(_one_segment()))
    path = tmp_path / "out.wav"
    write_wav(path, audio, RATE)
    read, rate = sf.read(path)
    assert rate == RATE
    assert read.shape == audio.shape


def test_sidecar_is_json_serialisable_and_describes_layers():
    session = load_session_dict(_one_segment(carrier_base=200.0))
    payload = sidecar(session, resolve(session))
    json.dumps(payload)
    assert payload["name"] == "t"
    assert payload["fidelity"] == "original"
    layer = payload["layers"][0]
    assert layer["group"] == "A"
    assert layer["start_s"] == 0.0
    assert layer["end_s"] == 4.0
    assert layer["carriers_start"]["left"] == [200.0]
    assert layer["carriers_start"]["right"] == [204.0]


def test_stacked_carriers_do_modulate_within_a_channel():
    # The shipped default (pairs=3) deliberately produces monaural amplitude
    # beats inside each channel: adjacent same-channel carriers are a beat
    # apart. This is the patent's stacked arrangement, not a defect, and the
    # visualizer's note must not claim otherwise.
    audio = render_session(
        load_session_dict(_one_segment(carrier_base=200.0, pairs=3))
    )
    envelope = np.abs(audio[RATE : 3 * RATE, 0])
    envelope = np.convolve(envelope, np.ones(200) / 200, mode="valid")
    envelope = envelope - np.mean(envelope)
    spectrum = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), 1.0 / RATE)
    band = (freqs > 1.0) & (freqs < 20.0)
    assert abs(freqs[band][np.argmax(spectrum[band])] - 4.0) < 0.5


def test_the_emerge_join_holds_its_level():
    data = {
        "name": "t", "title": "T", "fidelity": "original",
        "sample_rate": RATE,
        "segments": [
            {"duration": 6, "groups": [{"name": "A", "beat": 4.0, "pairs": 1,
                                        "harmonics": [1.0]}]},
        ],
        "emerge": {"duration": 4, "target_beat": 15.0},
    }
    session = load_session_dict(data)
    timeline = resolve(session)
    audio = render_session(session)
    boundary = timeline.layers[-1].start_sample
    smoothed = np.convolve(
        np.abs(audio[:, 0]), np.ones(2000) / 2000, mode="same"
    )
    before = float(np.mean(smoothed[boundary - RATE // 2 : boundary]))
    after = smoothed[boundary : boundary + RATE // 2]
    assert float(np.min(after)) > 0.5 * before
