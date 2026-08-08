"""Isochronic gating: the hard-edged pulse envelope isochronic stimuli need.

The engine's existing tremolo is sinusoidal. An isochronic tone is a train of
discrete pulses, and the difference is measurable: a pulse train has energy at
its rate AND its harmonics, a sinusoidal tremolo has energy only at its rate.
"""

import numpy as np
import pytest

from farfield.oscillators import gate_envelope
from farfield.render import render_session
from farfield.session import load_session_dict

RATE = 48000


def _session(group, seconds=2.0, **top):
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape",
        "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": seconds, "groups": [group]}],
    }
    data.update(top)
    return load_session_dict(data)


# --- the envelope itself -------------------------------------------------


def test_envelope_is_on_for_the_duty_fraction():
    env, _ = gate_envelope(10.0, RATE, RATE, duty=0.25, edge_s=0.0)
    # With zero-width edges the envelope is exactly a rectangle, so the mean
    # is the duty cycle to within one sample per cycle.
    assert env.mean() == pytest.approx(0.25, abs=1e-3)
    assert set(np.unique(env)) <= {0.0, 1.0}


def test_envelope_edges_take_the_requested_time():
    # 10 Hz, 5 ms edges: the rise occupies 5 ms = 240 samples at 48 kHz.
    env, _ = gate_envelope(10.0, RATE, RATE, duty=0.5, edge_s=0.005)
    first_cycle = env[: RATE // 10]
    rising = np.flatnonzero((first_cycle > 0.0) & (first_cycle < 1.0))
    # Two ramps per cycle, 240 samples each, minus the endpoints that land
    # exactly on 0 or 1.
    assert 460 <= len(rising) <= 482
    # And the ramp is monotone on the way up.
    up = first_cycle[: int(0.005 * RATE)]
    assert np.all(np.diff(up) >= 0)


def test_envelope_reaches_full_scale_and_true_zero():
    env, _ = gate_envelope(40.0, RATE, RATE, duty=0.5, edge_s=0.005)
    assert env.max() == pytest.approx(1.0, abs=1e-9)
    assert env.min() == 0.0


def test_raised_cosine_edges_beat_a_hard_switch_on_splatter():
    # The reason the edges exist at all: a 0 ms edge splatters broadband
    # energy. Compare the high-frequency content of the two envelopes.
    hard, _ = gate_envelope(40.0, RATE * 4, RATE, duty=0.5, edge_s=0.0)
    soft, _ = gate_envelope(40.0, RATE * 4, RATE, duty=0.5, edge_s=0.005)

    def hf_energy(x):
        spec = np.abs(np.fft.rfft(x - x.mean()))
        freqs = np.fft.rfftfreq(len(x), 1.0 / RATE)
        return float(np.sum(spec[freqs > 1000.0] ** 2))

    assert hf_energy(soft) < 0.02 * hf_energy(hard)


def test_envelope_phase_hands_off_in_cycles():
    a, handoff = gate_envelope(10.0, 1200, RATE, duty=0.5, edge_s=0.0)
    b, _ = gate_envelope(
        10.0, 1200, RATE, duty=0.5, edge_s=0.0, initial_phase=handoff
    )
    whole, _ = gate_envelope(10.0, 2400, RATE, duty=0.5, edge_s=0.0)
    assert np.array_equal(np.concatenate([a, b]), whole)


# --- through the renderer ------------------------------------------------


def _envelope_spectrum(channel, lo, hi):
    env = np.abs(channel)
    env = env - env.mean()
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), 1.0 / RATE)
    band = (freqs > lo) & (freqs < hi)
    return freqs[band], spec[band]


def test_gated_tone_has_beat_rate_energy_in_its_envelope():
    audio = render_session(_session({
        "name": "iso", "pairs": [{"mono": 400.0}], "harmonics": [1.0],
        "gate": {"rate_hz": 40.0, "depth": 1.0, "duty": 0.5, "edge_ms": 5.0},
    }, seconds=4))
    freqs, spec = _envelope_spectrum(audio[:, 0], 5.0, 300.0)
    assert freqs[np.argmax(spec)] == pytest.approx(40.0, abs=0.5)


def test_a_pulse_train_carries_harmonics_a_tremolo_does_not():
    """The measurable difference between isochronic and AM.

    An isochronic gate and a sinusoidal tremolo must not be the same
    stimulus wearing different names.
    """
    group = {"name": "g", "pairs": [{"mono": 400.0}], "harmonics": [1.0]}
    iso = render_session(_session(
        {**group, "gate": {"rate_hz": 40.0, "depth": 1.0, "duty": 0.5,
                           "edge_ms": 5.0}}, seconds=4))
    am = render_session(_session(
        {**group, "tremolo": {"rate_hz": 40.0, "depth": 1.0}}, seconds=4))

    def ratio(audio):
        freqs, spec = _envelope_spectrum(audio[:, 0], 5.0, 300.0)
        def at(f):
            return spec[np.argmin(np.abs(freqs - f))]
        return float(at(120.0) / at(40.0))

    # The gate's third harmonic is a real component of a square-ish pulse
    # train; the sinusoidal tremolo has essentially nothing there. Measured
    # here: 0.093 for the gate against 3e-4 for the tremolo. The gate's
    # third harmonic is well below the 1/3 a true square wave would give
    # because 5 ms edges are 0.2 of a cycle at 40 Hz — a substantial part of
    # the waveform is ramp, which is the point of having edges at all.
    assert ratio(iso) > 0.05
    assert ratio(iso) > 20.0 * ratio(am)


def test_gate_depth_controls_how_far_the_off_state_drops():
    full = render_session(_session({
        "name": "g", "pairs": [{"mono": 400.0}], "harmonics": [1.0],
        "gate": {"rate_hz": 20.0, "depth": 1.0, "duty": 0.5, "edge_ms": 5.0},
    }))
    half = render_session(_session({
        "name": "g", "pairs": [{"mono": 400.0}], "harmonics": [1.0],
        "gate": {"rate_hz": 20.0, "depth": 0.5, "duty": 0.5, "edge_ms": 5.0},
    }))
    # Sample well inside an off window (phase 0.75 of a 50 ms cycle).
    idx = int(0.75 * RATE / 20.0)
    window = slice(idx - 100, idx + 100)
    assert np.max(np.abs(full[window, 0])) < 1e-9
    # depth 0.5 leaves half the amplitude, and both renders are peak
    # normalised to the same ceiling, so the ratio is what is checked.
    assert np.max(np.abs(half[window, 0])) > 0.3 * np.max(np.abs(half[:, 0]))


def test_gate_phase_is_continuous_across_segments():
    # Segment length deliberately NOT a whole number of gate cycles: at a
    # whole number, restarting the phase at 0 would coincidentally match.
    group = {"name": "g", "pairs": [{"mono": 400.0}], "harmonics": [1.0],
             "gate": {"rate_hz": 10.0, "depth": 1.0, "duty": 0.5,
                      "edge_ms": 5.0}}
    split = render_session(load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 1.03, "groups": [group], "overlap": 0},
                     {"duration": 0.97, "groups": [group], "overlap": 0}],
    }))
    # A restart would put a gate edge at the seam. Locate the seam and check
    # the envelope there matches what a continuous 10 Hz gate predicts:
    # 1.03 s * 10 Hz = 10.3 cycles, phase 0.3 -> inside the ON plateau.
    seam = int(1.03 * RATE)
    assert np.max(np.abs(split[seam + 20 : seam + 200, 0])) > 0.1


def test_glide_of_the_gate_rate_is_supported():
    audio = render_session(_session({
        "name": "g", "pairs": [{"mono": 400.0}], "harmonics": [1.0],
        "gate": {"rate_hz": {"from": 10.0, "to": 20.0}, "depth": 1.0,
                 "duty": 0.5, "edge_ms": 5.0},
    }, seconds=8))

    def rate_in(seg):
        freqs, spec = _envelope_spectrum(audio[seg, 0], 3.0, 40.0)
        return freqs[np.argmax(spec)]

    assert rate_in(slice(0, 2 * RATE)) == pytest.approx(10.5, abs=1.5)
    assert rate_in(slice(-2 * RATE, None)) == pytest.approx(19.5, abs=1.5)


# --- validation ----------------------------------------------------------


@pytest.mark.parametrize("gate, message", [
    ({"depth": 1.0, "duty": 0.5}, "rate_hz"),
    ({"rate_hz": 40.0, "duty": 0.5}, "depth"),
    ({"rate_hz": 40.0, "depth": 1.0}, "duty"),
    ({"rate_hz": 0.0, "depth": 1.0, "duty": 0.5}, "positive"),
    ({"rate_hz": 40.0, "depth": 1.5, "duty": 0.5}, r"depth must be in \[0, 1\]"),
    ({"rate_hz": 40.0, "depth": 1.0, "duty": 0.0}, r"duty must be in \(0, 1\)"),
    ({"rate_hz": 40.0, "depth": 1.0, "duty": 1.0}, r"duty must be in \(0, 1\)"),
    ({"rate_hz": 40.0, "depth": 1.0, "duty": 0.5, "edge_ms": -1.0}, "edge_ms"),
    ({"rate_hz": 40.0, "depth": 1.0, "duty": 0.5, "edge_ms": 500.0}, "edge_ms"),
    # 2 x 10 ms at 40 Hz is 0.8 of a cycle, which does not fit in a 0.5 duty.
    ({"rate_hz": 40.0, "depth": 1.0, "duty": 0.5, "edge_ms": 10.0},
     "edges do not fit"),
])
def test_invalid_gates_are_rejected_at_load(gate, message):
    with pytest.raises(ValueError, match=message):
        _session({"name": "g", "pairs": [{"mono": 400.0}], "gate": gate})


def test_a_gliding_rate_is_checked_at_its_fastest_point():
    # Legal at 10 Hz, illegal at 60 Hz: catching this only at the start
    # would let the renderer silently clip the ramps mid-glide.
    with pytest.raises(ValueError, match="edges do not fit"):
        _session({"name": "g", "pairs": [{"mono": 400.0}],
                  "gate": {"rate_hz": {"from": 10.0, "to": 60.0},
                           "depth": 1.0, "duty": 0.5, "edge_ms": 8.0}})


def test_tremolo_depth_of_one_is_now_accepted():
    # 100% AM is the Experiment 1 calibration stimulus.
    audio = render_session(_session({
        "name": "g", "pairs": [{"mono": 500.0}], "harmonics": [1.0],
        "tremolo": {"rate_hz": 40.0, "depth": 1.0},
    }, seconds=2))
    freqs, spec = _envelope_spectrum(audio[:, 0], 5.0, 300.0)
    assert freqs[np.argmax(spec)] == pytest.approx(40.0, abs=0.5)


def test_tremolo_depth_above_one_is_still_rejected():
    with pytest.raises(ValueError, match=r"depth must be in \[0, 1\]"):
        _session({"name": "g", "pairs": [{"mono": 400.0}],
                  "tremolo": {"rate_hz": 4.0, "depth": 1.1}})


# --- byte identity -------------------------------------------------------


def test_an_absent_gate_changes_nothing():
    """The opt-in guarantee, stated as a test.

    A session with no gate key and one with gate: null must produce
    bit-identical audio, and neither may differ from the pin in
    tests/test_render_voices.py (which this suite does not restate).
    """
    group = {"name": "A", "beat": 4.0, "carrier_base": 200.0, "pairs": 3,
             "harmonics": [1.0, 0.35, 0.15]}
    absent = render_session(_session(group, seconds=4), seed=3)
    explicit_null = render_session(
        _session({**group, "gate": None}, seconds=4), seed=3
    )
    assert np.array_equal(absent, explicit_null)


def test_gate_appears_in_describe_and_the_sidecar():
    from farfield.cli import describe_lines
    from farfield.render import sidecar
    from farfield.timeline import resolve

    session = _session({
        "name": "iso", "pairs": [{"mono": 400.0}], "harmonics": [1.0],
        "gate": {"rate_hz": 40.0, "depth": 1.0, "duty": 0.5, "edge_ms": 5.0},
    })
    timeline = resolve(session)
    text = "\n".join(describe_lines(session, timeline))
    assert "gate 40.00 Hz duty 0.50" in text
    entry = sidecar(session, timeline)["layers"][0]["gate"]
    assert entry == {"rate_hz": 40.0, "depth": 1.0, "duty": 0.5,
                     "edge_ms": 5.0}


def test_sidecar_omits_gate_when_unused():
    from farfield.render import sidecar
    from farfield.timeline import resolve

    session = _session({"name": "A", "beat": 4.0, "harmonics": [1.0]})
    assert "gate" not in sidecar(session, resolve(session))["layers"][0]
