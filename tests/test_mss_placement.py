import numpy as np
import pytest

from farfield.render import render_session
from farfield.session import load_session_dict

RATE = 48000


def _demod(audio, f0, ch, rate=RATE):
    t = np.arange(len(audio)) / rate
    z = audio[:, ch] * np.exp(-2j * np.pi * f0 * t)
    return z.mean() * 2.0


def _placement_session(crossfeed_db=-3.4, phase_deg=40.0):
    return load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 20, "groups": [
            {"name": "g", "level_db": 0.0, "harmonics": [1.0],
             "high_ear": "right",
             "pairs": [{"center": 98.0, "beat": 1.5}],
             "placement": {"crossfeed_db": crossfeed_db,
                           "crossfeed_phase_deg": phase_deg}}]}],
    })


def test_crossfeed_level_matches_the_spec():
    audio = render_session(_placement_session(crossfeed_db=-3.4))
    # high member 98.75 Hz is the right-ear voice; its crossfeed copy is in L.
    main = abs(_demod(audio, 98.75, 1))
    cross = abs(_demod(audio, 98.75, 0))
    ratio_db = 20 * np.log10(cross / main)
    assert abs(ratio_db - (-3.4)) < 0.3, f"crossfeed {ratio_db:.2f} dB"


def test_crossfeed_phase_signs_are_opposite_for_the_two_members():
    audio = render_session(_placement_session(phase_deg=40.0))
    hi_ipd = np.degrees(np.angle(_demod(audio, 98.75, 0)
                                 / _demod(audio, 98.75, 1)))
    lo_ipd = np.degrees(np.angle(_demod(audio, 97.25, 1)
                                 / _demod(audio, 97.25, 0)))
    assert hi_ipd * lo_ipd < 0, f"same-sign IPDs {hi_ipd:.1f}, {lo_ipd:.1f}"
    assert abs(abs(hi_ipd) - 40.0) < 3.0
    assert abs(abs(lo_ipd) - 40.0) < 3.0


def test_placement_absent_leaves_ears_isolated():
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 20, "groups": [
            {"name": "g", "level_db": 0.0, "harmonics": [1.0],
             "pairs": [{"center": 98.0, "beat": 1.5}]}]}],
    })
    audio = render_session(session)
    leak_db = 20 * np.log10(abs(_demod(audio, 98.75, 0))
                            / abs(_demod(audio, 98.75, 1)))
    assert leak_db < -60.0, f"unexpected crossfeed {leak_db:.1f} dB"


def test_per_ear_tremolo_runs_at_different_rates():
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 60, "groups": [
            {"name": "g", "level_db": 0.0, "harmonics": [1.0],
             "pairs": [{"center": 49.0, "beat": 0.3}],
             "tremolo": {"left": {"rate_hz": 0.25, "depth": 0.83},
                         "right": {"rate_hz": 0.5, "depth": 0.83}}}]}],
    })
    audio = render_session(session)
    win = RATE // 2
    for ch, expected in ((0, 0.25), (1, 0.5)):
        env = np.sqrt(np.convolve(audio[:, ch] ** 2,
                                  np.ones(win) / win, "valid"))[::win // 4]
        dt = (win // 4) / RATE
        f = np.fft.rfftfreq(len(env), d=dt)
        p = np.abs(np.fft.rfft(env - env.mean())) ** 2
        band = (f > 0.05) & (f < 1.5)
        peak = f[band][np.argmax(p[band])]
        assert abs(peak - expected) < 0.05, (
            f"channel {ch} tremolo at {peak:.3f} Hz, expected {expected}")


def test_single_form_tremolo_unchanged():
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 10, "groups": [
            {"name": "g", "level_db": 0.0, "harmonics": [1.0],
             "pairs": [{"center": 200.0, "beat": 4.0}],
             "tremolo": {"rate_hz": 0.7, "depth": 0.5}}]}],
    }
    a = render_session(load_session_dict(data))
    b = render_session(load_session_dict(data))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bad,msg", [
    ({"crossfeed_db": 1.0, "crossfeed_phase_deg": 0.0}, "crossfeed_db"),
    ({"crossfeed_db": -70.0, "crossfeed_phase_deg": 0.0}, "crossfeed_db"),
    ({"crossfeed_db": -3.0, "crossfeed_phase_deg": 200.0}, "crossfeed_phase_deg"),
    ({"crossfeed_db": -3.0}, "crossfeed_phase_deg"),
])
def test_placement_validation(bad, msg):
    with pytest.raises(ValueError, match=msg):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5, "groups": [
                {"name": "g", "beat": 4.0, "placement": bad}]}],
        })


def test_placement_rejected_on_a_mono_pair():
    with pytest.raises(ValueError, match="mono"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5, "groups": [
                {"name": "g", "harmonics": [1.0],
                 "pairs": [{"mono": 50.0}],
                 "placement": {"crossfeed_db": -3.0,
                               "crossfeed_phase_deg": 10.0}}]}],
        })


def test_mixed_tremolo_forms_rejected():
    with pytest.raises(ValueError, match="tremolo"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5, "groups": [
                {"name": "g", "beat": 4.0,
                 "tremolo": {"rate_hz": 0.5, "depth": 0.5,
                             "left": {"rate_hz": 0.25, "depth": 0.5}}}]}],
        })


def test_rotation_and_placement_together_rejected():
    # They compose destructively (a crossfeed copy routed to the opposite
    # ear picks up the opposite rotation gain and counter-rotates against
    # its own parent), and no measured MSS material uses both on one group.
    with pytest.raises(ValueError, match="rotation"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5, "groups": [
                {"name": "g", "harmonics": [1.0],
                 "pairs": [{"center": 196.0, "beat": 4.0}],
                 "rotation": {"period_s": 30.0, "depth": 0.98,
                              "phase_deg": 0.0},
                 "placement": {"crossfeed_db": -3.4,
                               "crossfeed_phase_deg": 41.0}}]}],
        })
