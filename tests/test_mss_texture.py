import numpy as np
import pytest
from scipy.signal import welch

from farfield.render import render_session
from farfield.session import load_session_dict

RATE = 48000


def _texture_session(period_s=7.5, ild_amplitude_db=8.2, band=(2000, 8000)):
    return load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 60, "groups": [
            {"name": "g", "level_db": -40.0, "harmonics": [1.0],
             "pairs": [{"center": 196.0, "beat": 4.0}]}],
            "bed": {"level_db": -20.0, "color": "brown",
                    "stereo": {"mode": "crossfade", "lfo_period_s": 9.0,
                               "depth_db": 3.0}},
            "texture": {"band_hz": list(band), "level_db": -12.0,
                        "pan": {"period_s": period_s, "ild_amplitude_db": ild_amplitude_db,
                                "phase_deg": 0.0}}}],
    })


def _band_ild_trace(audio, lo, hi, rate=RATE, hop=None):
    from scipy.signal import butter, sosfilt
    sos = butter(4, [lo, hi], btype="bandpass", fs=rate, output="sos")
    l = sosfilt(sos, audio[:, 0]); r = sosfilt(sos, audio[:, 1])
    n = rate; hop = hop or rate // 4
    trace = []
    for s in range(0, len(l) - n, hop):
        trace.append(10 * np.log10((l[s:s+n] ** 2).mean()
                                   / (r[s:s+n] ** 2).mean()))
    return np.array(trace), hop / rate


def _fit(trace, dt, period_s):
    tt = np.arange(len(trace)) * dt
    w = 2 * np.pi / period_s
    A = np.column_stack([np.sin(w * tt), np.cos(w * tt), np.ones_like(tt)])
    c, *_ = np.linalg.lstsq(A, trace, rcond=None)
    return float(np.hypot(c[0], c[1]))


def test_texture_pans_at_its_own_period():
    audio = render_session(_texture_session(period_s=7.5))
    trace, dt = _band_ild_trace(audio, 3000, 6000)
    # scan periods; the texture's own must win
    best = max((7.5, 9.0, 5.0, 12.0), key=lambda p: _fit(trace, dt, p))
    assert best == 7.5, f"dominant pan period {best} s, expected 7.5"
    assert _fit(trace, dt, 7.5) > 3.0


def test_texture_pan_does_not_leak_below_its_band():
    audio = render_session(_texture_session())
    trace, dt = _band_ild_trace(audio, 400, 1200)
    assert _fit(trace, dt, 7.5) < 1.0, "texture pan leaks below its band"


def test_texture_is_band_limited():
    audio = render_session(_texture_session(band=(2000, 8000)))
    f, p = welch(audio[:, 0], fs=RATE, nperseg=2 ** 15)
    inband = p[(f > 3000) & (f < 6000)].mean()
    below = p[(f > 600) & (f < 1200)].mean()
    above = p[(f > 12000) & (f < 16000)].mean()
    assert 10 * np.log10(inband / above) > 12.0
    assert inband > below  # the texture dominates its own band over the bed


def test_texture_absent_is_byte_identical():
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 10, "groups": [
            {"name": "g", "level_db": 0.0, "harmonics": [1.0],
             "pairs": [{"center": 196.0, "beat": 4.0}]}],
            "bed": {"level_db": -20.0, "color": "brown"}}],
    }
    a = render_session(load_session_dict(data), seed=3)
    b = render_session(load_session_dict(data), seed=3)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bad,msg", [
    ({"band_hz": [8000, 2000], "level_db": -12.0,
      "pan": {"period_s": 7.5, "ild_amplitude_db": 8.0, "phase_deg": 0.0}}, "band_hz"),
    ({"band_hz": [10, 8000], "level_db": -12.0,
      "pan": {"period_s": 7.5, "ild_amplitude_db": 8.0, "phase_deg": 0.0}}, "band_hz"),
    ({"band_hz": [2000, 8000], "level_db": -12.0,
      "pan": {"period_s": 0.0, "ild_amplitude_db": 8.0, "phase_deg": 0.0}}, "period_s"),
    ({"band_hz": [2000, 8000], "level_db": -12.0,
      "pan": {"period_s": 7.5, "ild_amplitude_db": 40.0, "phase_deg": 0.0}}, "ild_amplitude_db"),
])
def test_texture_validation(bad, msg):
    with pytest.raises(ValueError, match=msg):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5,
                          "groups": [{"name": "g", "beat": 4.0}],
                          "bed": {"level_db": -20.0, "color": "brown"},
                          "texture": bad}],
        })


def test_texture_requires_a_bed():
    with pytest.raises(ValueError, match="bed"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5,
                          "groups": [{"name": "g", "beat": 4.0}],
                          "texture": {"band_hz": [2000, 8000],
                                      "level_db": -12.0,
                                      "pan": {"period_s": 7.5,
                                              "ild_amplitude_db": 8.0,
                                              "phase_deg": 0.0}}}],
        })
