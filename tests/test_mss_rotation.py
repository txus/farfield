import numpy as np
import pytest

from farfield.render import render_session
from farfield.session import load_session_dict

RATE = 48000


def _rot_session(phase_deg=0.0, depth=0.8, period_s=30.0, seconds=60):
    return load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [
            {"duration": seconds, "groups": [
                {"name": "g", "level_db": 0.0, "harmonics": [1.0],
                 "pairs": [{"center": 196.0, "beat": 4.0}],
                 "rotation": {"period_s": period_s, "depth": depth,
                              "phase_deg": phase_deg}}]},
        ],
    })


def _ild_trace(audio, f0, rate=RATE, block=RATE // 2):
    """Per-block ILD (dB) of the tone at f0, by complex demodulation."""
    t = np.arange(len(audio)) / rate
    n = len(audio) // block
    out = []
    for ch in (0, 1):
        z = audio[:, ch] * np.exp(-2j * np.pi * f0 * t)
        zb = z[:n * block].reshape(n, block).mean(axis=1)
        out.append(np.abs(zb))
    return 20 * np.log10(out[0] / out[1]), n, block / rate


def _fit_sine(trace, dt, period_s):
    tt = np.arange(len(trace)) * dt
    w = 2 * np.pi / period_s
    A = np.column_stack([np.sin(w * tt), np.cos(w * tt), np.ones_like(tt)])
    c, *_ = np.linalg.lstsq(A, trace, rcond=None)
    return float(np.hypot(c[0], c[1])), float(np.arctan2(c[1], c[0]))


def test_rotation_pans_at_the_configured_period_and_depth():
    audio = render_session(_rot_session())
    # 196 Hz centre, 4 Hz beat -> members at 194 and 198.
    trace, n, dt = _ild_trace(audio, 198.0)
    amp, _ = _fit_sine(trace, dt, 30.0)
    # depth 0.8 -> peak ILD 20*log10(sqrt(1.8/0.2)) = 9.54 dB per member
    # measured as a fitted sinusoid amplitude in dB (the trace is sinusoidal
    # in linear amplitude, so the dB fit reads lower than the peak).
    assert 6.0 < amp < 12.0, f"rotation ILD amplitude {amp:.2f} dB"


def test_pair_members_counter_rotate():
    audio = render_session(_rot_session())
    hi, n, dt = _ild_trace(audio, 198.0)
    lo, _, _ = _ild_trace(audio, 194.0)
    _, ph_hi = _fit_sine(hi, dt, 30.0)
    _, ph_lo = _fit_sine(lo, dt, 30.0)
    d = abs(np.degrees(ph_hi - ph_lo)) % 360.0
    assert abs(d - 180.0) < 5.0, f"members {d:.1f} deg apart, expected 180"


def test_rotation_preserves_total_power():
    audio = render_session(_rot_session())
    block = RATE // 2
    n = len(audio) // block
    p = (audio[:n * block] ** 2).sum(axis=1).reshape(n, block).mean(axis=1)
    mid = p[4:-4]  # skip session edges
    spread_db = 10 * np.log10(mid.max() / mid.min())
    assert spread_db < 0.5, f"power varies {spread_db:.2f} dB across the pan"


def test_rotation_phase_offset_shifts_the_lfo():
    a0 = render_session(_rot_session(phase_deg=0.0))
    a90 = render_session(_rot_session(phase_deg=90.0))
    t0, _, dt = _ild_trace(a0, 198.0)
    t90, _, _ = _ild_trace(a90, 198.0)
    _, p0 = _fit_sine(t0, dt, 30.0)
    _, p90 = _fit_sine(t90, dt, 30.0)
    d = (np.degrees(p90 - p0)) % 360.0
    assert abs(d - 90.0) < 5.0, f"phase shifted {d:.1f} deg, expected 90"


def test_rotation_lfo_is_continuous_across_a_seam():
    # Two segments of the same rotating group: the LFO runs on absolute
    # session time, so a fit over the whole session (spanning the seam)
    # matches a fit over the first segment alone.
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [
            {"duration": 40, "overlap": 4, "groups": [
                {"name": "g", "level_db": 0.0, "harmonics": [1.0],
                 "pairs": [{"center": 196.0, "beat": 4.0}],
                 "rotation": {"period_s": 30.0, "depth": 0.8,
                              "phase_deg": 0.0}}]},
            {"duration": 40, "groups": [
                {"name": "g", "level_db": 0.0, "harmonics": [1.0],
                 "pairs": [{"center": 196.0, "beat": 4.0}],
                 "rotation": {"period_s": 30.0, "depth": 0.8,
                              "phase_deg": 0.0}}]},
        ],
    })
    audio = render_session(session)
    trace, _, dt = _ild_trace(audio, 198.0)
    amp_all, ph_all = _fit_sine(trace, dt, 30.0)
    k = int(30 / dt)
    amp_first, ph_first = _fit_sine(trace[:k], dt, 30.0)
    d = abs(np.degrees(ph_all - ph_first)) % 360.0
    assert min(d, 360 - d) < 8.0, f"LFO phase jumps {d:.1f} deg at the seam"
    assert abs(amp_all - amp_first) < 1.5


def test_rotation_absent_is_byte_identical():
    plain = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 5, "groups": [
            {"name": "g", "level_db": 0.0, "harmonics": [1.0],
             "pairs": [{"center": 196.0, "beat": 4.0}]}]}],
    }
    a = render_session(load_session_dict(plain))
    b = render_session(load_session_dict(plain))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("bad,msg", [
    ({"period_s": 0.0, "depth": 0.8, "phase_deg": 0.0}, "period_s"),
    ({"period_s": 30.0, "depth": 0.0, "phase_deg": 0.0}, "depth"),
    # The ceiling is 0.99, not 0.95: the measured MSS rotation needs 0.98
    # to reach the tape's ILD swing under the equal-power law (see
    # mss-results.json rotation.pan_law_detail). 1.0 would null one ear.
    ({"period_s": 30.0, "depth": 1.0, "phase_deg": 0.0}, "depth"),
    ({"period_s": 30.0, "depth": 0.8}, "phase_deg"),
    ({"depth": 0.8, "phase_deg": 0.0}, "period_s"),
])
def test_rotation_validation(bad, msg):
    with pytest.raises(ValueError, match=msg):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5, "groups": [
                {"name": "g", "beat": 4.0, "rotation": bad}]}],
        })
