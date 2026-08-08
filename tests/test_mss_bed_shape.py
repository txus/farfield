import numpy as np
import pytest
from scipy.signal import welch

from farfield.noise import shaped_fft
from farfield.render import render_session
from farfield.session import BedShape, load_session_dict

RATE = 48000


def _slope_db_per_decade(f, p, lo, hi):
    m = (f >= lo) & (f <= hi) & (p > 0)
    return float(np.polyfit(np.log10(f[m]), 10 * np.log10(p[m]), 1)[0])


def test_shaped_fft_hump_peaks_and_falls_where_configured():
    rng = np.random.default_rng(4)
    shape = BedShape(peak_hz=100.0, rise_db_per_decade=7.7,
                     fall_db_per_decade=-26.3)
    x = shaped_fft(RATE * 40, rng, -20.0, RATE, shape=shape)
    f, p = welch(x, fs=RATE, nperseg=2 ** 16)
    band = (f > 25) & (f < 8000)
    peak = f[band][np.argmax(p[band])]
    assert abs(peak - 100.0) < 25.0, f"hump peaks at {peak:.0f} Hz"
    rise = _slope_db_per_decade(f, p, 30, 70)
    fall = _slope_db_per_decade(f, p, 300, 6000)
    # NO factor of 2: "dB per decade" reads the same on a PSD fit as on an
    # amplitude fit, because power = amplitude^2 cancels 10*log10 against
    # 20*log10 exactly. Ground truth: the single-slope path at -20 measures
    # -19.99 on this very estimator. The `fall` band is far enough from the
    # knee to read the asymptote; `rise` at 30-70 Hz sits within half a
    # decade of a 100 Hz knee and so reads shallower than its asymptote by
    # about a dB -- hence the asymmetric tolerances.
    assert abs(rise - 7.7) < 2.0, f"rise {rise:.1f} dB/decade"
    assert abs(fall - (-26.3)) < 1.5, f"fall {fall:.1f} dB/decade"


def test_shape_absent_is_byte_identical():
    a = shaped_fft(RATE, np.random.default_rng(7), -20.0, RATE)
    b = shaped_fft(RATE, np.random.default_rng(7), -20.0, RATE, shape=None)
    assert np.array_equal(a, b)


def test_bed_shape_renders_through_a_session():
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "output": {"edge_fade_s": 0.0},
        "segments": [{"duration": 40, "groups": [
            {"name": "g", "level_db": -40.0, "harmonics": [1.0],
             "pairs": [{"center": 196.0, "beat": 4.0}]}],
            "bed": {"level_db": -6.0, "color": "brown",
                    "shape": {"peak_hz": 100.0, "rise_db_per_decade": 7.7,
                              "fall_db_per_decade": -26.3}}}],
    })
    audio = render_session(session, seed=5)
    f, p = welch(audio[:, 0], fs=RATE, nperseg=2 ** 16)
    band = (f > 25) & (f < 150)
    peak = f[band][np.argmax(p[band])]
    assert abs(peak - 100.0) < 30.0


@pytest.mark.parametrize("bad,msg", [
    ({"peak_hz": 10.0, "rise_db_per_decade": 7.7,
      "fall_db_per_decade": -26.3}, "peak_hz"),
    ({"peak_hz": 100.0, "rise_db_per_decade": 0.0,
      "fall_db_per_decade": -26.3}, "rise_db_per_decade"),
    ({"peak_hz": 100.0, "rise_db_per_decade": 7.7,
      "fall_db_per_decade": 1.0}, "fall_db_per_decade"),
])
def test_bed_shape_validation(bad, msg):
    with pytest.raises(ValueError, match=msg):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5,
                          "groups": [{"name": "g", "beat": 4.0}],
                          "bed": {"level_db": -20.0, "color": "brown",
                                  "shape": bad}}],
        })


def test_shape_and_slope_together_rejected():
    with pytest.raises(ValueError, match="shape"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "segments": [{"duration": 5,
                          "groups": [{"name": "g", "beat": 4.0}],
                          "bed": {"level_db": -20.0, "color": "brown",
                                  "slope_db_per_decade": -18.0,
                                  "shape": {"peak_hz": 100.0,
                                            "rise_db_per_decade": 7.7,
                                            "fall_db_per_decade": -26.3}}}],
        })
