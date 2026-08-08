import numpy as np
from scipy.signal import find_peaks

from farfield.analysis import (
    ANALYSIS_RATE,
    carrier_frequencies,
    carrier_span,
    decimate_to,
    goertzel_track,
    spectrogram,
)
from tests.support import load_preset
from farfield.session import load_session_dict
from farfield.timeline import resolve

RATE = 48000


def _two_tones(f1: float, f2: float, seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)


def test_decimation_reduces_the_rate_by_the_expected_factor():
    signal, rate = decimate_to(_two_tones(200.0, 204.0), RATE, ANALYSIS_RATE)
    assert rate == ANALYSIS_RATE
    assert abs(len(signal) - 4 * ANALYSIS_RATE) <= 1


def test_decimation_preserves_the_low_band():
    signal, rate = decimate_to(_two_tones(200.0, 204.0), RATE, ANALYSIS_RATE)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / rate)
    assert abs(freqs[np.argmax(spectrum)] - 200.0) < 1.5


def test_spectrogram_resolves_four_hertz_spacing():
    signal, rate = decimate_to(_two_tones(200.0, 204.0), RATE, ANALYSIS_RATE)
    freqs, mags = spectrogram(signal, rate, 4096, 300, 190.0, 215.0)
    column = mags[len(mags) // 2]
    peaks, _ = find_peaks(column, height=0.3 * column.max())
    found = sorted(freqs[peaks])
    assert len(found) == 2
    assert abs(found[0] - 200.0) < 1.0
    assert abs(found[1] - 204.0) < 1.0


def test_spectrogram_crops_to_the_requested_band():
    signal, rate = decimate_to(_two_tones(200.0, 204.0), RATE, ANALYSIS_RATE)
    freqs, mags = spectrogram(signal, rate, 4096, 300, 190.0, 215.0)
    assert freqs[0] >= 190.0
    assert freqs[-1] <= 215.0
    assert mags.shape[1] == len(freqs)


def test_spectrogram_frame_count_follows_the_hop():
    signal, rate = decimate_to(_two_tones(200.0, 204.0), RATE, ANALYSIS_RATE)
    _, mags = spectrogram(signal, rate, 4096, 300, 190.0, 215.0)
    assert mags.shape[0] == 1 + (len(signal) - 4096) // 300


def test_carrier_frequencies_lists_every_tone():
    session = load_session_dict(
        {
            "name": "t", "title": "T", "fidelity": "original",
            "segments": [
                {"duration": 10,
                 "groups": [{"name": "A", "beat": 4.0, "carrier_base": 200.0,
                             "pairs": 3, "harmonics": [1.0]}]}
            ],
        }
    )
    assert carrier_frequencies(resolve(session)) == [
        200.0, 204.0, 208.0, 212.0
    ]


def test_carrier_frequencies_exclude_harmonic_partials():
    # Harmonics are mastering-level detail: they belong in validation (a
    # partial can break the fusion ceiling) but not in the meters, the
    # spectrogram band, or the sidecar's carrier lists.
    session = load_session_dict(
        {
            "name": "t", "title": "T", "fidelity": "original",
            "segments": [
                {"duration": 10,
                 "groups": [{"name": "A", "beat": 4.0, "carrier_base": 200.0,
                             "pairs": 3, "harmonics": [1.0, 0.5]}]}
            ],
        }
    )
    timeline = resolve(session)
    assert carrier_frequencies(timeline) == [200.0, 204.0, 208.0, 212.0]

    from farfield.render import sidecar
    layer = sidecar(session, timeline)["layers"][0]
    for side in ("left", "right"):
        carriers = layer["carriers_start"][side]
        assert carriers == sorted(set(carriers))  # deduped
        assert max(carriers) <= 212.0             # no 2f partials


def test_carrier_span_pads_the_extremes():
    session = load_session_dict(
        {
            "name": "t", "title": "T", "fidelity": "original",
            "segments": [
                {"duration": 10,
                 "groups": [{"name": "A", "beat": 4.0, "carrier_base": 200.0,
                             "pairs": 3, "harmonics": [1.0]}]}
            ],
        }
    )
    low, high = carrier_span(resolve(session), pad_hz=25.0)
    assert low == 175.0
    assert high == 237.0


def test_carrier_span_covers_groups_with_different_bases():
    low, high = carrier_span(resolve(load_preset("focus-10")))
    assert low < 140.0
    assert high > 200.0


def test_carrier_span_never_goes_below_zero():
    session = load_session_dict(
        {
            "name": "t", "title": "T", "fidelity": "original",
            "segments": [
                {"duration": 10,
                 "groups": [{"name": "A", "beat": 2.0, "carrier_base": 10.0,
                             "pairs": 1, "harmonics": [1.0]}]}
            ],
        }
    )
    assert carrier_span(resolve(session), pad_hz=25.0)[0] >= 0.0


def test_goertzel_isolates_a_tone_from_its_four_hertz_neighbour():
    t = np.arange(RATE * 2) / RATE
    pure_204 = np.sin(2 * np.pi * 204.0 * t)
    block = RATE // 4  # 0.25 s puts the neighbour exactly on a null
    at_204 = goertzel_track(pure_204, RATE, 204.0, block, block)
    at_200 = goertzel_track(pure_204, RATE, 200.0, block, block)
    assert at_200.max() < 0.05 * at_204.max()


def test_goertzel_tracks_amplitude_over_time():
    t = np.arange(RATE * 4) / RATE
    signal = np.sin(2 * np.pi * 200.0 * t)
    signal[RATE * 2 :] *= 0.25
    block = RATE // 4
    track = goertzel_track(signal, RATE, 200.0, block, block)
    early = track[: len(track) // 4].mean()
    late = track[-len(track) // 4 :].mean()
    assert 0.15 < late / early < 0.35


def test_goertzel_track_length_follows_the_hop():
    signal = np.sin(2 * np.pi * 200.0 * np.arange(RATE) / RATE)
    block = RATE // 4
    hop = block // 2
    track = goertzel_track(signal, RATE, 200.0, block, hop)
    assert len(track) == 1 + (RATE - block) // hop


def test_carrier_frequencies_cover_pairs_form_groups():
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape",
        "segments": [{"duration": 10, "groups": [
            {"name": "g", "harmonics": [1.0], "high_ear": "left",
             "pairs": [{"mono": 50.0}, {"center": 50.125, "beat": 0.75},
                        {"left": 304.8, "right": 300.0}]}]}],
    })
    freqs = carrier_frequencies(resolve(session))
    assert freqs == [49.75, 50.0, 50.5, 300.0, 304.8]


def test_carrier_frequencies_unchanged_for_stacks():
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "original",
        "segments": [{"duration": 10,
                      "groups": [{"name": "A", "beat": 4.0, "carrier_base": 200.0,
                                  "pairs": 3, "harmonics": [1.0]}]}],
    })
    assert carrier_frequencies(resolve(session)) == [200.0, 204.0, 208.0, 212.0]
