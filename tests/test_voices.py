import pytest

from farfield.session import Glide, Group, PairSpec, load_session_dict
from farfield.voices import Voice, expand_voices, voice_frequencies


def _group(**kw):
    data = {
        "name": "t", "title": "T", "fidelity": "original",
        "segments": [{"duration": "1:00", "groups": [dict(name="A", **kw)]}],
    }
    return load_session_dict(data).segments[0].groups[0]


def _freqs(voices, ear):
    return sorted((v.freq_start, v.freq_end) for v in voices if v.ear == ear)


def test_stack_form_reproduces_the_legacy_layout():
    g = _group(beat=4.0, carrier_base=200.0, pairs=3, harmonics=[1.0])
    voices = expand_voices(g)
    assert _freqs(voices, "left") == [(200.0, 200.0), (204.0, 204.0), (208.0, 208.0)]
    assert _freqs(voices, "right") == [(204.0, 204.0), (208.0, 208.0), (212.0, 212.0)]
    assert all(v.amplitude == pytest.approx(1.0 / 3.0) for v in voices)


def test_stack_form_harmonics_scale_around_the_base():
    g = _group(beat=4.0, carrier_base=200.0, pairs=1, harmonics=[1.0, 0.5])
    voices = expand_voices(g)
    # normalized amplitudes: 2/3 and 1/3; harmonic 2 has beat 8
    assert _freqs(voices, "right") == [(204.0, 204.0), (208.0, 208.0)]
    amps = sorted(v.amplitude for v in voices if v.ear == "right")
    assert amps == [pytest.approx(1.0 / 3.0), pytest.approx(2.0 / 3.0)]


def test_center_pair_right_high_by_default():
    g = _group(pairs=[{"center": 250.0, "beat": 4.0}], harmonics=[1.0])
    voices = expand_voices(g)
    assert _freqs(voices, "left") == [(248.0, 248.0)]
    assert _freqs(voices, "right") == [(252.0, 252.0)]


def test_center_pair_left_high_flips_the_ears():
    g = _group(pairs=[{"center": 250.0, "beat": 4.0}], high_ear="left",
               harmonics=[1.0])
    voices = expand_voices(g)
    assert _freqs(voices, "left") == [(252.0, 252.0)]
    assert _freqs(voices, "right") == [(248.0, 248.0)]


def test_center_pair_glide_moves_both_ends():
    g = _group(pairs=[{"center": 102.0, "beat": {"from": 4.0, "to": 2.0}}],
               harmonics=[1.0])
    voices = expand_voices(g)
    assert _freqs(voices, "left") == [(100.0, 101.0)]
    assert _freqs(voices, "right") == [(104.0, 103.0)]


def test_explicit_pair_places_measured_values():
    g = _group(pairs=[{"left": 304.8, "right": 300.0}], harmonics=[1.0])
    voices = expand_voices(g)
    assert _freqs(voices, "left") == [(304.8, 304.8)]
    assert _freqs(voices, "right") == [(300.0, 300.0)]


def test_mono_pair_is_one_voice_for_both_ears():
    g = _group(pairs=[{"mono": 50.0}], harmonics=[1.0])
    voices = expand_voices(g)
    assert len(voices) == 1
    assert voices[0].ear == "both"
    assert voices[0].freq_start == 50.0


def test_pairs_form_amplitude_splits_by_pair_count():
    g = _group(pairs=[{"mono": 50.0}, {"center": 50.125, "beat": 0.75}],
               harmonics=[1.0])
    voices = expand_voices(g)
    assert all(v.amplitude == pytest.approx(0.5) for v in voices)


def test_pairs_form_harmonics_keep_the_center():
    g = _group(pairs=[{"center": 100.0, "beat": 2.0}], harmonics=[1.0, 1.0])
    voices = expand_voices(g)
    # h=1: 99/101; h=2 keeps center 100, beat 4: 98/102
    assert _freqs(voices, "left") == [(98.0, 98.0), (99.0, 99.0)]
    assert _freqs(voices, "right") == [(101.0, 101.0), (102.0, 102.0)]


def test_explicit_and_mono_harmonics_scale_frequencies():
    g = _group(pairs=[{"left": 300.0, "right": 304.0}, {"mono": 50.0}],
               harmonics=[1.0, 0.5])
    voices = expand_voices(g)
    assert (600.0, 600.0) in _freqs(voices, "left")
    monos = sorted(v.freq_start for v in voices if v.ear == "both")
    assert monos == [50.0, 100.0]


def test_keys_are_stable_and_distinct():
    g = _group(pairs=[{"center": 100.0, "beat": 2.0}, {"mono": 50.0}],
               harmonics=[1.0, 0.5])
    keys = [v.key for v in expand_voices(g)]
    assert len(keys) == len(set(keys))
    assert (0, 1, "left") in keys and (1, 2, "both") in keys


def test_voice_frequencies_lists_every_endpoint():
    g = _group(pairs=[{"center": 100.0, "beat": {"from": 2.0, "to": 4.0}}],
               harmonics=[1.0])
    assert voice_frequencies(g) == [98.0, 99.0, 101.0, 102.0]
