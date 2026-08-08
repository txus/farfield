import numpy as np
import pytest

from tests.support import load_preset
from farfield.timeline import resolve
from farfield.voices import fundamental_voices

NAMES = ["focus-10-mss", "focus-12-mss", "focus-15-mss",
         "focus-21-mss"]

# 12-TET A440 pitches the MSS grid was retuned to (mss-results.json
# tuning_system): every rendered carrier must be a member of one of these
# pairs, i.e. within half its beat of one of these centres.
G_TRIAD_HZ = [49.0, 97.9988, 195.9979, 246.9416, 293.6649, 391.9951,
              493.8818, 587.3314, 783.9948, 987.7684]


@pytest.mark.parametrize("name", NAMES)
def test_preset_loads_and_is_measured_tier(name):
    session = load_preset(name)
    assert session.fidelity == "measured-mss"
    assert "mp3" in session.notes.lower()
    assert "music" in session.notes.lower()


@pytest.mark.parametrize("name", NAMES)
def test_every_carrier_sits_on_the_g_triad_grid(name):
    session = load_preset(name)
    timeline = resolve(session)
    for layer in timeline.layers:
        for v in fundamental_voices(layer.group):
            for f in (v.freq_start, v.freq_end):
                assert min(abs(f - c) for c in G_TRIAD_HZ) < 9.0, (
                    f"{name}: {f} Hz is not near any G-triad centre")


@pytest.mark.parametrize("name", NAMES)
def test_rotation_is_configured_above_the_low_layers(name):
    session = load_preset(name)
    rotating = [g.name for s in session.segments for g in s.groups
                if g.rotation is not None]
    assert rotating, f"{name} has no rotating groups"
    for s in session.segments:
        for g in s.groups:
            if g.rotation is not None:
                assert g.rotation.period_s == 30.0
                # 0.98, not the analysis's eps = 0.80: that eps is a
                # sideband index read under a LINEAR-pan model, and the
                # recordings' summed power (0.12 dB at f_lfo) settles the
                # law as equal-power. Solved against the same ILD
                # estimator, equal-power depth 0.98 is what reproduces the
                # tape's fitted ~14.4 dB ILD sinusoid; 0.80 would give
                # only 9.5 dB.
                assert abs(g.rotation.depth - 0.98) < 0.02


@pytest.mark.parametrize("name", NAMES)
def test_bed_and_texture_present_on_every_segment(name):
    session = load_preset(name)
    for i, s in enumerate(session.segments):
        assert s.pink is not None, f"{name} segment {i} has no bed"
        assert s.pink.shape is not None, f"{name} segment {i} bed has no hump"
        assert s.texture is not None, f"{name} segment {i} has no texture"
        assert s.texture.pan_period_s == 7.5
