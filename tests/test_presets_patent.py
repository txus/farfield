import numpy as np
import pytest

from tests.support import load_preset, preset_metadata, preset_names
from farfield.render import render_session
from farfield.timeline import resolve


def test_sleep_90_is_exactly_ninety_minutes():
    timeline = resolve(load_preset("sleep-90"))
    assert timeline.total_samples == 90 * 60 * 48000


def test_sleep_90_is_marked_patent_fidelity():
    assert load_preset("sleep-90").fidelity == "patent"


def test_sleep_90_has_the_patent_segment_boundaries():
    timeline = resolve(load_preset("sleep-90"))
    starts = sorted({layer.start_sample for layer in timeline.layers})
    expected_minutes = [0, 5, 20, 40, 65, 80]
    assert starts == [m * 60 * 48000 for m in expected_minutes]
    assert timeline.total_samples == 90 * 60 * 48000


def test_sleep_90_layers_two_groups_where_the_patent_says_so():
    session = load_preset("sleep-90")
    # Every segment but the sustained 40-65 minute hold carries two groups.
    counts = [len(s.groups) for s in session.segments]
    assert counts == [2, 2, 2, 1, 2, 2]


def test_sleep_90_uses_the_patent_level_offsets():
    session = load_preset("sleep-90")
    offsets = [
        sorted(g.level_db for g in segment.groups)
        for segment in session.segments
    ]
    assert offsets[0] == [-15.0, 0.0]
    assert offsets[1] == [-20.0, 0.0]
    assert offsets[2] == [-10.0, 0.0]


def test_sleep_90_pink_levels_match_the_patent():
    levels = [
        s.pink.level_db if s.pink else None for s in load_preset("sleep-90").segments
    ]
    assert levels == [-20.0, -15.0, -10.0, -10.0, -10.0, -15.0]


def test_sleep_90_descends_through_the_bands():
    session = load_preset("sleep-90")
    primaries = [
        max(s.groups, key=lambda g: g.level_db).beat_bounds()[0]
        for s in session.segments
    ]
    # alpha, theta, delta, deeper delta, then back up.
    assert primaries[0] > primaries[1] > primaries[2] > primaries[3]
    assert primaries[5] > primaries[3]


def test_sleep_90_has_no_emerge_block():
    assert load_preset("sleep-90").emerge is None


def test_wake_uses_the_patent_four_hundred_hertz_pair():
    group = load_preset("wake").segments[0].groups[0]
    assert group.carrier_base == 400.0
    assert group.pairs == 1
    assert group.beat_bounds() == (16.0, 16.0)


def test_wake_is_five_minutes():
    assert resolve(load_preset("wake")).total_samples == 5 * 60 * 48000


def test_wake_renders_the_four_hundred_and_sixteen_hertz_carrier():
    audio = render_session(load_preset("wake"))
    window = audio[48000 * 10 : 48000 * 12, 1]
    spectrum = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), 1.0 / 48000)
    assert abs(freqs[np.argmax(spectrum)] - 416.0) < 1.0


def test_preset_names_include_the_patent_tier():
    assert {"sleep-90", "wake"} <= set(preset_names())


def test_preset_metadata_carries_the_fidelity_tier():
    entry = next(m for m in preset_metadata() if m["name"] == "sleep-90")
    assert entry["fidelity"] == "patent"
    assert entry["title"]


def test_every_preset_loads_and_resolves():
    # 30 minutes comfortably exceeds every preset's fixed entry + emerge
    # time; presets without hold segments ignore the argument.
    for name in preset_names():
        timeline = resolve(load_preset(name, total_duration_s=1800.0))
        assert timeline.total_samples > 0
