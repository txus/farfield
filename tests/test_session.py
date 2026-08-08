import textwrap
from pathlib import Path

import pytest

from farfield.session import (
    FIDELITY_TIERS,
    Emerge,
    Glide,
    Group,
    Session,
    load_session,
    load_session_dict,
    parse_duration,
)

MINIMAL = {
    "name": "t",
    "title": "T",
    "fidelity": "original",
    "segments": [{"duration": "1:00", "groups": [{"name": "A", "beat": 4.0}]}],
}


def test_parse_duration_accepts_seconds():
    assert parse_duration(90) == 90.0
    assert parse_duration(1.5) == 1.5


def test_parse_duration_accepts_minutes_and_seconds():
    assert parse_duration("1:30") == 90.0
    assert parse_duration("90:00") == 5400.0


def test_parse_duration_accepts_hours():
    assert parse_duration("1:30:00") == 5400.0


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_parse_duration_rejects_negative():
    with pytest.raises(ValueError):
        parse_duration(-5)


def test_parse_duration_rejects_negative_components():
    with pytest.raises(ValueError):
        parse_duration("1:-30")


def test_defaults_fill_in_omitted_group_fields():
    data = dict(MINIMAL)
    data["defaults"] = {"carrier_base": 140.0, "pairs": 5, "level_db": -3.0}
    session = load_session_dict(data)
    group = session.segments[0].groups[0]
    assert group.carrier_base == 140.0
    assert group.pairs == 5
    assert group.level_db == -3.0


def test_group_defaults_apply_when_no_defaults_block():
    session = load_session_dict(MINIMAL)
    group = session.segments[0].groups[0]
    assert group.carrier_base == 200.0
    assert group.pairs == 3
    assert group.harmonics == (1.0, 0.35, 0.15)
    assert group.level_db == 0.0


def test_explicit_group_values_override_defaults():
    data = dict(MINIMAL)
    data["defaults"] = {"carrier_base": 140.0}
    data["segments"] = [
        {
            "duration": "1:00",
            "groups": [{"name": "A", "beat": 4.0, "carrier_base": 300.0}],
        }
    ]
    assert load_session_dict(data).segments[0].groups[0].carrier_base == 300.0


def test_beat_mapping_becomes_a_glide():
    data = dict(MINIMAL)
    data["segments"] = [
        {
            "duration": "1:00",
            "groups": [{"name": "A", "beat": {"from": 10.0, "to": 7.0}}],
        }
    ]
    beat = load_session_dict(data).segments[0].groups[0].beat
    assert beat == Glide(10.0, 7.0)


def test_beat_bounds_for_constant_and_glide():
    assert Group("A", 4.0, 200.0, 3, (1.0,), 0.0).beat_bounds() == (4.0, 4.0)
    assert Group(
        "A", Glide(10.0, 7.0), 200.0, 3, (1.0,), 0.0
    ).beat_bounds() == (10.0, 7.0)


def test_hold_segment_absorbs_the_remaining_time():
    data = dict(MINIMAL)
    data["segments"] = [
        {"duration": "2:00", "groups": [{"name": "A", "beat": 10.0}]},
        {"duration": "hold", "groups": [{"name": "A", "beat": 4.0}]},
    ]
    session = load_session_dict(data, total_duration_s=600.0)
    assert session.segments[1].duration_s == 480.0


def test_hold_sizing_compensates_for_crossfade_overlaps():
    # The 20 s crossfade pulls the hold earlier, so the hold must absorb
    # 20 s more for the resolved session to land on the requested total.
    data = dict(MINIMAL)
    data["segments"] = [
        {"duration": "2:00", "overlap": "0:20",
         "groups": [{"name": "A", "beat": 10.0}]},
        {"duration": "hold", "groups": [{"name": "A", "beat": 4.0}]},
    ]
    session = load_session_dict(data, total_duration_s=600.0)
    assert session.segments[1].duration_s == 500.0


def test_hold_segments_split_the_remainder_evenly():
    data = dict(MINIMAL)
    data["segments"] = [
        {"duration": "hold", "groups": [{"name": "A", "beat": 10.0}]},
        {"duration": "hold", "groups": [{"name": "A", "beat": 4.0}]},
    ]
    session = load_session_dict(data, total_duration_s=600.0)
    assert session.segments[0].duration_s == 300.0
    assert session.segments[1].duration_s == 300.0


def test_emerge_time_is_subtracted_before_holds_are_sized():
    data = dict(MINIMAL)
    data["segments"] = [{"duration": "hold", "groups": [{"name": "A", "beat": 4.0}]}]
    data["emerge"] = {"duration": "3:00", "target_beat": 15.0}
    session = load_session_dict(data, total_duration_s=600.0)
    # 600 - 180 emerge, plus the 2 s the emerge crossfade eats back.
    assert session.segments[0].duration_s == 422.0


def test_default_total_is_used_when_no_duration_argument():
    data = dict(MINIMAL)
    data["default_total"] = "10:00"
    data["segments"] = [{"duration": "hold", "groups": [{"name": "A", "beat": 4.0}]}]
    assert load_session_dict(data).segments[0].duration_s == 600.0


def test_hold_without_any_total_is_an_error():
    data = dict(MINIMAL)
    data["segments"] = [{"duration": "hold", "groups": [{"name": "A", "beat": 4.0}]}]
    with pytest.raises(ValueError, match="total"):
        load_session_dict(data)


def test_total_too_short_for_holds_is_an_error():
    data = dict(MINIMAL)
    data["segments"] = [
        {"duration": "9:00", "groups": [{"name": "A", "beat": 10.0}]},
        {"duration": "hold", "groups": [{"name": "A", "beat": 4.0}]},
    ]
    with pytest.raises(ValueError, match="too short"):
        load_session_dict(data, total_duration_s=540.0)


def test_emerge_defaults():
    data = dict(MINIMAL)
    data["emerge"] = {}
    assert load_session_dict(data).emerge == Emerge(180.0, 15.0)


def test_emerge_false_disables_it():
    data = dict(MINIMAL)
    data["emerge"] = False
    assert load_session_dict(data).emerge is None


def test_emerge_absent_yields_none():
    assert load_session_dict(MINIMAL).emerge is None


def test_pink_spec_defaults():
    data = dict(MINIMAL)
    data["segments"] = [
        {
            "duration": "1:00",
            "groups": [{"name": "A", "beat": 4.0}],
            "pink": {"level_db": -20.0},
        }
    ]
    pink = load_session_dict(data).segments[0].pink
    assert pink.level_db == -20.0
    assert pink.comb_sweep_hz == 0.125
    assert pink.pan_rate_hz == 0.05
    assert pink.algorithm == "fft"


def test_pink_absent_yields_none():
    assert load_session_dict(MINIMAL).segments[0].pink is None


def test_fidelity_must_be_a_known_tier():
    data = dict(MINIMAL)
    data["fidelity"] = "vibes"
    with pytest.raises(ValueError, match="fidelity"):
        load_session_dict(data)


def test_fidelity_tiers_are_the_documented_four():
    assert FIDELITY_TIERS == frozenset(
        {"measured-tape", "measured-mss", "patent", "original"})


def test_a_session_needs_at_least_one_segment():
    data = dict(MINIMAL)
    data["segments"] = []
    with pytest.raises(ValueError, match="segment"):
        load_session_dict(data)


def test_a_segment_needs_at_least_one_group():
    data = dict(MINIMAL)
    data["segments"] = [{"duration": "1:00", "groups": []}]
    with pytest.raises(ValueError, match="group"):
        load_session_dict(data)


def test_sample_rate_and_peak_defaults():
    session = load_session_dict(MINIMAL)
    assert session.sample_rate == 48000
    assert session.peak_dbfs == -3.0


def test_load_session_reads_yaml(tmp_path: Path):
    path = tmp_path / "s.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: demo
            title: Demo
            fidelity: original
            segments:
              - duration: "0:30"
                overlap: "0:05"
                groups:
                  - name: A
                    beat: 8.0
            """
        )
    )
    session = load_session(path)
    assert isinstance(session, Session)
    assert session.name == "demo"
    assert session.segments[0].overlap_s == 5.0
