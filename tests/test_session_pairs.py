import pytest

from farfield.session import (
    Glide,
    PairSpec,
    Tremolo,
    load_session_dict,
)


def _session(groups, defaults=None):
    data = {
        "name": "t", "title": "T", "fidelity": "original",
        "segments": [{"duration": "1:00", "groups": groups}],
    }
    if defaults:
        data["defaults"] = defaults
    return load_session_dict(data)


def test_center_pair_parses():
    g = _session([{"name": "A", "pairs": [{"center": 250.0, "beat": 4.0}]}]).segments[0].groups[0]
    assert g.pairs_spec == (PairSpec(kind="center", center=250.0, beat=4.0,
                                     left=None, right=None, mono=None),)


def test_center_pair_beat_glide():
    g = _session([{"name": "A", "pairs": [{"center": 102.0,
                                           "beat": {"from": 4.115, "to": 3.886}}]}]).segments[0].groups[0]
    assert g.pairs_spec[0].beat == Glide(4.115, 3.886)


def test_explicit_pair_parses():
    g = _session([{"name": "A", "pairs": [{"left": 304.8, "right": 300.0}]}]).segments[0].groups[0]
    p = g.pairs_spec[0]
    assert p.kind == "explicit" and p.left == 304.8 and p.right == 300.0


def test_mono_pair_parses():
    g = _session([{"name": "A", "pairs": [{"mono": 50.0}]}]).segments[0].groups[0]
    assert g.pairs_spec[0].kind == "mono" and g.pairs_spec[0].mono == 50.0


def test_mixed_pair_kinds_in_one_group():
    g = _session([{"name": "A", "pairs": [
        {"mono": 50.0}, {"center": 50.125, "beat": 0.75}]}]).segments[0].groups[0]
    assert [p.kind for p in g.pairs_spec] == ["mono", "center"]


def test_high_ear_defaults_right_and_parses_left():
    default = _session([{"name": "A", "beat": 4.0}]).segments[0].groups[0]
    assert default.high_ear == "right"
    left = _session([{"name": "A", "high_ear": "left",
                      "pairs": [{"center": 100.0, "beat": 1.5}]}]).segments[0].groups[0]
    assert left.high_ear == "left"


def test_tremolo_parses_with_rate_glide():
    g = _session([{"name": "B", "pairs": [{"center": 300.5, "beat": 3.8}],
                   "tremolo": {"rate_hz": {"from": 0.58, "to": 0.48},
                               "depth": 0.22}}]).segments[0].groups[0]
    assert g.tremolo == Tremolo(rate_hz=Glide(0.58, 0.48), depth=0.22)


def test_stack_form_still_parses_with_no_pairs_spec():
    g = _session([{"name": "A", "beat": 4.0}]).segments[0].groups[0]
    assert g.pairs_spec is None
    assert g.carrier_base == 200.0 and g.pairs == 3


def test_pairs_form_beat_bounds_come_from_first_pair():
    g = _session([{"name": "A", "pairs": [
        {"center": 102.0, "beat": {"from": 4.115, "to": 3.886}},
        {"mono": 50.0}]}]).segments[0].groups[0]
    assert g.beat_bounds() == (4.115, 3.886)


@pytest.mark.parametrize("pair,needle", [
    ({"center": 100.0, "beat": 4.0, "mono": 50.0}, "exactly one"),
    ({}, "exactly one"),
    ({"left": 300.0, "right": 304.0, "beat": 4.0}, "beat"),
    ({"mono": 50.0, "beat": 1.0}, "beat"),
    ({"center": 100.0}, "beat"),
    ({"left": 300.0}, "right"),
    ({"right": 300.0}, "left"),
])
def test_bad_pairs_are_rejected(pair, needle):
    with pytest.raises(ValueError, match=needle):
        _session([{"name": "A", "pairs": [pair]}])


def test_empty_pairs_list_is_rejected():
    with pytest.raises(ValueError, match="pairs"):
        _session([{"name": "A", "pairs": []}])


def test_mixing_stack_and_pairs_forms_is_rejected():
    with pytest.raises(ValueError, match="either"):
        _session([{"name": "A", "beat": 4.0, "carrier_base": 200.0,
                   "pairs": [{"center": 100.0, "beat": 1.5}]}])


def test_group_level_beat_on_pairs_form_is_rejected():
    with pytest.raises(ValueError, match="beat"):
        _session([{"name": "A", "beat": 4.0,
                   "pairs": [{"center": 100.0, "beat": 1.5}]}])


def test_bad_high_ear_is_rejected():
    with pytest.raises(ValueError, match="high_ear"):
        _session([{"name": "A", "high_ear": "up",
                   "pairs": [{"center": 100.0, "beat": 1.5}]}])


# depth 1.0 used to be rejected (the bound was [0, 1)). It is now accepted:
# 100% amplitude modulation is the textbook ASSR calibration stimulus and
# nothing physical forbids it. See
# tests/test_gate.py::test_tremolo_depth_of_one_is_now_accepted. No bundled
# preset sits at the boundary, so no existing render changed.
@pytest.mark.parametrize("trem,needle", [
    ({"rate_hz": 0.5, "depth": 1.01}, "depth"),
    ({"rate_hz": 0.5, "depth": -0.1}, "depth"),
    ({"rate_hz": 0.0, "depth": 0.2}, "rate"),
    ({"rate_hz": {"from": 0.5, "to": -0.1}, "depth": 0.2}, "rate"),
])
def test_bad_tremolo_is_rejected(trem, needle):
    with pytest.raises(ValueError, match=needle):
        _session([{"name": "A", "pairs": [{"center": 100.0, "beat": 1.5}],
                   "tremolo": trem}])


def test_emerge_on_a_pairs_form_primary_group_is_rejected():
    with pytest.raises(ValueError, match="emerge"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "original",
            "segments": [{"duration": "1:00", "groups": [
                {"name": "A", "pairs": [{"center": 100.0, "beat": 4.0}]}]}],
            "emerge": {},
        })
