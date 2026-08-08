"""Parsing and validation of the `sam:` group form.

Every rejection names the offending key, matching the house style set by
_parse_rotation / _parse_placement.
"""

import math

import pytest

from farfield.session import (
    SAM_CARRIER_CEILING_HZ,
    load_session_dict,
    sam_depth_from_arc,
)

BASE = {"carrier_hz": 300.0, "rate_hz": 40.0, "arc_deg": 180.0}


def _group(sam=None, **extra):
    group = {"name": "s"}
    if sam is not None:
        group["sam"] = sam
    group.update(extra)
    return load_session_dict({
        "name": "t", "title": "T", "fidelity": "patent",
        "segments": [{"duration": 4, "groups": [group]}],
    }).segments[0].groups[0]


def _reject(sam=None, **extra):
    with pytest.raises(ValueError) as exc:
        _group(sam, **extra)
    return str(exc.value)


def test_a_minimal_sam_group_parses():
    spec = _group(dict(BASE)).sam
    assert spec.carrier_hz == 300.0
    assert spec.rate_hz == 40.0
    assert spec.path == "closed"
    assert spec.arc_deg == 180.0


def test_arc_deg_reproduces_the_research_docs_derived_phi_p():
    # The doc derives phi_p ~= 0.48 rad for a full ear-to-ear arc at 300 Hz
    # from the 510 us maximum ITD, independently of this code path.
    assert sam_depth_from_arc(180.0, 300.0) == pytest.approx(0.481, abs=0.001)
    assert _group(dict(BASE)).sam.depth_rad == pytest.approx(0.481, abs=0.001)


def test_arc_deg_and_depth_rad_agree_at_the_same_setting():
    by_arc = _group(dict(BASE)).sam.depth_rad
    by_depth = _group(
        {"carrier_hz": 300.0, "rate_hz": 40.0, "depth_rad": by_arc}
    ).sam.depth_rad
    assert by_depth == pytest.approx(by_arc)


def test_peak_itd_is_phi_p_over_pi_f_s():
    spec = _group(dict(BASE)).sam
    assert spec.peak_itd_s() == pytest.approx(
        spec.depth_rad / (math.pi * spec.carrier_hz)
    )
    # A full arc at any carrier is the head's own 510 us traversal.
    assert spec.peak_itd_s() == pytest.approx(510e-6, rel=1e-3)


def test_a_sam_group_reports_its_modulation_rate_as_its_beat():
    assert _group(dict(BASE)).beat_bounds() == (40.0, 40.0)


def test_carrier_above_the_ceiling_is_rejected_with_the_reason():
    message = _reject({**BASE, "carrier_hz": 900.0, "arc_deg": 10.0})
    assert "carrier_hz" in message
    assert str(int(SAM_CARRIER_CEILING_HZ)) in message
    assert "interaural phase" in message


def test_carrier_below_twenty_hz_is_rejected():
    assert "carrier_hz" in _reject({**BASE, "carrier_hz": 5.0})


def test_missing_carrier_is_rejected():
    assert "carrier_hz" in _reject({"rate_hz": 40.0, "arc_deg": 90.0})


def test_missing_rate_is_rejected():
    assert "rate_hz" in _reject({"carrier_hz": 300.0, "arc_deg": 90.0})


def test_non_positive_rate_is_rejected():
    assert "rate_hz" in _reject({**BASE, "rate_hz": 0.0})


def test_rate_at_or_above_half_the_carrier_is_rejected():
    message = _reject({**BASE, "rate_hz": 150.0})
    assert "rate_hz" in message and "half the carrier" in message


def test_both_depth_parameterizations_at_once_are_rejected():
    message = _reject({**BASE, "depth_rad": 0.4})
    assert "arc_deg" in message and "depth_rad" in message


def test_neither_depth_parameterization_is_rejected():
    message = _reject({"carrier_hz": 300.0, "rate_hz": 40.0})
    assert "arc_deg" in message and "depth_rad" in message


def test_out_of_range_arc_is_rejected():
    assert "arc_deg" in _reject({**BASE, "arc_deg": 200.0})
    assert "arc_deg" in _reject({**BASE, "arc_deg": 0.0})


def test_out_of_range_depth_is_rejected():
    message = _reject({"carrier_hz": 300.0, "rate_hz": 40.0, "depth_rad": 2.0})
    assert "depth_rad" in message


def test_no_arc_can_reach_the_image_fold_under_the_carrier_ceiling():
    # Why _parse_sam has no fold check on the arc_deg branch: the widest
    # arc at the highest permitted carrier still lands inside the pi/2
    # deviation ceiling, so such a check would be unreachable.
    widest = sam_depth_from_arc(180.0, SAM_CARRIER_CEILING_HZ)
    assert widest < math.pi / 2.0
    assert _group(
        {"carrier_hz": SAM_CARRIER_CEILING_HZ, "rate_hz": 40.0,
         "arc_deg": 180.0}
    ).sam.depth_rad == pytest.approx(widest)


def test_unknown_path_is_rejected():
    assert "path" in _reject({**BASE, "path": "spiral"})


def test_steps_outside_the_discontinuous_path_is_rejected():
    message = _reject({**BASE, "path": "closed", "steps": 8})
    assert "steps" in message and "discontinuous" in message


def test_steps_must_be_an_integer_of_at_least_two():
    assert "steps" in _reject(
        {**BASE, "path": "discontinuous", "steps": 1}
    )
    assert "steps" in _reject(
        {**BASE, "path": "discontinuous", "steps": 4.5}
    )


def test_discontinuous_defaults_to_eight_steps():
    assert _group({**BASE, "path": "discontinuous"}).sam.steps == 8


def test_offset_deg_needs_both_ears():
    assert "offset_deg" in _reject({**BASE, "offset_deg": {"left": 10.0}})


def test_offset_deg_range_is_checked_per_ear():
    message = _reject({**BASE, "offset_deg": {"left": 0.0, "right": 400.0}})
    assert "offset_deg" in message and "right" in message


def test_offset_deg_rejects_unknown_keys():
    message = _reject(
        {**BASE, "offset_deg": {"left": 0.0, "right": 0.0, "up": 1.0}}
    )
    assert "offset_deg" in message and "up" in message


def test_offsets_are_carried_through():
    spec = _group({**BASE, "offset_deg": {"left": 15.0, "right": -15.0}}).sam
    assert (spec.offset_left_deg, spec.offset_right_deg) == (15.0, -15.0)


def test_unknown_sam_keys_are_rejected():
    assert "depth_deg" in _reject({**BASE, "depth_deg": 30.0})


def test_sam_must_be_a_mapping():
    assert "mapping" in _reject("closed")


@pytest.mark.parametrize(
    "key,value",
    [
        ("beat", 4.0),
        ("pairs", 3),
        ("carrier_base", 200.0),
        ("harmonics", [1.0]),
        ("high_ear", "left"),
    ],
)
def test_sam_excludes_the_paired_tone_keys(key, value):
    message = _reject(dict(BASE), **{key: value})
    assert key in message and "sam" in message


def test_sam_rejects_rotation_with_the_reason():
    message = _reject(
        dict(BASE), rotation={"period_s": 30.0, "depth": 0.5, "phase_deg": 0.0}
    )
    assert "rotation" in message and "ILD" in message


def test_sam_rejects_placement_with_the_reason():
    message = _reject(
        dict(BASE),
        placement={"crossfeed_db": -6.0, "crossfeed_phase_deg": 30.0},
    )
    assert "placement" in message and "crossfeed" in message


def test_sam_allows_tremolo():
    group = _group(dict(BASE), tremolo={"rate_hz": 0.2, "depth": 0.3})
    assert group.tremolo is not None and group.sam is not None


def test_emerge_rejects_a_sam_primary_group():
    with pytest.raises(ValueError) as exc:
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "patent", "emerge": True,
            "segments": [
                {"duration": 4, "groups": [{"name": "s", "sam": dict(BASE)}]}
            ],
        })
    assert "sam" in str(exc.value)


def test_a_sam_group_emits_no_voices_but_reports_its_carrier():
    from farfield.voices import expand_voices, voice_frequencies

    group = _group(dict(BASE))
    assert expand_voices(group) == ()
    assert voice_frequencies(group) == [300.0]
