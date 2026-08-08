import pytest

from farfield.session import FIDELITY_TIERS, load_session_dict


def _bed_session(bed=None, pink=None, fidelity="original"):
    seg = {"duration": "1:00", "groups": [{"name": "A", "beat": 4.0}]}
    if bed is not None:
        seg["bed"] = bed
    if pink is not None:
        seg["pink"] = pink
    return load_session_dict({
        "name": "t", "title": "T", "fidelity": fidelity, "segments": [seg],
    })


def test_bed_parses_with_color_and_surf():
    spec = _bed_session(bed={"level_db": -14.0, "color": "brown",
                             "surf": {"rate_hz": 0.21, "depth": 0.14}}).segments[0].pink
    assert spec.color == "brown"
    assert spec.surf_rate_hz == 0.21
    assert spec.surf_depth == 0.14
    assert spec.resolved_slope() == -20.0


def test_bed_slope_override():
    spec = _bed_session(bed={"level_db": -1.3, "color": "brown",
                             "slope_db_per_decade": -22.0}).segments[0].pink
    assert spec.resolved_slope() == -22.0


def test_bed_defaults_to_pink():
    spec = _bed_session(bed={"level_db": -18.0}).segments[0].pink
    assert spec.color == "pink"
    assert spec.resolved_slope() == -10.0
    assert spec.surf_rate_hz is None


def test_pink_alias_still_works_and_is_pink():
    spec = _bed_session(pink={"level_db": -18.0}).segments[0].pink
    assert spec.color == "pink"
    assert spec.comb_sweep_hz == 0.125


def test_declaring_both_bed_and_pink_is_rejected():
    with pytest.raises(ValueError, match="bed"):
        _bed_session(bed={"level_db": -10.0}, pink={"level_db": -10.0})


@pytest.mark.parametrize("bed,needle", [
    ({"level_db": -10.0, "color": "mauve"}, "color"),
    ({"level_db": -10.0, "slope_db_per_decade": -35.0}, "slope"),
    ({"level_db": -10.0, "slope_db_per_decade": 5.0}, "slope"),
    ({"level_db": -10.0, "surf": {"rate_hz": 0.0, "depth": 0.1}}, "rate"),
    ({"level_db": -10.0, "surf": {"rate_hz": 0.2, "depth": 1.0}}, "depth"),
])
def test_bad_beds_are_rejected(bed, needle):
    with pytest.raises(ValueError, match=needle):
        _bed_session(bed=bed)


def test_unknown_bed_algorithm_is_rejected_at_load():
    with pytest.raises(ValueError, match="algorithm"):
        _bed_session(bed={"level_db": -10.0, "algorithm": "kellet"})


def test_lfsr_with_a_non_pink_color_is_rejected_at_load():
    # noise.render_bed guards this too, but a load-time failure means
    # `describe` catches it instead of a long render dying halfway.
    with pytest.raises(ValueError, match="lfsr"):
        _bed_session(bed={"level_db": -10.0, "algorithm": "lfsr",
                          "color": "brown"})


def test_lfsr_with_pink_is_accepted():
    spec = _bed_session(bed={"level_db": -18.0, "algorithm": "lfsr"}
                        ).segments[0].pink
    assert spec.algorithm == "lfsr" and spec.color == "pink"


def test_measured_tape_is_a_valid_tier():
    assert "measured-tape" in FIDELITY_TIERS
    assert _bed_session(fidelity="measured-tape").fidelity == "measured-tape"


def test_tier_set_is_exactly_the_four():
    assert FIDELITY_TIERS == frozenset(
        {"measured-tape", "measured-mss", "patent", "original"})


def test_crossfade_stereo_parses():
    spec = _bed_session(bed={"level_db": -5.0, "color": "brown",
                             "stereo": {"mode": "crossfade",
                                        "lfo_period_s": 9.8,
                                        "depth_db": 3.25}}).segments[0].pink
    assert spec.stereo_mode == "crossfade"
    assert spec.lfo_period_s == 9.8
    assert spec.stereo_depth_db == 3.25
    assert spec.comb_enabled is False


def test_static_stereo_parses():
    spec = _bed_session(bed={"level_db": -0.7,
                             "stereo": {"mode": "static",
                                        "interaural_delay_us": 145.0}}).segments[0].pink
    assert spec.stereo_mode == "static"
    assert spec.interaural_delay_us == 145.0
    assert spec.comb_enabled is False


def test_default_stereo_is_pan_with_comb():
    spec = _bed_session(bed={"level_db": -18.0}).segments[0].pink
    assert spec.stereo_mode == "pan"
    assert spec.comb_enabled is True


def test_explicit_comb_survives_crossfade_mode():
    spec = _bed_session(bed={"level_db": -5.0, "comb_sweep_hz": 0.125,
                             "stereo": {"mode": "crossfade",
                                        "lfo_period_s": 9.8,
                                        "depth_db": 3.0}}).segments[0].pink
    assert spec.comb_enabled is True


def test_stereo_accepts_the_bare_pan_string():
    spec = _bed_session(bed={"level_db": -18.0,
                             "stereo": "pan"}).segments[0].pink
    assert spec.stereo_mode == "pan"
    assert spec.comb_enabled is True


@pytest.mark.parametrize("stereo,needle", [
    ({"mode": "orbit"}, "stereo"),
    ("orbit", "unknown stereo"),
    (["pan"], "string or mapping"),
    (3.5, "string or mapping"),
    ({"mode": "pan", "lfo_period_s": 9.8}, "mode"),
    ({"mode": "static", "depth_db": 3.0}, "mode"),
    ({"mode": "crossfade", "lfo_period_s": 0.0, "depth_db": 3.0}, "lfo_period_s"),
    ({"mode": "crossfade", "lfo_period_s": 9.8, "depth_db": 13.0}, "depth_db"),
    ({"mode": "crossfade", "lfo_period_s": 9.8}, "depth_db"),
    ({"mode": "crossfade", "depth_db": 3.0}, "lfo_period_s"),
    ({"mode": "crossfade", "lfo_period_s": 9.8, "depth_db": 3.0,
      "interaural_delay_us": 145.0}, "interaural_delay_us"),
    ({"mode": "static"}, "interaural_delay_us"),
    ({"mode": "static", "interaural_delay_us": 2000.0}, "interaural_delay_us"),
])
def test_bad_stereo_configs_are_rejected(stereo, needle):
    with pytest.raises(ValueError, match=needle):
        _bed_session(bed={"level_db": -5.0, "stereo": stereo})


def test_pink_alias_rejects_stereo():
    with pytest.raises(ValueError, match="pink alias"):
        _bed_session(pink={"level_db": -18.0,
                           "stereo": {"mode": "static",
                                      "interaural_delay_us": 100.0}})
