import numpy as np
import pytest

from tests.support import load_preset, preset_metadata, preset_names
from farfield.render import render_session
from farfield.timeline import resolve

MEASURED = ["focus-10", "focus-12", "focus-15", "focus-21"]


def test_measured_presets_exist():
    assert set(MEASURED) <= set(preset_names())


@pytest.mark.parametrize("name", MEASURED)
def test_measured_tier_and_no_emerge(name):
    session = load_preset(name)
    assert session.fidelity == "measured-tape"
    assert session.emerge is None


@pytest.mark.parametrize("name,total_s", [
    ("focus-10", 1800), ("focus-12", 2110),
    ("focus-15", 2305), ("focus-21", 2406),
])
def test_measured_totals_match_the_tape_timeline(name, total_s):
    assert resolve(load_preset(name)).total_samples == total_s * 48000


def test_measured_sorts_first_in_the_library():
    tiers = [e["fidelity"] for e in preset_metadata()]
    assert tiers[0] == "measured-tape"
    order = {"measured-tape": 0, "measured-mss": 1, "patent": 2, "original": 3}
    assert tiers == sorted(tiers, key=lambda t: order[t])


@pytest.mark.parametrize("name", MEASURED)
def test_measured_notes_disclose_method_and_idealization(name):
    notes = load_preset(name).notes.lower()
    assert "measured" in notes and "idealized" in notes
    assert "tape-analysis" in notes


def test_f12_ground_carries_the_mono_anchor():
    session = load_preset("focus-12")
    ground = next(g for s in session.segments for g in s.groups
                  if g.name == "ground")
    kinds = [p.kind for p in ground.pairs_spec]
    assert "mono" in kinds and "center" in kinds
    assert ground.high_ear == "left"


def test_f15_flips_polarity_mid_session():
    session = load_preset("focus-15")
    ears = []
    for seg in session.segments:
        for g in seg.groups:
            if g.name == "ground":
                ears.append(g.high_ear)
    assert "left" in ears and "right" in ears


def test_f15_deep_grid_is_verbatim():
    session = load_preset("focus-15")
    deep = next(g for s in session.segments for g in s.groups
                if g.name == "deep")
    pairs = [(p.left, p.right) for p in deep.pairs_spec]
    assert (304.8, 300.0) in pairs and (503.9, 511.2) in pairs


def test_f21_flips_polarity_at_the_seam():
    session = load_preset("focus-21")
    intro_ears = {g.high_ear for s in session.segments for g in s.groups
                  if g.name in ("introground", "introdelta")}
    body_ears = {g.high_ear for s in session.segments for g in s.groups
                 if g.name == "ground"}
    assert intro_ears == {"left"}
    assert body_ears == {"right"}


def test_f21_beta_middle_centre_is_755_not_750():
    session = load_preset("focus-21")
    bt755 = next(g for s in session.segments for g in s.groups
                 if g.name == "bt755")
    pair = bt755.pairs_spec[0]
    assert pair.kind == "center"
    assert pair.center == pytest.approx(755.0)
    assert pair.beat == pytest.approx(16.0)


def test_f10_beats_glide_and_pair_c_is_reversed():
    session = load_preset("focus-10")
    first_a = session.segments[0].groups[0]
    assert first_a.pairs_spec[0].beat.start == pytest.approx(4.115)
    c = next(g for s in session.segments for g in s.groups if g.name == "C")
    assert c.pairs_spec[0].left > c.pairs_spec[0].right  # 497.0 / 493.3


def test_measured_presets_match_the_measured_carriers():
    """Seam test: the rendered carriers match docs/tape-analysis/results.json.

    Values are the idealized (de-scaled) nominals of the measured pairs, read
    off the sidecar so this checks the whole load -> resolve -> voices chain,
    not just the YAML text.
    """
    from farfield.render import sidecar

    def layers(name):
        session = load_preset(name)
        return sidecar(session, resolve(session))["layers"]

    def group_at(name, group, index=0):
        matches = [l for l in layers(name) if l["group"] == group]
        return matches[index]["carriers_start"]

    # F12 ground, segment 1: measured fL 50.5 / fR 49.75, left ear high.
    f12_ground = group_at("focus-12", "ground")
    assert 50.5 in f12_ground["left"]
    assert 49.75 in f12_ground["right"]

    # F15 bridges: measured fL 175 / fR 179 and fL 262 / fR 266 -- the tape's
    # polarity note says the bridges have L LOW, i.e. high_ear right, from
    # their very first appearance (segment 3) and not only in segment 7.
    for index in (0, 1):
        br175 = group_at("focus-15", "br175", index)
        assert br175["left"] == [pytest.approx(175.0)]
        assert br175["right"] == [pytest.approx(179.0)]
        br262 = group_at("focus-15", "br262", index)
        assert br262["left"] == [pytest.approx(262.0)]
        assert br262["right"] == [pytest.approx(266.0)]

    # F15 deep grid: kept verbatim to 0.1 ppm.
    f15_deep = group_at("focus-15", "deep")
    assert pytest.approx(304.8) in f15_deep["left"]
    assert pytest.approx(503.9) in f15_deep["left"]

    # F10 group A, segment 1: measured fL 99.98 / fR 104.03, idealized to a
    # 102.0 centre with a 4.115 Hz beat.
    f10_a = group_at("focus-10", "A")
    assert f10_a["right"] == [pytest.approx(104.0575)]
    assert f10_a["left"] == [pytest.approx(99.9425)]

    # F21 ground, post-flip segment: measured fL 49.75 / fR 50.5, right ear
    # high (the polarity flips at the 220 s seam and stays right-high).
    f21_ground = group_at("focus-21", "ground")
    assert 50.5 in f21_ground["right"]
    assert 49.75 in f21_ground["left"]

    # F21 beta triplet's middle centre: 755 Hz, not the folklore's 750 --
    # sidecar carriers 747.0/763.0.
    f21_bt755 = group_at("focus-21", "bt755")
    assert f21_bt755["left"] == [pytest.approx(747.0)]
    assert f21_bt755["right"] == [pytest.approx(763.0)]


def test_f10_renders_the_measured_layout_briefly():
    session = load_preset("focus-10")
    # render only the first segment's worth by rendering the whole first
    # 20 s of the resolved timeline: cheap proxy — render the session at a
    # truncated copy is not supported, so spot-check via the sidecar instead.
    from farfield.render import sidecar
    payload = sidecar(session, resolve(session))
    a = next(l for l in payload["layers"] if l["group"] == "A")
    assert a["carriers_start"]["right"] == [pytest.approx(104.0575)]
    assert a["carriers_start"]["left"] == [pytest.approx(99.9425)]
