from farfield.cli import describe_lines
from farfield.render import sidecar
from farfield.session import load_session_dict
from farfield.timeline import resolve


def _session():
    return load_session_dict({
        "name": "m", "title": "M", "fidelity": "measured-tape", "sample_rate": 48000,
        "segments": [{
            "duration": 60,
            "groups": [
                {"name": "ground", "high_ear": "left", "harmonics": [1.0],
                 "pairs": [{"mono": 50.0}, {"center": 50.125, "beat": 0.75}]},
                {"name": "deep", "harmonics": [1.0],
                 "pairs": [{"left": 304.8, "right": 300.0}], "level_db": -28.0,
                 "tremolo": {"rate_hz": 0.5, "depth": 0.22}},
            ],
        }],
    })


def test_describe_shows_each_pair_and_the_polarity():
    text = "\n".join(describe_lines(_session(), resolve(_session())))
    assert "mono 50" in text
    assert "left" in text.lower()        # polarity surfaced
    assert "304.8" in text and "300.0" in text
    assert "tremolo" in text.lower()


def test_sidecar_carriers_come_from_voices():
    session = _session()
    payload = sidecar(session, resolve(session))
    ground = next(l for l in payload["layers"] if l["group"] == "ground")
    assert 50.0 in ground["carriers_start"]["left"]
    assert 50.5 in ground["carriers_start"]["left"]     # high ear = left
    assert 49.75 in ground["carriers_start"]["right"]
    deep = next(l for l in payload["layers"] if l["group"] == "deep")
    assert deep["carriers_start"]["left"] == [304.8]
    assert deep["carriers_start"]["right"] == [300.0]
    # pairs-form groups report their own pair count, not the stack default
    assert ground["pairs"] == 2
    assert deep["pairs"] == 1


def test_describe_shows_gliding_beat_and_tremolo_rate_as_ranges():
    session = load_session_dict({
        "name": "g", "title": "G", "fidelity": "measured-tape", "sample_rate": 48000,
        "segments": [{
            "duration": 60,
            "groups": [
                {"name": "glide", "harmonics": [1.0],
                 "pairs": [{"center": 250.0,
                            "beat": {"from": 4.115, "to": 3.886}}],
                 "tremolo": {"rate_hz": {"from": 0.58, "to": 0.48}, "depth": 0.2}},
            ],
        }],
    })
    text = "\n".join(describe_lines(session, resolve(session)))
    assert "4.12" in text and "3.89" in text
    assert "0.58" in text and "0.48" in text


def _stereo_bed_session(bed):
    return load_session_dict({
        "name": "s", "title": "S", "fidelity": "measured-tape", "sample_rate": 48000,
        "segments": [{
            "duration": 60,
            "groups": [{"name": "A", "beat": 4.0, "harmonics": [1.0]}],
            "bed": bed,
        }],
    })


def test_describe_names_the_stereo_stage_per_mode():
    crossfade = _stereo_bed_session({
        "level_db": -5.0, "color": "brown",
        "stereo": {"mode": "crossfade", "lfo_period_s": 9.82, "depth_db": 3.25}})
    text = "\n".join(describe_lines(crossfade, resolve(crossfade)))
    assert "crossfade 9.82 s / 3.25 dB" in text

    static = _stereo_bed_session({
        "level_db": -5.0, "color": "brown",
        "stereo": {"mode": "static", "interaural_delay_us": 145.0}})
    text = "\n".join(describe_lines(static, resolve(static)))
    assert "static lead 145 µs" in text

    # pan is the default and carries no parameters, so it gets no suffix
    pan = _stereo_bed_session({"level_db": -18.0})
    text = "\n".join(describe_lines(pan, resolve(pan)))
    assert "crossfade" not in text and "static lead" not in text


def test_sidecar_bed_entries_carry_the_stereo_stage():
    crossfade = _stereo_bed_session({
        "level_db": -5.0, "color": "brown",
        "stereo": {"mode": "crossfade", "lfo_period_s": 9.9, "depth_db": 2.83}})
    bed = sidecar(crossfade, resolve(crossfade))["pink"][0]
    assert bed["stereo_mode"] == "crossfade"
    assert bed["lfo_period_s"] == 9.9
    assert bed["stereo_depth_db"] == 2.83
    assert "interaural_delay_us" not in bed
    # the pre-existing keys all survive under the unchanged "pink" key
    for key in ("level_db", "algorithm", "color", "slope_db_per_decade",
                "surf_rate_hz", "surf_depth", "start_s", "end_s"):
        assert key in bed

    static = _stereo_bed_session({
        "level_db": -5.0, "color": "brown",
        "stereo": {"mode": "static", "interaural_delay_us": 145.0}})
    bed = sidecar(static, resolve(static))["pink"][0]
    assert bed["stereo_mode"] == "static"
    assert bed["interaural_delay_us"] == 145.0
    assert "lfo_period_s" not in bed and "stereo_depth_db" not in bed

    pan = _stereo_bed_session({"level_db": -18.0})
    bed = sidecar(pan, resolve(pan))["pink"][0]
    assert bed["stereo_mode"] == "pan"
    assert "lfo_period_s" not in bed


def test_sidecar_stack_output_is_unchanged():
    session = load_session_dict({
        "name": "t", "title": "T", "fidelity": "patent", "sample_rate": 48000,
        "segments": [{"duration": 10, "groups": [
            {"name": "A", "beat": 4.0, "carrier_base": 200.0, "pairs": 3,
             "harmonics": [1.0]}]}],
    })
    payload = sidecar(session, resolve(session))
    layer = payload["layers"][0]
    assert layer["carriers_start"]["left"] == [200.0, 204.0, 208.0]
    assert layer["carriers_start"]["right"] == [204.0, 208.0, 212.0]
