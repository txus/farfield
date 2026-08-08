import pytest

from farfield.session import Glide, load_session_dict
from farfield.timeline import Layer, PinkLayer, Timeline, resolve


def _session(**overrides) -> dict:
    data = {
        "name": "t",
        "title": "T",
        "fidelity": "original",
        "sample_rate": 1000,
        "segments": [
            {"duration": 10, "groups": [{"name": "A", "beat": 10.0}]},
            {"duration": 20, "groups": [{"name": "A", "beat": 4.0}]},
        ],
    }
    data.update(overrides)
    return data


def test_butt_joined_segments_are_contiguous():
    timeline = resolve(load_session_dict(_session()))
    first, second = timeline.layers
    assert first.start_sample == 0
    assert first.n_samples == 10_000
    assert second.start_sample == 10_000
    assert second.n_samples == 20_000
    assert timeline.total_samples == 30_000


def test_overlap_pulls_the_next_segment_earlier():
    data = _session()
    data["segments"][0]["overlap"] = 4
    timeline = resolve(load_session_dict(data))
    assert timeline.layers[1].start_sample == 6_000
    assert timeline.total_samples == 26_000


def test_overlap_creates_matching_fades():
    data = _session()
    data["segments"][0]["overlap"] = 4
    timeline = resolve(load_session_dict(data))
    assert timeline.layers[0].fade_out_samples == 4_000
    assert timeline.layers[1].fade_in_samples == 4_000


def test_first_segment_has_no_fade_in_and_last_no_fade_out():
    timeline = resolve(load_session_dict(_session()))
    assert timeline.layers[0].fade_in_samples == 0
    assert timeline.layers[-1].fade_out_samples == 0


def test_overlap_is_clamped_to_the_shorter_neighbour():
    data = _session()
    data["segments"][0]["overlap"] = 999
    timeline = resolve(load_session_dict(data))
    assert timeline.layers[0].fade_out_samples == 10_000


def test_every_group_in_a_segment_becomes_a_layer():
    data = _session()
    data["segments"] = [
        {
            "duration": 10,
            "groups": [
                {"name": "A", "beat": 10.0},
                {"name": "B", "beat": 4.0, "level_db": -15.0},
            ],
        }
    ]
    timeline = resolve(load_session_dict(data))
    assert len(timeline.layers) == 2
    assert {layer.group.name for layer in timeline.layers} == {"A", "B"}
    assert all(layer.start_sample == 0 for layer in timeline.layers)


def test_pink_becomes_its_own_layer_matching_the_segment_span():
    data = _session()
    data["segments"][0]["pink"] = {"level_db": -20.0}
    timeline = resolve(load_session_dict(data))
    assert len(timeline.pink_layers) == 1
    pink = timeline.pink_layers[0]
    assert pink.start_sample == 0
    assert pink.n_samples == 10_000
    assert pink.spec.level_db == -20.0


def test_segments_without_pink_produce_no_pink_layer():
    assert resolve(load_session_dict(_session())).pink_layers == ()


def test_emerge_appends_a_gliding_layer():
    data = _session(emerge={"duration": 6, "target_beat": 15.0})
    timeline = resolve(load_session_dict(data))
    emerge_layer = timeline.layers[-1]
    # The emerge block overlaps the previous segment by its 2 s crossfade.
    assert emerge_layer.start_sample == 28_000
    assert emerge_layer.n_samples == 6_000
    assert emerge_layer.group.beat == Glide(4.0, 15.0)
    assert timeline.total_samples == 34_000


def test_emerge_overlaps_the_segment_it_follows():
    data = _session(emerge={"duration": 6, "target_beat": 15.0})
    timeline = resolve(load_session_dict(data))
    previous, emerge_layer = timeline.layers[-2], timeline.layers[-1]
    overlap = emerge_layer.fade_in_samples
    assert overlap == 2_000
    assert emerge_layer.start_sample == (
        previous.start_sample + previous.n_samples - overlap
    )


def test_emerge_fades_the_previous_segment_and_its_pink_out():
    data = _session(emerge={"duration": 6, "target_beat": 15.0})
    data["segments"][1]["pink"] = {"level_db": -20.0}
    timeline = resolve(load_session_dict(data))
    overlap = timeline.layers[-1].fade_in_samples
    assert overlap > 0
    assert timeline.layers[-2].fade_out_samples == overlap
    assert timeline.pink_layers[-1].fade_out_samples == overlap


def test_emerge_starts_from_the_end_of_a_gliding_final_group():
    data = _session(emerge={"duration": 6, "target_beat": 15.0})
    data["segments"][1]["groups"] = [
        {"name": "A", "beat": {"from": 8.0, "to": 3.0}}
    ]
    timeline = resolve(load_session_dict(data))
    assert timeline.layers[-1].group.beat == Glide(3.0, 15.0)


def test_emerge_follows_the_loudest_group_of_the_last_segment():
    data = _session(emerge={"duration": 6, "target_beat": 15.0})
    data["segments"][1]["groups"] = [
        {"name": "quiet", "beat": 2.0, "level_db": -12.0},
        {"name": "loud", "beat": 6.0, "level_db": 0.0},
    ]
    timeline = resolve(load_session_dict(data))
    assert timeline.layers[-1].group.name == "loud"
    assert timeline.layers[-1].group.beat == Glide(6.0, 15.0)


def test_emerge_fades_in():
    data = _session(emerge={"duration": 6, "target_beat": 15.0})
    assert resolve(load_session_dict(data)).layers[-1].fade_in_samples == 2_000


def test_no_emerge_leaves_the_timeline_alone():
    timeline = resolve(load_session_dict(_session()))
    assert timeline.total_samples == 30_000


def test_timeline_carries_the_sample_rate():
    assert resolve(load_session_dict(_session())).sample_rate == 1000


def test_layers_are_ordered_by_start_sample():
    data = _session()
    data["segments"][0]["overlap"] = 4
    starts = [layer.start_sample for layer in resolve(load_session_dict(data)).layers]
    assert starts == sorted(starts)
