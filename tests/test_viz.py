import base64
import json
import re
from pathlib import Path

import numpy as np

from farfield.render import render_session
from farfield.session import load_session_dict
from farfield.timeline import resolve
from farfield.viz import build_payload, render_html, write_visualizer

SHORT = {
    "name": "demo",
    "title": "Demo",
    "fidelity": "original",
    "sample_rate": 48000,
    "segments": [
        {
            "duration": 6,
            "groups": [
                {"name": "A", "beat": 4.0, "carrier_base": 200.0,
                 "pairs": 3, "harmonics": [1.0]}
            ],
            "pink": {"level_db": -20.0},
        }
    ],
}


def _payload():
    session = load_session_dict(SHORT)
    timeline = resolve(session)
    return session, timeline, build_payload(
        session, timeline, render_session(session)
    )


def test_payload_embeds_the_session_sidecar():
    _, _, payload = _payload()
    assert payload["session"]["name"] == "demo"
    assert payload["session"]["layers"]


def test_payload_is_json_serialisable():
    _, _, payload = _payload()
    json.dumps(payload)


def test_spectrogram_band_covers_the_carriers():
    _, _, payload = _payload()
    spec = payload["spectrogram"]
    assert spec["freq_min"] <= 200.0
    assert spec["freq_max"] >= 212.0


def test_spectrogram_frames_are_capped():
    _, _, payload = _payload()
    assert 0 < payload["spectrogram"]["n_frames"] <= 4000


def test_spectrogram_payload_decodes_to_the_declared_shape():
    _, _, payload = _payload()
    spec = payload["spectrogram"]
    raw = base64.b64decode(spec["left"])
    assert len(raw) == spec["n_frames"] * spec["n_bins"]


def test_both_channels_are_present():
    _, _, payload = _payload()
    assert payload["spectrogram"]["left"]
    assert payload["spectrogram"]["right"]


def test_meters_cover_every_carrier_in_the_session():
    _, timeline, payload = _payload()
    freqs = sorted({m["freq"] for m in payload["meters"]["carriers"]})
    assert freqs == [200.0, 204.0, 208.0, 212.0]


def test_meter_tracks_decode_to_a_consistent_length():
    _, _, payload = _payload()
    lengths = {
        len(base64.b64decode(m["track"]))
        for m in payload["meters"]["carriers"]
    }
    assert len(lengths) == 1
    assert lengths.pop() > 0


def test_meters_are_labelled_by_channel():
    _, _, payload = _payload()
    channels = {m["channel"] for m in payload["meters"]["carriers"]}
    assert channels == {"left", "right"}


def test_audio_is_embedded_as_a_data_uri():
    _, _, payload = _payload()
    assert payload["audio_data_uri"].startswith("data:audio/flac;base64,")


def test_html_is_a_single_self_contained_document():
    session, timeline, payload = _payload()
    html = render_html(payload, render_session(session), session.sample_rate)
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "</html>" in html


def test_html_references_no_external_resources():
    session, _, payload = _payload()
    html = render_html(payload, render_session(session), session.sample_rate)
    external = re.findall(r'(?:src|href)\s*=\s*["\'](https?:)?//', html)
    assert external == []


def test_html_carries_no_script_or_style_imports():
    # Checked against the static shell only, not the substituted PAYLOAD
    # JSON: that JSON embeds megabytes of base64 audio/spectrogram data,
    # and a random 3-character trigram like "cdn" is near-certain to
    # appear somewhere in a blob that size by pure chance (~47 expected
    # occurrences for a 1.5 MB base64 string), regardless of how the page
    # is built. The template shell is what could actually reference an
    # external CDN, so that is what this test guards.
    session, _, payload = _payload()
    html = render_html(payload, render_session(session), session.sample_rate)
    before, _, after = html.partition("const PAYLOAD = ")
    _, _, after = after.partition("\n</script>")
    shell = before + after
    assert "cdn" not in shell.lower()
    assert "@import" not in html


def test_html_states_that_neither_channel_contains_the_beat():
    session, _, payload = _payload()
    html = render_html(payload, render_session(session), session.sample_rate)
    lowered = html.lower()
    assert "monaural" in lowered
    assert "speakers" in lowered
    assert "headphones" in lowered


def test_html_shows_the_fidelity_tier():
    session, _, payload = _payload()
    html = render_html(payload, render_session(session), session.sample_rate)
    # The payload carries the machine tier; the page maps it to the
    # plain-English label at render time, so both must be present.
    assert '"fidelity": "original"' in html
    assert "original design" in html.lower()


def test_write_visualizer_produces_a_file(tmp_path: Path):
    out = tmp_path / "demo.html"
    write_visualizer(load_session_dict(SHORT), out)
    assert out.exists()
    assert out.stat().st_size > 10_000
