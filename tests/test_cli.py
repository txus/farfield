import json
from pathlib import Path

import pytest
import soundfile as sf

from farfield.cli import describe_lines, format_duration, list_lines, main
import numpy as np

from tests.support import PRESET_DIR, load_preset, preset_path
from farfield.timeline import resolve


def test_format_duration_minutes_and_seconds():
    assert format_duration(90.0) == "1:30"
    assert format_duration(5.0) == "0:05"


def test_format_duration_hours():
    assert format_duration(5400.0) == "1:30:00"


def test_list_lines_group_by_fidelity_tier():
    lines = list_lines(PRESET_DIR)
    text = "\n".join(lines)
    assert "measured from the original tapes:" in text
    assert "measured from the MSS remasters:" in text
    assert "from the patents:" in text
    assert "original designs:" in text
    assert "sleep-90" in text
    assert "focus-10" in text


def test_list_lines_separates_tiers_with_blank_lines():
    lines = list_lines(PRESET_DIR)
    assert lines[0] != ""
    assert "" in lines  # blank separators between tier groups survive


def test_describe_shows_every_segment_and_layer():
    session = load_preset("sleep-90")
    text = "\n".join(describe_lines(session, resolve(session)))
    assert "sleep-90" in text
    assert "1:30:00" in text
    assert "-15.0" in text or "-15" in text
    assert "pink" in text.lower()


def test_describe_reports_the_beat_counts():
    session = load_preset("alert")
    text = "\n".join(describe_lines(session, resolve(session)))
    # Three pairs give three binaural beats and two monaural per channel.
    assert "3 binaural" in text
    assert "2 monaural" in text


def test_describe_surfaces_the_fidelity_tier():
    session = load_preset("focus-10")
    text = "\n".join(describe_lines(session, resolve(session)))
    assert "measured from the original tapes" in text.lower()


def test_main_list_exits_zero(capsys):
    assert main(["list", str(PRESET_DIR)]) == 0
    assert "sleep-90" in capsys.readouterr().out


def test_main_describe_exits_zero(capsys):
    assert main(["describe", str(preset_path("wake"))]) == 0
    assert "wake" in capsys.readouterr().out


def test_main_render_writes_a_wav(tmp_path: Path):
    out = tmp_path / "out.wav"
    assert main(["render", str(preset_path("wake")), "-o", str(out)]) == 0
    audio, rate = sf.read(out)
    assert rate == 48000
    assert audio.shape[1] == 2
    assert abs(len(audio) / rate - 300.0) < 1.0


def test_main_render_honours_duration(tmp_path: Path):
    # A bare hold session, so any requested duration is feasible. The
    # bundled open-ended presets carry fixed entry and emerge segments and
    # correctly reject totals shorter than those.
    src = tmp_path / "s.yaml"
    src.write_text(
        "name: d\ntitle: D\nfidelity: original\n"
        "segments:\n"
        "  - duration: hold\n"
        "    groups:\n"
        "      - {name: A, beat: 4.0, pairs: 1, harmonics: [1.0]}\n"
        "emerge: false\n"
    )
    out = tmp_path / "out.wav"
    assert main(["render", str(src), "-o", str(out), "--duration", "1:00"]) == 0
    audio, rate = sf.read(out)
    assert abs(len(audio) / rate - 60.0) < 1.0


def test_main_render_reports_an_infeasible_duration_cleanly(tmp_path: Path, capsys):
    # relaxation's fixed entry (3:00) plus emerge (3:00) exceed 1:00.
    out = tmp_path / "out.wav"
    assert main(["render", str(preset_path("relaxation")), "-o", str(out), "--duration", "1:00"]) == 2
    assert "too short" in capsys.readouterr().err


def test_main_render_writes_the_sidecar_on_request(tmp_path: Path):
    out = tmp_path / "out.wav"
    assert main(["render", str(preset_path("wake")), "-o", str(out), "--json"]) == 0
    payload = json.loads((tmp_path / "out.json").read_text())
    assert payload["name"] == "wake"
    assert payload["layers"]


def test_main_render_defaults_the_output_name(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["render", str(preset_path("wake"))]) == 0
    assert (tmp_path / "wake.wav").exists()


def test_missing_session_file_exits_two(capsys):
    assert main(["describe", "nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_no_arguments_exits_nonzero(capsys):
    with pytest.raises(SystemExit):
        main([])


PLOT_SESSION = (
    "name: plotdemo\ntitle: Plot Demo\nfidelity: original\n"
    "segments:\n"
    "  - duration: 6\n"
    "    groups:\n"
    "      - {name: A, beat: 4.0, carrier_base: 200.0, pairs: 3, harmonics: [1.0]}\n"
    "emerge: false\n"
)


def test_main_plot_writes_html(tmp_path: Path):
    src = tmp_path / "s.yaml"
    src.write_text(PLOT_SESSION)
    out = tmp_path / "out.html"
    assert main(["plot", str(src), "-o", str(out)]) == 0
    html = out.read_text()
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "monaural" in html.lower()


def test_main_plot_defaults_the_output_name(tmp_path: Path, monkeypatch):
    src = tmp_path / "s.yaml"
    src.write_text(PLOT_SESSION)
    monkeypatch.chdir(tmp_path)
    assert main(["plot", str(src)]) == 0
    assert (tmp_path / "plotdemo.html").exists()


def test_main_plot_rejects_an_unknown_preset(capsys):
    assert main(["plot", "nope"]) == 2


OVER_CEILING_SESSION = (
    "name: toohigh\ntitle: Too High\nfidelity: original\n"
    "segments:\n"
    "  - duration: 4\n"
    "    groups:\n"
    "      - {name: A, beat: 20.0, carrier_base: 1400.0, pairs: 3,\n"
    "         harmonics: [1.0, 0.5, 0.25]}\n"
    "emerge: false\n"
)


def test_describe_of_an_over_ceiling_session_exits_two(tmp_path: Path, capsys):
    src = tmp_path / "s.yaml"
    src.write_text(OVER_CEILING_SESSION)
    assert main(["describe", str(src)]) == 2
    assert "ceiling" in capsys.readouterr().err


def test_render_of_an_over_ceiling_session_exits_two(tmp_path: Path, capsys):
    src = tmp_path / "s.yaml"
    src.write_text(OVER_CEILING_SESSION)
    assert main(["render", str(src), "-o", str(tmp_path / "o.wav")]) == 2
    assert "ceiling" in capsys.readouterr().err


def test_render_fades_the_edges_by_default(tmp_path: Path):
    faded, plain = tmp_path / "f.wav", tmp_path / "p.wav"
    assert main(["render", str(preset_path("wake")), "-o", str(faded)]) == 0
    assert (
        main(["render", str(preset_path("wake")), "-o", str(plain), "--fade", "0"])
        == 0
    )
    af, _ = sf.read(faded)
    ap, rate = sf.read(plain)
    head = slice(0, rate // 10)
    tail = slice(-rate // 10, None)
    assert np.abs(af[head]).max() < 0.2 * np.abs(ap[head]).max()
    assert np.abs(af[tail]).max() < 0.2 * np.abs(ap[tail]).max()
    assert len(af) == len(ap)


def test_fixed_schedule_sessions_warn_when_duration_is_ignored(
    tmp_path: Path, capsys
):
    src = tmp_path / "fixed.yaml"
    src.write_text(
        "name: fx\ntitle: FX\nfidelity: original\n"
        "segments:\n"
        "  - duration: 30\n"
        "    groups:\n"
        "      - {name: A, beat: 4.0, pairs: 1, harmonics: [1.0]}\n"
        "emerge: false\n"
    )
    out = tmp_path / "out.wav"
    assert main(["render", str(src), "-o", str(out), "--duration", "2:00"]) == 0
    assert "fixed schedule" in capsys.readouterr().err
    audio, rate = sf.read(out)
    assert abs(len(audio) / rate - 30.0) < 1.0
