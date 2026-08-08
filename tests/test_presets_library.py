import pytest

from farfield.cli import list_lines
from tests.support import PRESET_DIR, load_preset, preset_names
from farfield.session import Glide
from farfield.timeline import resolve

ORIGINAL = ["alert", "attention", "concentration", "relaxation", "focus-3"]
MEASURED_TAPE = ["focus-10", "focus-12", "focus-15", "focus-21"]
# The Monroe Sound Science 3D remasters of the same four programs
# (tests/test_presets_mss.py holds their own shape assertions).
MEASURED_MSS = ["focus-10-mss", "focus-12-mss", "focus-15-mss", "focus-21-mss"]
PATENT = ["sleep-90", "wake", "sam-gamma"]


def test_all_expected_presets_exist():
    assert set(
        ORIGINAL + MEASURED_TAPE + MEASURED_MSS + PATENT
    ) == set(preset_names())


@pytest.mark.parametrize("name", ORIGINAL)
def test_original_designs_are_marked_original(name):
    assert load_preset(name).fidelity == "original"


@pytest.mark.parametrize("name", ORIGINAL)
def test_open_ended_presets_honour_a_duration_argument(name):
    timeline = resolve(load_preset(name, total_duration_s=1200.0))
    assert abs(timeline.total_samples / 48000 - 1200.0) < 1.0


@pytest.mark.parametrize("name", ORIGINAL)
def test_open_ended_presets_have_a_default_total(name):
    assert resolve(load_preset(name)).total_samples > 0


@pytest.mark.parametrize("name", ORIGINAL)
def test_open_ended_presets_emerge(name):
    session = load_preset(name)
    assert session.emerge is not None
    assert session.emerge.target_beat >= 12.0


def test_focus_3_explains_that_it_is_an_original_design():
    notes = load_preset("focus-3").notes
    assert "original design" in notes.lower()
    assert len(notes) > 200


def test_mood_minder_bands_are_ordered_as_described():
    def beat(name: str) -> float:
        session = load_preset(name)
        primary = max(session.segments[-1].groups, key=lambda g: g.level_db)
        return primary.beat_bounds()[1]

    assert beat("alert") > beat("attention") > beat("concentration")
    assert beat("concentration") > beat("relaxation")


@pytest.mark.parametrize("name", ORIGINAL)
def test_original_designs_open_with_a_glide_into_the_state(name):
    first = load_preset(name).segments[0]
    primary = max(first.groups, key=lambda g: g.level_db)
    assert isinstance(primary.beat, Glide)


def test_list_is_grouped_measured_first():
    text = "\n".join(list_lines(PRESET_DIR))
    assert (
        text.index("measured from the original tapes:")
        < text.index("measured from the MSS remasters:")
        < text.index("from the patents:")
        < text.index("original designs:")
    )
