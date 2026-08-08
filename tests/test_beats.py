import pytest

from farfield.beats import (
    CarrierStack,
    beat_counts,
    build_stack,
    stack_offsets,
    validate_stack,
)


def test_septon_reproduces_the_patent_example():
    stack = build_stack(200.0, 4.0, 3)
    assert stack.left == (200.0, 204.0, 208.0)
    assert stack.right == (204.0, 208.0, 212.0)


def test_single_pair_is_a_plain_binaural_beat():
    stack = build_stack(100.0, 4.0, 1)
    assert stack.left == (100.0,)
    assert stack.right == (104.0,)


def test_offsets_are_shared_by_render_and_describe_paths():
    left, right = stack_offsets(3)
    assert left == (0, 1, 2)
    assert right == (1, 2, 3)


def test_pairs_must_be_positive():
    with pytest.raises(ValueError):
        build_stack(200.0, 4.0, 0)


def test_beat_counts_match_the_patent_description():
    # Three pairs: three binaural beats, two monaural beats per channel.
    assert beat_counts(3) == (3, 2)
    assert beat_counts(1) == (1, 0)


def test_validate_accepts_a_stack_below_the_ceiling():
    validate_stack(build_stack(200.0, 4.0, 3))


def test_validate_rejects_a_stack_above_the_fusion_ceiling():
    with pytest.raises(ValueError, match="1500"):
        validate_stack(build_stack(1400.0, 50.0, 3))


def test_carrier_stack_is_hashable_and_frozen():
    stack = CarrierStack(left=(1.0,), right=(2.0,))
    hash(stack)
    with pytest.raises(Exception):
        stack.left = (3.0,)
