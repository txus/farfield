import pytest

from farfield.waveshape import (
    DEFAULT_HARMONICS,
    Harmonic,
    expand_harmonics,
    normalize_harmonics,
)


def test_ratios_become_one_based_harmonics():
    harmonics = expand_harmonics([1.0, 0.5, 0.25])
    assert harmonics == (
        Harmonic(1, 1.0),
        Harmonic(2, 0.5),
        Harmonic(3, 0.25),
    )


def test_zero_ratios_are_dropped():
    harmonics = expand_harmonics([1.0, 0.0, 0.25])
    assert [h.index for h in harmonics] == [1, 3]


def test_single_ratio_gives_a_pure_sine_beat():
    assert expand_harmonics([1.0]) == (Harmonic(1, 1.0),)


def test_empty_ratios_are_rejected():
    with pytest.raises(ValueError):
        expand_harmonics([])


def test_all_zero_ratios_are_rejected():
    with pytest.raises(ValueError):
        expand_harmonics([0.0, 0.0])


def test_negative_ratios_are_rejected():
    with pytest.raises(ValueError):
        expand_harmonics([1.0, -0.5])


def test_normalized_amplitudes_sum_to_one():
    harmonics = normalize_harmonics(expand_harmonics([2.0, 1.0, 1.0]))
    assert abs(sum(h.amplitude for h in harmonics) - 1.0) < 1e-12
    assert abs(harmonics[0].amplitude - 0.5) < 1e-12


def test_normalizing_preserves_indices():
    harmonics = normalize_harmonics(expand_harmonics([1.0, 0.0, 3.0]))
    assert [h.index for h in harmonics] == [1, 3]


def test_default_harmonics_are_the_documented_shape():
    assert DEFAULT_HARMONICS == (1.0, 0.35, 0.15)
