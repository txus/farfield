"""Carrier stack construction.

The stacked-carrier arrangement from US5356368A is the signature of the
technique: adjacent tones share a channel, so a stack of N pairs yields N
binaural beats plus N-1 monaural amplitude beats inside each channel.
"""

from __future__ import annotations

from dataclasses import dataclass

CARRIER_CEILING_HZ = 1500.0
"""Binaural fusion breaks down above roughly this carrier frequency."""


@dataclass(frozen=True)
class CarrierStack:
    left: tuple[float, ...]
    right: tuple[float, ...]


def stack_offsets(pairs: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Beat-frequency multipliers for each channel.

    Rendering and description both derive their frequencies from these, so
    the two paths cannot drift apart.
    """
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    return tuple(range(pairs)), tuple(range(1, pairs + 1))


def build_stack(base_hz: float, beat_hz: float, pairs: int) -> CarrierStack:
    left_offsets, right_offsets = stack_offsets(pairs)
    return CarrierStack(
        left=tuple(base_hz + k * beat_hz for k in left_offsets),
        right=tuple(base_hz + k * beat_hz for k in right_offsets),
    )


def beat_counts(pairs: int) -> tuple[int, int]:
    """(binaural beats, monaural amplitude beats per channel)."""
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    return pairs, pairs - 1


def validate_stack(
    stack: CarrierStack, ceiling_hz: float = CARRIER_CEILING_HZ
) -> None:
    highest = max(stack.left + stack.right)
    if highest > ceiling_hz:
        raise ValueError(
            f"highest carrier {highest:.1f} Hz exceeds the binaural fusion "
            f"ceiling of {ceiling_hz:.0f} Hz"
        )
