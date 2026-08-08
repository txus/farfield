"""Expand a group into concrete per-ear voices.

The renderer, validator, sidecar, analysis and CLI all consume voices, so
the two group forms (legacy carrier stack, free pairs) and both harmonic
rules live in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass

from farfield.beats import stack_offsets
from farfield.session import Glide, Group
from farfield.waveshape import expand_harmonics, normalize_harmonics


@dataclass(frozen=True)
class Voice:
    key: tuple
    ear: str  # "left" | "right" | "both"
    freq_start: float
    freq_end: float
    amplitude: float


def _beat_ends(beat) -> tuple[float, float]:
    if isinstance(beat, Glide):
        return beat.start, beat.end
    return float(beat), float(beat)


def expand_voices(group: Group) -> tuple[Voice, ...]:
    """Every per-ear sine a group emits.

    A SAM group emits NO voices: its single carrier is not an independent
    per-ear tone but one oscillator whose phase is split antiphase between
    the ears at render time, so it cannot be described as a Voice and is
    rendered by its own path in render._render_sam. Callers that walk
    voices to enumerate a group's tones must handle ``group.sam`` (see
    voice_frequencies below, and render.validate_timeline).
    """
    if group.sam is not None:
        return ()
    harmonics = normalize_harmonics(expand_harmonics(group.harmonics))
    voices: list[Voice] = []

    if group.pairs_spec is None:
        beat0, beat1 = _beat_ends(group.beat)
        left_offsets, right_offsets = stack_offsets(group.pairs)
        amp_div = group.pairs
        for h in harmonics:
            amplitude = h.amplitude * (1.0 / amp_div)
            # Multiply h*beat FIRST: the legacy renderer computed
            # base + k*(h*beat), and byte-identity requires the same
            # float-op association.
            beat_h0 = h.index * beat0
            beat_h1 = h.index * beat1
            for ear, offsets in (("left", left_offsets), ("right", right_offsets)):
                for k in offsets:
                    voices.append(Voice(
                        key=(k, h.index, ear),
                        ear=ear,
                        freq_start=group.carrier_base + k * beat_h0,
                        freq_end=group.carrier_base + k * beat_h1,
                        amplitude=amplitude,
                    ))
        return tuple(voices)

    amp_div = len(group.pairs_spec)
    high, low = (("left", "right") if group.high_ear == "left"
                 else ("right", "left"))
    for i, pair in enumerate(group.pairs_spec):
        for h in harmonics:
            amplitude = h.amplitude * (1.0 / amp_div)
            if pair.kind == "center":
                beat0, beat1 = _beat_ends(pair.beat)
                half0, half1 = h.index * beat0 / 2.0, h.index * beat1 / 2.0
                voices.append(Voice((i, h.index, high), high,
                                    pair.center + half0, pair.center + half1,
                                    amplitude))
                voices.append(Voice((i, h.index, low), low,
                                    pair.center - half0, pair.center - half1,
                                    amplitude))
            elif pair.kind == "explicit":
                voices.append(Voice((i, h.index, "left"), "left",
                                    pair.left * h.index, pair.left * h.index,
                                    amplitude))
                voices.append(Voice((i, h.index, "right"), "right",
                                    pair.right * h.index, pair.right * h.index,
                                    amplitude))
            else:  # mono
                voices.append(Voice((i, h.index, "both"), "both",
                                    pair.mono * h.index, pair.mono * h.index,
                                    amplitude))
    return tuple(voices)


def fundamental_voices(group: Group) -> tuple[Voice, ...]:
    """Only the fundamentals — harmonic partials excluded.

    Meters, the spectrogram's carrier span and the sidecar's carrier lists
    describe the *carriers* a listener is meant to hear; including every
    harmonic partial bloats them with tones that are mastering-level
    detail. Validation deliberately still walks every voice, harmonics
    included, since a partial can be what breaks the fusion ceiling.
    """
    return tuple(v for v in expand_voices(group) if v.key[1] == 1)


def voice_frequencies(group: Group) -> list[float]:
    # A SAM group has no Voices but does have a carrier, and the meters,
    # spectrogram band and sidecar all key off this list — reporting
    # nothing would hide the only tone the group actually emits.
    if group.sam is not None:
        return [round(group.sam.carrier_hz, 6)]
    found: set[float] = set()
    for voice in fundamental_voices(group):
        found.add(round(voice.freq_start, 6))
        found.add(round(voice.freq_end, 6))
    return sorted(found)
