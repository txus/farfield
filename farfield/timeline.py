"""Resolve a declarative session into timed render instructions.

Segments overlap and crossfade rather than butt-joining, matching the
patents' description of signal groups sounding together across transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from farfield.session import (
    EMERGE_FADE_IN_S,
    Emerge,
    Glide,
    Group,
    PinkSpec,
    Segment,
    Session,
    TextureSpec,
)
from farfield.voices import expand_voices

__all__ = [
    "EMERGE_FADE_IN_S",
    "Layer",
    "PinkLayer",
    "TextureLayer",
    "Timeline",
    "resolve",
]

SEAM_DRIFT_BOUND_CYCLES = 0.15
"""Maximum tolerated relative-phase drift, in cycles, between a coherent
seam's outgoing and incoming voice copies over the overlap.

At a relative phase error of phi cycles, a linear 50/50 sum of two
equal-amplitude copies has worst-case gain |cos(pi*phi)|, while the
equal-power law sums the same near-locked pair to sqrt(2)*|cos(pi*phi)| —
both laws degrade with the same cos(pi*phi) envelope, but equal-power
carries an extra sqrt(2) that turns its degradation into a swell rather
than linear's dip. Linear's worst-case dip therefore stays SMALLER in
magnitude than equal-power's worst-case over-sum for every phi below the
crossover where sqrt(2)*cos(pi*phi)^2 = 1, i.e. phi = 0.182 cycles
(cos(pi*phi) = 2**-0.25). Below that crossover linear is strictly the
better law even when imperfectly locked, so the bound is set with margin
inside it: at phi = 0.15 cycles, linear's worst dip is
20*log10(cos(0.15*pi)) ~= -1.0 dB, against the +2 to +3 dB over-sum
equal-power produces on the same near-locked seam (see fade_window).
For CORRELATED copies both laws null completely at anti-phase (the
sqrt(2)/1 factor scales a cos that reaches zero either way) and swell at
in-phase (+3 dB equal-power, 0 dB linear); the familiar "-3 dB dip"
bound belongs only to DEcorrelated pairs, where equal-power holds summed
power flat. Neither law is uniformly the "safer" fallback in the
abstract -- past 0.15 cycles the relative phase sweeps through
anti-phase somewhere in the overlap regardless, and equal-power is
simply the better of two imperfect options because its swelling
mid-crossfade gains never cut the summed level as deep as linear's
do."""


@dataclass(frozen=True)
class Layer:
    group: Group
    start_sample: int
    n_samples: int
    fade_in_samples: int
    fade_out_samples: int
    # A fade end is "coherent" when the adjacent segment carries a group of
    # the same name and level_db whose voice set stays phase-locked across
    # the overlap — every matched voice's relative-phase drift over the
    # overlap is within SEAM_DRIFT_BOUND_CYCLES (see _voices_phase_locked):
    # the two copies are near enough to a continuation of one tone (see R1)
    # that their crossfade can use a linear (amplitude-summing) law rather
    # than the equal-power law, which over-sums correlated signals by up to
    # +3 dB. This is deliberately conservative: matching the frequency only
    # AT the boundary is not enough — two glides that merely touch the same
    # boundary value can still drift tens of cycles apart across a long
    # overlap and dip several dB under a linear law, so the two
    # trajectories' full drift across the overlap is what is checked, not
    # just their endpoints. In practice this means the measured presets'
    # constant continuing layers are coherent, genuinely identical-slope
    # glides are coherent, and the emerge block's glide diverges from its
    # neighbour by 0.10-0.98 cycle over its 2 s overlap depending on the
    # preset: attention and concentration sit at the low end (0.10 cycles,
    # inside SEAM_DRIFT_BOUND_CYCLES) and are coherent; every other bundled
    # emerge seam drifts 0.5+ cycles and is not.
    coherent_fade_in: bool = False
    coherent_fade_out: bool = False


@dataclass(frozen=True)
class PinkLayer:
    spec: PinkSpec
    start_sample: int
    n_samples: int
    fade_in_samples: int
    fade_out_samples: int


@dataclass(frozen=True)
class TextureLayer:
    spec: TextureSpec
    start_sample: int
    n_samples: int
    fade_in_samples: int
    fade_out_samples: int


@dataclass(frozen=True)
class Timeline:
    sample_rate: int
    total_samples: int
    layers: tuple[Layer, ...]
    pink_layers: tuple[PinkLayer, ...]
    texture_layers: tuple[TextureLayer, ...] = ()


def _seam_drift_cycles(
    out_freq_start: float,
    out_freq_end: float,
    out_n_samples: int,
    in_freq_start: float,
    in_freq_end: float,
    in_n_samples: int,
    overlap_samples: int,
    rate: int,
) -> float:
    """Relative-phase drift, in cycles, between one voice's outgoing and
    incoming copies across a seam's overlap.

    Both copies' frequency trajectories are linear over their own layer
    (see oscillators.frequency_ramp). The overlap is the outgoing layer's
    *last* overlap_samples and the incoming layer's *first* overlap_samples,
    so this evaluates each trajectory at those two absolute times and
    integrates their difference (trapezoidal, exact for two linear
    trajectories up to a <=1-sample approximation: frequency_ramp is
    linspace(start, end, n), so the true endpoint is at
    overlap_samples/(n-1) rather than the overlap_samples/n used below --
    a sub-ppm difference on any real layer, well under the margin this
    bound is set with) over the overlap's T seconds:

        drift_cycles = 0.5 * ((f_out_start - f_in_start)
                              + (f_out_end - f_in_end)) * T

    Returned as the WORST running value, not the net integral: when the
    frequency difference changes sign inside the overlap (out ramps above
    then below the incoming copy, say 4->8 against 8->4 beats), the net
    integral can cancel to zero while the relative phase mid-overlap has
    already walked tens of cycles away and back. The difference of two
    linear trajectories is itself linear, so the running integral's only
    interior extremum sits at the sign change t* = T*d0/(d0-dT), with
    value 0.5*d0*t*; the drift that matters is the larger in magnitude of
    that extremum and the net.
    """
    if out_n_samples <= 0 or in_n_samples <= 0 or overlap_samples <= 0:
        return float("inf")
    out_span = out_freq_end - out_freq_start
    in_span = in_freq_end - in_freq_start
    f_out_start = out_freq_end - out_span * (overlap_samples / out_n_samples)
    f_out_end = out_freq_end
    f_in_start = in_freq_start
    f_in_end = in_freq_start + in_span * (overlap_samples / in_n_samples)
    overlap_s = overlap_samples / rate
    d0 = f_out_start - f_in_start
    d1 = f_out_end - f_in_end
    net = 0.5 * (d0 + d1) * overlap_s
    if d0 * d1 < 0.0:
        t_star = overlap_s * d0 / (d0 - d1)
        interior = 0.5 * d0 * t_star
        if abs(interior) > abs(net):
            return interior
    return net


def _voices_phase_locked(
    out_group: Group,
    in_group: Group,
    out_n_samples: int,
    in_n_samples: int,
    overlap_samples: int,
    rate: int,
) -> bool:
    """True when out_group's voice set exactly matches in_group's (same
    per-voice (key, ear)) and every matched voice's relative-phase drift
    across the overlap is within SEAM_DRIFT_BOUND_CYCLES.

    A constant continuing tone (drift 0) and a glide whose slope continues
    identically into the next segment (drift 0) both pass. A glide that
    merely touches the same boundary frequency but then diverges — or two
    constant tones at different frequencies — accumulates drift across the
    overlap and correctly fails this even though a naive "same value at the
    join" check would pass it.
    """
    # SAM groups expand to NO voices, so two *different* SAM groups would
    # compare {} == {} and be judged coherent by the loop below. Both
    # accumulators (carrier and modulator) are handed off across the seam,
    # so an identical SamSpec on both sides is a true continuation with
    # zero drift (SamSpec is frozen and has no glide fields) and the linear
    # law is exactly right; anything else is not a continuation at all.
    if out_group.sam is not None or in_group.sam is not None:
        return out_group.sam == in_group.sam and not (
            expand_voices(out_group) or expand_voices(in_group)
        )
    out_voices = {voice.key: voice for voice in expand_voices(out_group)}
    in_voices = {voice.key: voice for voice in expand_voices(in_group)}
    if out_voices.keys() != in_voices.keys():
        return False
    for key, out_voice in out_voices.items():
        in_voice = in_voices[key]
        drift = _seam_drift_cycles(
            out_voice.freq_start, out_voice.freq_end, out_n_samples,
            in_voice.freq_start, in_voice.freq_end, in_n_samples,
            overlap_samples, rate,
        )
        if abs(drift) > SEAM_DRIFT_BOUND_CYCLES:
            return False
    return True


def _primary_group(segment: Segment) -> Group:
    return max(segment.groups, key=lambda g: g.level_db)


def _emerge_segment(last: Segment, emerge: Emerge) -> Segment:
    primary = _primary_group(last)
    _, ending_beat = primary.beat_bounds()
    return Segment(
        duration_s=emerge.duration_s,
        overlap_s=0.0,
        groups=(
            replace(primary, beat=Glide(ending_beat, emerge.target_beat)),
        ),
        pink=None,
    )


def resolve(session: Session) -> Timeline:
    rate = session.sample_rate
    segments = list(session.segments)
    emerging = session.emerge is not None
    if emerging:
        segments.append(_emerge_segment(segments[-1], session.emerge))

    lengths = [int(round(s.duration_s * rate)) for s in segments]

    # Overlap between segment i and i+1, clamped to both their lengths.
    overlaps: list[int] = []
    for i in range(len(segments) - 1):
        requested = int(round(segments[i].overlap_s * rate))
        overlaps.append(max(0, min(requested, lengths[i], lengths[i + 1])))
    if emerging:
        # The emerge block crossfades with what precedes it rather than
        # butting on: the outgoing segment (and its pink noise) fades out
        # across exactly the window the emerge glide fades in over.
        overlaps[-1] = min(
            int(round(EMERGE_FADE_IN_S * rate)), lengths[-2], lengths[-1]
        )

    starts: list[int] = [0]
    for i in range(1, len(segments)):
        starts.append(starts[i - 1] + lengths[i - 1] - overlaps[i - 1])

    # For each gap i (between segment i and i+1), the set of (name, level_db)
    # keys whose voice set stays phase-locked across it (see
    # _voices_phase_locked): those layers' adjoining fade ends are coherent
    # and must use a linear crossfade law instead of equal-power.
    # NOTE: a duplicated (name, level_db) within one segment collapses to a
    # single dict entry here (pre-existing schema looseness, not new in this
    # pass) — parked rather than fixed.
    coherent_keys_at_gap: list[set[tuple[str, float]]] = []
    for i in range(len(segments) - 1):
        if overlaps[i] <= 0:
            coherent_keys_at_gap.append(set())
            continue
        out_by_key = {(g.name, g.level_db): g for g in segments[i].groups}
        in_by_key = {(g.name, g.level_db): g for g in segments[i + 1].groups}
        coherent_keys_at_gap.append({
            key for key in out_by_key.keys() & in_by_key.keys()
            if _voices_phase_locked(
                out_by_key[key], in_by_key[key],
                lengths[i], lengths[i + 1], overlaps[i], rate,
            )
        })

    layers: list[Layer] = []
    pink_layers: list[PinkLayer] = []
    texture_layers: list[TextureLayer] = []
    for i, segment in enumerate(segments):
        fade_in = overlaps[i - 1] if i > 0 else 0
        fade_out = overlaps[i] if i < len(overlaps) else 0
        for group in segment.groups:
            key = (group.name, group.level_db)
            coherent_out = i < len(overlaps) and key in coherent_keys_at_gap[i]
            coherent_in = i > 0 and key in coherent_keys_at_gap[i - 1]
            layers.append(
                Layer(
                    group=group,
                    start_sample=starts[i],
                    n_samples=lengths[i],
                    fade_in_samples=fade_in,
                    fade_out_samples=fade_out,
                    coherent_fade_in=coherent_in,
                    coherent_fade_out=coherent_out,
                )
            )
        if segment.pink is not None:
            pink_layers.append(
                PinkLayer(
                    spec=segment.pink,
                    start_sample=starts[i],
                    n_samples=lengths[i],
                    fade_in_samples=fade_in,
                    fade_out_samples=fade_out,
                )
            )
        if segment.texture is not None:
            texture_layers.append(
                TextureLayer(
                    spec=segment.texture,
                    start_sample=starts[i],
                    n_samples=lengths[i],
                    fade_in_samples=fade_in,
                    fade_out_samples=fade_out,
                )
            )

    total = max(
        (layer.start_sample + layer.n_samples for layer in layers), default=0
    )
    layers.sort(key=lambda layer: layer.start_sample)
    return Timeline(
        sample_rate=rate,
        total_samples=total,
        layers=tuple(layers),
        pink_layers=tuple(pink_layers),
        texture_layers=tuple(texture_layers),
    )
