"""Sum resolved layers into stereo audio.

Levels within a session are relative, exactly as the patents state them
("Group B, 15 dB below Group A"); absolute output level is decided once, at
normalisation.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from farfield.beats import CARRIER_CEILING_HZ
from farfield.session import (
    SAM_CARRIER_CEILING_HZ,
    SAM_HEAD_SEPARATION_M,
    SAM_SPEED_OF_SOUND_MS,
)
from farfield.noise import render_bed, render_texture
from farfield.oscillators import TWO_PI, gate_envelope, phase_track
from farfield.session import Glide, SamSpec, Session, TremoloSplit
from farfield.timeline import Layer, PinkLayer, TextureLayer, Timeline, resolve
from farfield.voices import expand_voices, fundamental_voices


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def fade_window(
    n_samples: int,
    fade_in: int,
    fade_out: int,
    coherent_in: bool = False,
    coherent_out: bool = False,
) -> np.ndarray:
    """Fade in/out ramps for a layer's crossfades at segment boundaries.

    Independent signals (or beds, which never pass coherent_in/out) use the
    equal-power sin/cos law so overlapping crossfades hold perceived level
    steady. A phase-aligned continuation of the *same* tone (see R1 handoff)
    is correlated with its neighbour, so equal-power over-sums it by up to
    +3 dB; those ends use a linear law instead, where an out+in pair of
    ramps sums to exactly 1 at every overlap sample — *except* when the
    collision branch below rescales a too-short window, which breaks that
    exact-sum property for both laws (pre-existing shape, not new here).
    Coherence is decided in timeline._voices_phase_locked by bounding the
    relative-phase drift between the two copies across the whole overlap,
    not just by matching their frequency at the boundary — a same-value
    join can still diverge across a long overlap (see SEAM_DRIFT_BOUND_CYCLES).
    The emerge block's fade-in drifts 0.10-0.98 cycle from the outgoing tone
    across its 2 s overlap depending on the preset; on most bundled presets
    that fails SEAM_DRIFT_BOUND_CYCLES and the seam is NOT coherent, keeping
    the equal-power law like any other decorrelating pair, but attention and
    concentration sit at the low end (0.10 cycles, inside the bound) and are
    coherent.
    """
    window = np.ones(n_samples, dtype=np.float64)
    fade_in = min(fade_in, n_samples)
    fade_out = min(fade_out, n_samples)
    if fade_in + fade_out > n_samples:
        # The two ramps would overlap and the fade-out write would clobber
        # the fade-in's tail. Scale both down proportionally so they
        # exactly partition the window with no overwrite and no step.
        ratio = n_samples / (fade_in + fade_out)
        fade_in = int(fade_in * ratio)
        fade_out = n_samples - fade_in
    if fade_in > 0:
        if coherent_in:
            window[:fade_in] = np.linspace(0.0, 1.0, fade_in, endpoint=False)
        else:
            ramp = np.linspace(0.0, np.pi / 2.0, fade_in, endpoint=False)
            window[:fade_in] = np.sin(ramp)
    if fade_out > 0:
        if coherent_out:
            window[n_samples - fade_out :] = np.linspace(
                1.0, 0.0, fade_out, endpoint=False
            )
        else:
            ramp = np.linspace(0.0, np.pi / 2.0, fade_out, endpoint=False)
            window[n_samples - fade_out :] = np.cos(ramp)
    return window


_EAR_COLUMN = {"left": 0, "right": 1}


def sam_modulator(phase: np.ndarray, spec: SamSpec) -> np.ndarray:
    """The unit-amplitude modulator m(Phi) for a SAM path type.

    Every path is a pure function of the modulator's ACCUMULATED PHASE and
    is 2*pi periodic in it. That is what makes the seam handoff work: the
    handoff wraps the phase mod 2*pi, and none of these needs an unwrapped
    phase or an absolute time. In particular the discontinuous path's
    sample-and-hold grid is anchored to phase (a step that divides 2*pi),
    never to sample index, so it survives the wrap unchanged.
    """
    if spec.path == "closed":
        # The patents' literal equation: a source traversing a closed orbit,
        # its interaural-delay projection sinusoidal.
        return np.sin(phase)
    if spec.path == "open":
        # "Oscillate between two points": the AZIMUTH is sinusoidal and the
        # delay follows sin(theta), so the source decelerates at the turns.
        # Normalised by sin(theta_max) so the peak deviation is still phi_p,
        # i.e. open and closed reach the same ITD extremes and differ only
        # in the trajectory between them. Degenerates to closed as the arc
        # narrows.
        theta = math.asin(
            min(
                1.0,
                spec.depth_rad
                / (math.pi * spec.carrier_hz)
                / (SAM_HEAD_SEPARATION_M / SAM_SPEED_OF_SOUND_MS),
            )
        )
        if theta <= 1e-12:
            return np.sin(phase)
        return np.sin(theta * np.sin(phase)) / math.sin(theta)
    # discontinuous: jump cuts between `steps` fixed positions per cycle.
    step = 2.0 * np.pi / spec.steps
    return np.sin(np.floor(phase / step) * step)


RENDER_BLOCK_SECONDS = 60.0
"""Length of one tonal render block.

Tonal layers render block-by-block so every per-voice temporary (frequency
ramp slice, phase track, tone buffer, envelope slice) is block-length
rather than layer-length: a long layer's working set is bounded at a few
hundred MB regardless of layer duration, while the mix buffer itself stays
full-length. Oscillator phase carries across block edges exactly — each
block starts at the wrapped phase the previous block ended on — so a
blocked render is phase-identical to a whole-layer render up to float
rounding, and block size is an implementation detail, not a parameter of
the sound (see tests/test_render_blocks.py's block-size invariance)."""


def _ramp_slice(
    start: float, end: float, n_samples: int, b0: int, b1: int
) -> np.ndarray:
    """The [b0, b1) slice of ``frequency_ramp(start, end, n_samples)``.

    Evaluated pointwise from the same linear trajectory (both endpoints
    inclusive over the FULL layer), so a block sees exactly the frequencies
    the whole-layer ramp would give it at those absolute sample indices.
    """
    if n_samples == 1:
        return np.full(b1 - b0, float(start), dtype=np.float64)
    idx = np.arange(b0, b1, dtype=np.float64)
    return float(start) + (float(end) - float(start)) * (idx / (n_samples - 1))


def _fade_window_slice(
    n_samples: int,
    fade_in: int,
    fade_out: int,
    coherent_in: bool,
    coherent_out: bool,
    b0: int,
    b1: int,
) -> np.ndarray:
    """The [b0, b1) slice of ``fade_window(...)``, evaluated pointwise.

    Same clamping and collision rules as fade_window; the ramps are
    functions of the absolute sample index within the layer, so a block
    boundary falling inside a fade region sees the identical gain values
    the whole-layer window carries there.
    """
    fade_in = min(fade_in, n_samples)
    fade_out = min(fade_out, n_samples)
    if fade_in + fade_out > n_samples:
        ratio = n_samples / (fade_in + fade_out)
        fade_in = int(fade_in * ratio)
        fade_out = n_samples - fade_in
    window = np.ones(b1 - b0, dtype=np.float64)
    idx = np.arange(b0, b1, dtype=np.float64)
    if fade_in > 0:
        rising = idx < fade_in
        if rising.any():
            frac = idx[rising] / fade_in
            window[rising] = frac if coherent_in else np.sin(np.pi / 2.0 * frac)
    if fade_out > 0:
        falling = idx >= n_samples - fade_out
        if falling.any():
            frac = (idx[falling] - (n_samples - fade_out)) / fade_out
            window[falling] = (
                1.0 - frac if coherent_out else np.cos(np.pi / 2.0 * frac)
            )
    return window


def _render_layer(
    out: np.ndarray,
    layer: Layer,
    sample_rate: int,
    phases: dict,
    block_samples: int,
) -> None:
    """Render one tonal layer into ``out`` (the full mix buffer), blocked.

    Phase continuity across block edges is the load-bearing property: every
    oscillator (voice tones, tremolo LFOs, the gate's cycle counter, SAM's
    carrier and modulator) is an accumulator whose block starts at the
    wrapped phase the previous block ended on, and every other per-sample
    quantity (frequency ramps, fade windows, the rotation LFO) is a pure
    function of the absolute sample index — so sample k has the same value
    whether it falls mid-block or first-of-block, and no state is ever
    re-initialized at a boundary. Wrapping mod 2*pi at block edges is safe
    for the same reason the seam handoff's wrap is: every consumer of a
    phase is 2*pi-periodic in it (sin, and sam_modulator by construction).

    The cross-layer handoff contract is unchanged: on return, phases[key]
    holds each oscillator's phase AT sample ``n - fade_out_samples``, which
    is what a continuing layer starting at that absolute sample needs.
    """
    group = layer.group
    n = layer.n_samples
    h = n - layer.fade_out_samples
    gain = db_to_gain(group.level_db)
    rotation = group.rotation
    placement = group.placement
    sam = group.sam
    voices = expand_voices(group)

    # Running per-oscillator phase (seeded from the cross-layer store) and
    # the layer handoffs to publish once the whole layer has rendered.
    running: dict = {}
    handoffs: dict = {}

    def _phase_block(key, freq, blen, b0):
        """One oscillator's phase track for this block, state carried."""
        initial = running.get(key)
        if initial is None:
            initial = phases.get(key, 0.0)
        track, end = phase_track(
            freq, blen, sample_rate, initial_phase=initial
        )
        if b0 <= h < b0 + blen:
            # track[i] is the phase AT sample b0+i, so this is exactly the
            # handoff phase_track would report for the whole layer.
            handoffs[key] = float(track[h - b0] % TWO_PI)
        elif h == n and b0 + blen == n:
            handoffs[key] = end
        running[key] = end
        return track

    for b0 in range(0, n, block_samples):
        b1 = min(n, b0 + block_samples)
        blen = b1 - b0
        buf = np.zeros((blen, 2), dtype=np.float64)

        if rotation is not None:
            idx = np.arange(
                layer.start_sample + b0,
                layer.start_sample + b1,
                dtype=np.float64,
            )
            s = np.sin(
                2.0 * np.pi * idx / (sample_rate * rotation.period_s)
                + np.radians(rotation.phase_deg)
            )
            rot_a = np.sqrt((1.0 + rotation.depth * s) / 2.0)
            rot_b = np.sqrt((1.0 - rotation.depth * s) / 2.0)

        def _place(samples, ear):
            if rotation is None:
                if ear == "both":
                    buf[:, 0] += samples
                    buf[:, 1] += samples
                else:
                    buf[:, _EAR_COLUMN[ear]] += samples
                return
            # A single shared LFO phase for every voice in the group: a
            # pair's left/right members already sit on opposite output
            # channels (see `near` below), so the shared phase alone makes
            # them counter-rotate 180 degrees apart in ILD. Adding a +pi
            # offset for right-ear voices (as an earlier version of this
            # law specified) cancels that channel swap instead of
            # reinforcing it, collapsing the pair to co-rotation (measured
            # 0.0 deg apart, not the ~180 deg the tape shows) — so no
            # ear-based offset is applied here.
            near = 1 if ear == "right" else 0
            buf[:, near] += samples * rot_a
            buf[:, 1 - near] += samples * rot_b

        if sam is not None:
            # S_L/S_R per US 2013/0010967 A1:
            #     S_L = A*sin(Phi_c + phi_p*m(Phi_m) + phi_L)
            #     S_R = A*sin(Phi_c - phi_p*m(Phi_m) + phi_R)
            # A SAM group emits no Voices and takes no rotation or
            # placement (rejected at parse time), so it bypasses the voice
            # loop; tremolo and the crossfade envelope below still apply.
            carrier_phase = _phase_block(
                (group.name, "sam", "carrier"), sam.carrier_hz, blen, b0
            )
            modulator_phase = _phase_block(
                (group.name, "sam", "modulator"), sam.rate_hz, blen, b0
            )
            deviation = sam.depth_rad * sam_modulator(modulator_phase, sam)
            buf[:, 0] = np.sin(
                carrier_phase + deviation + math.radians(sam.offset_left_deg)
            )
            buf[:, 1] = np.sin(
                carrier_phase - deviation + math.radians(sam.offset_right_deg)
            )

        for voice in voices:
            freq = (
                voice.freq_start
                if voice.freq_start == voice.freq_end
                else _ramp_slice(voice.freq_start, voice.freq_end, n, b0, b1)
            )
            key = (group.name,) + voice.key
            track = _phase_block(key, freq, blen, b0)
            _place(voice.amplitude * np.sin(track), voice.ear)

            if placement is not None and voice.ear != "both":
                # The crossfeed copy is the same oscillator at a constant
                # phase offset, so it needs no accumulator of its own:
                # sin(track + offset) IS its whole trajectory.
                low_ear = "right" if group.high_ear == "left" else "left"
                sign = 1.0 if voice.ear == low_ear else -1.0
                cross = (
                    voice.amplitude * db_to_gain(placement.crossfeed_db)
                ) * np.sin(
                    track + sign * np.radians(placement.crossfeed_phase_deg)
                )
                _place(cross, "left" if voice.ear == "right" else "right")

        if isinstance(group.tremolo, TremoloSplit):
            for ch, (ear, spec) in enumerate((("left", group.tremolo.left),
                                              ("right", group.tremolo.right))):
                rate_traj = (
                    _ramp_slice(spec.rate_hz.start, spec.rate_hz.end, n, b0, b1)
                    if isinstance(spec.rate_hz, Glide)
                    else float(spec.rate_hz)
                )
                track = _phase_block((group.name, "tremolo", ear),
                                     rate_traj, blen, b0)
                buf[:, ch] *= 1.0 - spec.depth * (0.5 + 0.5 * np.sin(track))
        elif group.tremolo is not None:
            tremolo = group.tremolo
            rate_traj = (
                _ramp_slice(tremolo.rate_hz.start, tremolo.rate_hz.end,
                            n, b0, b1)
                if isinstance(tremolo.rate_hz, Glide)
                else float(tremolo.rate_hz)
            )
            track = _phase_block((group.name, "tremolo"), rate_traj, blen, b0)
            buf *= (1.0 - tremolo.depth * (0.5 + 0.5 * np.sin(track)))[:, None]

        # Isochronic gating. Opt-in: with group.gate None this block does
        # not execute at all. The gate's phase is in CYCLES (see
        # gate_envelope), carried across blocks exactly like the sine
        # accumulators; the layer handoff at a mid-block h costs one extra
        # gate_envelope call in the single block that contains h.
        if group.gate is not None:
            gate = group.gate
            gate_rate = (
                _ramp_slice(gate.rate_hz.start, gate.rate_hz.end, n, b0, b1)
                if isinstance(gate.rate_hz, Glide)
                else float(gate.rate_hz)
            )
            gate_key = (group.name, "gate")
            initial = running.get(gate_key)
            if initial is None:
                initial = phases.get(gate_key, 0.0)
            shape, gate_end = gate_envelope(
                gate_rate,
                blen,
                sample_rate,
                duty=gate.duty,
                edge_s=gate.edge_ms / 1000.0,
                initial_phase=initial,
            )
            if b0 <= h < b1:
                _, handoffs[gate_key] = gate_envelope(
                    gate_rate,
                    blen,
                    sample_rate,
                    duty=gate.duty,
                    edge_s=gate.edge_ms / 1000.0,
                    initial_phase=initial,
                    handoff_index=h - b0,
                )
            elif h == n and b1 == n:
                handoffs[gate_key] = gate_end
            running[gate_key] = gate_end
            buf *= (1.0 - gate.depth * (1.0 - shape))[:, None]

        window = _fade_window_slice(
            n,
            layer.fade_in_samples,
            layer.fade_out_samples,
            layer.coherent_fade_in,
            layer.coherent_fade_out,
            b0,
            b1,
        )
        buf *= (gain * window)[:, None]
        start = layer.start_sample + b0
        out[start : start + blen] += buf

    phases.update(handoffs)


def _render_pink_layer(
    layer: PinkLayer, sample_rate: int, seed: int
) -> np.ndarray:
    if layer.spec.level_db > -10.0 and layer.spec.color == "pink":
        warnings.warn(
            f"pink sound at {layer.spec.level_db:+.1f} dB is less than 10 dB "
            "below the beat signals; US5213562A recommends at least 10 dB",
            UserWarning,
            stacklevel=2,
        )
    rng = np.random.default_rng(seed)
    stereo = render_bed(layer.n_samples, sample_rate, layer.spec, rng)
    envelope = fade_window(
        layer.n_samples, layer.fade_in_samples, layer.fade_out_samples
    )
    return stereo * db_to_gain(layer.spec.level_db) * envelope[:, None]


def _render_texture_layer(
    layer: TextureLayer, sample_rate: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stereo = render_texture(layer.n_samples, sample_rate, layer.spec, rng)
    envelope = fade_window(
        layer.n_samples, layer.fade_in_samples, layer.fade_out_samples
    )
    return stereo * db_to_gain(layer.spec.level_db) * envelope[:, None]


def validate_timeline(timeline: Timeline) -> None:
    """Fail fast: check every layer's voices before synthesising.

    A voice that breaks the fusion ceiling or dips negative anywhere in the
    session should be reported before minutes of audio are generated, and
    `describe` can pre-flight a config with it.
    """
    for layer in timeline.layers:
        sam = layer.group.sam
        if sam is not None and sam.carrier_hz > SAM_CARRIER_CEILING_HZ:
            # Belt and braces: _parse_sam already rejects this, but a Group
            # can be built directly in Python and the ceiling is a physical
            # limit on the technique, not a YAML nicety.
            raise ValueError(
                f"sam carrier {sam.carrier_hz:.1f} Hz exceeds the "
                f"{SAM_CARRIER_CEILING_HZ:.0f} Hz interaural-phase "
                "localisation ceiling"
            )
        for voice in expand_voices(layer.group):
            highest = max(voice.freq_start, voice.freq_end)
            lowest = min(voice.freq_start, voice.freq_end)
            if highest > CARRIER_CEILING_HZ:
                raise ValueError(
                    f"highest carrier {highest:.1f} Hz exceeds the binaural "
                    f"fusion ceiling of {CARRIER_CEILING_HZ:.0f} Hz"
                )
            if lowest < 0.0:
                raise ValueError(
                    f"negative frequency {lowest:.1f} Hz is not renderable"
                )


def render_timeline(
    timeline: Timeline, seed: int = 0, block_samples: int | None = None
) -> np.ndarray:
    """Render a resolved timeline to a full-length (n, 2) float64 mix.

    Tonal layers render block-wise (see RENDER_BLOCK_SECONDS) so their
    working set is bounded; the mix buffer and the noise layers stay
    full-length. ``block_samples`` overrides the block length — the audio
    is invariant to it up to float rounding (tests/test_render_blocks.py),
    so it exists for those tests, not for tuning.
    """
    validate_timeline(timeline)
    if block_samples is None:
        block_samples = max(
            1, int(round(RENDER_BLOCK_SECONDS * timeline.sample_rate))
        )
    out = np.zeros((timeline.total_samples, 2), dtype=np.float64)
    phases: dict = {}
    for layer in timeline.layers:
        _render_layer(out, layer, timeline.sample_rate, phases, block_samples)
    for index, pink in enumerate(timeline.pink_layers):
        block = _render_pink_layer(pink, timeline.sample_rate, seed + index)
        start = pink.start_sample
        out[start : start + pink.n_samples] += block
    for index, texture in enumerate(timeline.texture_layers):
        block = _render_texture_layer(
            texture, timeline.sample_rate, seed + 1000 + index
        )
        start = texture.start_sample
        out[start : start + texture.n_samples] += block
    return out


def normalize(audio: np.ndarray, peak_dbfs: float) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    return audio * (db_to_gain(peak_dbfs) / peak)


def _edge_fade(audio: np.ndarray, sample_rate: int, edge_fade_s: float) -> np.ndarray:
    """Raised-cosine ramp on the head and tail of the final mix.

    Removes the session's hard-cut first/last sample (and the static-delay
    lead-in DC hold it exposes) without touching any resolved timing: this
    runs on the mix only, after the timeline has been fully rendered.
    """
    if edge_fade_s <= 0.0:
        return audio
    n = min(int(round(edge_fade_s * sample_rate)), len(audio) // 2)
    if n <= 0:
        return audio
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n, endpoint=False)))
    window = np.ones(len(audio), dtype=np.float64)
    window[:n] = ramp
    window[len(audio) - n :] = ramp[::-1]
    return audio * window[:, None]


def render_session(session: Session, seed: int = 0) -> np.ndarray:
    timeline = resolve(session)
    audio = normalize(render_timeline(timeline, seed=seed), session.peak_dbfs)
    return _edge_fade(audio, session.sample_rate, session.edge_fade_s)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), audio, sample_rate, subtype="PCM_24")


def sidecar(session: Session, timeline: Timeline) -> dict:
    rate = timeline.sample_rate

    def carriers(group, *, end: bool) -> dict:
        # Fundamentals only, deduped: the sidecar's carrier lists drive the
        # visualizer's meters, so harmonic partials would double-count them.
        voices = fundamental_voices(group)
        attr = "freq_end" if end else "freq_start"

        def ears(*wanted: str) -> list[float]:
            return sorted({
                round(getattr(v, attr), 6) for v in voices if v.ear in wanted
            })

        return {"left": ears("left", "both"), "right": ears("right", "both")}

    layers = []
    for layer in timeline.layers:
        start_beat, end_beat = layer.group.beat_bounds()
        layers.append(
            {
                "group": layer.group.name,
                "level_db": layer.group.level_db,
                "start_s": layer.start_sample / rate,
                "end_s": (layer.start_sample + layer.n_samples) / rate,
                "beat_start": start_beat,
                "beat_end": end_beat,
                "pairs": (
                    len(layer.group.pairs_spec)
                    if layer.group.pairs_spec is not None
                    else layer.group.pairs
                ),
                "carriers_start": carriers(layer.group, end=False),
                "carriers_end": carriers(layer.group, end=True),
                # Only present when in use: a null on every layer of every
                # existing session would be noise in a file people read.
                **(
                    {
                        "gate": {
                            "rate_hz": (
                                {
                                    "from": layer.group.gate.rate_hz.start,
                                    "to": layer.group.gate.rate_hz.end,
                                }
                                if isinstance(layer.group.gate.rate_hz, Glide)
                                else layer.group.gate.rate_hz
                            ),
                            "depth": layer.group.gate.depth,
                            "duty": layer.group.gate.duty,
                            "edge_ms": layer.group.gate.edge_ms,
                        }
                    }
                    if layer.group.gate is not None
                    else {}
                ),
                "rotation": (
                    {
                        "period_s": layer.group.rotation.period_s,
                        "depth": layer.group.rotation.depth,
                        "phase_deg": layer.group.rotation.phase_deg,
                    }
                    if layer.group.rotation is not None
                    else None
                ),
                "sam": (
                    {
                        "carrier_hz": layer.group.sam.carrier_hz,
                        "rate_hz": layer.group.sam.rate_hz,
                        "depth_rad": layer.group.sam.depth_rad,
                        "arc_deg": layer.group.sam.arc_deg,
                        "path": layer.group.sam.path,
                        "steps": (
                            layer.group.sam.steps
                            if layer.group.sam.path == "discontinuous"
                            else None
                        ),
                        "offset_left_deg": layer.group.sam.offset_left_deg,
                        "offset_right_deg": layer.group.sam.offset_right_deg,
                        "peak_itd_us": layer.group.sam.peak_itd_s() * 1e6,
                    }
                    if layer.group.sam is not None
                    else None
                ),
            }
        )

    # The key stays "pink" — the visualizer reads payload.session.pink — but
    # the entries describe whatever colour the bed actually is.
    def _bed_entry(p: PinkLayer) -> dict:
        entry = {
            "level_db": p.spec.level_db,
            "algorithm": p.spec.algorithm,
            "color": p.spec.color,
            "slope_db_per_decade": p.spec.resolved_slope(),
            "surf_rate_hz": p.spec.surf_rate_hz,
            "surf_depth": p.spec.surf_depth,
            "surf_phase_deg": p.spec.surf_phase_deg,
            "stereo_mode": p.spec.stereo_mode,
            "start_s": p.start_sample / rate,
            "end_s": (p.start_sample + p.n_samples) / rate,
        }
        # Mode-specific stereo parameters are only meaningful (and only set)
        # for the mode that uses them, so they appear only when in use rather
        # than as a row of nulls on every bed.
        if p.spec.lfo_period_s is not None:
            entry["lfo_period_s"] = p.spec.lfo_period_s
        if p.spec.stereo_depth_db is not None:
            entry["stereo_depth_db"] = p.spec.stereo_depth_db
        if p.spec.interaural_delay_us is not None:
            entry["interaural_delay_us"] = p.spec.interaural_delay_us
        return entry

    pink = [_bed_entry(p) for p in timeline.pink_layers]

    def _texture_entry(t: TextureLayer) -> dict:
        return {
            "band_hz": list(t.spec.band_hz),
            "level_db": t.spec.level_db,
            "pan": {
                "period_s": t.spec.pan_period_s,
                "ild_amplitude_db": t.spec.pan_ild_amplitude_db,
                "phase_deg": t.spec.pan_phase_deg,
            },
            "surf_rate_hz": t.spec.surf_rate_hz,
            "surf_depth": t.spec.surf_depth,
            "surf_phase_deg": t.spec.surf_phase_deg,
            "start_s": t.start_sample / rate,
            "duration_s": t.n_samples / rate,
        }

    texture = [_texture_entry(t) for t in timeline.texture_layers]

    return {
        "name": session.name,
        "title": session.title,
        "fidelity": session.fidelity,
        "notes": session.notes,
        "sample_rate": rate,
        "duration_s": timeline.total_samples / rate,
        "layers": layers,
        "pink": pink,
        "texture": texture,
    }


def fade_edges(audio: np.ndarray, sample_rate: int, fade_s: float) -> np.ndarray:
    """Raised-cosine fade-in and fade-out on a rendered (n, 2) mix.

    Each fade is clamped to a quarter of the render so short sessions are
    never faded into silence."""
    if fade_s <= 0:
        return audio
    n = min(int(round(fade_s * sample_rate)), len(audio) // 4)
    if n < 1:
        return audio
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
    audio = audio.copy()
    audio[:n] *= ramp[:, None]
    audio[-n:] *= ramp[::-1, None]
    return audio
