from dataclasses import replace

import numpy as np
import pytest

from farfield.oscillators import TWO_PI, frequency_ramp, render_tone
from tests.support import load_preset
from farfield.render import render_session, render_timeline
from farfield.session import load_session_dict
from farfield.timeline import resolve

RATE = 48000


def test_handoff_index_matches_continuous_oscillator_constant():
    # 137.3 Hz * 2 s = 274.6 cycles: far from an integer, the regression case.
    n, h = RATE * 3, RATE * 2
    _, handoff = render_tone(137.3, n, RATE, handoff_index=h)
    _, expected = render_tone(137.3, h, RATE)
    d = (handoff - expected) % TWO_PI
    assert min(d, TWO_PI - d) < 1e-9


def test_handoff_index_matches_continuous_oscillator_glide():
    n, h = RATE * 3, RATE * 2
    ramp = frequency_ramp(200.0, 100.0, n)
    _, handoff = render_tone(ramp, n, RATE, handoff_index=h)
    _, expected = render_tone(ramp[:h], h, RATE)
    d = (handoff - expected) % TWO_PI
    assert min(d, TWO_PI - d) < 1e-9


def test_handoff_index_none_reproduces_legacy():
    n = RATE
    _, legacy = render_tone(313.7, n, RATE)
    _, explicit = render_tone(313.7, n, RATE, handoff_index=n)
    assert legacy == explicit


def _two_segment_session():
    group = {"name": "g", "pairs": [{"center": 137.3, "beat": 0.0001}],
             "harmonics": [1.0]}
    return load_session_dict({
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": 6, "overlap": 2, "groups": [dict(group)]},
            {"duration": 6, "groups": [dict(group)]},
        ],
    })


def test_continuing_layer_holds_level_through_the_seam():
    # Before R1: 137.3 Hz * 2 s overlap = 274.6 cycles -> ~144 deg phase
    # error between the outgoing and incoming copies, a deep partial
    # cancellation. After R1+R3: the copies are phase-aligned and the
    # coherent seam uses a linear (equal-gain) crossfade, so the level
    # through the seam is flat rather than dipping or over-summing.
    audio = render_session(_two_segment_session())
    env = np.sqrt(np.convolve(audio[:, 0] ** 2, np.ones(4800) / 4800, "valid"))
    seam = env[4 * RATE: 6 * RATE + RATE // 2]
    steady = np.median(env[RATE: 3 * RATE])
    dip_db = 20 * np.log10(seam.min() / steady)
    assert dip_db > -1.0, f"seam dips {dip_db:.1f} dB"


def test_tremolo_phase_hands_off_at_the_handoff_sample():
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": 6, "overlap": 2, "groups": [
                {"name": "g", "pairs": [{"center": 200.0, "beat": 0.0001}],
                 "harmonics": [1.0],
                 "tremolo": {"rate_hz": 0.731, "depth": 0.8}}]},
            {"duration": 6, "groups": [
                {"name": "g", "pairs": [{"center": 200.0, "beat": 0.0001}],
                 "harmonics": [1.0],
                 "tremolo": {"rate_hz": 0.731, "depth": 0.8}}]},
        ],
    }
    audio = render_session(load_session_dict(data))
    # The tremolo envelope of a continuous oscillator at 0.731 Hz: compare
    # the rendered envelope's phase before and after the seam by fitting a
    # sinusoid to the smoothed envelope in [1,3] s and checking its
    # extrapolation into [7,9] s stays in phase (residual < 15% of swing).
    env = np.sqrt(np.convolve(audio[:, 0] ** 2, np.ones(2400) / 2400, "valid"))
    t = np.arange(len(env)) / RATE
    w = 2 * np.pi * 0.731

    def fit(seg):
        tt, ee = t[seg], env[seg]
        A = np.column_stack([np.sin(w * tt), np.cos(w * tt), np.ones_like(tt)])
        coef, *_ = np.linalg.lstsq(A, ee, rcond=None)
        return coef

    pre = fit(slice(RATE, 3 * RATE))
    post_seg = slice(7 * RATE, 9 * RATE)
    predicted = pre[0] * np.sin(w * t[post_seg]) + pre[1] * np.cos(w * t[post_seg]) + pre[2]
    residual = np.abs(predicted - env[post_seg]).mean()
    swing = env[post_seg].max() - env[post_seg].min()
    assert residual < 0.15 * swing


def test_mix_edges_are_faded_by_default():
    audio = render_session(_two_segment_session())
    assert np.abs(audio[0]).max() < 1e-6
    assert np.abs(audio[-1]).max() < 1e-6
    # 10%..90% rise spread over roughly the configured 30 ms
    env = np.abs(audio[: RATE // 2, 0])
    peak = np.median(np.abs(audio[RATE: 2 * RATE, 0]).max())
    i10 = np.argmax(env > 0.1 * peak)
    i90 = np.argmax(env > 0.9 * peak)
    assert 0.010 * RATE < (i90 - i10) < 0.080 * RATE


def test_edge_fade_zero_reproduces_hard_edges():
    s = _two_segment_session()
    audio = render_session(replace(s, edge_fade_s=0.0))
    assert np.abs(audio[0]).max() > 1e-4 or np.abs(audio[1]).max() > 1e-4


def test_bad_edge_fade_rejected():
    with pytest.raises(ValueError, match="edge_fade_s"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "output": {"edge_fade_s": -0.1},
            "segments": [{"duration": 5,
                          "groups": [{"name": "a", "beat": 4.0}]}],
        })


def test_bad_edge_fade_rejected_above_upper_bound():
    with pytest.raises(ValueError, match="edge_fade_s"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "output": {"edge_fade_s": 5.1},
            "segments": [{"duration": 5,
                          "groups": [{"name": "a", "beat": 4.0}]}],
        })


def test_non_numeric_edge_fade_rejected():
    with pytest.raises(ValueError, match="edge_fade_s"):
        load_session_dict({
            "name": "t", "title": "T", "fidelity": "measured-tape",
            "output": {"edge_fade_s": "abc"},
            "segments": [{"duration": 5,
                          "groups": [{"name": "a", "beat": 4.0}]}],
        })


def test_edge_fade_clamps_on_a_session_shorter_than_two_fades():
    # Minor 6: a render shorter than 2*edge_fade_s must not crash or
    # produce a negative-length window; _edge_fade clamps each end to
    # len(audio)//2 samples.
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [{"duration": 0.05,
                      "groups": [{"name": "a", "beat": 4.0}]}],
    }
    audio = render_session(load_session_dict(data))
    assert len(audio) == round(0.05 * RATE)
    assert np.abs(audio[0]).max() < 1e-6
    assert np.abs(audio[-1]).max() < 1e-6


def test_coherent_seam_is_flat_after_equal_gain():
    audio = render_session(_two_segment_session())
    env = np.sqrt(np.convolve(audio[:, 0] ** 2, np.ones(4800) / 4800, "valid"))
    steady = np.median(env[RATE: 3 * RATE])
    seam = env[4 * RATE: 6 * RATE]
    excursion_db = 20 * np.abs(np.log10(seam / steady)).max()
    assert excursion_db < 0.3, f"seam excursion {excursion_db:.2f} dB"


def _non_coherent_level_change_session_dict():
    return {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": 6, "overlap": 2, "groups": [
                {"name": "g", "pairs": [{"center": 137.3, "beat": 0.0001}],
                 "harmonics": [1.0], "level_db": 0.0}]},
            {"duration": 6, "groups": [
                {"name": "g", "pairs": [{"center": 137.3, "beat": 0.0001}],
                 "harmonics": [1.0], "level_db": -6.0}]},
        ],
    }


def test_non_coherent_seam_keeps_equal_power():
    # Same group name, different level: NOT coherent; equal-power stays.
    data = _non_coherent_level_change_session_dict()
    timeline = resolve(load_session_dict(data))
    layers = timeline.layers
    assert layers[0].coherent_fade_out is False
    assert layers[1].coherent_fade_in is False


def test_non_coherent_level_change_seam_has_no_deep_cancellation():
    # A wiring check alone (flags only) wouldn't catch a broken fade law;
    # confirm the audio itself: an equal-power crossfade between a 0 dB
    # steady level and a -6 dB steady level should never dip more than
    # ~1 dB below the quieter (incoming) side's own steady level.
    audio = render_session(load_session_dict(
        _non_coherent_level_change_session_dict()))
    env = np.sqrt(np.convolve(audio[:, 0] ** 2, np.ones(4800) / 4800, "valid"))
    quiet_steady = np.median(env[7 * RATE: 9 * RATE])
    seam = env[4 * RATE: 6 * RATE]
    dip_db = 20 * np.log10(seam.min() / quiet_steady)
    assert dip_db > -1.0, f"seam dips {dip_db:.1f} dB below the quiet side"


def test_beat_jump_across_seam_is_not_coherent():
    # Same name and level_db, but the beat jumps (12 Hz -> 4 Hz) rather than
    # continuing smoothly across the boundary: this must NOT be flagged
    # coherent, or the moving carrier decorrelates from its "continuation"
    # and the linear law dips (this is the review's repro: -2.97 dB on the
    # simulated old (name, level_db)-only rule; verified below via the flag
    # and via the actual audio, which should stay flat under equal-power).
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": 20, "overlap": 6, "groups": [
                {"name": "g", "pairs": [{"center": 137.3, "beat": 12.0}],
                 "harmonics": [1.0], "level_db": 0.0}]},
            {"duration": 20, "groups": [
                {"name": "g", "pairs": [{"center": 137.3, "beat": 4.0}],
                 "harmonics": [1.0], "level_db": 0.0}]},
        ],
    }
    session = load_session_dict(data)
    timeline = resolve(session)
    layers = timeline.layers
    assert layers[0].coherent_fade_out is False
    assert layers[1].coherent_fade_in is False

    audio = render_session(session)
    env = np.sqrt(np.convolve(audio[:, 1] ** 2, np.ones(RATE) / RATE, "valid"))
    steady = np.median(env[2 * RATE: 5 * RATE])
    seam = env[14 * RATE: 26 * RATE]
    excursion_db = 20 * np.abs(np.log10(seam / steady)).max()
    assert excursion_db < 0.5, f"seam excursion {excursion_db:.2f} dB"


def _glide_voice_session_dict(seg2_beat):
    # Stack form with carrier_base=0.0 and pairs=1 puts the whole group beat
    # trajectory directly on the right ear's voice frequency (right offset
    # k=1 -> freq(t) = 0 + 1*beat(t)); the left ear (k=0) is a fixed 0 Hz
    # "silent" voice, irrelevant to what these tests measure (right ear).
    # seg1 glides 260 Hz -> 200 Hz over 6 s (slope -10 Hz/s); the overlap is
    # its last 2 s, so seg1's own trajectory across the overlap is exactly
    # 220 Hz -> 200 Hz.
    return {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": 6, "overlap": 2, "groups": [
                {"name": "g", "beat": {"from": 260.0, "to": 200.0},
                 "carrier_base": 0.0, "pairs": 1, "harmonics": [1.0],
                 "level_db": 0.0}]},
            {"duration": 6, "groups": [
                {"name": "g", "beat": seg2_beat,
                 "carrier_base": 0.0, "pairs": 1, "harmonics": [1.0],
                 "level_db": 0.0}]},
        ],
    }


def test_glide_that_continues_across_the_seam_is_coherent():
    # seg2 glides 220 -> 160 Hz over 6 s, the same -10 Hz/s slope as seg1:
    # over the 2 s overlap (its first 2 s) it renders exactly seg1's tail,
    # 220 -> 200 Hz. Drift is 0 cycles -> coherent, and the audio is flat.
    data = _glide_voice_session_dict({"from": 220.0, "to": 160.0})
    session = load_session_dict(data)
    timeline = resolve(session)
    layers = timeline.layers
    assert layers[0].coherent_fade_out is True
    assert layers[1].coherent_fade_in is True

    audio = render_session(session)
    env = np.sqrt(np.convolve(audio[:, 1] ** 2, np.ones(4800) / 4800, "valid"))
    steady = np.median(env[RATE: 3 * RATE])
    seam = env[4 * RATE: 6 * RATE]
    excursion_db = 20 * np.abs(np.log10(seam / steady)).max()
    assert excursion_db < 0.3, f"seam excursion {excursion_db:.2f} dB"


def test_trajectories_that_cross_inside_the_overlap_are_not_coherent():
    # The net drift integral of two trajectories that CROSS mid-overlap can
    # cancel to zero while the relative phase walks cycles away and back:
    # seg1's tail runs 220 -> 200 Hz while seg2's head runs 205 -> 215 Hz,
    # so the difference is +15 Hz at the start and -15 Hz at the end -- net
    # integral 0, but the running integral peaks at 7.5 cycles where they
    # cross. The guard must use the running extremum, not the net.
    from farfield.timeline import _seam_drift_cycles

    n = 6 * RATE
    drift = _seam_drift_cycles(260.0, 200.0, n, 205.0, 235.0, n, 2 * RATE, RATE)
    assert abs(drift) > 7.0, f"net-only integral slipped through: {drift}"

    data = _glide_voice_session_dict({"from": 205.0, "to": 235.0})
    timeline = resolve(load_session_dict(data))
    assert timeline.layers[0].coherent_fade_out is False
    assert timeline.layers[1].coherent_fade_in is False


def test_glide_that_only_touches_the_boundary_value_is_not_coherent():
    # The rejected round-1 rule: seg2 is CONSTANT at 200 Hz, so the
    # boundary frequency matches seg1's ending value exactly, but the two
    # trajectories diverge across the overlap (seg1's tail is 220 -> 200,
    # seg2 stays at 200 the whole time): drift = 0.5*((220-200)+(200-200))*2
    # = 20 cycles, far past SEAM_DRIFT_BOUND_CYCLES. Must NOT be coherent,
    # and the audio must stay within equal-power's bound (never worse than
    # about -3.5 dB), not dip further under a wrongly-applied linear law.
    data = _glide_voice_session_dict(200.0)
    session = load_session_dict(data)
    timeline = resolve(session)
    layers = timeline.layers
    assert layers[0].coherent_fade_out is False
    assert layers[1].coherent_fade_in is False

    audio = render_session(session)
    env = np.sqrt(np.convolve(audio[:, 1] ** 2, np.ones(4800) / 4800, "valid"))
    steady = np.median(env[RATE: 3 * RATE])
    seam = env[4 * RATE: 6 * RATE]
    dip_db = 20 * np.log10(seam.min() / steady)
    assert dip_db > -3.5, f"seam dips {dip_db:.2f} dB"


def test_emerge_seam_stays_equal_power():
    # A drift that is genuinely large (well past SEAM_DRIFT_BOUND_CYCLES)
    # keeps the emerge seam on equal-power, which is bounded only when the
    # two copies are uncorrelated -- this session's glide (10 -> 40 Hz
    # beat over a 120 s emerge, sampled through a 2 s overlap starting
    # from a completely different steady beat) drifts far enough (~0.5
    # cycles on the k=1 voice) that the seam is not coherent, and the
    # assertion is on the flags that decide the law, not a level bound
    # equal-power does not actually guarantee (see
    # test_emerge_seam_is_within_bound_on_a_near_locked_preset for the
    # near-locked case, where the two laws' bounds actually matter). A
    # smaller target_beat (e.g. 15, as in an earlier revision of this
    # test) drifts only ~0.08 cycles here -- inside the 0.15 bound, so it
    # would now flip coherent and no longer exercise this path.
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": 10, "overlap": 2, "groups": [
                {"name": "g", "beat": 10.0, "level_db": 0.0,
                 "carrier_base": 200.0, "pairs": 1, "harmonics": [1.0]}]},
        ],
        "emerge": {"duration": 120, "target_beat": 40.0},
    }
    session = load_session_dict(data)
    timeline = resolve(session)
    layers = timeline.layers
    assert layers[0].coherent_fade_out is False
    assert layers[1].coherent_fade_in is False


def test_emerge_seam_is_within_bound_on_a_near_locked_preset():
    # attention's and concentration's emerge glides drift only 0.10 cycles
    # across their 2 s overlap (the incoming glide barely moves off the
    # outgoing tone's beat before the window closes) -- inside
    # SEAM_DRIFT_BOUND_CYCLES (0.15), so these two bundled presets' emerge
    # seams are coherent and use the linear law. Before this bound was
    # raised from 0.05, both sat on the equal-power side and measured a
    # +2.95 dB swell with no compensating dip (RED, verified by flipping
    # SEAM_DRIFT_BOUND_CYCLES back to 0.05 locally: attention +2.95/+0.39,
    # concentration +2.95/+0.39 dB peak/dip against this same measurement).
    # Under the new rule both sides of the seam should stay close to flat.
    #
    # total_duration_s shrinks only the preset's "hold" segment (the fixed
    # entry and emerge segments keep their declared lengths, so the seam
    # itself and its drift are unaffected) -- this keeps the render under
    # a second instead of tens of seconds.
    #
    # The measurement excludes the pink bed (render_timeline with
    # pink_layers=()): beds never carry a coherence flag and always use
    # equal-power independently of this rule, so including them would mix
    # an unrelated crossfade into the number this test is about. The 0.5 s
    # (24000-sample) smoothing window is chosen, not arbitrary: attention's
    # own harmonic stack beats at 16/32/48 Hz, and a rectangular window of
    # exactly RATE/16 samples (or a multiple of it) has its first spectral
    # nulls exactly at 16 Hz and its multiples, killing that ripple instead
    # of leaking it into the excursion measurement (a shorter, e.g. 4800-
    # sample, window leaves several dB of beat ripple in the envelope that
    # has nothing to do with the seam law).
    win = 24000
    for name in ("attention", "concentration"):
        session = load_preset(name, total_duration_s=400.0)
        timeline = resolve(session)
        tone_only = replace(timeline, pink_layers=())
        layers = sorted(timeline.layers, key=lambda layer: layer.start_sample)
        out_layer = next(
            layer for layer in layers
            if layer.group.name == "hold" and layer.fade_out_samples > 0
        )
        assert out_layer.coherent_fade_out, f"{name}: expected a coherent seam"
        seam_start = (
            out_layer.start_sample + out_layer.n_samples
            - out_layer.fade_out_samples
        )
        seam_end = seam_start + out_layer.fade_out_samples

        audio = render_timeline(tone_only, seed=11)
        lo, hi = seam_start - 3 * RATE, seam_end + RATE
        chunk = audio[lo:hi, 1]
        env = np.sqrt(np.convolve(chunk ** 2, np.ones(win) / win, "valid"))
        offset = win // 2
        steady = np.median(env[
            seam_start - 2 * RATE - lo - offset: seam_start - RATE - lo - offset
        ])
        seam = env[seam_start - lo - offset: seam_end - lo - offset]
        excursion_db = 20 * np.abs(np.log10(seam / steady)).max()
        assert excursion_db < 1.2, (
            f"{name}: emerge seam excursion {excursion_db:.2f} dB"
        )


def test_bed_keeps_equal_power_alongside_a_coherent_tonal_seam():
    # A continuing constant tonal group is coherent (linear), but the bed
    # crossfading alongside it must never pick up a coherence flag — beds
    # are independent noise streams and always keep the equal-power law.
    # Decorrelated equal-power noise sums flat in power through a crossfade,
    # so the bed's own smoothed power should stay close to its steady level.
    # stereo: {mode: static} disables the default panning LFO (period 20 s
    # at the default 0.05 Hz), which would otherwise swing power between
    # channels over this test's timescale and swamp the seam measurement.
    bed_spec = {"level_db": -12.0,
                "stereo": {"mode": "static", "interaural_delay_us": 0.0}}
    data = {
        "name": "t", "title": "T", "fidelity": "measured-tape", "sample_rate": RATE,
        "segments": [
            {"duration": 6, "overlap": 2, "groups": [
                {"name": "g", "beat": 4.0, "level_db": 0.0,
                 "carrier_base": 200.0, "pairs": 1, "harmonics": [1.0]}],
             "bed": dict(bed_spec)},
            {"duration": 6, "groups": [
                {"name": "g", "beat": 4.0, "level_db": 0.0,
                 "carrier_base": 200.0, "pairs": 1, "harmonics": [1.0]}],
             "bed": dict(bed_spec)},
        ],
    }
    session = load_session_dict(data)
    timeline = resolve(session)
    # PinkLayer carries no coherence fields at all — beds never see them.
    for p in timeline.pink_layers:
        assert not hasattr(p, "coherent_fade_in")
        assert not hasattr(p, "coherent_fade_out")
    # The tonal group itself IS coherent here, confirming the bed's
    # equal-power behaviour isn't just an artifact of a non-coherent seam.
    tonal = timeline.layers
    assert tonal[0].coherent_fade_out is True
    assert tonal[1].coherent_fade_in is True

    # Isolate the bed's power by silencing the tone (-120 dB, inaudible)
    # rather than removing it, so the geometry (and hence the crossfade
    # windows) is identical to the session actually checked for coherence.
    data["segments"][0]["groups"][0]["level_db"] = -120.0
    data["segments"][1]["groups"][0]["level_db"] = -120.0
    bed_audio = render_session(load_session_dict(data), seed=7)
    power = np.convolve(
        bed_audio[:, 0] ** 2 + bed_audio[:, 1] ** 2, np.ones(4800) / 4800, "valid"
    )
    steady = np.median(power[RATE: 3 * RATE])
    seam = power[4 * RATE: 6 * RATE]
    # 2.0 dB, not tighter: a single noise seed's smoothed power wanders
    # ~1-1.2 dB on this timescale by itself (measured across seeds 1-7);
    # the failure this guards against -- beds wrongly picking up linear
    # fades -- measures 3.6 dB, comfortably past the bound either way.
    excursion_db = 10 * np.abs(np.log10(seam / steady)).max()
    assert excursion_db < 2.0, f"bed power excursion {excursion_db:.2f} dB"
