"""Block-boundary integrity of the blocked tonal renderer.

render_timeline renders tonal layers block-by-block (RENDER_BLOCK_SECONDS)
to bound the working set. Block size must be an implementation detail:
every oscillator carries its wrapped phase across block edges and every
other per-sample quantity (frequency ramps, fade windows, the rotation
LFO) is a pure function of the absolute sample index, so the audio may
not depend on where the boundaries fall — no click, no envelope step, no
LFO reset. The block lengths used here (7997 = 11*727, 13001 prime,
6001 prime) divide neither each other nor any segment boundary of the
fixture, so every block edge lands mid-signal, mid-fade, mid-glide.
"""

import numpy as np

from farfield.render import render_timeline
from farfield.session import load_session_dict
from farfield.timeline import resolve

RATE = 8000


def _session(with_gate: bool = True):
    """Two overlapping segments exercising every stateful per-sample path:
    gliding stack pairs (frequency ramps + phase accumulators), a tremolo
    glide (LFO accumulator), a rotating pairs group (absolute-index LFO),
    a crossfed pair (constant-offset copy), a SAM group (carrier and
    modulator accumulators), and optionally an isochronic gate (cycle
    accumulator with shaped edges); the 1 s overlap puts fade windows and
    seam handoffs across block boundaries too."""
    second_groups = [
        {"name": "stack", "beat": 7.0, "carrier_base": 200.0,
         "pairs": 2, "harmonics": [1.0, 0.3], "level_db": 0.0,
         "tremolo": {"rate_hz": 2.0, "depth": 0.4}},
        {"name": "xfeed", "level_db": -10.0,
         "pairs": [{"center": 120.0, "beat": 1.5}],
         "placement": {"crossfeed_db": -3.4, "crossfeed_phase_deg": 41.0}},
    ]
    if with_gate:
        second_groups.append(
            {"name": "gated", "level_db": -8.0,
             "pairs": [{"center": 250.0, "beat": 5.0}],
             "gate": {"rate_hz": 8.0, "depth": 1.0, "duty": 0.5,
                      "edge_ms": 10.0}})
    return load_session_dict({
        "name": "blocks", "title": "Blocks", "fidelity": "original",
        "sample_rate": RATE,
        "segments": [
            {"duration": 3.0, "overlap": 1.0, "groups": [
                {"name": "stack", "beat": {"from": 4.0, "to": 7.0},
                 "carrier_base": 200.0, "pairs": 2,
                 "harmonics": [1.0, 0.3], "level_db": 0.0,
                 "tremolo": {"rate_hz": {"from": 0.5, "to": 2.0},
                             "depth": 0.4}},
                {"name": "rot", "level_db": -6.0,
                 "pairs": [{"center": 300.0, "beat": 4.0}],
                 "rotation": {"period_s": 2.0, "depth": 0.8,
                              "phase_deg": 30.0}},
                {"name": "sam", "level_db": -3.0,
                 "sam": {"carrier_hz": 300.0, "rate_hz": 40.0,
                         "arc_deg": 120.0}},
            ]},
            {"duration": 3.0, "groups": second_groups},
        ],
    })


def _boundaries(timeline, block_samples):
    """Absolute sample indices where some layer starts a new block."""
    edges = set()
    for layer in timeline.layers:
        b = block_samples
        while b < layer.n_samples:
            edges.add(layer.start_sample + b)
            b += block_samples
    return sorted(edges)


def test_audio_is_invariant_to_block_size():
    # The strongest boundary proof: two coprime block lengths put their
    # edges at entirely different samples, so any state error at an edge
    # (a re-seeded phase, a restarted envelope) would make the two renders
    # disagree by far more than accumulated float rounding.
    timeline = resolve(_session())
    a = render_timeline(timeline, block_samples=7997)
    b = render_timeline(timeline, block_samples=13001)
    assert np.max(np.abs(a - b)) < 1e-9


def test_blocked_render_matches_a_single_block():
    # block_samples beyond the longest layer renders each layer in one
    # block — the mathematically unblocked path — so this pins the blocked
    # render to the whole-layer computation, not just to itself. The
    # tolerance is looser than the blocked-vs-blocked test's because the
    # single-block path accumulates phase UNWRAPPED across the whole layer
    # (thousands of radians), where float rounding per sample scales with
    # the phase magnitude; the disagreement is the unblocked path's own
    # rounding, at about -170 dBFS.
    timeline = resolve(_session())
    blocked = render_timeline(timeline, block_samples=6001)
    whole = render_timeline(timeline, block_samples=10**9)
    assert np.max(np.abs(blocked - whole)) < 3e-8


def test_no_click_at_block_boundaries():
    # A click is a discontinuity in value or slope, which the second
    # difference exposes: a boundary sample must not be a curvature
    # outlier against the rest of the signal. The gate is excluded here
    # because its raised-cosine edges are the signal's legitimate
    # curvature maxima and would slacken the comparison.
    timeline = resolve(_session(with_gate=False))
    block = 6001
    audio = render_timeline(timeline, block_samples=block)
    d2 = np.diff(audio, n=2, axis=0)
    near = np.zeros(len(d2), dtype=bool)
    for edge in _boundaries(timeline, block):
        lo = max(0, edge - 3)
        near[lo : min(len(d2), edge + 3)] = True
    assert near.any()
    smooth_max = np.abs(d2[~near]).max()
    boundary_max = np.abs(d2[near]).max()
    assert boundary_max <= smooth_max * 1.05


def test_handoff_is_block_size_invariant():
    # The cross-layer handoff (each oscillator's phase at n - fade_out)
    # must come out the same wherever the blocks fall, or a seam would
    # click only for some block sizes. The seam samples themselves are
    # covered by the invariance test; this isolates the handoff store.
    # Comparison is circular: a phase landing at the wrap point may be
    # reported as ~0 or ~2*pi (or ~1.0 cycles for the gate) depending on
    # rounding, and those are the same angle, not a disagreement.
    from farfield.render import _render_layer

    def circular_delta(a, b, modulus):
        d = abs(a - b) % modulus
        return min(d, modulus - d)

    timeline = resolve(_session())
    layer = timeline.layers[0]
    reference = None
    for block in (7997, 13001, 10**9):
        out = np.zeros((timeline.total_samples, 2))
        phases: dict = {}
        _render_layer(out, layer, RATE, phases, block)
        if reference is None:
            reference = phases
        else:
            assert phases.keys() == reference.keys()
            for key in reference:
                modulus = 1.0 if "gate" in key else 2.0 * np.pi
                assert circular_delta(
                    phases[key], reference[key], modulus
                ) < 1e-9, key
