"""The declarative session model and its YAML front end.

A session is an ordered list of segments; each segment layers one or more
signal groups at relative levels, optionally over pink noise. Layering
multiple groups per segment is what makes the patent protocols expressible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from farfield.waveshape import DEFAULT_HARMONICS

FIDELITY_TIERS = frozenset({"measured-tape", "measured-mss", "patent", "original"})
"""Provenance tiers, strongest first: measured from the original tapes,
measured from the MSS remasters, from the patents, original designs."""

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_PEAK_DBFS = -3.0
DEFAULT_EDGE_FADE_S = 0.030
DEFAULT_CARRIER_BASE = 200.0
DEFAULT_PAIRS = 3
DEFAULT_EMERGE_DURATION_S = 180.0
DEFAULT_EMERGE_TARGET_BEAT = 15.0
EMERGE_FADE_IN_S = 2.0
"""The emerge block crossfades in over this long, fading the previous
segment (and its pink noise) out across the same window."""
DEFAULT_COMB_SWEEP_HZ = 0.125
DEFAULT_PAN_RATE_HZ = 0.05
DEFAULT_PINK_ALGORITHM = "fft"

HOLD = "hold"


@dataclass(frozen=True)
class Glide:
    start: float
    end: float


@dataclass(frozen=True)
class PairSpec:
    kind: str  # "center" | "explicit" | "mono"
    center: float | None = None
    beat: float | Glide | None = None
    left: float | None = None
    right: float | None = None
    mono: float | None = None


@dataclass(frozen=True)
class Tremolo:
    rate_hz: float | Glide
    depth: float


@dataclass(frozen=True)
class TremoloSplit:
    left: Tremolo
    right: Tremolo


@dataclass(frozen=True)
class Gate:
    """A hard on/off envelope with ramped edges — an isochronic tone.

    Distinct from Tremolo, which is sinusoidal. An isochronic stimulus is
    a train of discrete pulses: the carrier is at full level for `duty` of
    each cycle and silent for the rest, with a raised-cosine ramp of
    `edge_ms` on each edge so the switch does not splatter broadband energy
    across the spectrum. `depth` scales how far the "off" state drops:
    1.0 is true silence, 0.5 dips to half amplitude.
    """

    rate_hz: float | Glide
    depth: float
    duty: float
    edge_ms: float


@dataclass(frozen=True)
class Rotation:
    period_s: float
    depth: float
    phase_deg: float


@dataclass(frozen=True)
class Placement:
    crossfeed_db: float
    crossfeed_phase_deg: float


@dataclass(frozen=True)
class BedShape:
    peak_hz: float
    rise_db_per_decade: float
    fall_db_per_decade: float


@dataclass(frozen=True)
class TextureSpec:
    band_hz: tuple[float, float]
    level_db: float
    pan_period_s: float
    pan_ild_amplitude_db: float
    pan_phase_deg: float
    # Common-mode swell, same shape as PinkSpec's. The texture needs its
    # own because it, not the bed, is what carries 1.5-8 kHz: measured on
    # a rendered MSS preset the texture is 62% of 1500-3000 Hz and 92% of
    # 4500-6000 Hz, so a swell applied only to the bed cannot reach the
    # bands the gesture is heard in.
    surf_rate_hz: float | None = None
    surf_depth: float = 0.0
    surf_phase_deg: float = 0.0


SAM_CARRIER_CEILING_HZ = 700.0
"""Highest carrier a SAM group may use.

Not the same quantity as beats.CARRIER_CEILING_HZ (1500 Hz), which bounds
binaural-beat FUSION. SAM localises by interaural phase alone, and an
interaural phase difference is only an unambiguous azimuth while half a
carrier wavelength exceeds the head's ear-to-ear path difference. Above
roughly 700 Hz one phase difference maps to more than one direction and
the phantom image splits or jumps sides instead of sweeping. The patents'
own worked carriers, 300 and 440 Hz, sit under it.
"""

SAM_HEAD_SEPARATION_M = 0.175
"""Ear-to-ear acoustic path used to convert an arc angle into phase.

Midpoint of the patents' stated 15-25 cm; with c below it gives the
maximum physical ITD of 510 us that the research doc derives.
"""

SAM_SPEED_OF_SOUND_MS = 343.0

SAM_PATHS = ("closed", "open", "discontinuous")

SAM_MAX_DEPTH_RAD = math.pi / 2.0
"""Beyond a quarter carrier period of one-way deviation the swept ITD runs
past half a carrier period and the phantom image folds back rather than
widening, so this bounds depth_rad."""


def sam_depth_from_arc(arc_deg: float, carrier_hz: float) -> float:
    """Peak phase deviation phi_p for a swept arc of ``arc_deg`` total.

    Spherical-head ITD model: a source at azimuth theta carries an
    interaural delay of (d/c)*sin(theta), so an arc spanning +/-arc_deg/2
    reaches a peak delay of (d/c)*sin(arc_deg/2) and hence a peak one-way
    carrier phase deviation of pi*f_s*(d/c)*sin(arc_deg/2).

    Sanity check against the research doc, which derives phi_p ~= 0.48 rad
    for a full ear-to-ear arc at 300 Hz independently of this function:
    sam_depth_from_arc(180.0, 300.0) == 0.4807...
    """
    return (
        math.pi
        * carrier_hz
        * (SAM_HEAD_SEPARATION_M / SAM_SPEED_OF_SOUND_MS)
        * math.sin(math.radians(arc_deg) / 2.0)
    )


@dataclass(frozen=True)
class SamSpec:
    """Spatial Angle Modulation on one shared carrier.

    S_L = A*sin(Phi_c + phi_p*m(Phi_m) + phi_L)
    S_R = A*sin(Phi_c - phi_p*m(Phi_m) + phi_R)

    with m the unit-amplitude modulator selected by ``path``, evaluated on
    the modulator's own accumulated phase so every path type is 2*pi
    periodic and survives the seam handoff's phase wrap.
    """

    carrier_hz: float
    rate_hz: float
    depth_rad: float
    path: str = "closed"
    steps: int = 8
    offset_left_deg: float = 0.0
    offset_right_deg: float = 0.0
    arc_deg: float | None = None  # reporting only; None when depth was given

    def peak_itd_s(self) -> float:
        """Interaural time difference at a sweep extreme, seconds."""
        return self.depth_rad / (math.pi * self.carrier_hz)


@dataclass(frozen=True)
class Group:
    name: str
    beat: float | Glide
    carrier_base: float
    pairs: int
    harmonics: tuple[float, ...]
    level_db: float
    pairs_spec: tuple[PairSpec, ...] | None = None
    high_ear: str = "right"
    tremolo: Tremolo | TremoloSplit | None = None
    gate: Gate | None = None
    rotation: Rotation | None = None
    placement: Placement | None = None
    sam: SamSpec | None = None

    def beat_bounds(self) -> tuple[float, float]:
        if self.sam is not None:
            # A SAM group has no binaural beat: its instantaneous frequency
            # difference is 2*phi_p*f_m*cos(2*pi*f_m*t), mean zero. The
            # perceived beat rate is the modulation rate, and that is the
            # honest thing to report on the beat axis (sidecar, describe,
            # the visualizer's beat meter).
            return self.sam.rate_hz, self.sam.rate_hz
        if self.pairs_spec is not None:
            for pair in self.pairs_spec:
                if pair.kind == "center":
                    if isinstance(pair.beat, Glide):
                        return pair.beat.start, pair.beat.end
                    return float(pair.beat), float(pair.beat)
                if pair.kind == "explicit":
                    beat = abs(pair.left - pair.right)
                    return beat, beat
            return 0.0, 0.0
        if isinstance(self.beat, Glide):
            return self.beat.start, self.beat.end
        return float(self.beat), float(self.beat)


@dataclass(frozen=True)
class PinkSpec:
    level_db: float
    comb_sweep_hz: float
    pan_rate_hz: float
    algorithm: str
    color: str = "pink"
    slope_db_per_decade: float | None = None
    shape: BedShape | None = None
    surf_rate_hz: float | None = None
    surf_depth: float = 0.0
    surf_phase_deg: float = 0.0
    stereo_mode: str = "pan"
    lfo_period_s: float | None = None
    stereo_depth_db: float | None = None
    interaural_delay_us: float | None = None
    comb_enabled: bool = True

    def resolved_slope(self) -> float:
        """Return the effective slope: override if set, else color default."""
        if self.slope_db_per_decade is not None:
            return self.slope_db_per_decade
        return {"pink": -10.0, "brown": -20.0}[self.color]


@dataclass(frozen=True)
class Segment:
    duration_s: float | None
    overlap_s: float
    groups: tuple[Group, ...]
    pink: PinkSpec | None
    texture: TextureSpec | None = None


@dataclass(frozen=True)
class Emerge:
    duration_s: float
    target_beat: float


@dataclass(frozen=True)
class Session:
    name: str
    title: str
    fidelity: str
    notes: str
    sample_rate: int
    peak_dbfs: float
    segments: tuple[Segment, ...]
    emerge: Emerge | None
    edge_fade_s: float = DEFAULT_EDGE_FADE_S


def parse_duration(value: str | int | float) -> float:
    """Seconds from a number, ``M:SS``, or ``H:MM:SS``."""
    if isinstance(value, bool):
        raise ValueError(f"invalid duration {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        parts = value.strip().split(":")
        if not all(part.strip().replace(".", "", 1).isdigit() for part in parts):
            raise ValueError(f"invalid duration {value!r}")
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            seconds = float(parts[0]) * 60.0 + float(parts[1])
        elif len(parts) == 3:
            seconds = (
                float(parts[0]) * 3600.0
                + float(parts[1]) * 60.0
                + float(parts[2])
            )
        else:
            raise ValueError(f"invalid duration {value!r}")
    else:
        raise ValueError(f"invalid duration {value!r}")
    if seconds < 0.0:
        raise ValueError(f"duration must not be negative: {value!r}")
    return seconds


def _parse_beat(raw: Any) -> float | Glide:
    if isinstance(raw, dict):
        try:
            return Glide(float(raw["from"]), float(raw["to"]))
        except KeyError as exc:
            raise ValueError(
                "a gliding beat needs both 'from' and 'to'"
            ) from exc
    return float(raw)


def _parse_rate(raw: Any) -> float | Glide:
    if isinstance(raw, dict):
        rate = Glide(float(raw["from"]), float(raw["to"]))
        if rate.start <= 0.0 or rate.end <= 0.0:
            raise ValueError("tremolo rate must be positive")
        return rate
    rate = float(raw)
    if rate <= 0.0:
        raise ValueError("tremolo rate must be positive")
    return rate


def _parse_tremolo(raw: Any) -> Tremolo | TremoloSplit | None:
    if raw is None:
        return None
    has_single = "rate_hz" in raw or "depth" in raw
    has_split = "left" in raw or "right" in raw
    if has_single and has_split:
        raise ValueError(
            "tremolo must use either the single form {rate_hz, depth} or "
            "the split form {left, right}, not both"
        )
    if has_split:
        if "left" not in raw or "right" not in raw:
            raise ValueError("a split tremolo needs both 'left' and 'right'")
        return TremoloSplit(
            left=_parse_tremolo(raw["left"]),
            right=_parse_tremolo(raw["right"]),
        )
    depth = float(raw["depth"])
    # The upper bound is inclusive: 1.0 is 100% amplitude modulation, the
    # textbook ASSR calibration stimulus, and nothing physical forbids it.
    # No bundled preset sits at the boundary, so widening the old [0, 1)
    # bound changed no existing render.
    if not 0.0 <= depth <= 1.0:
        raise ValueError("tremolo depth must be in [0, 1]")
    return Tremolo(rate_hz=_parse_rate(raw["rate_hz"]), depth=depth)


MAX_GATE_EDGE_MS = 100.0


def _parse_gate(raw: Any) -> Gate | None:
    """Parse an isochronic gate. Absent means absent — see render._render_layer.

    Kept deliberately separate from tremolo rather than folded in as a
    "shape" option: the two have different parameters (a gate has duty and
    edges, a tremolo has neither) and a group may legitimately carry both,
    a slow tremolo swell over a fast pulse train.
    """
    if raw is None:
        return None
    for key in ("rate_hz", "depth", "duty"):
        if key not in raw:
            raise ValueError(f"gate needs {key!r}")

    rate = _parse_rate(raw["rate_hz"])  # shares tremolo's positivity check

    depth = float(raw["depth"])
    if not 0.0 <= depth <= 1.0:
        raise ValueError("gate depth must be in [0, 1]")

    duty = float(raw["duty"])
    if not 0.0 < duty < 1.0:
        raise ValueError("gate duty must be in (0, 1)")

    edge_ms = float(raw.get("edge_ms", 5.0))
    if not 0.0 <= edge_ms <= MAX_GATE_EDGE_MS:
        raise ValueError(
            f"gate edge_ms must be in [0, {MAX_GATE_EDGE_MS:.0f}], got {edge_ms}"
        )

    # Both ramps have to fit inside the on-time, at the FASTEST rate the gate
    # ever reaches: a glide that starts legal and ends illegal must be caught
    # here, not silently clipped in the renderer.
    fastest = rate.start if isinstance(rate, Glide) else rate
    if isinstance(rate, Glide):
        fastest = max(rate.start, rate.end)
    edge_fraction = edge_ms * fastest / 1000.0
    if 2.0 * edge_fraction > duty:
        raise ValueError(
            f"gate edges do not fit: two {edge_ms:g} ms ramps are "
            f"{2.0 * edge_fraction:.3f} of a cycle at {fastest:g} Hz, "
            f"but duty is only {duty:g}"
        )
    return Gate(rate_hz=rate, depth=depth, duty=duty, edge_ms=edge_ms)


def _parse_placement(raw: Any) -> Placement | None:
    if raw is None:
        return None
    if "crossfeed_db" not in raw:
        raise ValueError("placement needs 'crossfeed_db'")
    if "crossfeed_phase_deg" not in raw:
        raise ValueError("placement needs 'crossfeed_phase_deg'")
    crossfeed_db = float(raw["crossfeed_db"])
    if not -60.0 <= crossfeed_db < 0.0:
        raise ValueError(
            f"placement crossfeed_db must be in [-60, 0), got {crossfeed_db}"
        )
    crossfeed_phase_deg = float(raw["crossfeed_phase_deg"])
    if not -180.0 <= crossfeed_phase_deg <= 180.0:
        raise ValueError(
            "placement crossfeed_phase_deg must be in [-180, 180], "
            f"got {crossfeed_phase_deg}"
        )
    return Placement(
        crossfeed_db=crossfeed_db, crossfeed_phase_deg=crossfeed_phase_deg
    )


def _parse_rotation(raw: Any) -> Rotation | None:
    if raw is None:
        return None
    for key in ("period_s", "depth", "phase_deg"):
        if key not in raw:
            raise ValueError(f"rotation needs {key!r}")
    period_s = float(raw["period_s"])
    if period_s <= 0.0:
        raise ValueError(f"rotation period_s must be positive, got {period_s}")
    depth = float(raw["depth"])
    # Ceiling 0.99, not 0.95: the measured MSS rotation needs 0.98 to reach
    # the tape's own ILD swing under the equal-power law (see the rotation
    # note in docs/tape-analysis/mss-results.json). 1.0 would null one ear
    # outright, so the bound stays below it.
    if not 0.0 < depth <= 0.99:
        raise ValueError(f"rotation depth must be in (0, 0.99], got {depth}")
    phase_deg = float(raw["phase_deg"])
    if not math.isfinite(phase_deg):
        raise ValueError(f"rotation phase_deg must be finite, got {phase_deg}")
    return Rotation(period_s=period_s, depth=depth, phase_deg=phase_deg)


def _parse_sam(raw: Any) -> SamSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"sam must be a mapping, got {type(raw).__name__}")

    known = {
        "carrier_hz", "rate_hz", "depth_rad", "arc_deg", "path", "steps",
        "offset_deg",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"sam has unknown key(s) {unknown!r}; expected some of "
            f"{sorted(known)}"
        )

    for key in ("carrier_hz", "rate_hz"):
        if key not in raw:
            raise ValueError(f"sam needs {key!r}")

    carrier_hz = float(raw["carrier_hz"])
    if not 20.0 <= carrier_hz <= SAM_CARRIER_CEILING_HZ:
        raise ValueError(
            f"sam carrier_hz must be in [20, {SAM_CARRIER_CEILING_HZ:.0f}], "
            f"got {carrier_hz}: SAM localises by interaural phase alone, "
            "and an interaural phase difference only maps to one azimuth "
            f"below about {SAM_CARRIER_CEILING_HZ:.0f} Hz - above it the "
            "phantom image splits or jumps sides instead of sweeping "
            "(the patents' own examples are 300 and 440 Hz)"
        )

    rate_hz = float(raw["rate_hz"])
    if rate_hz <= 0.0:
        raise ValueError(f"sam rate_hz must be positive, got {rate_hz}")
    if rate_hz >= carrier_hz / 2.0:
        raise ValueError(
            f"sam rate_hz must stay below half the carrier "
            f"({carrier_hz / 2.0:.1f} Hz), got {rate_hz}: above that the "
            "modulation sidebands fold over the carrier and there is no "
            "coherent phantom position left to sweep"
        )

    has_arc = "arc_deg" in raw
    has_depth = "depth_rad" in raw
    if has_arc == has_depth:
        raise ValueError(
            "sam needs exactly one of 'arc_deg' or 'depth_rad' "
            "(arc_deg is the swept arc's full included angle; depth_rad is "
            "the patents' phi_p directly)"
        )
    arc_deg = None
    if has_arc:
        arc_deg = float(raw["arc_deg"])
        if not 0.0 < arc_deg <= 180.0:
            raise ValueError(
                f"sam arc_deg must be in (0, 180], got {arc_deg}"
            )
        # No fold check here: the widest arc at the highest permitted
        # carrier is pi*700*(0.175/343) = 1.122 rad, inside
        # SAM_MAX_DEPTH_RAD. An arc angle cannot reach the fold while the
        # carrier ceiling holds, so a check would be dead code. depth_rad,
        # which is stated directly rather than derived from anatomy, can
        # reach it and is bounded below.
        depth_rad = sam_depth_from_arc(arc_deg, carrier_hz)
    else:
        depth_rad = float(raw["depth_rad"])
        if not 0.0 < depth_rad <= SAM_MAX_DEPTH_RAD:
            raise ValueError(
                f"sam depth_rad must be in (0, {SAM_MAX_DEPTH_RAD:.6f}], "
                f"got {depth_rad}"
            )

    path = str(raw.get("path", "closed"))
    if path not in SAM_PATHS:
        raise ValueError(
            f"unknown sam path {path!r}; expected one of {list(SAM_PATHS)}"
        )

    steps = 8
    if "steps" in raw:
        if path != "discontinuous":
            raise ValueError(
                f"sam steps only applies to path 'discontinuous', not {path!r}"
            )
        raw_steps = raw["steps"]
        if isinstance(raw_steps, bool) or not isinstance(raw_steps, int):
            raise ValueError(f"sam steps must be an integer, got {raw_steps!r}")
        steps = int(raw_steps)
        if steps < 2:
            raise ValueError(f"sam steps must be at least 2, got {steps}")

    offset_left_deg = 0.0
    offset_right_deg = 0.0
    if "offset_deg" in raw:
        offsets = raw["offset_deg"]
        if not isinstance(offsets, dict):
            raise ValueError(
                "sam offset_deg must be a mapping with 'left' and 'right', "
                f"got {type(offsets).__name__}"
            )
        extra = sorted(set(offsets) - {"left", "right"})
        if extra:
            raise ValueError(
                f"sam offset_deg has unknown key(s) {extra!r}; expected "
                "'left' and 'right'"
            )
        for key in ("left", "right"):
            if key not in offsets:
                raise ValueError(f"sam offset_deg needs {key!r}")
        offset_left_deg = float(offsets["left"])
        offset_right_deg = float(offsets["right"])
        for key, value in (
            ("left", offset_left_deg), ("right", offset_right_deg)
        ):
            if not -180.0 <= value <= 180.0:
                raise ValueError(
                    f"sam offset_deg {key!r} must be in [-180, 180], "
                    f"got {value}"
                )

    return SamSpec(
        carrier_hz=carrier_hz,
        rate_hz=rate_hz,
        depth_rad=depth_rad,
        path=path,
        steps=steps,
        offset_left_deg=offset_left_deg,
        offset_right_deg=offset_right_deg,
        arc_deg=arc_deg,
    )


def _parse_texture(raw: Any, has_bed: bool) -> TextureSpec | None:
    if raw is None:
        return None
    if not has_bed:
        raise ValueError("texture requires an accompanying 'bed' on the same segment")
    for key in ("band_hz", "level_db", "pan"):
        if key not in raw:
            raise ValueError(f"texture needs {key!r}")
    band_hz = raw["band_hz"]
    if len(band_hz) != 2:
        raise ValueError("texture band_hz must have exactly two values")
    lo, hi = float(band_hz[0]), float(band_hz[1])
    if not lo < hi:
        raise ValueError(f"texture band_hz must be ascending, got {band_hz!r}")
    if not (20.0 < lo < 20000.0 and 20.0 < hi < 20000.0):
        raise ValueError(
            f"texture band_hz values must be in (20, 20000), got {band_hz!r}"
        )
    level_db = float(raw["level_db"])
    pan = raw["pan"]
    for key in ("period_s", "ild_amplitude_db", "phase_deg"):
        if key not in pan:
            raise ValueError(f"texture pan needs {key!r}")
    period_s = float(pan["period_s"])
    if period_s <= 0.0:
        raise ValueError(f"texture pan period_s must be positive, got {period_s}")
    # Named ild_amplitude_db, not depth_db: a bed's stereo.depth_db is a
    # PEAK-TO-PEAK ILD while this is the sinusoid's AMPLITUDE, and the two
    # would silently differ by 2x under one name.
    ild_amplitude_db = float(pan["ild_amplitude_db"])
    if not 0.0 < ild_amplitude_db <= 30.0:
        raise ValueError(
            f"texture pan ild_amplitude_db must be in (0, 30], got {ild_amplitude_db}"
        )
    phase_deg = float(pan["phase_deg"])
    if not math.isfinite(phase_deg):
        raise ValueError(f"texture pan phase_deg must be finite, got {phase_deg}")
    surf_rate_hz = None
    surf_depth = 0.0
    surf_phase_deg = 0.0
    if "surf" in raw:
        surf = raw["surf"]
        surf_rate_hz = float(surf["rate_hz"])
        surf_depth = float(surf["depth"])
        surf_phase_deg = float(surf.get("phase_deg", 0.0))
        if surf_rate_hz <= 0.0:
            raise ValueError(
                f"texture surf rate_hz must be positive, got {surf_rate_hz}"
            )
        if not 0.0 <= surf_depth < 1.0:
            raise ValueError(
                f"texture surf depth must be in [0, 1), got {surf_depth}"
            )
        if not math.isfinite(surf_phase_deg):
            raise ValueError(
                f"texture surf phase_deg must be finite, got {surf_phase_deg}"
            )
    return TextureSpec(
        band_hz=(lo, hi),
        level_db=level_db,
        pan_period_s=period_s,
        pan_ild_amplitude_db=ild_amplitude_db,
        pan_phase_deg=phase_deg,
        surf_rate_hz=surf_rate_hz,
        surf_depth=surf_depth,
        surf_phase_deg=surf_phase_deg,
    )


def _parse_bed_shape(raw: Any) -> BedShape | None:
    if raw is None:
        return None
    for key in ("peak_hz", "rise_db_per_decade", "fall_db_per_decade"):
        if key not in raw:
            raise ValueError(f"shape needs {key!r}")
    peak_hz = float(raw["peak_hz"])
    if not 30.0 <= peak_hz <= 1000.0:
        raise ValueError(f"shape peak_hz must be in [30, 1000], got {peak_hz}")
    rise_db_per_decade = float(raw["rise_db_per_decade"])
    if not 0.0 < rise_db_per_decade <= 24.0:
        raise ValueError(
            f"shape rise_db_per_decade must be in (0, 24], got {rise_db_per_decade}"
        )
    fall_db_per_decade = float(raw["fall_db_per_decade"])
    if not -48.0 <= fall_db_per_decade < 0.0:
        raise ValueError(
            f"shape fall_db_per_decade must be in [-48, 0), got {fall_db_per_decade}"
        )
    return BedShape(
        peak_hz=peak_hz,
        rise_db_per_decade=rise_db_per_decade,
        fall_db_per_decade=fall_db_per_decade,
    )


def _parse_pair(raw: dict) -> PairSpec:
    has_center = "center" in raw
    has_explicit = "left" in raw or "right" in raw
    has_mono = "mono" in raw
    if sum([has_center, has_explicit, has_mono]) != 1:
        raise ValueError(
            "a pair must use exactly one form: {center, beat}, "
            "{left, right}, or {mono}"
        )
    if has_center:
        if "beat" not in raw:
            raise ValueError("a center pair needs a beat")
        return PairSpec(kind="center", center=float(raw["center"]),
                        beat=_parse_beat(raw["beat"]))
    if has_explicit:
        if "beat" in raw:
            raise ValueError("an explicit pair implies its beat; drop the beat key")
        if "left" not in raw:
            raise ValueError("an explicit pair needs left")
        if "right" not in raw:
            raise ValueError("an explicit pair needs right")
        return PairSpec(kind="explicit", left=float(raw["left"]),
                        right=float(raw["right"]))
    if "beat" in raw:
        raise ValueError("a mono pair has no beat")
    return PairSpec(kind="mono", mono=float(raw["mono"]))


def _parse_high_ear(raw: dict, defaults: dict) -> str:
    high_ear = str(raw.get("high_ear", defaults.get("high_ear", "right")))
    if high_ear not in ("left", "right"):
        raise ValueError(
            f"high_ear must be 'left' or 'right', got {high_ear!r}"
        )
    return high_ear


def _parse_group(raw: dict, defaults: dict) -> Group:
    if "name" not in raw:
        raise ValueError("every group needs a name")

    sam = _parse_sam(raw.get("sam"))
    if sam is not None:
        # sam: is a third group FORM, alternative to the stack and pairs
        # forms rather than an addition to either: it emits exactly one
        # carrier, whose two ears are the modulation itself, so every key
        # that describes a set of paired tones is meaningless here and is
        # rejected rather than silently ignored.
        for key in ("beat", "pairs", "carrier_base", "harmonics", "high_ear"):
            if key in raw:
                raise ValueError(
                    f"group {raw['name']!r} uses 'sam', so {key!r} does not "
                    "apply: a SAM group is a single shared carrier, not a "
                    "set of binaural pairs"
                )
        if "rotation" in raw:
            raise ValueError(
                f"group {raw['name']!r} cannot combine 'sam' and 'rotation': "
                "rotation is an amplitude (ILD) pan while SAM is an "
                "interaural-phase-only phantom, and superimposing them makes "
                "the two position cues disagree"
            )
        if "placement" in raw:
            raise ValueError(
                f"group {raw['name']!r} cannot combine 'sam' and 'placement': "
                "crossfeed rewrites the interaural phase relationship that "
                "SAM exists to synthesise"
            )
        return Group(
            name=str(raw["name"]),
            beat=0.0,
            carrier_base=DEFAULT_CARRIER_BASE,
            pairs=DEFAULT_PAIRS,
            harmonics=(1.0,),
            level_db=float(raw.get("level_db", defaults.get("level_db", 0.0))),
            tremolo=_parse_tremolo(raw.get("tremolo")),
            pairs_spec=None,
            sam=sam,
        )

    harmonics = raw.get(
        "harmonics", defaults.get("harmonics", DEFAULT_HARMONICS)
    )
    high_ear = _parse_high_ear(raw, defaults)
    tremolo = _parse_tremolo(raw.get("tremolo"))
    gate = _parse_gate(raw.get("gate"))
    rotation = _parse_rotation(raw.get("rotation"))
    placement_raw = raw.get("placement")
    if rotation is not None and placement_raw is not None:
        # No measurement models the combination, and the two compose
        # destructively as written: a crossfeed copy routed to the opposite
        # ear picks up the opposite rotation gain and counter-rotates
        # against its own parent, collapsing the pair's ILD swing. In the
        # measured MSS material they never co-occur (the delta pair has
        # crossfeed and no rotation; theta and above rotate and have no
        # crossfeed), so this is rejected rather than guessed at.
        raise ValueError(
            "a group cannot combine 'rotation' and 'placement': no measured "
            "material uses both and the two do not compose"
        )
    placement = _parse_placement(raw.get("placement"))
    common = dict(
        name=str(raw["name"]),
        harmonics=tuple(float(h) for h in harmonics),
        level_db=float(raw.get("level_db", defaults.get("level_db", 0.0))),
        high_ear=high_ear,
        tremolo=tremolo,
        gate=gate,
        rotation=rotation,
        placement=placement,
    )

    raw_pairs = raw.get("pairs", defaults.get("pairs", DEFAULT_PAIRS))
    if isinstance(raw_pairs, list):
        if "beat" in raw:
            raise ValueError(
                "use either the stack form or a pairs list, not both "
                "(a pairs-form group must not set a group-level beat)"
            )
        if "carrier_base" in raw:
            raise ValueError(
                "use either the stack form or a pairs list, not both "
                "('carrier_base' is a stack-form setting)"
            )
        if not raw_pairs:
            raise ValueError(f"group {raw['name']!r} has an empty pairs list")
        pairs_spec = tuple(_parse_pair(p) for p in raw_pairs)
        if placement is not None and any(p.kind == "mono" for p in pairs_spec):
            raise ValueError(
                f"group {raw['name']!r} cannot use placement crossfeed on a "
                "mono pair"
            )
        return Group(
            beat=0.0,
            carrier_base=DEFAULT_CARRIER_BASE,
            pairs=DEFAULT_PAIRS,
            pairs_spec=pairs_spec,
            **common,
        )

    if not isinstance(raw_pairs, int):
        raise ValueError(
            f"group {raw['name']!r} has an invalid 'pairs' value: {raw_pairs!r}"
        )
    if "beat" not in raw:
        raise ValueError(f"group {raw['name']!r} needs a beat")
    return Group(
        beat=_parse_beat(raw["beat"]),
        carrier_base=float(
            raw.get(
                "carrier_base",
                defaults.get("carrier_base", DEFAULT_CARRIER_BASE),
            )
        ),
        pairs=int(raw_pairs),
        pairs_spec=None,
        **common,
    )


def _parse_pink(raw: dict | None, is_alias: bool = False) -> PinkSpec | None:
    if raw is None:
        return None

    # The pink: alias path rejects the new keys
    if is_alias:
        forbidden = {"color", "surf", "slope_db_per_decade", "stereo"}
        if forbidden & set(raw.keys()):
            raise ValueError(
                f"pink alias does not support color/surf/slope_db_per_decade/stereo; "
                f"use 'bed' instead"
            )

    # Parse color with validation
    color = str(raw.get("color", "pink"))
    if color not in ("pink", "brown"):
        raise ValueError(f"unknown bed color {color!r}; expected 'pink' or 'brown'")

    # Parse algorithm with validation. Checking at load keeps a typo from
    # surviving `describe` and only surfacing minutes into a render; the
    # render-time guard in noise.render_bed stays as a belt-and-braces check.
    algorithm = str(raw.get("algorithm", DEFAULT_PINK_ALGORITHM))
    if algorithm not in ("fft", "lfsr"):
        raise ValueError(
            f"unknown bed algorithm {algorithm!r}; expected 'fft' or 'lfsr'"
        )
    if algorithm == "lfsr" and color != "pink":
        raise ValueError(
            f"the lfsr algorithm only generates pink noise; color {color!r} "
            "requires the fft algorithm"
        )

    # Parse slope_db_per_decade with validation
    slope = None
    if "slope_db_per_decade" in raw:
        slope = float(raw["slope_db_per_decade"])
        if not -30.0 <= slope <= 0.0:
            raise ValueError(
                f"slope_db_per_decade must be in [-30, 0], got {slope}"
            )

    # Parse shape with validation; shape and slope_db_per_decade are mutually
    # exclusive so there's only one authority for the bed's spectral curve.
    shape = _parse_bed_shape(raw.get("shape"))
    if shape is not None and slope is not None:
        raise ValueError(
            "bed cannot set both 'shape' and 'slope_db_per_decade'; "
            "shape supersedes the single slope"
        )

    # Parse surf with validation
    surf_rate_hz = None
    surf_depth = 0.0
    surf_phase_deg = 0.0
    if "surf" in raw:
        surf = raw["surf"]
        surf_rate_hz = float(surf["rate_hz"])
        surf_depth = float(surf["depth"])
        surf_phase_deg = float(surf.get("phase_deg", 0.0))
        if surf_rate_hz <= 0.0:
            raise ValueError(
                f"surf rate_hz must be positive, got {surf_rate_hz}"
            )
        if not 0.0 <= surf_depth < 1.0:
            raise ValueError(
                f"surf depth must be in [0, 1), got {surf_depth}"
            )
        if not math.isfinite(surf_phase_deg):
            raise ValueError(
                f"surf phase_deg must be finite, got {surf_phase_deg}"
            )

    # Parse stereo with validation
    stereo_mode = "pan"
    lfo_period_s = None
    stereo_depth_db = None
    interaural_delay_us = None
    comb_enabled = True

    if "stereo" in raw:
        stereo_raw = raw["stereo"]
        # Handle string form (just "pan")
        if isinstance(stereo_raw, str):
            if stereo_raw != "pan":
                raise ValueError(
                    f"unknown stereo {stereo_raw!r}; expected 'pan' or a mapping"
                )
            stereo_mode = "pan"
        # Handle mapping form
        elif isinstance(stereo_raw, dict):
            if "mode" not in stereo_raw:
                raise ValueError("stereo mapping must have 'mode' key")

            mode = str(stereo_raw["mode"])
            if mode not in ("pan", "crossfade", "static"):
                raise ValueError(
                    f"unknown stereo mode {mode!r}; "
                    "expected 'pan', 'crossfade', or 'static'"
                )

            stereo_mode = mode

            # Validate mode-specific parameters
            if mode == "pan":
                # pan mode rejects lfo_period_s, depth_db and interaural_delay_us
                if (
                    "lfo_period_s" in stereo_raw
                    or "depth_db" in stereo_raw
                    or "interaural_delay_us" in stereo_raw
                ):
                    raise ValueError(
                        "mode 'pan' does not support "
                        "lfo_period_s/depth_db/interaural_delay_us"
                    )
                # pan keeps the comb_enabled default initialized above

            elif mode == "crossfade":
                # crossfade mode requires lfo_period_s and depth_db
                if "interaural_delay_us" in stereo_raw:
                    raise ValueError(
                        "mode 'crossfade' does not support interaural_delay_us"
                    )

                if "lfo_period_s" not in stereo_raw:
                    raise ValueError("crossfade mode requires 'lfo_period_s'")
                if "depth_db" not in stereo_raw:
                    raise ValueError("crossfade mode requires 'depth_db'")

                lfo_period_s = float(stereo_raw["lfo_period_s"])
                if lfo_period_s <= 0.0:
                    raise ValueError(
                        f"lfo_period_s must be positive, got {lfo_period_s}"
                    )

                stereo_depth_db = float(stereo_raw["depth_db"])
                if not 0.0 < stereo_depth_db <= 12.0:
                    raise ValueError(
                        f"depth_db must be in (0, 12], got {stereo_depth_db}"
                    )

                comb_enabled = "comb_sweep_hz" in raw

            elif mode == "static":
                # static mode requires interaural_delay_us
                if "lfo_period_s" in stereo_raw or "depth_db" in stereo_raw:
                    raise ValueError(
                        "mode 'static' does not support lfo_period_s/depth_db"
                    )

                if "interaural_delay_us" not in stereo_raw:
                    raise ValueError("static mode requires 'interaural_delay_us'")

                interaural_delay_us = float(stereo_raw["interaural_delay_us"])
                if not -1000.0 <= interaural_delay_us <= 1000.0:
                    raise ValueError(
                        "interaural_delay_us must be in [-1000, 1000], "
                        f"got {interaural_delay_us}"
                    )

                comb_enabled = "comb_sweep_hz" in raw
        else:
            raise ValueError(
                "stereo must be a string or mapping, "
                f"got {type(stereo_raw).__name__}"
            )
    # With no "stereo" key at all, every field keeps its initialization above.

    return PinkSpec(
        level_db=float(raw["level_db"]),
        comb_sweep_hz=float(raw.get("comb_sweep_hz", DEFAULT_COMB_SWEEP_HZ)),
        pan_rate_hz=float(raw.get("pan_rate_hz", DEFAULT_PAN_RATE_HZ)),
        algorithm=algorithm,
        color=color,
        slope_db_per_decade=slope,
        shape=shape,
        surf_rate_hz=surf_rate_hz,
        surf_depth=surf_depth,
        surf_phase_deg=surf_phase_deg,
        stereo_mode=stereo_mode,
        lfo_period_s=lfo_period_s,
        stereo_depth_db=stereo_depth_db,
        interaural_delay_us=interaural_delay_us,
        comb_enabled=comb_enabled,
    )


def _parse_emerge(raw: Any) -> Emerge | None:
    if raw is None or raw is False:
        return None
    if raw is True:
        raw = {}
    return Emerge(
        duration_s=parse_duration(raw.get("duration", DEFAULT_EMERGE_DURATION_S)),
        target_beat=float(raw.get("target_beat", DEFAULT_EMERGE_TARGET_BEAT)),
    )


def _size_hold_segments(
    segments: list[Segment],
    total_duration_s: float | None,
    emerge: Emerge | None,
) -> tuple[Segment, ...]:
    hold_indices = [i for i, s in enumerate(segments) if s.duration_s is None]
    if not hold_indices:
        return tuple(segments)
    if total_duration_s is None:
        raise ValueError(
            "this session has 'hold' segments, so it needs a total duration "
            "(pass --duration or set default_total)"
        )
    fixed = sum(s.duration_s for s in segments if s.duration_s is not None)
    emerge_s = emerge.duration_s if emerge else 0.0
    # Each crossfade pulls the following segment earlier, so the declared
    # overlaps are added back to keep the resolved total on target. The last
    # segment's own overlap never applies to a following segment, but the
    # emerge block joins with a fixed crossfade of its own.
    overlap_total = sum(s.overlap_s for s in segments[:-1])
    if emerge is not None:
        overlap_total += EMERGE_FADE_IN_S
    remaining = total_duration_s - fixed - emerge_s + overlap_total
    if remaining <= 0.0:
        raise ValueError(
            f"total duration {total_duration_s:.0f}s is too short: fixed "
            f"segments and emerge already take {fixed + emerge_s:.0f}s"
        )
    share = remaining / len(hold_indices)
    sized = list(segments)
    for i in hold_indices:
        sized[i] = Segment(
            duration_s=share,
            overlap_s=segments[i].overlap_s,
            groups=segments[i].groups,
            pink=segments[i].pink,
            texture=segments[i].texture,
        )
    return tuple(sized)


def load_session_dict(
    data: dict, total_duration_s: float | None = None
) -> Session:
    fidelity = str(data.get("fidelity", "original"))
    if fidelity not in FIDELITY_TIERS:
        raise ValueError(
            f"unknown fidelity {fidelity!r}; expected one of "
            f"{sorted(FIDELITY_TIERS)}"
        )

    raw_segments = data.get("segments") or []
    if not raw_segments:
        raise ValueError("a session needs at least one segment")

    defaults = data.get("defaults") or {}
    segments: list[Segment] = []
    for raw in raw_segments:
        raw_groups = raw.get("groups") or []
        if not raw_groups:
            raise ValueError("every segment needs at least one group")
        raw_duration = raw.get("duration")
        duration_s = (
            None
            if raw_duration == HOLD
            else parse_duration(raw_duration)
        )

        # Handle bed: vs pink: - they are mutually exclusive
        has_bed = "bed" in raw
        has_pink = "pink" in raw
        if has_bed and has_pink:
            raise ValueError("use bed or pink, not both")

        pink_spec = None
        if has_bed:
            pink_spec = _parse_pink(raw.get("bed"), is_alias=False)
        elif has_pink:
            pink_spec = _parse_pink(raw.get("pink"), is_alias=True)

        texture_spec = _parse_texture(raw.get("texture"), has_bed)

        segments.append(
            Segment(
                duration_s=duration_s,
                overlap_s=parse_duration(raw.get("overlap", 0.0)),
                groups=tuple(_parse_group(g, defaults) for g in raw_groups),
                pink=pink_spec,
                texture=texture_spec,
            )
        )

    emerge = _parse_emerge(data.get("emerge"))
    if emerge is not None:
        primary = max(segments[-1].groups, key=lambda g: g.level_db)
        if primary.sam is not None:
            raise ValueError(
                "emerge cannot glide a sam group (group "
                f"{primary.name!r} in the final segment): a SAM group has no "
                "binaural beat to glide, only a modulation rate"
            )
        if primary.pairs_spec is not None:
            raise ValueError(
                "emerge cannot glide a pairs-form group "
                f"(group {primary.name!r} in the final segment)"
            )
    total = total_duration_s
    if total is None and "default_total" in data:
        total = parse_duration(data["default_total"])

    output = data.get("output") or {}
    raw_edge_fade_s = output.get("edge_fade_s", DEFAULT_EDGE_FADE_S)
    try:
        edge_fade_s = float(raw_edge_fade_s)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"output.edge_fade_s must be a number, got {raw_edge_fade_s!r}"
        ) from exc
    if not 0.0 <= edge_fade_s <= 5.0:
        raise ValueError(
            f"output.edge_fade_s must be in [0.0, 5.0], got {edge_fade_s}"
        )

    return Session(
        name=str(data.get("name", "untitled")),
        title=str(data.get("title", data.get("name", "untitled"))),
        fidelity=fidelity,
        notes=str(data.get("notes", "")),
        sample_rate=int(data.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        peak_dbfs=float(output.get("peak_dbfs", DEFAULT_PEAK_DBFS)),
        segments=_size_hold_segments(segments, total, emerge),
        emerge=emerge,
        edge_fade_s=edge_fade_s,
    )


def load_session(
    path: Path, total_duration_s: float | None = None
) -> Session:
    data = yaml.safe_load(Path(path).read_text())
    return load_session_dict(data, total_duration_s=total_duration_s)
