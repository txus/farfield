"""Command line front end."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from farfield.beats import beat_counts
from farfield.render import (
    fade_edges,
    render_session,
    sidecar,
    validate_timeline,
    write_wav,
)
from farfield.session import (
    Glide,
    Session,
    TremoloSplit,
    load_session,
    parse_duration,
)
from farfield.timeline import Timeline, resolve

TIER_HEADINGS = {
    "measured-tape": "measured from the original tapes",
    "measured-mss": "measured from the MSS remasters",
    "patent": "from the patents",
    "original": "original designs",
}
"""Plain-English group headings for the machine-readable fidelity tiers.
The machine values stay in the YAML and the sidecar; these are what a
person sees in `list`."""

TIER_LABELS = {**TIER_HEADINGS, "original": "original design"}
"""Singular variant for a single session's `describe` line."""


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


_TIER_ORDER = {"measured-tape": 0, "measured-mss": 1, "patent": 2, "original": 3}


def list_lines(directory: Path) -> list[str]:
    entries = []
    for p in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(p.read_text())
        entries.append(
            {
                "path": str(p),
                "title": str(data.get("title", p.stem)),
                "fidelity": str(data.get("fidelity", "original")),
            }
        )
    entries.sort(key=lambda e: (_TIER_ORDER.get(e["fidelity"], 9), e["path"]))
    lines: list[str] = []
    current_tier: str | None = None
    for entry in entries:
        if entry["fidelity"] != current_tier:
            current_tier = entry["fidelity"]
            lines.append("")
            lines.append(f"{TIER_HEADINGS.get(current_tier, current_tier)}:")
        lines.append(f"  {entry['path']:<30} {entry['title']}")
    if lines and lines[0] == "":
        lines = lines[1:]
    return lines


def _glide_text(value: float | Glide) -> str:
    if isinstance(value, Glide):
        return f"{value.start:.2f} -> {value.end:.2f}"
    return f"{float(value):.2f}"


def _pair_description(pair, high_ear: str) -> str:
    if pair.kind == "mono":
        return f"mono {pair.mono:.1f} Hz"
    if pair.kind == "explicit":
        beat = abs(pair.left - pair.right)
        return f"L {pair.left:.1f} / R {pair.right:.1f} (beat {beat:.1f} Hz)"
    # center
    beat_text = _glide_text(pair.beat)
    high = "left" if high_ear == "left" else "right"
    if isinstance(pair.beat, Glide):
        half_text = _glide_text(Glide(pair.beat.start / 2.0, pair.beat.end / 2.0))
    else:
        half_text = f"{float(pair.beat) / 2.0:.2f}"
    return f"{pair.center:.1f} Hz ±{half_text} (beat {beat_text} Hz, {high} high)"


def _tremolo_suffix(tremolo) -> str:
    if tremolo is None:
        return ""
    if isinstance(tremolo, TremoloSplit):
        return (
            f"  tremolo L {_glide_text(tremolo.left.rate_hz)} Hz"
            f" depth {tremolo.left.depth:.2f}"
            f" / R {_glide_text(tremolo.right.rate_hz)} Hz"
            f" depth {tremolo.right.depth:.2f}"
        )
    return f"  tremolo {_glide_text(tremolo.rate_hz)} Hz depth {tremolo.depth:.2f}"


def _gate_suffix(gate) -> str:
    if gate is None:
        return ""
    return (
        f"  gate {_glide_text(gate.rate_hz)} Hz"
        f" duty {gate.duty:.2f} depth {gate.depth:.2f}"
        f" edge {gate.edge_ms:g} ms"
    )


def describe_lines(session: Session, timeline: Timeline) -> list[str]:
    rate = timeline.sample_rate
    lines = [
        f"{session.name} — {session.title}",
        f"  fidelity: {TIER_LABELS.get(session.fidelity, session.fidelity)}",
        f"  duration: {format_duration(timeline.total_samples / rate)}",
        "",
    ]
    for layer in timeline.layers:
        start = format_duration(layer.start_sample / rate)
        end = format_duration((layer.start_sample + layer.n_samples) / rate)
        group = layer.group
        if group.sam is not None:
            sam = group.sam
            arc = (
                f"arc {sam.arc_deg:.0f} deg"
                if sam.arc_deg is not None
                else f"phi_p {sam.depth_rad:.3f} rad"
            )
            steps = f" x{sam.steps}" if sam.path == "discontinuous" else ""
            lines.append(
                f"    {start} - {end}  {group.name}  SAM: "
                f"{sam.carrier_hz:.1f} Hz carrier, {sam.rate_hz:.2f} Hz "
                f"{sam.path}{steps}, {arc} "
                f"(+/-{sam.peak_itd_s() * 1e6:.0f} us ITD)  "
                f"{group.level_db:+.1f} dB{_tremolo_suffix(group.tremolo)}"
            )
            continue
        if group.pairs_spec is not None:
            suffix = _tremolo_suffix(group.tremolo) + _gate_suffix(group.gate)
            for i, pair in enumerate(group.pairs_spec):
                desc = _pair_description(pair, group.high_ear)
                lines.append(
                    f"    {start} - {end}  {group.name}  pair {i}: {desc}  "
                    f"{group.level_db:+.1f} dB{suffix}"
                )
            continue
        binaural_beats, monaural = beat_counts(group.pairs)
        beat_start, beat_end = group.beat_bounds()
        beat = (
            f"{beat_start:.2f} Hz"
            if beat_start == beat_end
            else f"{beat_start:.2f} -> {beat_end:.2f} Hz"
        )
        lines.append(
            f"  {start:>8} - {end:<8} {group.name:<10} {beat:<20} "
            f"{group.level_db:+6.1f} dB   "
            f"base {group.carrier_base:.0f} Hz, "
            f"{binaural_beats} binaural / {monaural} monaural per channel"
        )
    for bed in timeline.pink_layers:
        start = format_duration(bed.start_sample / rate)
        end = format_duration((bed.start_sample + bed.n_samples) / rate)
        if bed.spec.shape is not None:
            # A shaped bed has no single slope; resolved_slope() would report
            # the unused colour default, which reads as a flat lie.
            detail = (
                f"hump {bed.spec.shape.peak_hz:.0f} Hz"
                f" {bed.spec.shape.rise_db_per_decade:+.1f}"
                f"/{bed.spec.shape.fall_db_per_decade:+.1f}"
            )
        else:
            detail = f"{bed.spec.algorithm} {bed.spec.resolved_slope():+.0f} dB/dec"
        if bed.spec.surf_rate_hz is not None:
            detail += (
                f", surf {bed.spec.surf_rate_hz:.2f} Hz "
                f"depth {bed.spec.surf_depth:.2f}"
            )
        # Stereo stage: pan is the default and carries no parameters, so it
        # gets no suffix; the two measured-tier modes name their own numbers.
        if bed.spec.stereo_mode == "crossfade":
            detail += (
                f", crossfade {bed.spec.lfo_period_s:.2f} s"
                f" / {bed.spec.stereo_depth_db:.2f} dB"
            )
        elif bed.spec.stereo_mode == "static":
            detail += f", static lead {bed.spec.interaural_delay_us:.0f} µs"
        lines.append(
            f"  {start:>8} - {end:<8} {bed.spec.color:<10} "
            f"{detail:<20} {bed.spec.level_db:+6.1f} dB"
        )
    for texture in timeline.texture_layers:
        start = format_duration(texture.start_sample / rate)
        end = format_duration((texture.start_sample + texture.n_samples) / rate)
        spec = texture.spec
        detail = (
            f"{spec.band_hz[0]:.0f}-{spec.band_hz[1]:.0f} Hz"
            f", pan {spec.pan_period_s:.2f} s"
            f" / {spec.pan_ild_amplitude_db:.1f} dB"
        )
        lines.append(
            f"  {start:>8} - {end:<8} {'texture':<10} "
            f"{detail:<20} {spec.level_db:+6.1f} dB"
        )
    if session.notes:
        lines.extend(["", *(f"  {line}" for line in session.notes.splitlines())])
    return lines


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="farfield", description="Layered binaural beat generator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render a session to a WAV file")
    render.add_argument("source", help="path to a session YAML")
    render.add_argument("-o", "--output", default=None)
    render.add_argument("--duration", default=None)
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--json", action="store_true", help="also write the sidecar")
    render.add_argument(
        "--fade",
        type=float,
        default=8.0,
        help="fade-in/out seconds (default 8, 0 disables)",
    )

    listing = sub.add_parser(
        "list", help="list the session YAMLs in a directory (default: ./presets)"
    )
    listing.add_argument("directory", nargs="?", default="presets")

    describe = sub.add_parser("describe", help="print a session timeline")
    describe.add_argument("source", help="path to a session YAML")
    describe.add_argument("--duration", default=None)

    plot = sub.add_parser("plot", help="write a self-contained HTML visualizer")
    plot.add_argument("source", help="path to a session YAML")
    plot.add_argument("-o", "--output", default=None)
    plot.add_argument("--duration", default=None)
    plot.add_argument("--seed", type=int, default=0)
    plot.add_argument(
        "--fade",
        type=float,
        default=8.0,
        help="fade-in/out seconds on the embedded audio (default 8, 0 disables)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "list":
        directory = Path(args.directory)
        if not directory.is_dir():
            print(f"no such directory: {directory}", file=sys.stderr)
            return 2
        print("\n".join(list_lines(directory)))
        return 0

    # Loading *and* executing sit inside the guard: an unrenderable session
    # (a stack over the fusion ceiling, say) should report one line on
    # stderr, not a traceback.
    try:
        total = parse_duration(args.duration) if args.duration else None
        path = Path(args.source)
        if not path.is_file():
            raise FileNotFoundError(f"no such session file: {path}")
        session = load_session(path, total_duration_s=total)

        if total is not None:
            actual = resolve(session).total_samples / session.sample_rate
            if abs(actual - total) > 1.0:
                print(
                    "note: this session has a fixed schedule; --duration was "
                    f"ignored (rendering {format_duration(actual)})",
                    file=sys.stderr,
                )

        if args.command == "describe":
            timeline = resolve(session)
            validate_timeline(timeline)
            print("\n".join(describe_lines(session, timeline)))
            return 0

        if args.command == "plot":
            from farfield.viz import write_visualizer

            output = (
                Path(args.output) if args.output else Path(f"{session.name}.html")
            )
            write_visualizer(session, output, seed=args.seed, fade_s=args.fade)
            print(f"wrote {output}")
            return 0

        output = Path(args.output) if args.output else Path(f"{session.name}.wav")
        audio = fade_edges(
            render_session(session, seed=args.seed), session.sample_rate, args.fade
        )
        write_wav(output, audio, session.sample_rate)
        if args.json:
            output.with_suffix(".json").write_text(
                json.dumps(sidecar(session, resolve(session)), indent=2)
            )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"wrote {output} ({format_duration(len(audio) / session.sample_rate)})")
    return 0
