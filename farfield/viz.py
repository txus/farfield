"""Emit a self-contained HTML visualizer for a rendered session.

All spectral analysis happens here, in Python. The page plays the actual
rendered audio (embedded as FLAC — lossless, and roughly a fifth of the
WAV's size for this kind of tonal material), so this library remains the
only synthesis implementation.
"""

from __future__ import annotations

import base64
import io
import tempfile
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from farfield.analysis import (
    ANALYSIS_NFFT,
    ANALYSIS_RATE,
    carrier_frequencies,
    carrier_span,
    decimate_to,
    goertzel_track,
    spectrogram,
)
from farfield.render import fade_edges, render_session, sidecar
from farfield.session import Session
from farfield.timeline import Timeline, resolve

MAX_FRAMES = 4000
DYNAMIC_RANGE_DB = 60.0
METER_HOP_S = 0.25
TEMPLATE_PATH = Path(__file__).parent / "viz_template.html"


def _to_bytes(mags: np.ndarray, ceiling: float) -> bytes:
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(mags, 1e-12) / ceiling)
    scaled = np.clip((db + DYNAMIC_RANGE_DB) / DYNAMIC_RANGE_DB, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8).tobytes()


def _encode(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _audio_data_uri(audio: np.ndarray, sample_rate: int) -> str:
    # Short renders embed lossless FLAC, which every browser plays. Longer
    # ones embed Ogg Vorbis: the noise bed defeats FLAC's prediction, so a
    # lossless embed of a full session is a multi-hundred-MB page that no
    # browser will open, while each channel's frequency content -- which is
    # what the beats are -- survives lossy coding.
    if len(audio) <= 60 * sample_rate:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="FLAC", subtype="PCM_16")
        return "data:audio/flac;base64," + _encode(buffer.getvalue())
    # The vorbis encoder in libsndfile 1.2.2 segfaults on one multi-minute
    # write call; feeding it 10 s blocks through a SoundFile handle is fine.
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as handle:
        tmp = Path(handle.name)
    try:
        with sf.SoundFile(
            tmp, "w", samplerate=sample_rate, channels=2,
            format="OGG", subtype="VORBIS",
        ) as out:
            step = 10 * sample_rate
            for i in range(0, len(audio), step):
                out.write(audio[i : i + step])
        data = tmp.read_bytes()
    finally:
        tmp.unlink()
    return "data:audio/ogg;base64," + _encode(data)


def build_payload(
    session: Session, timeline: Timeline, audio: np.ndarray
) -> dict:
    rate = timeline.sample_rate
    fmin, fmax = carrier_span(timeline)

    channels = []
    for column in (0, 1):
        decimated, analysis_rate = decimate_to(
            audio[:, column], rate, ANALYSIS_RATE
        )
        usable = max(1, len(decimated) - ANALYSIS_NFFT)
        hop = max(1, int(np.ceil(usable / MAX_FRAMES)))
        freqs, mags = spectrogram(
            decimated, analysis_rate, ANALYSIS_NFFT, hop, fmin, fmax
        )
        channels.append((freqs, mags, hop / analysis_rate))

    ceiling = max(float(mags.max()) for _, mags, _ in channels) or 1.0
    left_freqs, left_mags, hop_s = channels[0]
    _, right_mags, _ = channels[1]

    meter_block = max(1, int(rate * METER_HOP_S))
    meter_ceiling = 1e-12
    tracks = []
    for freq in carrier_frequencies(timeline):
        for column, name in ((0, "left"), (1, "right")):
            track = goertzel_track(
                audio[:, column], rate, freq, meter_block, meter_block
            )
            meter_ceiling = max(meter_ceiling, float(track.max()))
            tracks.append({"freq": freq, "channel": name, "raw": track})

    return {
        "session": sidecar(session, timeline),
        "spectrogram": {
            "freq_min": float(left_freqs[0]),
            "freq_max": float(left_freqs[-1]),
            "n_bins": int(len(left_freqs)),
            "n_frames": int(left_mags.shape[0]),
            "hop_s": float(hop_s),
            "window_s": ANALYSIS_NFFT / ANALYSIS_RATE,
            "left": _encode(_to_bytes(left_mags, ceiling)),
            "right": _encode(_to_bytes(right_mags, ceiling)),
        },
        "meters": {
            "hop_s": METER_HOP_S,
            "carriers": [
                {
                    "freq": t["freq"],
                    "channel": t["channel"],
                    "track": _encode(
                        np.clip(t["raw"] / meter_ceiling * 255.0, 0, 255)
                        .astype(np.uint8)
                        .tobytes()
                    ),
                }
                for t in tracks
            ],
        },
        "audio_data_uri": _audio_data_uri(audio, rate),
    }


def render_html(payload: dict, audio: np.ndarray, sample_rate: int) -> str:
    """Fill the template. ``audio`` and ``sample_rate`` are accepted so a
    caller can render a page from a payload alone, and are only used if the
    payload has no embedded audio."""
    payload = dict(payload)
    if not payload.get("audio_data_uri"):
        payload["audio_data_uri"] = _audio_data_uri(audio, sample_rate)
    template = TEMPLATE_PATH.read_text()
    return template.replace(
        "__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")
    )


def write_visualizer(
    session: Session, path: Path, seed: int = 0, fade_s: float = 8.0
) -> None:
    timeline = resolve(session)
    audio = fade_edges(render_session(session, seed=seed), session.sample_rate, fade_s)
    payload = build_payload(session, timeline, audio)
    Path(path).write_text(render_html(payload, audio, session.sample_rate))
