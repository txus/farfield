# 🌌 farfield

[![tests](https://github.com/txus/farfield/actions/workflows/test.yml/badge.svg)](https://github.com/txus/farfield/actions/workflows/test.yml)

Mind awake, body asleep — binaural-beat synthesis and analysis engine.

An open-source engine for the territory Robert Monroe mapped — from his
expired patents and our own spectral analysis of the classic gateway-era
programs.

Farfield renders layered binaural-beat sessions from plain YAML
descriptions, and ships the measurement pipeline its presets were built
with. It exists for personal experimentation and research: render your
own custom meditation sessions and have fun with it.

## What it does

- **Layered sessions**: multiple simultaneous carrier pairs with
  scheduled beat frequencies, phase-coherent segment crossfades, and
  glide-free seams.
- **Spatial angle modulation (SAM)**: antiphase sinusoidal phase
  modulation of a shared carrier —
  `S_L/R(t) = A·sin[2πf_s·t ± φ_p·sin(2πf_m·t)]` — which produces a
  40 Hz beat percept where a conventional binaural pair won't fuse. See
  [`docs/monroe-sound-science.md`](docs/monroe-sound-science.md) for the
  math and the detector that verifies it.
- **Noise beds**: pink/brown noise with slow amplitude modulation
  ("surf"), stereo decorrelation, and band-limited moving textures.
- **An analysis pipeline** (`farfield.analysis`): coherent FFTs, envelope
  demodulation, and mid/side complex projection that separates true phase
  modulation from amplitude panning. It measures carriers, beat rates,
  modulation depths and layer timelines from any stereo recording — it's
  how the measured presets were made, and how the test suite verifies
  every render against its numbers.
- **16 bundled presets** (in [`presets/`](presets/)) in four provenance
  groups, strongest first:
  measured from the original tapes, measured from the MSS remasters, from
  the patents, and original designs. The measurement data behind the
  first two groups is published in
  [`docs/tape-analysis/`](docs/tape-analysis/).

The measured presets were reverse engineered by spectral analysis of
personal copies of the classic gateway-era recordings and their modern
remasters; the rest implement techniques from the now-public-domain
patents (US 3,884,218; US 5,213,562; US 5,356,368) and their abandoned
SAM successors. No audio from any recording is included or reproduced —
presets are measured numbers, re-synthesized from scratch.

## Quick start

Requires Python ≥ 3.11.

```sh
git clone https://github.com/txus/farfield && cd farfield

# run straight from the repo with uv, no install:
uv run farfield render presets/focus-10.yaml

# or install the CLI on your PATH:
uv tool install .    # with uv
pip install .        # with plain pip
farfield render presets/focus-10.yaml
```

## Usage

```sh
farfield list                          # list the presets in ./presets
farfield list path/to/dir              # or in any directory
farfield render presets/focus-10.yaml  # render a session to WAV
farfield describe presets/focus-10-mss.yaml   # print a session timeline
farfield plot presets/focus-10-mss.yaml       # self-contained HTML visualizer
```

The CLI takes plain paths — sessions are YAML files, and the bundled
presets in [`presets/`](presets/) are ordinary examples of them, not a
registry. Renders get an 8 s fade-in/out (`--fade 0` disables), `--seed`
makes them reproducible, and `--json` writes a sidecar describing every
layer.

## Roadmap

- [x] **Layered sessions** — stacked carrier pairs, mono anchors,
  per-ear polarity, beat and carrier glides
- [x] **Spatial angle modulation (SAM)** — with `closed`, `open` and
  `discontinuous` paths
- [x] **Same-ear tremolo** — independent rate and depth per ear
- [x] **Shaped noise beds** — two-slope spectra, two-stream crossfade
  panning, fixed interaural delay
- [x] **Moving band-limited textures** — the MSS-era "3D" shimmer
- [x] **Quadrature surf** — common-mode swell locked to the texture pan
- [x] **Phase-coherent crossfades** — click-free seams, verified per render
- [x] **Measurement pipeline** — carriers, beats, modulation depths and
  layer timelines from any stereo recording
- [ ] **Isochronic tones** — first-class pulsed (gated) tones; the
  technique most requested in the wider community, including in Tom
  Campbell's freely published audio experiments
- [ ] **Monaural beats** — both members summed pre-pan into both ears
- [ ] **Carrier FM** — per-group LFO on carrier frequency
- [ ] **Arbitrary SAM spatial paths** — any periodic azimuth trajectory
  θ(t) via ITD(θ) = (d/c)·sin θ
- [ ] **Real-time streaming output** — render to an audio device instead
  of a file

## License

- Code and presets: [Apache License 2.0](LICENSE).
- Measurement data (`docs/tape-analysis/*.json`):
  [CC0 1.0](docs/tape-analysis/LICENSE).
- Prose docs: [CC BY 4.0](docs/LICENSE).

Provenance and trademark details live in [NOTICE](NOTICE).

---

farfield is an independent project, not affiliated with, endorsed by, or
sponsored by The Monroe Institute, Interstate Industries, Inc.
(Hemi-Sync®), or any related entity.
