# Gateway tape spectral analysis — Focus 10, 12, 15, 21

Measured from personal copies of four Gateway Experience tracks ("Free
Flow 10", "Free Flow 12", "Exploring Focus 15", "Free Flow Journey in
Focus 21"). Full machine-readable detail, including the 2-minute
timelines and per-measurement confidence notes, lives in `results.json`,
`f21-results.json` and `bed-results.json` alongside this file.
Frequencies below are facts measured from the audio, not reproductions of
any published material.

## Method

Decimate 44.1 kHz → 2940 Hz; coherent 60 s Hann FFT per block and channel
(0.0167 Hz bins) with a 12 dB local-tonality gate — the median-over-time
gating is what rejects voice, music and the spoken intros. Carrier
frequencies pinned with 600 s windows and quadratic peak interpolation.
Every claimed beat was verified independently by FFT of the analytic
envelope of the band-passed mono sum; all verification values are in the
JSON files. No claimed pair failed verification.

**One estimator per kind of thing.** A bed-level ratio has a continuum on
top and a single spectral line on the bottom, and those want different
estimators. Continua are measured with Welch's averaged periodogram
(2^17 points, 0.34–0.37 Hz bins): a continuum's per-bin power is one
noisy chi-square(2) sample, and only segment-averaging makes it stable.
Carriers are measured with a single coherent full-window FFT
(0.008–0.017 Hz bins): a line's power stays in one or two bins at any
resolution, so averaging buys it nothing while costing resolution. Bin
width is what makes this matter: as bins narrow, a line's height is
unchanged while the continuum floor's power *per bin* — a density —
shrinks with the bin, so the same real tone towers higher over the same
floor the finer you look. Using Welch for both sides conflates "is this
tone real" with "how coarse is my continuum estimator": F21's 50.4987 Hz
reference carrier reads at 0.11× the local floor at Welch resolution and
clears a 3× presence guard (3.9×) at full-window resolution — the same
tone in the same window. The F10/F12/F15 bed figures below are on a
single-estimator (Welch) footing and each number stays paired with the
estimator that produced it, since switching estimator shifts a ratio by
the bin-width factor (~16 dB here). See
`f21-results.json`'s `noise_bed.level_estimator` and
`tests/test_bed_acceptance.py::_carrier_power_coherent`.

**Single-window line measurements at low SNR are phase-variant.** At
~4× local-floor SNR a single coherent window reads
|tone + that window's noise|², which swings ±3 dB with the tone's phase
against the noise. Robust figures therefore average several disjoint
voice-free windows, with wow screening: complex demodulation exposes
windows where tape wow smears the line across bins (drift ±35 mHz per
10 s against 8.3 mHz bins collapses the presence guard to 0.5×/0.04×
versus 2.7–3.4× in clean windows), and those windows are excluded as
casualties, not level changes. On the render side no averaging is
needed: a render separates — the tonal mix re-renders bit-exactly
without the bed, the bed is the exact difference, and line power read
off the noiseless tonal render is phase-invariant outright.

**Known traps.** 60 Hz mains harmonics masquerade as carriers (caught by
frequency-wander screening, 101 ppm vs 6 ppm); overlapping exit bands
can demodulate as a phantom 40 Hz; the 50 Hz cluster misreads as 0.5 Hz
unless notched before demodulation.

## Focus 10 — an older, analog-mastered bed

No round-number grid fits this track: it drifts ~0.3% overall, and the two
members of each pair drift *relative to each other*, so the beats themselves
glide. This is not a speed error (a 5.7% beat change cannot come from a
0.3% speed change) — the beat glide is in the master.

| Pair | Left | Right | Beat | Span | Level |
|---|---|---|---|---|---|
| A | 99.98 Hz | 104.03 Hz | +4.05 Hz (glides 4.115 → 3.886) | 100–1370 s | 0 dB |
| B | 298.63 Hz | 302.48 Hz | +3.85 Hz (glides 3.867 → 3.691) | 350–1900 s | −0.5 dB |
| C | 497.01 Hz | 493.31 Hz | −3.70 Hz (ears reversed) | 420–1570 s | −6 dB |

Pair B is additionally amplitude-modulated at ±0.58 → 0.48 Hz, modulation
index ≈ 0.22, the *same* rate in both ears — a deliberate slow tremolo with
no binaural component. Second harmonics sit at −29 dB. The bed is the
loudest element and is *not* surf: envelope modulation 5.7% vs the ~4% a
stationary Gaussian bed shows.

## Focus 12 — digital master, 51.3 ppm fast, and a mono anchor

Every carrier is an exactly round nominal × 1.0000513 (residuals
< 0.0006 Hz across 18 tones). The track also carries a strong 15734 Hz
tone — the NTSC video line rate — so the master almost certainly passed
through a video-based PCM adaptor, which is where the 51 ppm came from.
The left ear is always the high member. De-scaled nominals:

| Layer | Pairs (L/R) | Beat | Span |
|---|---|---|---|
| sub-delta | 50.5/49.75 | 0.75 Hz | whole track |
| delta | 100.75/99.25 | 1.50 Hz | whole track |
| theta ×3 | 202/198, 252/248, 302/298 | 4.0 Hz | 355–1925 s |
| alpha ×3 | 405/395, 505/495, 605/595 | 10.0 Hz | 595–1745 s |
| exit | 483/467, 608/592 | 16 Hz | 1950–2110 s |
| exit | 632/568 | 64 Hz | 1950–2110 s, +12 dB |

Plus a **mono 50.000 Hz anchor** present in both ears (phase-locked,
−10° ± 6° over 30 blocks). Against the 50.5 L and 49.75 R members it
produces *monaural* beats of 0.5 Hz in the left ear and 0.25 Hz in the
right; envelope demodulation shows 0.25, 0.50 and 0.75 Hz simultaneously.
Naive L-vs-R peak matching can never see this structure. The bed swells
at 0.21 Hz (13.8% envelope modulation) — the second harmonic of its
0.102 Hz pan LFO, not an independent surf modulator; one LFO drives both
balance and swell (see the bed analysis below).

## Focus 15 — correct speed, reversed polarity, and an unexplained grid

Runs at correct speed (which is what proves F12 is the fast one, not
everything else slow). Same layer vocabulary as F12, but the track
**swaps ear polarity mid-session**: L-high through the intro, then from
410–1900 s the right ear is the high member and every beat sign reverses.
Bridging pairs at 175/179 and 262/266 (4 Hz). The exit block is F12's exit
tone set with the channels mirrored.

The 19-minute deep section (470–1600 s) is nearly silent except four pairs:

| Left | Right | Beat |
|---|---|---|
| 304.8 | 300.0 | +4.8 Hz |
| 503.9 | 511.2 | −7.3 Hz |
| 606.6 | 599.3 | +7.3 Hz |
| 807.3 | 800.0 | +7.3 Hz |

These are held bit-accurate on a 0.1 Hz grid to ±0.1 ppm (0.002 Hz over
16 minutes) yet sit at non-round values with a repeated 7.3 Hz split. No
generating rule was found; intermodulation was ruled out (nothing present
is loud enough). Reported as measured, unexplained.

No bed on any track is pink noise: all sit between brown (−20) and
−30 dB/decade.

## Focus 21 — the full ladder

Full detail in `f21-results.json`. Runs at correct speed (−4.2 ppm; the
NTSC pilot is present but the master is not the 51.3 ppm-fast F12
lineage). All beats verified by envelope demodulation.

| Layer | Pairs (L/R after the flip) | Beat | Span |
|---|---|---|---|
| mono anchor | 50.000 both ears | — (0.25/0.5/0.75 monaural) | most of track |
| sub-delta | 49.75/50.5 | 0.75 Hz | 215–2280 s (L-high before 215 s) |
| delta | 99.25/100.75 | 1.5 Hz | 215–2280 s |
| theta ×3 | 198/202, 248/252, 298/302 | 4.0 Hz | 230–2260 s |
| alpha ×3 | 395/405, 495/505, 595/605 | 10.0 Hz | 600–840 s and 1980–2280 s |
| beta ×3 | 592/608, 747/763, 892/908 | 16.0 Hz | 900–1980 s |
| exit | 467/483, 592/608 | 16 Hz | 2335–2406 s |
| exit | 568/632 | 64 Hz | 2335–2406 s |

Polarity: left-high intro, flips to right-high across a ~15 s crossfade
at 215–230 s and stays there — F15's device, used once at the seam. The
50-cluster is F12's construction with the ears swapped. No glides
anywhere; layers enter and exit by crossfade. The middle beta centre is
**755 Hz** (747/763, held to 0.005 Hz for 18 minutes), not the folklore's
750. Bed: brown-ish (−18.6 dB/decade), strongly surf-modulated (28.9%
envelope modulation at 0.099 Hz) — the deepest, slowest surf of the four
tapes.

Unexplained, F15-style: a 276.676 Hz tone, identical in both ears and
near-antiphase, 18 dB prominent, deep-section only, on no grid and not an
intermodulation product.

## Folklore comparison

The "focus.txt" frequency table that has long circulated in binaural-beat
communities, claiming these programs' recipes, is **right** about 100/104
at 4 Hz, the 100/200/250/300 Hz centres with 1.5/4 Hz beats (exact once
the 51 ppm scale is removed), and on F21 the 4 Hz low carriers, 16 Hz
beta, and the 600/900 Hz centres. It is **wrong or silent** on: the
16 Hz claim for the earlier tapes (exit-signal only, on 475/600 Hz
carriers, paired with an unmentioned 64 Hz), the entire 10 Hz alpha
layer, the 50 Hz mono-anchor cluster, ear polarity and its mid-session
reversal, F15's deep section, F21's 755 Hz beta centre and dominant
0.75/1.5 Hz pairs, the 64 Hz exits, and the premise that Focus 10 uses
fixed round frequencies at all.

## Bed spatial behaviour and levels

Full detail in `bed-results.json`.

**No flanger exists on any tape.** Comb search in the bed's own energy
band (150 Hz–2 kHz continuum, discrete lines masked, empirical null from
24 matched brown-noise realisations): every candidate delay from 2.5 to
30 ms shows only 0.1–0.8 dB peak-to-peak ripple, and the per-frame
quefrency track is uncorrelated frame to frame. Audible flanging needs
notches tens of dB deep, swept coherently. Absent.

**What moves instead:** F12 and F15 cross-fade **two fully independent
noise streams** (interchannel coherence ≈ 0) under one shared sinusoidal
LFO — 9.82 s period on F12, 9.65 s on F15 (same production setting;
1.8% apart), 3.25 and 4.50 dB peak-to-peak. Ping-pong, not wrap-around
(harmonic ratios rule out a sawtooth). F21 runs the same construction
(coherence 0.003–0.013 across 250 Hz–8 kHz) at **9.90 s, 2.83 dB p-p**,
with the envelope swell at the LFO fundamental (F15's arrangement, not
F12's second-harmonic swell): the 1.5–8 kHz ILD trace carries a line at
0.1010 Hz standing 70–235× over its local median (H2/H1 = 0.10,
H3/H1 = 0.008), while a sinusoid fit at half that rate recovers under
1.5% of the fundamental's amplitude — no 20 s component exists. F10's
bed does not move at all: one mono stream in both ears with a fixed
145 µs left-lead (analogue azimuth) and independent tape hiss above
~5 kHz; its "phasey" colour is the static 3.4 kHz interchannel null that
delay creates. The "flanging that surfs" percept is the decorrelated
crossfade at correct (much louder) level — a flanger would be the wrong
reproduction.

**Bed levels, band-integrated.** Comparing one FFT bin of bed noise
against a carrier's coherent peak understates a continuum; the correct
figure integrates the bed's power across the band (RMS of the
tone-excised continuum vs the strongest carrier's RMS, same window):
F10 **−0.7 dB**, F12 **+7.8 dB**, F15 **+34.6 dB**. The beds are not
background — on F12/F15 they are LOUDER than the carriers, dominating
the mix while the tones sit inside them. F21's robust figure — mean of
four clean windows out of six across the voice-free 1100–1820 s stretch
(18.98/18.04/19.86/19.65 dB, with two wow-casualty outliers at
26.52/31.97 dB excluded per the wow screen above), bed side in physical
bin-width-scaled power so 44.1 kHz tape and 48 kHz render measurements
share units — is **14.34 dB** over the 50.4987 Hz reference in that
cross-rate convention. `f21-results.json` carries both the single-window
record (`bed_level_rms_rel_db`) and the robust calibration figure
(`bed_level_rms_rel_db_robust`).

## Implications for this project's engine

Features the measured structure needs:

1. **Ear-polarity control** — which ear gets the high carrier, per group,
   and it matters: F15 flips it mid-session as (presumably) a deepening cue.
2. **Beat glide independent of carrier glide** — F10's beats drift while
   carrier centres barely move.
3. **Free per-pair carriers** — 202/198 + 252/248 + 302/298 is three
   independent bases sharing one beat; F15's deep pairs fit no grid.
4. **Mono anchor tones** — a single in-phase carrier in both ears, beating
   monaurally against each ear's pair member.
5. **Same-ear AM (tremolo)** — F10 pair B's 0.5 Hz envelope in both ears.
6. **Surf/brown beds, RMS-referenced** — the patents' pink noise appears
   on none of these tracks; beds must be level-calibrated per tape, and
   F12/F15/F21-style beds need a two-independent-streams crossfade mode
   driven by a ~9.7–9.9 s LFO (a single-stream equal-power pan has
   coherence 1 and cannot produce the measured spatial churn); F10-style
   beds need a fixed interaural delay option.

The presets built from these tables (the bundled `focus-10/12/15/21`)
are measured numbers, but any preset is still our arrangement of them,
not a copy of the recordings.
