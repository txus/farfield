# Monroe Sound Science remasters — spectral analysis

Measured from personal copies of the four MSS3D releases of the same
programs analysed in `report.md`: Freeflow 10, Freeflow 12, Free Flow 15 and
Freeflow 21. Machine-readable detail, including per-track layer schedules and
every uncertainty, is in `mss-results.json`. Frequencies below are facts
measured from the audio. TMI publishes no numbers for the MSS era, so there is
nothing here to check against except the tapes.

Source: 320 kbps CBR mp3, 48 kHz stereo, 2400–2695 s each.

## Method

Same pipeline as the tape work — decimate, coherent long FFTs for lines, Welch
for continua, envelope demodulation to verify every beat, multiple disjoint
voice-free windows with the scatter reported — plus two additions these files
require.

First, **beats are measured separately in L, R and L+R**. On a tape a
binaural pair puts one tone in each ear, so the beat exists only in the sum.
Here several layers beat inside each ear as well, and only the per-ear envelope
distinguishes the two cases.

Second, the modulation analysis is done on the **mid and side channels**, with
`r_u = S(f0+f_m)/M(f0)` and `r_l = S(f0−f_m)/M(f0)` as complex projections at
exact frequencies. Then

    a = (r_u + conj(r_l))/2   is the pure amplitude-pan component,   ε   = 2|a|
    b = (r_u − conj(r_l))/2   is the pure SAM (antiphase PM) component, φ_p = 2|b|

This separation is exact for any modulator phase, and it is required: the
obvious test — "are the sidebands in phase or in quadrature with the carrier?"
— does not work, because an unknown LFO phase makes a pan look like SAM and
vice versa.

Two implementation rules. Never sum the three-bin Hann core as *complex*
values: the ±0.5 side lobes cancel the centre bin and the ratios come out
random. Use single-frequency projection, the single centre bin, or magnitudes.
And in the mono-sum envelope, every line at an exact difference between two
coexisting carriers is an ordinary intermodulation reading, not a hidden
layer: the 30–350 Hz bands carry strong lines at 46.9, 48.1, 48.4, 49.6,
49.9, 50.9, 54.9, 93.7, 97.7 and 101.7 Hz, all of which look like gamma and
every one of which is a carrier difference (97.25 − 48.85 = 48.40, and so
on) — the same phantom-gamma trap `f21-results.json` documents.

## The headline: everything was retuned to a G major chord

The tape grid was round decimals — 50, 100, 200, 250, 300, 400, 500, 600, 755,
900 Hz. MSS moved every centre to the nearest twelve-tone-equal-tempered pitch
at A4 = 440, and the pitches it chose spell a G major triad in successive
octaves.

| Tape centre | MSS centre | Note | Deviation from 12-TET |
|---|---|---|---|
| 100 | 97.9988 | G2 | −0.1 mHz |
| 200 | 195.9979 | G3 | +0.2 mHz |
| 250 | 246.9416 | B3 | −0.1 mHz |
| 300 | 293.6649 | D4 | +0.1 mHz |
| 400 | 391.9951 | G4 | −0.3 mHz |
| 500 | 493.8818 | B4 | −1.5 mHz |
| 600 | 587.3314 | D5 | +1.9 mHz |
| 755 | 783.9948 | G5 | +3.9 mHz |
| 900 | 987.7684 | B5 | +1.8 mHz |
| 50 | 49.000 | G1 | +0.6 mHz |

Every centre lands within 4 mHz — 7 ppm — of the exact equal-tempered value,
with 0.0–0.6 mHz scatter across disjoint windows. Just intonation is ruled out:
a just third and fifth on G are 245.00 and 294.00 Hz, and the measured values
are the tempered 246.9416 and 293.6649.

This is what makes the grid read about 2% flat while the beats do not move: it
is a retune, not a speed change. A 2% speed change would have dragged the 4 Hz
beat to 3.92; the measured theta beat is **4.001 Hz**.

## The layer vocabulary, shared by all four programs

All four tracks are assembled from the same six layers at the same frequencies,
agreeing across tracks to under a millihertz. What differs between programs is
only which layers run and when.

| Layer | Pairs (low/high) | Centre notes | Beat | Spatial |
|---|---|---|---|---|
| ground | 48.850 / 49.150 | G1 | 0.300 Hz | hard-panned, static |
| delta | 97.2484 / 98.7492 | G2 | 1.5008 Hz | static, −3.4 dB crossfeed |
| theta | 194.00/198.00, 244.941/248.942, 291.665/295.665 | G3 B3 D4 | 4.001 Hz | rotating |
| alpha | 386.995/396.995, 488.883/498.881, 582.331/592.332 | G4 B4 D5 | 10.000 Hz | rotating |
| "7.333" | 490.219/497.548, 583.663/590.999, 780.328/787.658 | B4 D5 G5 | 7.333 Hz | rotating |
| beta | 579.082/595.580, 775.745/792.245, 979.515/996.022 | D5 G5 B5 | 16.500 Hz | rotating |
| exit | 458.167/474.167, 567.333/607.333, 588.188/629.656 | Bb4, D5, — | 16.000, 40.000, 41.468 Hz | static |

Every beat was verified by envelope demodulation, with the second harmonic
present in each case. Levels fall about 2.2 dB per step up each triad and about
13 dB per layer.

Two beat rates are new numbers rather than kept ones. The tape era's 16.0 Hz
beta became **16.500**, held to 6 mHz across three pairs and eighteen minutes.
And F15's unexplained 7.3 Hz split became **7.333** — kept, and moved onto the
G grid, which means whoever built the MSS masters treated that odd number as
deliberate. It still has no explanation.

## The rotation — what MSS actually added

Every carrier from the theta layer upward is swept left to right by a pure
sinusoidal amplitude pan:

- **period 30.000 s** (f_m = 0.033333 Hz), identical on all four programs
- **pan depth ε = 0.80 ± 0.011** across 30 measured tones, i.e. peak ILD ±19 dB
- second harmonic of the ILD trace at 0.001 of the fundamental — a clean sine
- total power constant to 0.45 dB across the cycle: a pan, not two tremolos
- **the two members of every pair counter-rotate**, 179.5–180.9° apart

That last point is the design. When the low member of a pair is hard left the
high member is hard right — an ordinary binaural pair — and fifteen seconds
later they have swapped ears and the beat's polarity has reversed. The tape era
used a single mid-session polarity flip (F15 at 410 s, F21 at 215 s) as a
deepening cue. MSS turned that gesture into a continuous 30-second rotation and
applied it to every layer at once, with different layers sitting at different
points in the cycle.

It is also a measurement trap. Any window that is a whole number of 30 s cycles
— which every sensible 120 s or 240 s analysis window is — averages the ILD to
zero and the tones read as dead-centre mono, with interaural differences below
0.2 dB and 1.3°. They are not mono. They are moving.

## SAM: absent

At the modulation rate that actually exists (1/30 Hz), the SAM component of
every carrier is at or below its own noise floor: **φ_p between 0.0021 and
0.0155 rad against a floor of 0.0079–0.0128 rad**, on 30 tones across four
tracks. The pan component in the same measurement is ε = 0.80 — between 50 and
380 times larger. The equivalent swept ITD is at most 24 µs and typically 6 µs.

Widening the search to any f_m from 1 to 100 Hz — the patents' claimed 40–70 Hz
gamma range and everything around it — the strongest non-carrier side-channel
line near any carrier bounds **φ_p ≤ 0.045–0.123 rad**, equivalent to a swept
ITD of at most 26–108 µs. The patents' own arithmetic puts a full ear-to-ear
arc at φ_p ≈ 0.48 rad at 300 Hz. The bound here is four to ten times tighter
than that at gamma rates and thirty to sixty times tighter at 1/30 Hz.

Three independent checks agree. GCC-PHAT ITD tracking (350 ms windows) finds no
periodic ITD at any rate on any track: fitted amplitudes of 3–8 µs against
40–96 µs frame-to-frame scatter. The mono-sum envelope shows the layers' own
beats and nothing else. And the method is demonstrably not blind — the same
data yields the bed's 7.5 s ILD pan at 4.6–4.9 dB with a second-harmonic ratio
of 0.0001.

Note that the presence of a mono-sum envelope line is not by itself evidence
either way: summing two tones of different frequency produces an envelope at
their difference whatever ears they came from. What distinguishes SAM is the
sideband structure around a *shared* carrier.

The only gamma-rate content anywhere in the four programs is in the exit block,
where 567.333/607.333 beats at exactly **40.000 Hz** (verified in L, R and L+R
envelopes) and 588.188/629.656 at 41.468 Hz. Those are ordinary beats on
separate carriers. The tape era's 64 Hz exit pair is gone; 40 Hz replaced it.

## The mp3 question

At 320 kbps the concern is that joint-stereo coding zeroed a side channel and
manufactured the mono readings. It did not. The side channel carries bed noise
only 3.0–6.5 dB below the mid channel at 194 Hz, so side information is being
transmitted there; the ±0.0333 Hz pan sidebands of those same tones come out of
the side channel at 37–40 dB over the local side floor; and the ground pair's
29–49 dB interchannel isolation and the delta pair's ±33–48° IPD survive intact
in the same files. The encoder is preserving interaural detail across the whole
band in question.

The one artifact that is real: a brickwall near **16 kHz**, with the continuum
falling from −88.7 dB at 12.8 kHz to −104 dB at 16.1 kHz and −188 dB by 20 kHz.
Bandwidth-limited source or encoder setting; irrelevant to anything measured here.

## The bed, and what "3D" turned out to mean

One bed design, shared by all four programs, with every figure agreeing to
0.2 dB across tracks; the shape and level figures below come from nine
voice-free 120 s windows across all four tracks, cross-window scatter under
0.6 dB. Detail in `mss-results.json` under `bed.spectrum_shape_v2`,
`bed.bed_vs_carrier_db_v2` and `bed.panned_hf_element`.

**Shape.** Not a straight-line noise colour: the continuum's argmax is
**88.1 ± 1.5 Hz**, and the fall from the knee is not one line — about
−37 dB/decade just above it, shallower through 630–1600 Hz, lifted by the
moving texture over 2–4 kHz, then a cliff above 6.3 kHz. Under the acceptance
test's own two-slope fit the tape reads **+1.84 ± 0.46 dB/decade** over
25–150 Hz and **−27.30 ± 0.06** over 150–8000 Hz; those are the like-for-like
targets. Total mono RMS is −27.3 dBFS.

**Level.** The bed's continuum, integrated 20 Hz–20 kHz against the 194 Hz
carrier's projected power, is **+15.4 dB** (range 15.03–16.09). (A
median-PSD-times-bandwidth reading of the 150–8000 Hz band gives −8.6 dB
instead; on a band this steep that estimator sits ~18 dB under the same
band's integral, and it is recorded in `mss-results.json` only as the
superseded `bed_vs_carrier_db`.) An estimator-independent check agrees: Free
Flow 10 carries no carrier above 300 Hz all session, so a plain
350 Hz–20 kHz band RMS of the tape is pure bed, and it reads −1.04 dB against
the 194 Hz carrier.

**Stereo construction** is the tape era's: two independent noise streams,
interchannel coherence 0.06–0.07 across 300–3000 Hz. There is no static
interaural delay — the 500–5000 Hz cross-correlation is even in lag and peaks
at 0.15 normalized, with no delayed copy (F10's tape bed had a fixed 145 µs
left lead; that is gone). No reverb tail — envelope autocorrelation is
0.012–0.022 at 5 to 200 ms.

**The moving texture.** A separate band-limited element pans while nothing
below ~800 Hz does. Its per-band ILD amplitude, fit on a 1 s/0.25 s block
trace, runs 0.27 dB at 800 Hz, 0.58 at 1 kHz, 0.62 at 1.6, 1.37 at 2, 2.66 at
2.5, 4.09 at 3.2, 6.70 at 4, **7.80 dB at 4.5–5.6 kHz**, 3.52 at 6.3 and 0.88
at 8 — one common LFO phase from 891 Hz to 7 kHz, agreeing to ±3°. The element
carries at least 72% of its peak band's energy: it dominates there. Its LFO is
**7.5000 s ± 0.0001** on all four tracks, sinusoidal to a second-harmonic
ratio of 0.0001, with under 1 dB of sum-level modulation.

The texture is **two decorrelated streams cross-faded, not a panned mono
source**: magnitude-squared coherence across the moving band measures
0.24–0.38 (0.38 at 2.8–3.5 kHz, 0.35 at 3.5–4.5, 0.24 at 4.5–5.6) where one
mono source panned between the ears would read close to 1.0, against a
0.03–0.15 floor elsewhere in the bed. That is the mechanism behind the
listener's "goes left to right and then reappears from the left": with
decorrelated streams the ear does not fuse the outgoing and the incoming sound
into one object crossing the middle — the gesture wraps rather than retracing.
It is also how a large ILD swing and a low coherence coexist, which a mono
pan model cannot produce at any depth.

The four tracks share the setting but not the audio: raw cross-track
correlation of the 2.8–6.3 kHz band is |r| < 0.003 and the LFO phase at
t = 600 s differs by track. Same production preset, four renders.

**The pan is ILD-only.** The ITD trace over the same windows shows no periodic
component at 7.5 s or any other rate. A genuine head-related pan would sweep
ITD to ±600 µs in step with the ILD, and it does not. So the immersive 3D
format of the marketing copy is, in these files, three things: a
5 kHz shimmer panned at 7.5 s, a crossfeed treatment on the 98 Hz pair, and the
30-second counter-rotation of the carriers. There is no HRTF and no ITD
anywhere in the release.

## The surf

The bed's total loudness breathes at the texture's own 7.5 s rate. Figures in
`mss-results.json` under `bed.surf_v2`, from twenty-two voice-free 120 s
windows across all four tracks:

- broadband, 20 Hz–20 kHz: **2.6–4.6% amplitude AM, grand mean 3.4%**
- spectrally shaped: about 3.4% below 300 Hz, rising to a **10% peak at
  400–630 Hz**, back to 2.3% by 2.5 kHz
- under a second estimator, the mono-sum envelope spectrum, the 7.5 s line
  stands **25–30 dB over its local median** through 1.5–8 kHz

A swell this size is invisible to an unsmoothed, full-bandwidth Hilbert
envelope of the mono sum: that statistic is dominated by the noise's own
sample-to-sample fluctuation (stationary Gaussian noise already reads 52.3%),
and the mono sum's `(L+R)²` cross term is not modulated at the fundamental by
a common-mode swell. Measuring slow structure requires smoothing (0.30 s) and
a **per-ear decomposition**: fit each ear's smoothed amplitude envelope at
7.5 s, then split the two complex fits into `S = (L+R)/2`, the common mode,
and `P = (L−R)/2`, the pan. `S` is a clean readout because an ideal
equal-power crossfade gives `S = 0` identically: a swell-free render measures
S = 0.1–1.0% against P = 33%, so the estimator does not leak pan into swell.

**Mechanism.** Modulation at the pan *fundamental* rules out an
amplitude-preserving `a+b=1` crossfade, whose signature is at 2×. Unequal
stream levels under an exactly counter-phased crossfade would move at the
fundamental but *exactly in phase or antiphase* with the ILD, leaving a DC
offset of `10log₁₀(P₁/P₂)` on the ILD trace. Two per-ear gain LFOs at one
rate whose phase offset departs from 180° give a sum going as
`cos((φ₁−φ₂)/2)·sin(θ+…)` and a difference going as
`sin((φ₁−φ₂)/2)·cos(θ+…)` — **90° apart for any offset**. The tape shows `S`
leading `P` by **+85.7 ± 1.3°** on Free Flow 10's six windows and +72 to
+103° over all twenty-two, with a mean ILD in the moving band of −0.04 to
−0.19 dB. Exact quadrature with no DC offset: the second mechanism.

**Both elements swell, not just the bed.** Rendering a preset three ways —
tones only, tones + bed, full — and differencing separates the elements
exactly (the mix is linear); the share of the tape's swell each band carries:

| band | tones | bed | texture |
|---|---|---|---|
| 20–300 Hz | 27% | 73% | 0% |
| 300–1500 | 13% | 87% | 1% |
| 1500–3000 | 0% | 38% | **62%** |
| 3000–4500 | 0% | 8% | **92%** |
| 4500–6000 | 0% | 8% | **92%** |
| 6000–8000 | 0% | 34% | 66% |

1.5–8 kHz is the *texture's* band, not the bed's: a bed-only swell of 4.7%
leaves 4500–6000 Hz at 0.2–0.6% against the tape's 5.4–6.2%. The texture
carries its own swell at the same rate and the same quadrature relationship
to its *own* pan (+87 to +97° against the tape's +85 to +100°). The two
depths differ — 4.7% on the bed, fitted to the broadband and 20–300 Hz
figures, and 6.0% on the texture, fitted to 4500–6000 Hz — because the
tape's swell is spectrally shaped and the two elements sit in different
parts of that shape.

The shape does not fall out of the two-element split. If the profile were two
flat depths in varying proportion, the bed-only region would read one
constant depth; on Free Flow 10, which has no carrier above 300 Hz and no
moving element below about 800 Hz, the 250 and 315 Hz thirds read 2.4% and
3.1% while the 400, 500 and 630 Hz thirds read 9.9%, 9.8% and 8.8% — a
threefold range inside a band where only the bed exists. The bed's own swell
is itself frequency-dependent. Reproducing the tape's profile needs a
swelling element with its own band shape, and none is invented here.

**Two things unexplained rather than smoothed over.** Above 6 kHz the common
mode reads 13–33% at continuum levels 70–85 dB down; that is encoder
bit-allocation noise tracking programme loudness, identical in magnitude on
all four tracks whatever they contain, and it is excluded from every figure
here. And the 3.15–4 kHz thirds carry a common mode of 0.25× the pan
amplitude *in antiphase*, reproducible to three figures across four
independent tracks — deterministic in the pan, so a property of the tape's
own pan law, but which departure from equal power produces it is not
established.

**What the presets render.** A bed at peak 88 Hz, +16.0/−35.0 dB/decade, and
a 1000–4800 Hz two-stream texture, sitting within 1.3 dB rms of the tape's
whole third-octave continuum from 25 Hz to 12.5 kHz. The surf renders as a
common-mode swell on **both** noise elements at each segment's texture pan
phase minus 90° — depth 0.090 on the bed and 0.114 on the texture — measuring
3.3–4.3% broadband against the tape's 2.8–4.6% and 5.9–6.3% at 4.5–6 kHz
against the tape's 5.4–6.2%. Disclosed residuals, not chased: each element's
swell is flat across its own band where the tape's is shaped (the render
reads 5.5% at 1500–3000 Hz against the tape's 1.8–2.8, and 4.1–4.5% at
300–1500 against the tape's 3.8–8.9); the 16 kHz third octave sits inside
the encoder lowpass, so it says nothing about the master; and the render's
ILD profile is broader than the tape's — 2.6 dB at 1.4–1.8 kHz and 5.9 at
5.6–7.1 against the tape's 0.62 and 3.52 — because one 4th-order Butterworth
band over a two-slope bed cannot make skirts as steep as the original's.

## Music

These remasters contain actual music, which the tapes did not. It is
intermittent, not a continuous pad. A G-based pentatonic intro pad (G4, C5, D5,
E5, G5, plus a 915.5 Hz partial on no pitch of the grid) runs 150–300 s on
three tracks and 260–400 s on Freeflow 10. An A major sonority (A4, E5, G#5,
A5, B5, C#6) appears briefly mid-session on Freeflow 12 at 1560, 1860 and
2280 s and on Freeflow 21 at 1110 s — unrelated to the beat grid, a
musical event rather than a layer. All bed and carrier figures above come from
windows containing none of it; outside the music spans the count of unexplained
tonal lines above 320 Hz is zero per 30-second block.

## Per program

**Freeflow 10** is the simplest of the four and the most changed. Its layer set
is ground + delta + rotating theta and nothing else — no alpha, no 7.333, no
16.5, and only a residual 607.33 Hz tone where the others have a full exit
block. Nothing of the F10 tape survives: the tape's 100/104 at 4.05 Hz, its
298.6/302.5, its 497/493.3 with reversed ears, its beat glides and its static
145 µs bed are all gone. MSS Freeflow 10 was rebuilt from the shared vocabulary
rather than remastered from the F10 recipe.

**Freeflow 12** keeps its tape's shape best: theta from 240 s to the end, alpha
for the 28-minute middle (780–2460 s), full exit block. The tape's 0.75 Hz
sub-delta became 0.300 Hz and its 50.000 Hz mono anchor — with the 0.25/0.5/0.75
Hz monaural beats that made that tape interesting — is gone.

**Free Flow 15** hands its 20-minute deep section (860–2060 s) to the 7.333 Hz
layer, with alpha on either side of it, and drops the ground pair for exactly
that stretch. The tape's deep 4.8 Hz pair is gone; the 7.3 Hz split is what
carried over.

**Freeflow 21** has the fullest architecture: a ladder in
(G1 → G2 → theta → alpha → 7.333 → 16.5), the 16.5 Hz beta layer owning
1040–1960 s, and the reverse ladder out. The tape's 755 Hz beta centre and its
unexplained 276.676 Hz deep-section tone are both gone, and its 28.9% surf is
down to the 3.4% measured on all four tracks.

## Scorecard against the tape era

**Kept**: the 4.0 Hz theta triplet and the 10.0 Hz alpha triplet beat-for-beat;
the 1.5 Hz delta pair; F15's 7.3 Hz oddity; a 16 Hz exit pair; the
three-pairs-one-beat architecture; the two-independent-streams bed.

**Retuned**: every carrier centre to 12-TET G major; sub-delta 0.75 → 0.300 Hz;
beta 16.0 → 16.500 Hz; the 64 Hz exit → 40.000 Hz; the bed LFO 9.7 → 7.5 s and
band-limited to 2–8 kHz; the surf 13.8–28.9% at 9.65–9.90 s → 3.4% at 7.5 s,
locked in quadrature to that LFO.

**Added**: the 30 s counter-rotation on every carrier; per-ear tremolo on the
ground pair (index 0.83, 0.250 Hz left and 0.500 Hz right — different rates in
the two ears); crossfeed on the delta pair; music; a 40 Hz exit beat.

**Removed**: the 50 Hz mono anchors and their monaural beats; mid-session
polarity flips; all glides; F10's entire recipe; F15's 4.8 Hz pair; F21's
755 Hz centre and 276.676 Hz anomaly; F10's fixed bed delay; the 64 Hz exit.

The overall direction is toward a cleaner, more regular, more musical
construction: one tuning system, one beat per layer, one modulation gesture
applied uniformly, no glides, no anomalies. The idiosyncrasies that made the
tapes hard to reverse-engineer — the drifting beats, the mono anchors, the
unexplained grids — were engineered out. What was engineered in is spatial
motion.

## SAM and the engine

**SAM needs nothing here, because there is none here.** The engine implements
SAM from the patents (see `docs/monroe-sound-science.md` and the `sam-gamma`
preset), and that is a defensible thing to do on its own merits, but it does
not bring a render closer to these recordings: the measured MSS-era "spatial"
mechanism is an amplitude pan, and the patents' one genuinely new idea does
not appear in the four programs that carry the brand.

The presets built from these tables (the bundled `focus-*-mss` set) sit
alongside the tape-measured set: the numbers are measured, but any preset
remains our arrangement of them, not a copy of the recordings.
