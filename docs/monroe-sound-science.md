# Monroe Sound Science — what it is and how this engine supports it

Companion to the patent work this engine is built on: what "Monroe Sound
Science" (MSS) names, the mathematics of Spatial Angle Modulation (SAM),
and how each MSS component maps onto the engine.

## The brand

Interstate Industries (dba Monroe Products) has sold Hemi-Sync®
recordings since 1985 and has been independent of The Monroe Institute
(TMI) since 2017. TMI kept the research programs and patent applications
and brands its post-2017 sound work as **Monroe Sound Science** — the
Hemi-Sync techniques plus incremental additions (SAM, gamma layering,
3D-audio delivery), not a new physics.

## What MSS names (TMI's own component list)

Binaural beats, monaural beats, isochronic tones, frequency modulation,
amplitude modulation, phase modulation, and Spatial Angle Modulation
(SAM); layered with pink noise, music, and verbal guidance; delivered in
3D-audio formats. The "more than 50 techniques" line is a marketing
count, never enumerated. The gamma framing leans on Orch-OR/microtubule
language — rationale copy, not a signal spec. REBAL is a taught
visualization carried by narration, not an audio component; nothing to
model. **TMI publishes no frequency numbers, no bed spectra, no
per-Focus-level recipes for the MSS era — there is no MSS equivalent of
the Hemi-Sync patents.**

## SAM — the one new mechanism, fully specified and free

Patent applications US 2013/0010967 A1 and US 2015/0016613 A1 (Atwater &
Turner, assignee The Monroe Institute). **Both abandoned, never granted**
— published prior art, freely implementable; only the SAM™ trademark is
live.

Core synthesis, verbatim from the patents:

```
S_L(t) = A·sin[2π·f_s·t + φ_p·sin(2π·f_m·t) + φ_L]
S_R(t) = A·sin[2π·f_s·t − φ_p·sin(2π·f_m·t) + φ_R]
```

Six degrees of freedom: amplitude A, carrier f_s, modulation rate f_m,
peak phase deviation φ_p, per-channel offsets φ_L/φ_R (aim the arc
off-axis). Only concrete numbers in the patents: example carriers 300 and
440 Hz; conventional binaural ceiling cited ~30 Hz; SAM claimed working
at 40–70 Hz; ear separation 15–25 cm. No arc angles, no φ_p values, no
worked examples.

What the equations reduce to:

- **Antiphase sinusoidal phase modulation of one shared carrier** — a
  sinusoidally swept ITD of peak magnitude φ_p/(π·f_s) seconds, plus a
  static ITD bias from (φ_L−φ_R). No ILD, no HRTF, no Doppler: an
  ITD-only phantom source. The antiphase sign is what makes SAM SAM:
  same-sign modulation on both channels is common-mode FM of a diotic
  tone, spatially inert.
- **A binaural beat whose Δf is FM'd around zero**: instantaneous
  frequency difference = 2·φ_p·f_m·cos(2π·f_m·t), mean zero. That is the
  gamma trick — the effective beat rate is f_m (40–70 Hz) while the
  instantaneous Δf stays small, dodging the binaural fusion limit.
- **Carrier ≲700 Hz is forced by physics**: ITD/IPD localization is only
  unambiguous below ~750–1500 Hz, which is why the patents' examples sit
  at 300/440.
- **Arc calibration**: max physical ITD ≈ 0.175 m / 343 m/s ≈ 510 µs;
  `φ_p = π·f_s·(d/c)·sin(arc/2)`, so at f_s = 300 Hz a full ear-to-ear
  arc needs φ_p ≈ 0.4809 rad. Smaller φ_p = narrower arc. Verified by
  complex demodulation of rendered audio: IPD sweep amplitude 0.9617 rad
  against the predicted 2·φ_p = 0.9617, implied ITD 510.2 µs against
  510.2 µs predicted.

**Path types.** The patents name three, qualitatively; the engine
implements them as modulator shapes, all antiphase: `closed` is the
patents' literal `sin Φ` (a closed orbit, sinusoidal ITD projection);
`open` is `sin(θ·sin Φ)/sin θ` (azimuth sinusoidal, ITD following
`sin θ`, so the source decelerates at the turns); `discontinuous` is
`sin(⌊Φ/Δ⌋·Δ)`, a phase-anchored sample-and-hold. Open and closed reach
identical ITD extremes and differ only in the trajectory between them.

**Detection.** SAM survives mono summing: L+R collapses to an AM signal
whose envelope's lowest line is 2·f_m (f_m appears only with an
asymmetric offset) — which is why it works over speakers. A conventional
pair beating at rate R does *not* vanish under the same test: its
rectified mono-sum envelope keeps the fundamental at R with harmonics at
2R, 3R, … (the 2R line only ~14 dB down). The working detector is
therefore the *contrast* between the 2·f_m and f_m envelope lines, not
the 2·f_m line alone: symmetric SAM measures +195 dB contrast on
rendered audio, a conventional 40 Hz pair −4.9 dB. A second, stronger
discriminator on real material: the IPD of a conventional pair is an
unbounded ramp (slope = the beat frequency) while SAM's is bounded and
oscillatory with mean-zero slope; the ramp must be fitted and removed
before measuring the sweep, or a plain binaural beat scores a spurious
IPD sweep amplitude of ~2 rad. Implemented as
`farfield.analysis.sam_signature`.

## Support matrix vs this engine

| Component | Status |
|---|---|
| Binaural beats, stacked pairs, mono anchors, ear polarity | supported |
| Monaural beats | trivial (sum pre-pan into both ears) |
| AM/tremolo, noise beds, crossfade/static stereo | supported |
| Isochronic tones | modest: duty-cycled gate shape on the tremolo |
| Carrier FM | modest: per-group LFO on carrier frequency |
| **SAM** | **implemented** — `sam:` group primitive; see the `sam-gamma` preset |
| Arbitrary spatial paths / 3D | modest: generalize the modulator to any periodic θ(t) via ITD(θ) = (d/c)·sin θ; beyond that is HRTF decoration, not beat structure |
| Gamma layering | f_m = 40–70 Hz in the SAM form (plain binaural Δf at those rates fails to fuse — TMI's own stated reason for SAM) |
| MSS per-program recipes | TMI publishes zero numbers; see `docs/tape-analysis/` for our own measurements of MSS-era releases |

## Sources

- [US 2013/0010967 A1](https://patents.google.com/patent/US20130010967A1/en)
  and [US 2015/0016613 A1](https://patents.google.com/patent/US20150016613A1/en)
  (both abandoned)
- [TMI on Monroe Sound Science](https://www.monroeinstitute.org/pages/monroe-sound-science)
- [TMI UK on Spatial Angle Modulation](https://www.monroeinstituteuk.org/spatial-angle-modulaton-sam/)
