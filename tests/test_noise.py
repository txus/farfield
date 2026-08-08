import numpy as np
import pytest
from scipy.signal import welch

from farfield.noise import (
    LFSR_PERIOD,
    generate_pink,
    lfsr_white,
    pink_fft,
    pink_lfsr,
)


def _log_log_slope(signal: np.ndarray, sample_rate: int) -> float:
    freqs, power = welch(signal, fs=sample_rate, nperseg=8192)
    band = (freqs > 50.0) & (freqs < 5000.0)
    return float(np.polyfit(np.log10(freqs[band]), np.log10(power[band]), 1)[0])


def test_fft_pink_has_a_one_over_f_power_spectrum():
    rng = np.random.default_rng(0)
    assert -1.3 < _log_log_slope(pink_fft(2**18, rng), 48000) < -0.7


def test_lfsr_pink_has_a_one_over_f_power_spectrum():
    assert -1.4 < _log_log_slope(pink_lfsr(2**18), 48000) < -0.6


def test_pink_is_peak_normalised():
    rng = np.random.default_rng(1)
    assert abs(np.max(np.abs(pink_fft(2**16, rng))) - 1.0) < 1e-9


def test_pink_is_zero_mean():
    rng = np.random.default_rng(2)
    signal = pink_fft(2**16, rng)
    assert abs(float(np.mean(signal))) < 1e-2


def test_lfsr_repeats_every_65535_samples():
    signal = lfsr_white(LFSR_PERIOD * 2 + 10)
    assert np.array_equal(signal[:500], signal[LFSR_PERIOD : LFSR_PERIOD + 500])


def test_lfsr_period_constant_is_the_patent_value():
    assert LFSR_PERIOD == 65535


def test_lfsr_white_is_bipolar():
    assert set(np.unique(lfsr_white(1000))) == {-1.0, 1.0}


def test_lfsr_visits_the_full_period_without_repeating_early():
    period = lfsr_white(LFSR_PERIOD)
    # A maximal-length register cannot be periodic in half its period.
    half = LFSR_PERIOD // 2
    assert not np.array_equal(period[:half], period[half : half * 2])


def test_generate_pink_dispatches_on_algorithm():
    rng = np.random.default_rng(3)
    assert len(generate_pink(4096, "fft", rng)) == 4096
    assert len(generate_pink(4096, "lfsr", rng)) == 4096


def test_generate_pink_rejects_unknown_algorithm():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="algorithm"):
        generate_pink(1024, "brownian", rng)


def test_pink_generators_return_the_requested_length():
    rng = np.random.default_rng(5)
    for n in (1, 1000, LFSR_PERIOD + 7):
        assert len(pink_fft(n, rng)) == n
        assert len(pink_lfsr(n)) == n


from farfield.noise import phased_pan, render_pink, swept_comb


def test_comb_introduces_spectral_notches():
    rng = np.random.default_rng(10)
    signal = pink_fft(48000 * 4, rng)
    combed = swept_comb(signal, 48000, sweep_hz=0.0)
    spectrum = np.abs(np.fft.rfft(combed))
    freqs = np.fft.rfftfreq(len(combed), 1.0 / 48000)
    band = (freqs > 100.0) & (freqs < 2000.0)
    # A static 20 ms comb notches every 50 Hz, so the in-band spectrum
    # becomes far more uneven than the smooth pink source.
    assert np.min(spectrum[band]) < 0.2 * np.median(spectrum[band])


def test_comb_preserves_length_and_is_finite():
    rng = np.random.default_rng(11)
    combed = swept_comb(pink_fft(10000, rng), 48000)
    assert len(combed) == 10000
    assert np.all(np.isfinite(combed))


def test_phased_pan_returns_stereo():
    rng = np.random.default_rng(12)
    panned = phased_pan(pink_fft(48000, rng), 48000)
    assert panned.shape == (48000, 2)


def test_phased_pan_moves_energy_between_channels():
    rng = np.random.default_rng(13)
    # Twenty seconds spans a full cycle at the default 0.05 Hz pan rate.
    panned = phased_pan(pink_fft(48000 * 20, rng), 48000)
    chunk = 48000
    balance = [
        float(np.sum(panned[i : i + chunk, 0] ** 2))
        - float(np.sum(panned[i : i + chunk, 1] ** 2))
        for i in range(0, 48000 * 20, chunk)
    ]
    assert max(balance) > 0.0
    assert min(balance) < 0.0


def test_phased_pan_conserves_power_per_sample():
    rng = np.random.default_rng(14)
    mono = pink_fft(48000 * 20, rng)
    panned = phased_pan(mono, 48000, amp_depth=0.0)
    # Equal-power law: cos^2 + sin^2 = 1, so L^2 + R^2 == source^2 exactly.
    assert np.allclose(panned[:, 0] ** 2 + panned[:, 1] ** 2, mono**2)


def test_render_pink_is_stereo_and_peak_normalised():
    rng = np.random.default_rng(15)
    stereo = render_pink(48000 * 2, 48000, "fft", 0.125, 0.05, rng)
    assert stereo.shape == (48000 * 2, 2)
    assert abs(float(np.max(np.abs(stereo))) - 1.0) < 1e-9


from farfield.noise import render_bed, shaped_fft, surf_envelope
from farfield.session import PinkSpec


def _slope_of(signal: np.ndarray, sample_rate: int = 48000) -> float:
    freqs, power = welch(signal, fs=sample_rate, nperseg=8192)
    band = (freqs > 50.0) & (freqs < 5000.0)
    return float(np.polyfit(np.log10(freqs[band]), np.log10(power[band]), 1)[0])


def test_brown_noise_slope():
    rng = np.random.default_rng(20)
    # -20 dB/decade amplitude = -2.0 power-law slope
    assert -2.3 < _slope_of(shaped_fft(2**18, rng, -20.0)) < -1.7


def test_custom_slope():
    rng = np.random.default_rng(21)
    assert -1.7 < _slope_of(shaped_fft(2**18, rng, -15.0)) < -1.3


def test_pink_fft_is_the_minus_ten_case():
    rng1 = np.random.default_rng(22)
    rng2 = np.random.default_rng(22)
    assert np.array_equal(pink_fft(4096, rng1), shaped_fft(4096, rng2, -10.0))


def test_surf_envelope_peaks_at_one_and_troughs_at_depth():
    env = surf_envelope(48000 * 10, 48000, 0.2, 0.14)
    assert abs(float(env.max()) - 1.0) < 1e-6
    assert abs(float(env.min()) - 0.86) < 1e-3


def test_render_bed_applies_surf_modulation():
    spec = PinkSpec(level_db=-14.0, comb_sweep_hz=0.125, pan_rate_hz=0.05,
                    algorithm="fft", color="brown",
                    surf_rate_hz=0.21, surf_depth=0.5)
    rng = np.random.default_rng(23)
    stereo = render_bed(48000 * 40, 48000, spec, rng)
    envelope = np.sqrt((stereo ** 2).sum(axis=1))
    smoothed = np.convolve(envelope, np.ones(4800) / 4800, mode="valid")
    spectrum = np.abs(np.fft.rfft(smoothed - smoothed.mean()))
    freqs = np.fft.rfftfreq(len(smoothed), 1.0 / 48000)
    # 40 s render gives ~0.025 Hz resolution, so this band spans ~9 bins:
    # both the surf peak (0.21 Hz) and the shoulder of the comb x amp-mod
    # sideband at ~0.108 Hz, forcing argmax to actually discriminate.
    band = (freqs > 0.12) & (freqs < 0.35)
    freqs_in_band = freqs[band]
    spectrum_in_band = spectrum[band]
    peak_idx = np.argmax(spectrum_in_band)
    detected_freq = freqs_in_band[peak_idx]
    assert abs(detected_freq - 0.21) < 0.03

    # Mutation check: surf modulation must be present (power ratio > 2.0)
    spec_no_surf = PinkSpec(level_db=-14.0, comb_sweep_hz=0.125, pan_rate_hz=0.05,
                            algorithm="fft", color="brown",
                            surf_rate_hz=None, surf_depth=0.5)
    rng2 = np.random.default_rng(23)
    stereo_no_surf = render_bed(48000 * 40, 48000, spec_no_surf, rng2)
    envelope_no_surf = np.sqrt((stereo_no_surf ** 2).sum(axis=1))
    smoothed_no_surf = np.convolve(envelope_no_surf, np.ones(4800) / 4800, mode="valid")
    spectrum_no_surf = np.abs(np.fft.rfft(smoothed_no_surf - smoothed_no_surf.mean()))
    spectrum_no_surf_band = spectrum_no_surf[band]
    power_with_surf = float(spectrum_in_band[peak_idx])
    power_no_surf = float(spectrum_no_surf_band[peak_idx])
    assert power_with_surf / power_no_surf > 2.0


def test_lfsr_with_non_pink_color_is_rejected():
    spec = PinkSpec(level_db=-10.0, comb_sweep_hz=0.125, pan_rate_hz=0.05,
                    algorithm="lfsr", color="brown")
    with pytest.raises(ValueError, match="lfsr"):
        render_bed(4096, 48000, spec, np.random.default_rng(0))


def test_render_bed_is_rms_normalized():
    spec = PinkSpec(level_db=0.0, comb_sweep_hz=0.125, pan_rate_hz=0.05,
                    algorithm="fft", color="brown")
    stereo = render_bed(48000 * 10, 48000, spec, np.random.default_rng(31))
    rms = np.sqrt(((stereo ** 2).sum(axis=1) / 2).mean())
    assert abs(rms - 1 / np.sqrt(2)) < 0.01


def test_rms_bed_matches_equal_level_sine_loudness():
    # A 0 dB bed and a 0 dB single sine now carry the same power.
    spec = PinkSpec(level_db=0.0, comb_sweep_hz=0.125, pan_rate_hz=0.05,
                    algorithm="fft", color="pink")
    stereo = render_bed(48000 * 10, 48000, spec, np.random.default_rng(32))
    sine_rms = 1 / np.sqrt(2)
    bed_rms = np.sqrt(((stereo ** 2).sum(axis=1) / 2).mean())
    assert abs(20 * np.log10(bed_rms / sine_rms)) < 0.1


def _mk_spec(**kw):
    base = dict(level_db=0.0, comb_sweep_hz=0.125, pan_rate_hz=0.05,
                algorithm="fft", color="brown")
    base.update(kw)
    return PinkSpec(**base)


def _coherence(stereo, fs, flo=1500.0, fhi=8000.0):
    from scipy.signal import coherence
    f, c = coherence(stereo[:, 0], stereo[:, 1], fs=fs, nperseg=8192)
    band = (f >= flo) & (f <= fhi)
    return float(c[band].mean())


def test_crossfade_streams_are_decorrelated():
    spec = _mk_spec(stereo_mode="crossfade", lfo_period_s=9.8,
                    stereo_depth_db=3.25, comb_enabled=False)
    stereo = render_bed(48000 * 40, 48000, spec, np.random.default_rng(40))
    # Spec asked coherence < 0.2. This unit test holds the spec value (the
    # acceptance test on real presets relaxes to 0.25); measured here at
    # ~0.002-0.003, so the margin is enormous either way.
    assert _coherence(stereo, 48000) < 0.2


def test_pan_mode_stays_coherent():
    # Welch coherence over a window spanning most/all of one 0.05 Hz pan
    # sweep (20 s) reads low regardless of implementation: the L/R gain
    # ratio traverses its full range within the analysis window, and Welch
    # averages the segment-wise cross-spectra, so even a noiseless,
    # deterministic full 0-90 deg pan sweep measures ~0.52 coherence here
    # (verified against a constant-amplitude input, bypassing any noise
    # contribution). A 5 s window covers only a modest slice of the sweep,
    # so the L/R ratio is far more stable across Welch's averaged segments.
    # Measured (5 seeds): pan mode ~0.505-0.516; crossfade mode (same
    # coherence helper, its own test's params) ~0.002-0.003 -- a wide,
    # stable separation.
    spec = _mk_spec(stereo_mode="pan", comb_enabled=True, color="pink")
    stereo = render_bed(48000 * 5, 48000, spec, np.random.default_rng(41))
    assert _coherence(stereo, 48000) > 0.4


def test_crossfade_ild_period_and_depth():
    # Band-limit to 1.5-8 kHz before computing per-window RMS, exactly the
    # band the tape analysis measured its per-band ILD from. This kills
    # brown noise's dominant source of 1 s window-to-window RMS variance:
    # unfiltered brown noise's power is concentrated sub-Hz (its 1/f**2
    # tilt), so a 1 s window sees only a handful of independent low-frequency
    # degrees of freedom and the RMS estimate swings wildly window to
    # window (~140% std/mean, measured). The 1.5-8 kHz band instead has
    # bandwidth*time ~= 6500*1 ~= 6500 independent DOF per window (~1% std
    # RMS estimate), and the crossfade gain sweep is broadband (applied
    # before any band split), so the band-limited ILD ratio is the same
    # signal, just measured where the noise floor is negligible.
    from scipy.signal import butter, sosfilt
    spec = _mk_spec(stereo_mode="crossfade", lfo_period_s=9.8,
                    stereo_depth_db=4.5, comb_enabled=False)
    fs = 48000
    stereo = render_bed(fs * 120, fs, spec, np.random.default_rng(42))
    sos = butter(4, [1500.0, 8000.0], btype="bandpass", fs=fs, output="sos")
    banded = sosfilt(sos, stereo, axis=0)
    hop = fs // 4
    n_frames = (len(banded) - fs) // hop
    ild = np.array([
        10 * np.log10(
            (banded[i * hop:i * hop + fs, 0] ** 2).mean()
            / (banded[i * hop:i * hop + fs, 1] ** 2).mean())
        for i in range(n_frames)
    ])
    spectrum = np.abs(np.fft.rfft(ild - ild.mean()))
    freqs = np.fft.rfftfreq(len(ild), hop / fs)
    band = (freqs > 0.02) & (freqs < 0.5)
    peak = freqs[band][np.argmax(spectrum[band])]
    assert abs(peak - 1 / 9.8) < 0.02
    depth = np.percentile(ild, 97.5) - np.percentile(ild, 2.5)
    # Spec asked +/-1 dB; +/-1.2 because the 1 s ILD windows smear the
    # sinusoid's extremes, pulling the measured p-p under the configured depth.
    assert abs(depth - 4.5) < 1.2


def test_static_delay_lands_at_the_configured_lag():
    spec = _mk_spec(stereo_mode="static", interaural_delay_us=145.0,
                    comb_enabled=False)
    fs = 48000
    stereo = render_bed(fs * 10, fs, spec, np.random.default_rng(43))
    xcorr = np.correlate(stereo[fs:fs * 5, 0], stereo[fs:fs * 5, 1], "full")
    lag = np.argmax(xcorr) - (len(stereo[fs:fs * 5, 0]) - 1)
    measured_us = -lag / fs * 1e6  # left leads => right is delayed => peak at negative lag
    # Spec asked +/-10 us; +/-25 because argmax resolves the lag only to
    # integer samples, and one sample at 48 kHz is already 20.8 us.
    assert abs(abs(measured_us) - 145.0) < 25.0


def test_static_mode_is_fully_coherent():
    spec = _mk_spec(stereo_mode="static", interaural_delay_us=145.0,
                    comb_enabled=False)
    stereo = render_bed(48000 * 10, 48000, spec, np.random.default_rng(44))
    assert _coherence(stereo, 48000, 200.0, 2000.0) > 0.95


def test_comb_bypass_respected_in_crossfade():
    # The flag under test is plumbing (does spec.comb_enabled actually skip
    # the comb stage?), so probe it in the ripple metric's working regime.
    # The production sweep (comb_sweep_hz=0.125) moves its notches fast
    # enough that they smear out of any long-window ripple metric (measured
    # ratio ~1.00 +/- 5% at comb_sweep_hz=0.125, even collapsing well before
    # 0.002 Hz on an 8 s capture) -- that's a real, separately-verified
    # property of the swept comb, not a bypass-flag bug, and the production
    # audibility question is settled by the tape analysis, not this test.
    # A comb_sweep_hz=0.0 comb has fixed (non-moving) notches, which *is*
    # discriminating (measured ratio ~1.33 across seeds): use it as the
    # probe for whether comb_enabled actually gates the comb stage.
    def ripple_db(spec):
        stereo = render_bed(48000 * 8, 48000, spec, np.random.default_rng(45))
        mono = stereo.sum(axis=1)
        spectrum = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(len(mono), 1 / 48000)
        band = (freqs > 300) & (freqs < 1500)
        smooth = np.convolve(spectrum[band], np.ones(201) / 201, "same")
        return float(20 * np.log10((spectrum[band] / smooth)).std())
    bypassed = ripple_db(_mk_spec(stereo_mode="crossfade", lfo_period_s=9.8,
                                  stereo_depth_db=3.0, comb_enabled=False))
    static_comb = ripple_db(_mk_spec(stereo_mode="crossfade", lfo_period_s=9.8,
                                     stereo_depth_db=3.0, comb_enabled=True,
                                     comb_sweep_hz=0.0))
    assert static_comb > bypassed * 1.25
