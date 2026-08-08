import numpy as np

from farfield.oscillators import TWO_PI, frequency_ramp, render_tone


def test_constant_tone_peaks_at_its_frequency():
    sample_rate = 48000
    samples, _ = render_tone(440.0, sample_rate, sample_rate)
    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(sample_rate, 1.0 / sample_rate)
    assert abs(freqs[np.argmax(spectrum)] - 440.0) < 1.0


def test_chained_render_matches_a_single_long_render():
    sample_rate = 48000
    first, phase = render_tone(200.0, 1000, sample_rate)
    second, _ = render_tone(200.0, 1000, sample_rate, initial_phase=phase)
    whole, _ = render_tone(200.0, 2000, sample_rate)
    assert np.allclose(np.concatenate([first, second]), whole, atol=1e-9)


def test_final_phase_is_wrapped():
    _, phase = render_tone(440.0, 48000, 48000)
    assert 0.0 <= phase < TWO_PI


def test_first_sample_uses_the_initial_phase():
    samples, _ = render_tone(100.0, 16, 48000, initial_phase=np.pi / 2)
    assert abs(samples[0] - 1.0) < 1e-12


def test_amplitude_scales_the_output():
    samples, _ = render_tone(100.0, 4800, 48000, amplitude=0.25)
    assert abs(np.max(np.abs(samples)) - 0.25) < 1e-3


def test_frequency_ramp_spans_the_requested_range():
    ramp = frequency_ramp(10.0, 4.0, 5)
    assert ramp[0] == 10.0
    assert ramp[-1] == 4.0
    assert len(ramp) == 5


def test_glide_sweeps_the_instantaneous_frequency():
    sample_rate = 48000
    n = sample_rate * 2
    samples, _ = render_tone(frequency_ramp(100.0, 300.0, n), n, sample_rate)
    head = np.abs(np.fft.rfft(samples[: sample_rate // 4]))
    tail = np.abs(np.fft.rfft(samples[-sample_rate // 4 :]))
    freqs = np.fft.rfftfreq(sample_rate // 4, 1.0 / sample_rate)
    assert freqs[np.argmax(head)] < 150.0
    assert freqs[np.argmax(tail)] > 250.0
