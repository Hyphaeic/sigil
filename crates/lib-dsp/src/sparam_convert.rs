//! S-parameter to time-domain conversion.

use crate::causality::enforce_causality;
use crate::error::{DspError, DspResult};
use crate::fft::FftEngine;
use crate::interpolation::{interpolate_linear, uniform_frequency_grid};
use crate::passivity::enforce_passivity;
use lib_types::sparams::SParameters;
use lib_types::units::{Hertz, Seconds};
use lib_types::waveform::Waveform;
use num_complex::Complex64;

/// Configuration for S-parameter to time-domain conversion.
#[derive(Clone, Debug)]
pub struct ConversionConfig {
    /// Number of FFT points (must be power of 2).
    pub num_fft_points: usize,

    /// Input port index (0-based).
    pub input_port: usize,

    /// Output port index (0-based).
    pub output_port: usize,

    /// Bit time (UI) for pulse response generation.
    pub bit_time: Seconds,

    /// Enforce causality on the response.
    pub enforce_causality: bool,

    /// Enforce passivity on S-parameters.
    pub enforce_passivity: bool,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            num_fft_points: 8192,
            input_port: 0,
            output_port: 1,
            bit_time: Seconds::from_ps(31.25), // PCIe Gen 5
            enforce_causality: true,
            enforce_passivity: true,
        }
    }
}

/// Convert S-parameters to impulse response.
pub fn sparam_to_impulse(
    sparams: &SParameters,
    config: &ConversionConfig,
) -> DspResult<Waveform> {
    if sparams.is_empty() {
        return Err(DspError::InsufficientData { needed: 2, got: 0 });
    }

    // Get the transfer function
    let mut transfer = sparams.get_parameter(config.output_port, config.input_port);

    // Enforce passivity if requested
    let mut sparams_copy = sparams.clone();
    if config.enforce_passivity {
        enforce_passivity(&mut sparams_copy)?;
        transfer = sparams_copy.get_parameter(config.output_port, config.input_port);
    }

    // Interpolate to uniform frequency grid
    let (f_min, f_max) = sparams.frequency_range().ok_or(DspError::InsufficientData {
        needed: 2,
        got: sparams.len(),
    })?;

    let target_freqs = uniform_frequency_grid(f_min, f_max, config.num_fft_points / 2);
    let interpolated = interpolate_linear(&sparams.frequencies, &transfer, &target_freqs)?;

    // Enforce causality if requested
    let causal = if config.enforce_causality {
        // Pad to full FFT size with Hermitian symmetry
        let mut full_spectrum = vec![Complex64::new(0.0, 0.0); config.num_fft_points];
        for (i, &val) in interpolated.iter().enumerate() {
            full_spectrum[i] = val;
        }
        // Mirror for Hermitian symmetry
        for i in 1..config.num_fft_points / 2 {
            full_spectrum[config.num_fft_points - i] = full_spectrum[i].conj();
        }

        enforce_causality(&full_spectrum)?
    } else {
        // Just mirror for IFFT
        let mut full_spectrum = vec![Complex64::new(0.0, 0.0); config.num_fft_points];
        for (i, &val) in interpolated.iter().enumerate() {
            full_spectrum[i] = val;
        }
        for i in 1..config.num_fft_points / 2 {
            full_spectrum[config.num_fft_points - i] = full_spectrum[i].conj();
        }
        full_spectrum
    };

    // IFFT to time domain
    let mut engine = FftEngine::new();
    let impulse_complex = engine.ifft(&causal)?;

    // Extract real part
    let samples: Vec<f64> = impulse_complex.iter().map(|c| c.re).collect();

    // Compute time step from frequency range
    let df = (f_max.0 - f_min.0) / (config.num_fft_points / 2 - 1) as f64;
    let dt = Seconds(1.0 / (config.num_fft_points as f64 * df));

    Ok(Waveform {
        samples,
        dt,
        t_start: Seconds::ZERO,
    })
}

/// Convert impulse response to pulse response.
///
/// Integrates the impulse to get step response, then convolves with
/// a rectangular pulse of duration `bit_time`.
pub fn impulse_to_pulse(impulse: &Waveform, bit_time: Seconds) -> Waveform {
    let samples_per_bit = (bit_time.0 / impulse.dt.0).round() as usize;

    // Integrate impulse to get step response
    let mut step: Vec<f64> = Vec::with_capacity(impulse.samples.len());
    let mut cumsum = 0.0;
    for &sample in &impulse.samples {
        cumsum += sample * impulse.dt.0;
        step.push(cumsum);
    }

    // Pulse response = step(t) - step(t - bit_time)
    let mut pulse = vec![0.0; impulse.samples.len()];
    for i in 0..impulse.samples.len() {
        let step_now = step[i];
        let step_delayed = if i >= samples_per_bit {
            step[i - samples_per_bit]
        } else {
            0.0
        };
        pulse[i] = step_now - step_delayed;
    }

    Waveform {
        samples: pulse,
        dt: impulse.dt,
        t_start: impulse.t_start,
    }
}

/// Full conversion from S-parameters to pulse response.
pub fn sparam_to_pulse(
    sparams: &SParameters,
    config: &ConversionConfig,
) -> DspResult<Waveform> {
    let impulse = sparam_to_impulse(sparams, config)?;
    Ok(impulse_to_pulse(&impulse, config.bit_time))
}

/// Compute group delay from S-parameters.
pub fn compute_group_delay(sparams: &SParameters, port_out: usize, port_in: usize) -> Vec<f64> {
    sparams.group_delay(port_out, port_in)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_sparams() -> SParameters {
        let mut sp = SParameters::new(2, lib_types::units::Ohms::Z0_50);

        // Create a simple lossy line model
        for i in 0..100 {
            let f = (i as f64 + 1.0) * 1e8; // 100 MHz to 10 GHz
            let loss = (-f * 1e-11).exp(); // Exponential loss
            let phase = -f * 1e-10; // Linear phase (delay)

            let mut m = Array2::zeros((2, 2));
            m[[0, 0]] = Complex64::new(0.1, 0.0);
            m[[1, 0]] = Complex64::from_polar(loss, phase);
            m[[0, 1]] = Complex64::from_polar(loss, phase);
            m[[1, 1]] = Complex64::new(0.1, 0.0);

            sp.add_point(Hertz(f), m);
        }

        sp
    }

    #[test]
    fn test_sparam_to_impulse() {
        let sp = create_test_sparams();
        let config = ConversionConfig {
            num_fft_points: 1024,
            ..Default::default()
        };

        let impulse = sparam_to_impulse(&sp, &config).unwrap();

        assert!(!impulse.samples.is_empty());
        assert!(impulse.dt.0 > 0.0);
    }

    #[test]
    fn test_impulse_to_pulse() {
        // Create a simple impulse
        let mut samples = vec![0.0; 1000];
        samples[100] = 1.0;

        let impulse = Waveform::new(samples, Seconds::from_ps(1.0), Seconds::ZERO);
        let pulse = impulse_to_pulse(&impulse, Seconds::from_ps(50.0));

        // Pulse should have width of ~50 samples
        let nonzero: Vec<_> = pulse
            .samples
            .iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > 0.01)
            .collect();

        assert!(!nonzero.is_empty());
    }
}
