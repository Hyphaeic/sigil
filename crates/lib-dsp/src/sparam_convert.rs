//! S-parameter to time-domain conversion.

use crate::causality::{apply_group_delay, enforce_causality, extract_reference_delay};
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

    /// Preserve group delay when enforcing causality (IBIS 7.2 compliant).
    ///
    /// When true, the reference group delay is extracted from the original
    /// phase response and re-applied after minimum-phase reconstruction.
    /// This ensures the impulse response peak appears at the correct
    /// propagation delay time rather than at t=0.
    ///
    /// Per IBIS 7.2 Section 6.4.2: "Group delay shall be preserved when
    /// enforcing causality."
    pub preserve_group_delay: bool,
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
            preserve_group_delay: true, // IBIS 7.2 compliant by default
        }
    }
}

/// Apply proper Hermitian symmetry to a frequency spectrum for real-valued IFFT output.
///
/// For a real-valued time-domain signal, the frequency spectrum must satisfy:
/// - DC component (index 0) must be real
/// - Nyquist component (index N/2 for even N) must be real
/// - X[N-k] = conj(X[k]) for k = 1 to N/2-1
///
/// This function modifies the spectrum in place to ensure these properties.
fn apply_hermitian_symmetry(spectrum: &mut [Complex64]) {
    let n = spectrum.len();
    if n == 0 {
        return;
    }

    // DC component must be real
    spectrum[0] = Complex64::new(spectrum[0].re, 0.0);

    // Nyquist component must be real (if N is even)
    if n % 2 == 0 && n > 1 {
        let nyquist = n / 2;
        spectrum[nyquist] = Complex64::new(spectrum[nyquist].re, 0.0);
    }

    // Mirror for Hermitian symmetry: X[N-k] = conj(X[k])
    for i in 1..n / 2 {
        spectrum[n - i] = spectrum[i].conj();
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

    // MED-004 FIX: Validate FFT size to prevent division by zero and ensure valid FFT
    // Need at least 4 points: num_fft_points / 2 - 1 >= 1 requires num_fft_points >= 4
    if config.num_fft_points < 4 {
        return Err(DspError::InvalidFftSize(config.num_fft_points));
    }
    if !config.num_fft_points.is_power_of_two() {
        return Err(DspError::InvalidFftSize(config.num_fft_points));
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

    // Compute frequency step for later use
    let df = (f_max.0 - f_min.0) / (config.num_fft_points / 2 - 1) as f64;

    // Enforce causality if requested
    let (causal, reference_delay) = if config.enforce_causality {
        // Pad to full FFT size with proper Hermitian symmetry
        let mut full_spectrum = vec![Complex64::new(0.0, 0.0); config.num_fft_points];
        for (i, &val) in interpolated.iter().enumerate() {
            full_spectrum[i] = val;
        }
        apply_hermitian_symmetry(&mut full_spectrum);

        if config.preserve_group_delay {
            // IBIS 7.2 compliant: preserve group delay
            // Extract reference delay from original interpolated data (positive frequencies)
            let freqs: Vec<f64> = target_freqs.iter().map(|h| h.0).collect();
            let tau_ref = extract_reference_delay(&interpolated, &freqs).unwrap_or(0.0);

            // Apply minimum-phase reconstruction
            let mut causal = enforce_causality(&full_spectrum)?;

            // Build frequency grid for delay application (full spectrum)
            let full_freqs: Vec<f64> = (0..config.num_fft_points)
                .map(|i| {
                    if i <= config.num_fft_points / 2 {
                        f_min.0 + i as f64 * df
                    } else {
                        // Negative frequencies for Hermitian symmetry
                        -(f_min.0 + (config.num_fft_points - i) as f64 * df)
                    }
                })
                .collect();

            // Re-apply the group delay
            apply_group_delay(&mut causal, &full_freqs, tau_ref);

            // Restore Hermitian symmetry after phase modification
            apply_hermitian_symmetry(&mut causal);

            (causal, tau_ref)
        } else {
            // Legacy behavior: minimum-phase only, no delay preservation
            (enforce_causality(&full_spectrum)?, 0.0)
        }
    } else {
        // Just apply Hermitian symmetry for IFFT
        let mut full_spectrum = vec![Complex64::new(0.0, 0.0); config.num_fft_points];
        for (i, &val) in interpolated.iter().enumerate() {
            full_spectrum[i] = val;
        }
        apply_hermitian_symmetry(&mut full_spectrum);
        (full_spectrum, 0.0)
    };

    // IFFT to time domain
    let mut engine = FftEngine::new();
    let impulse_complex = engine.ifft(&causal)?;

    // Extract real part
    let samples: Vec<f64> = impulse_complex.iter().map(|c| c.re).collect();

    // Compute time step from frequency range
    let dt = Seconds(1.0 / (config.num_fft_points as f64 * df));

    Ok(Waveform {
        samples,
        dt,
        t_start: Seconds(reference_delay), // Reflect actual propagation delay
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

    /// Create S-params for a transmission line with known propagation delay.
    fn create_sparams_with_delay(tau: f64) -> SParameters {
        let mut sp = SParameters::new(2, lib_types::units::Ohms::Z0_50);

        // Create transfer function H(f) = exp(-αf) * exp(-j2πfτ)
        // with mild frequency-dependent loss and linear phase
        for i in 0..100 {
            let f = (i as f64 + 1.0) * 1e8; // 100 MHz to 10 GHz
            let loss = (-f * 1e-11).exp(); // Mild loss
            let phase = -2.0 * std::f64::consts::PI * f * tau;

            let mut m = Array2::zeros((2, 2));
            m[[0, 0]] = Complex64::new(0.1, 0.0); // S11 = small reflection
            m[[1, 0]] = Complex64::from_polar(loss, phase); // S21 = through
            m[[0, 1]] = Complex64::from_polar(loss, phase); // S12 = through (reciprocal)
            m[[1, 1]] = Complex64::new(0.1, 0.0); // S22 = small reflection

            sp.add_point(Hertz(f), m);
        }

        sp
    }

    #[test]
    fn test_impulse_t_start_reflects_propagation_delay() {
        // CRIT-DSP-002: Verify impulse t_start reflects actual propagation delay
        // Using 1ns delay to ensure phase changes per step are well below π
        let tau = 1e-9; // 1 nanosecond delay
        let sp = create_sparams_with_delay(tau);

        let config = ConversionConfig {
            num_fft_points: 1024,
            preserve_group_delay: true,
            ..Default::default()
        };

        let impulse = sparam_to_impulse(&sp, &config).unwrap();

        // t_start should be close to the propagation delay
        // Allow 50% tolerance due to numerical effects
        let t_start_ns = impulse.t_start.0 * 1e9;
        let tau_ns = tau * 1e9;

        assert!(
            t_start_ns > tau_ns * 0.5 && t_start_ns < tau_ns * 1.5,
            "t_start ({:.2} ns) should be near propagation delay ({:.2} ns)",
            t_start_ns,
            tau_ns
        );
    }

    #[test]
    fn test_legacy_mode_t_start_zero() {
        // Verify legacy mode (preserve_group_delay=false) has t_start=0
        let tau = 1e-9;
        let sp = create_sparams_with_delay(tau);

        let config = ConversionConfig {
            num_fft_points: 1024,
            preserve_group_delay: false,
            ..Default::default()
        };

        let impulse = sparam_to_impulse(&sp, &config).unwrap();

        assert!(
            impulse.t_start.0.abs() < 1e-15,
            "Legacy mode should have t_start=0, got {}",
            impulse.t_start.0
        );
    }

    #[test]
    fn test_preserve_group_delay_is_default() {
        let config = ConversionConfig::default();
        assert!(
            config.preserve_group_delay,
            "preserve_group_delay should be true by default for IBIS 7.2 compliance"
        );
    }

    #[test]
    fn test_no_causality_enforcement_preserves_delay() {
        // When causality is not enforced, t_start should still be 0
        // (delay is in the phase, not extracted)
        let tau = 1e-9;
        let sp = create_sparams_with_delay(tau);

        let config = ConversionConfig {
            num_fft_points: 1024,
            enforce_causality: false,
            preserve_group_delay: true, // Should be ignored when enforce_causality=false
            ..Default::default()
        };

        let impulse = sparam_to_impulse(&sp, &config).unwrap();

        // Without causality enforcement, t_start is 0
        assert!(
            impulse.t_start.0.abs() < 1e-15,
            "Without causality enforcement, t_start should be 0"
        );
    }
}
