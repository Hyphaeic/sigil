//! Causality enforcement using Hilbert transform.
//!
//! This module provides causality enforcement with optional group delay preservation
//! per IBIS 7.2 Section 6.4.2: "Group delay shall be preserved when enforcing causality."

use crate::error::{DspError, DspResult};
use crate::fft::FftEngine;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Enforce causality on a frequency response using minimum-phase reconstruction.
///
/// A causal system has impulse response h(t) = 0 for t < 0. This is enforced
/// by computing the minimum-phase response from the magnitude spectrum.
pub fn enforce_causality(h: &[Complex64]) -> DspResult<Vec<Complex64>> {
    let n = h.len();
    if !n.is_power_of_two() {
        return Err(DspError::InvalidFftSize(n));
    }

    // Extract magnitudes
    let magnitudes: Vec<f64> = h.iter().map(|c| c.norm().max(1e-15)).collect();

    // Compute log magnitude
    let log_mag: Vec<f64> = magnitudes.iter().map(|m| m.ln()).collect();

    // Compute cepstrum via IFFT
    let mut engine = FftEngine::new();
    let mut cepstrum: Vec<Complex64> = log_mag
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    engine.ifft_inplace(&mut cepstrum)?;

    // Apply causal window for minimum-phase reconstruction:
    // - DC (index 0): keep real, zero imaginary
    // - Positive frequencies (1 to N/2-1): double
    // - Nyquist (index N/2, if N is even): keep real, zero imaginary
    // - Negative frequencies (N/2+1 to N-1): zero

    // DC component: keep real, zero imaginary
    cepstrum[0] = Complex64::new(cepstrum[0].re, 0.0);

    // Positive frequencies: double
    for i in 1..n / 2 {
        cepstrum[i] *= 2.0;
    }

    // Nyquist (if N is even): keep real, zero imaginary
    if n % 2 == 0 && n > 1 {
        cepstrum[n / 2] = Complex64::new(cepstrum[n / 2].re, 0.0);
    }

    // Negative frequencies: zero
    for i in (n / 2 + 1)..n {
        cepstrum[i] = Complex64::new(0.0, 0.0);
    }

    // FFT back to frequency domain
    engine.fft_inplace(&mut cepstrum)?;

    // Exponentiate to get minimum-phase response
    let causal: Vec<Complex64> = cepstrum
        .iter()
        .map(|c| {
            let mag = c.re.exp();
            let phase = c.im;
            Complex64::from_polar(mag, phase)
        })
        .collect();

    Ok(causal)
}

/// Check if a response is approximately causal.
///
/// Returns the ratio of energy before t=0 to total energy.
pub fn causality_metric(impulse: &[f64]) -> f64 {
    let total_energy: f64 = impulse.iter().map(|x| x * x).sum();
    if total_energy == 0.0 {
        return 0.0;
    }

    // Assuming first half is t < 0 for symmetric FFT output
    let n = impulse.len();
    let acausal_energy: f64 = impulse[n / 2..].iter().map(|x| x * x).sum();

    acausal_energy / total_energy
}

/// Unwrap phase in-place to remove 2π discontinuities.
///
/// Phase unwrapping ensures continuity by adding/subtracting 2π when
/// consecutive phase differences exceed π.
fn unwrap_phase_inplace(phases: &mut [f64]) {
    let mut offset = 0.0;
    for i in 1..phases.len() {
        let diff = phases[i] - phases[i - 1] + offset;
        if diff > PI {
            offset -= 2.0 * PI;
        } else if diff < -PI {
            offset += 2.0 * PI;
        }
        phases[i] += offset;
    }
}

/// Compute median of a slice, filtering out non-finite values.
fn median(data: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = data.iter().filter(|x| x.is_finite()).copied().collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted.len();
    if len % 2 == 0 {
        (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
    } else {
        sorted[len / 2]
    }
}

/// Extract reference group delay from frequency-domain data.
///
/// Uses median of group delay values for robustness against noise.
/// Group delay is computed as τ = -dφ/(2π·df) using central differences.
///
/// # Arguments
///
/// * `h` - Frequency-domain transfer function samples
/// * `frequencies` - Frequency values in Hz corresponding to each sample
///
/// # Returns
///
/// The median group delay in seconds.
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 3 data points are provided
/// - Length mismatch between `h` and `frequencies`
pub fn extract_reference_delay(h: &[Complex64], frequencies: &[f64]) -> DspResult<f64> {
    if h.len() < 3 || frequencies.len() < 3 {
        return Err(DspError::InsufficientData {
            needed: 3,
            got: h.len().min(frequencies.len()),
        });
    }
    if h.len() != frequencies.len() {
        return Err(DspError::LengthMismatch {
            expected: frequencies.len(),
            actual: h.len(),
        });
    }

    // Compute unwrapped phase
    let mut phases: Vec<f64> = h.iter().map(|c| c.arg()).collect();
    unwrap_phase_inplace(&mut phases);

    // Compute group delay: τ = -dφ/(2π·df) using central differences
    let mut delays = Vec::with_capacity(frequencies.len());
    for i in 1..frequencies.len() - 1 {
        let df = frequencies[i + 1] - frequencies[i - 1];
        if df.abs() > 1e-10 {
            let dphase = phases[i + 1] - phases[i - 1];
            delays.push(-dphase / (2.0 * PI * df));
        }
    }

    Ok(median(&delays))
}

/// Apply linear phase term to restore group delay.
///
/// Multiplies each frequency bin by exp(-j·2π·f·τ), which corresponds
/// to a time shift of τ seconds in the time domain.
///
/// # Arguments
///
/// * `h` - Frequency-domain samples to modify in-place
/// * `frequencies` - Frequency values in Hz corresponding to each sample
/// * `delay` - Time delay in seconds to apply
pub fn apply_group_delay(h: &mut [Complex64], frequencies: &[f64], delay: f64) {
    for (i, &freq) in frequencies.iter().enumerate() {
        if i < h.len() {
            let phase_shift = -2.0 * PI * freq * delay;
            h[i] *= Complex64::from_polar(1.0, phase_shift);
        }
    }
}

/// Enforce causality with group delay preservation (IBIS 7.2 compliant).
///
/// # Algorithm
///
/// 1. Extract reference group delay τ_ref from original phase
/// 2. Apply minimum-phase reconstruction (Hilbert transform)
/// 3. Re-apply linear phase term exp(-j·2π·f·τ_ref)
///
/// This preserves the physical propagation delay of the channel while
/// still enforcing causality on the response.
///
/// # IBIS 7.2 Reference
///
/// Section 6.4.2: "Group delay shall be preserved when enforcing causality."
///
/// # Arguments
///
/// * `h` - Frequency-domain transfer function (must be power-of-2 length for FFT)
/// * `frequencies` - Frequency values in Hz corresponding to each sample
///
/// # Returns
///
/// A tuple of (causal response, reference delay in seconds).
///
/// # Errors
///
/// Returns an error if causality enforcement fails or insufficient data.
pub fn enforce_causality_with_delay_preservation(
    h: &[Complex64],
    frequencies: &[f64],
) -> DspResult<(Vec<Complex64>, f64)> {
    // Step 1: Extract reference delay from the positive frequency portion
    // We use the first half of frequencies (positive frequencies) for delay extraction
    let half = h.len().min(frequencies.len()) / 2;
    let tau_ref = if half >= 3 {
        extract_reference_delay(&h[..half], &frequencies[..half]).unwrap_or(0.0)
    } else {
        0.0
    };

    // Step 2: Standard minimum-phase reconstruction
    let mut causal = enforce_causality(h)?;

    // Step 3: Re-apply the group delay
    apply_group_delay(&mut causal, frequencies, tau_ref);

    Ok((causal, tau_ref))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causality_enforcement_preserves_magnitude() {
        let n = 64;
        let h: Vec<Complex64> = (0..n)
            .map(|i| {
                let f = i as f64 / n as f64;
                Complex64::from_polar((-f).exp(), -f * std::f64::consts::PI)
            })
            .collect();

        let causal = enforce_causality(&h).unwrap();

        // Magnitudes should be preserved
        for (orig, caus) in h.iter().zip(causal.iter()) {
            assert!((orig.norm() - caus.norm()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_unwrap_phase_inplace() {
        // Phase that wraps from +π to -π
        let mut phases = vec![2.8, 3.0, 3.1, -3.0, -2.8];
        unwrap_phase_inplace(&mut phases);

        // After unwrapping, phase should be monotonically increasing
        for i in 1..phases.len() {
            assert!(
                (phases[i] - phases[i - 1]).abs() < PI,
                "Phase jump too large at index {}: {} -> {}",
                i,
                phases[i - 1],
                phases[i]
            );
        }
    }

    #[test]
    fn test_median_odd_length() {
        let data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        assert!((median(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((median(&data) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_median_with_nan() {
        let data = vec![1.0, f64::NAN, 3.0, 5.0];
        // NaN should be filtered out, median of [1, 3, 5] = 3
        assert!((median(&data) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_extract_reference_delay_known_delay() {
        // Create transfer function with known 1ns delay: H(f) = exp(-j*2π*f*τ)
        // Using smaller delay to ensure phase change per step is well below π
        // for reliable unwrapping (Δφ = -2π * Δf * τ ≈ -0.2π per step)
        let tau = 1e-9; // 1 nanosecond
        let n = 100;
        let f_max = 10e9; // 10 GHz

        let frequencies: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0) * f_max / n as f64).collect();

        let h: Vec<Complex64> = frequencies
            .iter()
            .map(|&f| {
                let phase = -2.0 * PI * f * tau;
                Complex64::from_polar(1.0, phase)
            })
            .collect();

        let extracted_delay = extract_reference_delay(&h, &frequencies).unwrap();

        // Should be within 10% of actual delay
        let error = (extracted_delay - tau).abs() / tau;
        assert!(
            error < 0.1,
            "Extracted delay {} differs from expected {} by {:.1}%",
            extracted_delay,
            tau,
            error * 100.0
        );
    }

    #[test]
    fn test_extract_reference_delay_insufficient_data() {
        let h = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let freqs = vec![1e9, 2e9];

        let result = extract_reference_delay(&h, &freqs);
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_reference_delay_length_mismatch() {
        let h = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let freqs = vec![1e9, 2e9, 3e9, 4e9]; // Length mismatch

        let result = extract_reference_delay(&h, &freqs);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_group_delay() {
        let tau = 1e-9; // 1 ns delay
        let freqs = vec![1e9, 2e9, 3e9];
        let mut h = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        apply_group_delay(&mut h, &freqs, tau);

        // Check that phases are correct: phase = -2π*f*τ
        for (i, &freq) in freqs.iter().enumerate() {
            let expected_phase = -2.0 * PI * freq * tau;
            let actual_phase = h[i].arg();
            // Phases should match (modulo 2π)
            let phase_diff = (expected_phase - actual_phase).abs();
            let normalized_diff = phase_diff - (phase_diff / (2.0 * PI)).round() * 2.0 * PI;
            assert!(
                normalized_diff.abs() < 1e-10,
                "Phase mismatch at freq {}: expected {}, got {}",
                freq,
                expected_phase,
                actual_phase
            );
        }
    }

    #[test]
    fn test_enforce_causality_with_delay_preservation() {
        // Create a 64-point transfer function with known delay
        let n = 64;
        let tau = 2e-9; // 2 ns delay
        let df = 100e6; // 100 MHz spacing

        let frequencies: Vec<f64> = (0..n).map(|i| i as f64 * df).collect();

        let h: Vec<Complex64> = frequencies
            .iter()
            .map(|&f| {
                let loss = (-f * 1e-11).exp();
                let phase = -2.0 * PI * f * tau;
                Complex64::from_polar(loss, phase)
            })
            .collect();

        let (causal, extracted_tau) = enforce_causality_with_delay_preservation(&h, &frequencies)
            .expect("Causality enforcement should succeed");

        // Verify extracted delay is close to original
        let delay_error = (extracted_tau - tau).abs();
        assert!(
            delay_error < tau * 0.2,
            "Extracted delay {} differs too much from expected {}",
            extracted_tau,
            tau
        );

        // Verify magnitudes are preserved
        for (i, (&orig, &caus)) in h.iter().zip(causal.iter()).enumerate() {
            let mag_diff = (orig.norm() - caus.norm()).abs();
            assert!(
                mag_diff < 1e-6,
                "Magnitude mismatch at index {}: original {}, causal {}",
                i,
                orig.norm(),
                caus.norm()
            );
        }

        // Verify result has correct length
        assert_eq!(causal.len(), n);
    }

    #[test]
    fn test_delay_preservation_vs_standard_causality() {
        // Show that standard causality destroys delay while preservation keeps it
        let n = 64;
        let tau = 3e-9; // 3 ns delay
        let df = 100e6;

        let frequencies: Vec<f64> = (0..n).map(|i| i as f64 * df).collect();

        let h: Vec<Complex64> = frequencies
            .iter()
            .map(|&f| {
                let phase = -2.0 * PI * f * tau;
                Complex64::from_polar(1.0, phase)
            })
            .collect();

        // Standard causality enforcement
        let standard = enforce_causality(&h).expect("Standard causality should work");

        // Delay-preserving causality
        let (preserved, _) = enforce_causality_with_delay_preservation(&h, &frequencies)
            .expect("Delay-preserving causality should work");

        // Extract group delay from both results
        let standard_delay = extract_reference_delay(&standard[..n / 2], &frequencies[..n / 2])
            .unwrap_or(0.0);
        let preserved_delay = extract_reference_delay(&preserved[..n / 2], &frequencies[..n / 2])
            .unwrap_or(0.0);

        // Standard causality should have near-zero delay (minimum phase)
        assert!(
            standard_delay.abs() < tau * 0.5,
            "Standard causality should have reduced delay, got {}",
            standard_delay
        );

        // Preserved causality should maintain original delay
        assert!(
            (preserved_delay - tau).abs() < tau * 0.3,
            "Preserved delay {} should be close to original {}",
            preserved_delay,
            tau
        );
    }
}
