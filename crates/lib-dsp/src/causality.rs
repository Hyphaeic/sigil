//! Causality enforcement using Hilbert transform.

use crate::error::{DspError, DspResult};
use crate::fft::FftEngine;
use num_complex::Complex64;

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
}
