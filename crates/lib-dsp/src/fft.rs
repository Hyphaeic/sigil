//! FFT/IFFT operations using rustfft.
//!
//! This module provides a high-level wrapper around rustfft with:
//! - Planner caching for repeated transforms
//! - Real-to-complex and complex-to-real transforms
//! - Convenience methods for common operations

use crate::error::{DspError, DspResult};
use num_complex::Complex64;
use realfft::{RealFftPlanner, RealToComplex, ComplexToReal};
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// FFT engine with cached planners.
pub struct FftEngine {
    /// Complex FFT planner.
    complex_planner: FftPlanner<f64>,

    /// Real FFT planner.
    real_planner: RealFftPlanner<f64>,
}

impl FftEngine {
    /// Create a new FFT engine.
    pub fn new() -> Self {
        Self {
            complex_planner: FftPlanner::new(),
            real_planner: RealFftPlanner::new(),
        }
    }

    /// Perform forward FFT on complex data in-place.
    pub fn fft_inplace(&mut self, data: &mut [Complex64]) -> DspResult<()> {
        let len = data.len();
        if !len.is_power_of_two() {
            return Err(DspError::InvalidFftSize(len));
        }

        let fft = self.complex_planner.plan_fft_forward(len);
        fft.process(data);
        Ok(())
    }

    /// Perform inverse FFT on complex data in-place.
    pub fn ifft_inplace(&mut self, data: &mut [Complex64]) -> DspResult<()> {
        let len = data.len();
        if !len.is_power_of_two() {
            return Err(DspError::InvalidFftSize(len));
        }

        let fft = self.complex_planner.plan_fft_inverse(len);
        fft.process(data);

        // Normalize
        let scale = 1.0 / len as f64;
        for x in data.iter_mut() {
            *x *= scale;
        }

        Ok(())
    }

    /// Perform forward FFT on complex data, returning new buffer.
    pub fn fft(&mut self, data: &[Complex64]) -> DspResult<Vec<Complex64>> {
        let mut result = data.to_vec();
        self.fft_inplace(&mut result)?;
        Ok(result)
    }

    /// Perform inverse FFT on complex data, returning new buffer.
    pub fn ifft(&mut self, data: &[Complex64]) -> DspResult<Vec<Complex64>> {
        let mut result = data.to_vec();
        self.ifft_inplace(&mut result)?;
        Ok(result)
    }

    /// Perform forward real-to-complex FFT.
    ///
    /// Input: N real samples
    /// Output: N/2 + 1 complex samples (Hermitian symmetry exploited)
    pub fn rfft(&mut self, data: &[f64]) -> DspResult<Vec<Complex64>> {
        let len = data.len();
        if !len.is_power_of_two() {
            return Err(DspError::InvalidFftSize(len));
        }

        let r2c = self.real_planner.plan_fft_forward(len);
        let mut input = data.to_vec();
        let mut output = r2c.make_output_vec();

        r2c.process(&mut input, &mut output)
            .map_err(|e| DspError::NumericalInstability(e.to_string()))?;

        Ok(output)
    }

    /// Perform inverse complex-to-real FFT.
    ///
    /// Input: N/2 + 1 complex samples
    /// Output: N real samples
    pub fn irfft(&mut self, data: &[Complex64], output_len: usize) -> DspResult<Vec<f64>> {
        if !output_len.is_power_of_two() {
            return Err(DspError::InvalidFftSize(output_len));
        }

        let expected_input_len = output_len / 2 + 1;
        if data.len() != expected_input_len {
            return Err(DspError::LengthMismatch {
                expected: expected_input_len,
                actual: data.len(),
            });
        }

        let c2r = self.real_planner.plan_fft_inverse(output_len);
        let mut input = data.to_vec();
        let mut output = c2r.make_output_vec();

        c2r.process(&mut input, &mut output)
            .map_err(|e| DspError::NumericalInstability(e.to_string()))?;

        // Normalize
        let scale = 1.0 / output_len as f64;
        for x in output.iter_mut() {
            *x *= scale;
        }

        Ok(output)
    }

    /// Get a cached forward FFT plan.
    pub fn get_fft_forward(&mut self, len: usize) -> Arc<dyn Fft<f64>> {
        self.complex_planner.plan_fft_forward(len)
    }

    /// Get a cached inverse FFT plan.
    pub fn get_fft_inverse(&mut self, len: usize) -> Arc<dyn Fft<f64>> {
        self.complex_planner.plan_fft_inverse(len)
    }
}

impl Default for FftEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the power spectrum (magnitude squared) of a signal.
pub fn power_spectrum(signal: &[f64]) -> DspResult<Vec<f64>> {
    let mut engine = FftEngine::new();
    let spectrum = engine.rfft(signal)?;
    Ok(spectrum.iter().map(|c| c.norm_sqr()).collect())
}

/// Compute the magnitude spectrum of a signal.
pub fn magnitude_spectrum(signal: &[f64]) -> DspResult<Vec<f64>> {
    let mut engine = FftEngine::new();
    let spectrum = engine.rfft(signal)?;
    Ok(spectrum.iter().map(|c| c.norm()).collect())
}

/// Compute the phase spectrum of a signal (in radians).
pub fn phase_spectrum(signal: &[f64]) -> DspResult<Vec<f64>> {
    let mut engine = FftEngine::new();
    let spectrum = engine.rfft(signal)?;
    Ok(spectrum.iter().map(|c| c.arg()).collect())
}

/// Zero-pad a signal to the next power of 2.
pub fn zero_pad_to_pow2(signal: &[f64]) -> Vec<f64> {
    let len = signal.len();
    let new_len = len.next_power_of_two();

    if new_len == len {
        signal.to_vec()
    } else {
        let mut result = signal.to_vec();
        result.resize(new_len, 0.0);
        result
    }
}

/// Zero-pad a signal to a specific length.
pub fn zero_pad(signal: &[f64], new_len: usize) -> Vec<f64> {
    let mut result = signal.to_vec();
    if new_len > signal.len() {
        result.resize(new_len, 0.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_fft_ifft_roundtrip() {
        let mut engine = FftEngine::new();

        // Create a simple signal
        let n = 64;
        let signal: Vec<Complex64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                Complex64::new((2.0 * PI * 4.0 * t).sin(), 0.0)
            })
            .collect();

        let spectrum = engine.fft(&signal).unwrap();
        let recovered = engine.ifft(&spectrum).unwrap();

        // Check roundtrip accuracy
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!((orig.re - rec.re).abs() < 1e-10);
            assert!((orig.im - rec.im).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip() {
        let mut engine = FftEngine::new();

        // Create a simple signal
        let n = 64;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 4.0 * t).sin()
            })
            .collect();

        let spectrum = engine.rfft(&signal).unwrap();
        let recovered = engine.irfft(&spectrum, n).unwrap();

        // Check roundtrip accuracy
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!((orig - rec).abs() < 1e-10);
        }
    }

    #[test]
    fn test_invalid_fft_size() {
        let mut engine = FftEngine::new();
        let data: Vec<Complex64> = vec![Complex64::new(1.0, 0.0); 100]; // Not power of 2

        let result = engine.fft(&data);
        assert!(matches!(result, Err(DspError::InvalidFftSize(100))));
    }
}
