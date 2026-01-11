//! High-performance convolution using overlap-save FFT method.
//!
//! This module provides efficient convolution for large bitstreams,
//! using parallel processing with Rayon.

use crate::error::{DspError, DspResult};
use crate::fft::FftEngine;
use lib_types::waveform::Waveform;
use lib_types::units::Seconds;
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::Fft;
use std::sync::Arc;

/// High-performance convolution engine.
///
/// Uses the overlap-save method with pre-computed impulse FFT
/// for efficient processing of long signals.
pub struct ConvolutionEngine {
    /// Pre-computed FFT of impulse response.
    impulse_fft: Vec<Complex64>,

    /// FFT size (power of 2).
    fft_size: usize,

    /// Overlap size (impulse length - 1).
    overlap: usize,

    /// Valid output size per chunk.
    valid_size: usize,

    /// Original impulse length.
    impulse_len: usize,

    /// Cached FFT plans.
    fft_forward: Arc<dyn Fft<f64>>,
    fft_inverse: Arc<dyn Fft<f64>>,
}

impl ConvolutionEngine {
    /// Create a convolution engine for a given impulse response.
    ///
    /// # Arguments
    ///
    /// * `impulse` - The impulse response to convolve with
    ///
    /// # FFT Size Selection
    ///
    /// The FFT size is chosen to balance:
    /// - Efficiency (larger is better for FFT)
    /// - Memory (smaller uses less RAM)
    /// - Parallelism (more chunks = more parallel work)
    pub fn new(impulse: &[f64]) -> DspResult<Self> {
        let impulse_len = impulse.len();
        if impulse_len == 0 {
            return Err(DspError::InsufficientData { needed: 1, got: 0 });
        }

        // Choose FFT size: 4x impulse length, minimum 1024
        let fft_size = (impulse_len * 4).next_power_of_two().max(1024);
        let overlap = impulse_len - 1;
        let valid_size = fft_size - overlap;

        // Create FFT plans
        let mut engine = FftEngine::new();
        let fft_forward = engine.get_fft_forward(fft_size);
        let fft_inverse = engine.get_fft_inverse(fft_size);

        // Compute impulse FFT
        let mut impulse_fft: Vec<Complex64> = impulse
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        impulse_fft.resize(fft_size, Complex64::new(0.0, 0.0));

        fft_forward.process(&mut impulse_fft);

        Ok(Self {
            impulse_fft,
            fft_size,
            overlap,
            valid_size,
            impulse_len,
            fft_forward,
            fft_inverse,
        })
    }

    /// Create from a waveform impulse response.
    pub fn from_waveform(impulse: &Waveform) -> DspResult<Self> {
        Self::new(&impulse.samples)
    }

    /// Convolve an input signal with the impulse response.
    ///
    /// Uses parallel processing for large inputs.
    pub fn convolve(&self, input: &[f64]) -> Vec<f64> {
        let input_len = input.len();
        let output_len = input_len + self.impulse_len - 1;

        // Determine number of chunks
        let num_chunks = (input_len + self.valid_size - 1) / self.valid_size;

        if num_chunks <= 2 {
            // Small input: process sequentially
            self.convolve_sequential(input, output_len)
        } else {
            // Large input: process in parallel
            self.convolve_parallel(input, output_len, num_chunks)
        }
    }

    /// Sequential convolution for small inputs.
    fn convolve_sequential(&self, input: &[f64], output_len: usize) -> Vec<f64> {
        let mut output = vec![0.0; output_len];

        let mut chunk_idx = 0;
        let mut input_pos = 0;

        while input_pos < input.len() {
            let chunk_end = (input_pos + self.fft_size).min(input.len());
            let chunk_result = self.convolve_chunk(&input[input_pos..chunk_end]);

            // Overlap-add
            let output_start = chunk_idx * self.valid_size;
            for (i, &val) in chunk_result.iter().enumerate() {
                let out_idx = output_start + i;
                if out_idx < output_len {
                    output[out_idx] += val;
                }
            }

            input_pos += self.valid_size;
            chunk_idx += 1;
        }

        output
    }

    /// Parallel convolution for large inputs.
    fn convolve_parallel(&self, input: &[f64], output_len: usize, num_chunks: usize) -> Vec<f64> {
        // Process chunks in parallel
        let chunk_results: Vec<(usize, Vec<f64>)> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * self.valid_size;
                let end = (start + self.fft_size).min(input.len());

                let chunk_result = self.convolve_chunk(&input[start..end]);
                (chunk_idx, chunk_result)
            })
            .collect();

        // Combine results with overlap-add
        let mut output = vec![0.0; output_len];

        for (chunk_idx, chunk_result) in chunk_results {
            let output_start = chunk_idx * self.valid_size;
            for (i, &val) in chunk_result.iter().enumerate() {
                let out_idx = output_start + i;
                if out_idx < output_len {
                    output[out_idx] += val;
                }
            }
        }

        output
    }

    /// Convolve a single chunk using FFT.
    fn convolve_chunk(&self, chunk: &[f64]) -> Vec<f64> {
        // Zero-pad chunk to FFT size
        let mut chunk_fft: Vec<Complex64> = chunk
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        chunk_fft.resize(self.fft_size, Complex64::new(0.0, 0.0));

        // Forward FFT
        self.fft_forward.process(&mut chunk_fft);

        // Multiply with impulse FFT (frequency domain convolution)
        for (c, h) in chunk_fft.iter_mut().zip(self.impulse_fft.iter()) {
            *c = *c * *h;
        }

        // Inverse FFT
        self.fft_inverse.process(&mut chunk_fft);

        // Extract real part and normalize
        let scale = 1.0 / self.fft_size as f64;
        chunk_fft.iter().map(|c| c.re * scale).collect()
    }

    /// Convolve a waveform, returning a new waveform.
    pub fn convolve_waveform(&self, input: &Waveform) -> Waveform {
        let samples = self.convolve(&input.samples);
        Waveform {
            samples,
            dt: input.dt,
            t_start: input.t_start,
        }
    }

    /// Get the impulse response length.
    pub fn impulse_length(&self) -> usize {
        self.impulse_len
    }

    /// Get the FFT size being used.
    pub fn fft_size(&self) -> usize {
        self.fft_size
    }
}

/// Direct convolution (for comparison/validation).
///
/// This is O(n*m) and should only be used for short signals.
pub fn direct_convolve(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let output_len = signal.len() + kernel.len() - 1;
    let mut output = vec![0.0; output_len];

    for (i, &s) in signal.iter().enumerate() {
        for (j, &k) in kernel.iter().enumerate() {
            output[i + j] += s * k;
        }
    }

    output
}

/// Simple FFT-based convolution (single chunk).
///
/// Good for moderate-sized signals where overlap-save overhead isn't worth it.
pub fn fft_convolve(signal: &[f64], kernel: &[f64]) -> DspResult<Vec<f64>> {
    let output_len = signal.len() + kernel.len() - 1;
    let fft_size = output_len.next_power_of_two();

    let mut engine = FftEngine::new();

    // Zero-pad both signals
    let mut signal_fft: Vec<Complex64> = signal
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    signal_fft.resize(fft_size, Complex64::new(0.0, 0.0));

    let mut kernel_fft: Vec<Complex64> = kernel
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    kernel_fft.resize(fft_size, Complex64::new(0.0, 0.0));

    // Forward FFT
    engine.fft_inplace(&mut signal_fft)?;
    engine.fft_inplace(&mut kernel_fft)?;

    // Multiply
    for (s, k) in signal_fft.iter_mut().zip(kernel_fft.iter()) {
        *s = *s * *k;
    }

    // Inverse FFT
    engine.ifft_inplace(&mut signal_fft)?;

    // Extract real part and truncate to output length
    Ok(signal_fft[..output_len].iter().map(|c| c.re).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_convolve_impulse() {
        // Convolving with a delta function should return the input
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0]; // Delta function

        let result = direct_convolve(&signal, &kernel);
        assert_eq!(result, signal);
    }

    #[test]
    fn test_direct_convolve_shift() {
        // Convolving with [0, 1] should shift by one sample
        let signal = vec![1.0, 2.0, 3.0, 4.0];
        let kernel = vec![0.0, 1.0];

        let result = direct_convolve(&signal, &kernel);
        assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fft_convolve_matches_direct() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let kernel = vec![1.0, 0.5, 0.25];

        let direct = direct_convolve(&signal, &kernel);
        let fft = fft_convolve(&signal, &kernel).unwrap();

        for (d, f) in direct.iter().zip(fft.iter()) {
            assert!((d - f).abs() < 1e-10);
        }
    }

    #[test]
    fn test_convolution_engine() {
        let signal = vec![1.0; 10000]; // Long signal
        let kernel = vec![1.0, 0.5, 0.25, 0.125];

        let engine = ConvolutionEngine::new(&kernel).unwrap();
        let result = engine.convolve(&signal);

        // Verify output length
        assert_eq!(result.len(), signal.len() + kernel.len() - 1);

        // Compare with direct convolution for first few samples
        let direct = direct_convolve(&signal[..100], &kernel);
        for (i, (&d, &r)) in direct.iter().zip(result.iter()).enumerate().take(50) {
            assert!((d - r).abs() < 1e-10, "Mismatch at index {}: {} vs {}", i, d, r);
        }
    }
}
