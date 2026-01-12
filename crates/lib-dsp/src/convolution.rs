//! High-performance convolution using overlap-save FFT method.
//!
//! This module provides efficient convolution for large bitstreams,
//! using parallel processing with Rayon.
//!
//! # Transient Handling (HIGH-DSP-005 Fix)
//!
//! Per IBIS 7.2 Section 11.3: "The statistical eye shall be computed from
//! steady-state waveform data only. A warm-up period of at least 3x the
//! impulse response duration shall be discarded."
//!
//! The `ConvolutionEngine` provides methods to query the transient length
//! and to convolve while automatically discarding the initial transient.

use crate::error::{DspError, DspResult};
use crate::fft::FftEngine;
use lib_types::waveform::Waveform;
use lib_types::units::{Hertz, Seconds};
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::Fft;
use std::sync::Arc;

/// FFT sizing strategy for convolution.
///
/// # HIGH-DSP-004 and HIGH-PHYS-007 Fix
///
/// Controls the frequency resolution (Δf) of the convolution FFT.
/// Different strategies optimize for different channel characteristics.
#[derive(Clone, Debug)]
pub enum FftSizeStrategy {
    /// Automatic sizing: 4x impulse length, minimum 1024.
    ///
    /// Good default for most channels. Fast and memory-efficient.
    Auto,

    /// Bandwidth-based sizing: ensure Δf captures narrowband features.
    ///
    /// Required for channels with sharp resonances (connector stubs, vias).
    /// Ensures Δf ≤ f_resonance / Q for Q-factor resonances.
    ///
    /// # Example
    /// ```no_run
    /// use lib_dsp::FftSizeStrategy;
    /// use lib_types::units::Hertz;
    /// // Capture resonances with Q=50 at 10 GHz
    /// // Requires Δf ≤ 10 GHz / 50 = 200 MHz
    /// let strategy = FftSizeStrategy::Bandwidth { min_delta_f: Hertz(200e6) };
    /// ```
    Bandwidth { min_delta_f: Hertz },

    /// User-specified fixed size.
    ///
    /// For manual tuning or matching external tool configurations.
    /// Must be a power of 2.
    Fixed { size: usize },
}

impl Default for FftSizeStrategy {
    fn default() -> Self {
        FftSizeStrategy::Auto
    }
}

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
    /// Uses automatic sizing (4x impulse length, minimum 1024).
    /// For custom FFT sizing, use [`ConvolutionEngine::with_strategy`].
    pub fn new(impulse: &[f64]) -> DspResult<Self> {
        Self::with_strategy(impulse, Seconds(1.0), FftSizeStrategy::Auto)
    }

    /// Create a convolution engine with a custom FFT sizing strategy.
    ///
    /// # HIGH-DSP-004 and HIGH-PHYS-007 Fix
    ///
    /// Allows control over frequency resolution to capture narrowband resonances.
    ///
    /// # Arguments
    ///
    /// * `impulse` - The impulse response to convolve with
    /// * `dt` - Time step of the impulse (used for bandwidth-based sizing)
    /// * `strategy` - FFT sizing strategy
    ///
    /// # FFT Size Selection Strategies
    ///
    /// - **Auto**: 4x impulse length (fast, may miss narrow resonances)
    /// - **Bandwidth**: Ensures Δf ≤ min_delta_f (captures resonances with Q > f/Δf)
    /// - **Fixed**: User-specified size (must be power of 2)
    pub fn with_strategy(
        impulse: &[f64],
        dt: Seconds,
        strategy: FftSizeStrategy,
    ) -> DspResult<Self> {
        let impulse_len = impulse.len();
        if impulse_len == 0 {
            return Err(DspError::InsufficientData { needed: 1, got: 0 });
        }

        // HIGH-DSP-004 and HIGH-PHYS-007 FIX: Select FFT size based on strategy
        let fft_size = match strategy {
            FftSizeStrategy::Auto => {
                // Original heuristic: 4x impulse length, minimum 1024
                (impulse_len * 4).next_power_of_two().max(1024)
            }
            FftSizeStrategy::Bandwidth { min_delta_f } => {
                // Bandwidth-based: ensure Δf ≤ min_delta_f
                // Δf = 1 / (N × dt), so N = 1 / (Δf × dt)
                let n_from_bandwidth = (1.0 / (min_delta_f.0 * dt.0)).ceil() as usize;

                // Also ensure we don't go smaller than impulse requires
                let n_from_impulse = impulse_len * 2;

                n_from_bandwidth.max(n_from_impulse).next_power_of_two()
            }
            FftSizeStrategy::Fixed { size } => {
                if !size.is_power_of_two() {
                    return Err(DspError::InvalidFftSize(size));
                }
                if size < impulse_len {
                    return Err(DspError::InvalidConfig(format!(
                        "FFT size {} is smaller than impulse length {}",
                        size, impulse_len
                    )));
                }
                size
            }
        };
        let overlap = impulse_len - 1;
        let valid_size = fft_size - overlap;

        // Log FFT size selection rationale
        let delta_f = if dt.0 > 0.0 {
            1.0 / (fft_size as f64 * dt.0)
        } else {
            0.0
        };

        tracing::debug!(
            "ConvolutionEngine: FFT size={}, impulse_len={}, Δf={:.1} MHz",
            fft_size,
            impulse_len,
            delta_f * 1e-6
        );

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
    ///
    /// Uses automatic FFT sizing. For custom sizing, use
    /// [`ConvolutionEngine::from_waveform_with_strategy`].
    pub fn from_waveform(impulse: &Waveform) -> DspResult<Self> {
        Self::with_strategy(&impulse.samples, impulse.dt, FftSizeStrategy::Auto)
    }

    /// Create from a waveform with a custom FFT sizing strategy.
    ///
    /// # HIGH-DSP-004 and HIGH-PHYS-007 Fix
    ///
    /// Uses the waveform's dt for bandwidth-based FFT sizing.
    pub fn from_waveform_with_strategy(
        impulse: &Waveform,
        strategy: FftSizeStrategy,
    ) -> DspResult<Self> {
        Self::with_strategy(&impulse.samples, impulse.dt, strategy)
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

    /// Sequential convolution for small inputs using overlap-save method.
    fn convolve_sequential(&self, input: &[f64], output_len: usize) -> Vec<f64> {
        let mut output = vec![0.0; output_len];

        let mut chunk_idx = 0;
        let mut input_pos: isize = -(self.overlap as isize); // Start before input for proper overlap

        while (input_pos as usize) < input.len() + self.overlap {
            // Build input chunk with overlap from previous section
            let mut chunk = vec![0.0; self.fft_size];
            for i in 0..self.fft_size {
                let src_idx = input_pos + i as isize;
                if src_idx >= 0 && (src_idx as usize) < input.len() {
                    chunk[i] = input[src_idx as usize];
                }
                // Otherwise remains 0 (zero-padding)
            }

            let chunk_result = self.convolve_chunk(&chunk);

            // Overlap-save: discard first `overlap` samples (invalid due to circular conv)
            // Keep the remaining `valid_size` samples
            let output_start = chunk_idx * self.valid_size;
            for i in 0..self.valid_size {
                let out_idx = output_start + i;
                if out_idx < output_len {
                    output[out_idx] = chunk_result[self.overlap + i];
                }
            }

            input_pos += self.valid_size as isize;
            chunk_idx += 1;
        }

        output
    }

    /// Parallel convolution for large inputs using overlap-save method.
    fn convolve_parallel(&self, input: &[f64], output_len: usize, num_chunks: usize) -> Vec<f64> {
        let overlap = self.overlap;
        let valid_size = self.valid_size;
        let fft_size = self.fft_size;
        let input_len = input.len();

        // Process chunks in parallel
        let chunk_results: Vec<(usize, Vec<f64>)> = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                // For overlap-save, input position starts before actual input
                let input_pos = (chunk_idx * valid_size) as isize - overlap as isize;

                // Build input chunk with proper overlap
                let mut chunk = vec![0.0; fft_size];
                for i in 0..fft_size {
                    let src_idx = input_pos + i as isize;
                    if src_idx >= 0 && (src_idx as usize) < input_len {
                        chunk[i] = input[src_idx as usize];
                    }
                    // Otherwise remains 0 (zero-padding)
                }

                let chunk_result = self.convolve_chunk(&chunk);
                (chunk_idx, chunk_result)
            })
            .collect();

        // Combine results: take only valid samples from each chunk
        let mut output = vec![0.0; output_len];

        for (chunk_idx, chunk_result) in chunk_results {
            // Overlap-save: discard first `overlap` samples, keep `valid_size`
            let output_start = chunk_idx * valid_size;
            for i in 0..valid_size {
                let out_idx = output_start + i;
                if out_idx < output_len {
                    output[out_idx] = chunk_result[overlap + i];
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

    /// Get the number of transient samples to discard.
    ///
    /// The initial transient is `impulse_len - 1` samples, where the
    /// convolution hasn't reached steady state (ISI not fully accumulated).
    ///
    /// # HIGH-DSP-005 Fix
    ///
    /// Per IBIS 7.2 Section 11.3: "The statistical eye shall be computed
    /// from steady-state waveform data only."
    #[inline]
    pub fn transient_samples(&self) -> usize {
        self.impulse_len.saturating_sub(1)
    }

    /// Get the recommended warmup samples to discard.
    ///
    /// Per IBIS 7.2 Section 11.3: "A warm-up period of at least 3x the
    /// impulse response duration shall be discarded."
    ///
    /// This returns `3 * impulse_len` which is more conservative than
    /// `transient_samples()`.
    #[inline]
    pub fn warmup_samples(&self) -> usize {
        self.impulse_len * 3
    }

    /// Convolve and return only the steady-state portion.
    ///
    /// # HIGH-DSP-005 Fix
    ///
    /// Automatically discards the initial transient where ISI hasn't
    /// reached steady state, per IBIS 7.2 Section 11.3.
    ///
    /// # Arguments
    ///
    /// * `input` - Input signal samples
    /// * `discard_3x` - If true, discards 3x impulse length (IBIS recommended).
    ///                  If false, discards only the minimum transient.
    ///
    /// # Returns
    ///
    /// The steady-state portion of the convolution output.
    pub fn convolve_steady_state(&self, input: &[f64], discard_3x: bool) -> Vec<f64> {
        let full_output = self.convolve(input);

        let discard = if discard_3x {
            self.warmup_samples()
        } else {
            self.transient_samples()
        };

        if discard >= full_output.len() {
            // Not enough output for steady-state
            Vec::new()
        } else {
            full_output[discard..].to_vec()
        }
    }

    /// Convolve a waveform and return only the steady-state portion.
    ///
    /// # HIGH-DSP-005 Fix
    ///
    /// Automatically discards the initial transient, adjusting t_start
    /// to reflect the new starting point.
    pub fn convolve_waveform_steady_state(&self, input: &Waveform, discard_3x: bool) -> Waveform {
        let discard = if discard_3x {
            self.warmup_samples()
        } else {
            self.transient_samples()
        };

        let full_output = self.convolve(&input.samples);

        if discard >= full_output.len() {
            // Not enough output for steady-state
            Waveform {
                samples: Vec::new(),
                dt: input.dt,
                t_start: input.t_start,
            }
        } else {
            let samples = full_output[discard..].to_vec();
            let t_start = Seconds(input.t_start.0 + discard as f64 * input.dt.0);
            Waveform {
                samples,
                dt: input.dt,
                t_start,
            }
        }
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

    #[test]
    fn test_transient_samples() {
        let kernel = vec![1.0, 0.5, 0.25, 0.125]; // 4 samples
        let engine = ConvolutionEngine::new(&kernel).unwrap();

        // Transient = impulse_len - 1 = 3
        assert_eq!(engine.transient_samples(), 3);

        // Warmup = 3 * impulse_len = 12
        assert_eq!(engine.warmup_samples(), 12);
    }

    #[test]
    fn test_convolve_steady_state() {
        let kernel = vec![1.0, 0.5, 0.25, 0.125]; // 4 samples
        let signal = vec![1.0; 100];

        let engine = ConvolutionEngine::new(&kernel).unwrap();

        // Full convolution
        let full = engine.convolve(&signal);
        assert_eq!(full.len(), 103); // 100 + 4 - 1

        // Steady-state with minimum transient discard
        let steady = engine.convolve_steady_state(&signal, false);
        assert_eq!(steady.len(), 100); // 103 - 3

        // Steady-state with 3x warmup discard
        let steady_3x = engine.convolve_steady_state(&signal, true);
        assert_eq!(steady_3x.len(), 91); // 103 - 12

        // Verify steady-state portion matches
        for (i, (&s, &f)) in steady.iter().zip(full[3..].iter()).enumerate() {
            assert!(
                (s - f).abs() < 1e-10,
                "Mismatch at steady-state index {}: {} vs {}",
                i, s, f
            );
        }
    }

    #[test]
    fn test_convolve_waveform_steady_state() {
        let kernel = vec![1.0, 0.5, 0.25, 0.125];
        let signal = vec![1.0; 100];
        let dt = Seconds::from_ps(10.0);
        let t_start = Seconds::ZERO;

        let input = Waveform::new(signal, dt, t_start);
        let engine = ConvolutionEngine::new(&kernel).unwrap();

        let steady = engine.convolve_waveform_steady_state(&input, false);

        // Check length
        assert_eq!(steady.samples.len(), 100);

        // Check t_start is advanced by transient time
        let expected_t_start = Seconds::from_ps(30.0); // 3 samples * 10 ps
        assert!((steady.t_start.0 - expected_t_start.0).abs() < 1e-20);
    }
}
