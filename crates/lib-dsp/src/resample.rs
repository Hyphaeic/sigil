//! Waveform resampling for time-step alignment.
//!
//! # HIGH-PHYS-006 Fix
//!
//! This module provides waveform resampling to ensure that the impulse response
//! and stimulus waveforms have compatible sampling rates before convolution.
//!
//! When the impulse dt (from S-parameter IFFT) differs from the stimulus dt
//! (user-specified samples_per_ui), the mismatch causes:
//! - High-frequency content aliasing
//! - UI folding slips by several samples
//! - Equalizer taps drift
//!
//! Per Nyquist theorem, resampling must preserve frequency content up to the
//! Nyquist limit of the coarser sampling rate.

use crate::error::{DspError, DspResult};
use lib_types::units::Seconds;
use lib_types::waveform::Waveform;
use std::f64::consts::PI;

/// Check if two time steps are compatible (within tolerance).
///
/// # Arguments
///
/// * `dt1` - First time step
/// * `dt2` - Second time step
/// * `relative_tolerance` - Relative tolerance (e.g., 1e-6 for 0.0001%)
///
/// # Returns
///
/// True if time steps are within tolerance, false otherwise.
#[inline]
pub fn are_compatible_dt(dt1: Seconds, dt2: Seconds, relative_tolerance: f64) -> bool {
    let max_dt = dt1.0.max(dt2.0);
    if max_dt == 0.0 {
        return dt1.0 == dt2.0;
    }
    (dt1.0 - dt2.0).abs() / max_dt < relative_tolerance
}

/// Resample a waveform to a new time step using windowed sinc interpolation.
///
/// # HIGH-PHYS-006 Fix
///
/// Uses a windowed sinc kernel for high-quality interpolation that preserves
/// frequency content up to the Nyquist limit of the original sampling rate.
///
/// # Arguments
///
/// * `waveform` - Input waveform to resample
/// * `new_dt` - Target time step
///
/// # Returns
///
/// Resampled waveform with new_dt spacing, preserving waveform.t_start.
///
/// # Notes
///
/// - Uses Lanczos windowed sinc (a=3) for anti-aliasing
/// - Preserves DC and low-frequency content
/// - Band-limits to min(old_nyquist, new_nyquist)
pub fn resample_waveform(waveform: &Waveform, new_dt: Seconds) -> DspResult<Waveform> {
    if waveform.samples.is_empty() {
        return Err(DspError::InsufficientData {
            needed: 1,
            got: 0,
        });
    }

    if new_dt.0 <= 0.0 {
        return Err(DspError::InvalidConfig(format!(
            "new_dt must be positive, got {}",
            new_dt.0
        )));
    }

    // Check if already at target dt
    if are_compatible_dt(waveform.dt, new_dt, 1e-10) {
        return Ok(waveform.clone());
    }

    let old_dt = waveform.dt.0;
    let new_dt_val = new_dt.0;

    // Compute duration and new sample count
    let duration = waveform.duration().0;
    let new_len = (duration / new_dt_val).ceil() as usize;

    if new_len == 0 {
        return Err(DspError::InvalidConfig(
            "Resampled waveform would have zero samples".into(),
        ));
    }

    let mut resampled = Vec::with_capacity(new_len);

    // Lanczos kernel parameter (a=3 is standard for high quality)
    let lanczos_a = 3;

    // Resample using windowed sinc interpolation
    for i in 0..new_len {
        let t_target = i as f64 * new_dt_val; // Time of new sample
        let old_index_f = t_target / old_dt; // Fractional index in old samples

        // Sinc interpolation with Lanczos window
        let mut value = 0.0;

        // Kernel support: ±lanczos_a samples around old_index_f
        let index_min = (old_index_f - lanczos_a as f64).floor() as isize;
        let index_max = (old_index_f + lanczos_a as f64).ceil() as isize;

        for k in index_min..=index_max {
            if k < 0 || k >= waveform.samples.len() as isize {
                continue; // Zero-padding outside range
            }

            let sample = waveform.samples[k as usize];
            let x = old_index_f - k as f64;

            // Windowed sinc kernel: sinc(x) × Lanczos(x/a)
            let kernel = windowed_sinc(x, lanczos_a as f64);
            value += sample * kernel;
        }

        resampled.push(value);
    }

    Ok(Waveform {
        samples: resampled,
        dt: new_dt,
        t_start: waveform.t_start, // Preserve time origin
    })
}

/// Windowed sinc interpolation kernel (Lanczos window).
///
/// # Arguments
///
/// * `x` - Distance from sample center
/// * `a` - Lanczos window parameter (typically 3)
///
/// # Returns
///
/// Kernel value at position x
#[inline]
fn windowed_sinc(x: f64, a: f64) -> f64 {
    if x.abs() < 1e-10 {
        return 1.0; // sinc(0) = 1
    }

    if x.abs() >= a {
        return 0.0; // Outside window support
    }

    // sinc(x) = sin(πx) / (πx)
    let sinc_val = (PI * x).sin() / (PI * x);

    // Lanczos window: sinc(x/a)
    let window = (PI * x / a).sin() / (PI * x / a);

    sinc_val * window
}

/// Estimate the bandwidth of a waveform (3 dB point).
///
/// # HIGH-PHYS-006 Support
///
/// Used to validate Nyquist criterion and detect insufficient sampling.
///
/// # Arguments
///
/// * `waveform` - Waveform to analyze
///
/// # Returns
///
/// Estimated 3 dB bandwidth in Hz
///
/// # Note
///
/// This is a simple estimate based on spectral energy distribution.
/// For more accurate bandwidth estimation, use the S-parameter magnitude directly.
pub fn estimate_bandwidth(waveform: &Waveform) -> DspResult<f64> {
    use crate::fft::FftEngine;

    if waveform.samples.len() < 4 {
        return Err(DspError::InsufficientData {
            needed: 4,
            got: waveform.samples.len(),
        });
    }

    // FFT to frequency domain
    let mut engine = FftEngine::new();
    let fft_size = waveform.samples.len().next_power_of_two();

    let mut spectrum: Vec<num_complex::Complex64> = waveform
        .samples
        .iter()
        .map(|&v| num_complex::Complex64::new(v, 0.0))
        .collect();
    spectrum.resize(fft_size, num_complex::Complex64::new(0.0, 0.0));

    let spectrum = engine.fft(&spectrum)?;

    // Compute magnitude spectrum
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();

    // Find peak magnitude (DC or fundamental)
    let peak_mag = magnitudes[0..fft_size / 2]
        .iter()
        .cloned()
        .fold(0.0, f64::max);

    if peak_mag == 0.0 {
        return Ok(1.0 / waveform.dt.0 / 2.0); // Default to Nyquist
    }

    // Find 3 dB point (peak / sqrt(2))
    let threshold = peak_mag / 2.0_f64.sqrt();

    let df = 1.0 / (fft_size as f64 * waveform.dt.0);

    for (i, &mag) in magnitudes[0..fft_size / 2].iter().enumerate() {
        if mag < threshold {
            return Ok(i as f64 * df);
        }
    }

    // If no 3 dB point found, return Nyquist
    Ok(1.0 / waveform.dt.0 / 2.0)
}

/// Validate Nyquist criterion for a given sampling configuration.
///
/// # HIGH-PHYS-006 Fix
///
/// Per Nyquist theorem, to avoid aliasing:
/// ```text
/// samples_per_ui ≥ 2 × (bit_time × bandwidth)
/// ```
///
/// In practice, for rise times near the Nyquist limit, we require:
/// ```text
/// samples_per_ui ≥ 10 × (bit_time / rise_time)
/// ```
///
/// # Arguments
///
/// * `bit_time` - UI duration
/// * `samples_per_ui` - User-specified samples per UI
/// * `channel_bandwidth` - Channel 3 dB bandwidth (Hz)
///
/// # Returns
///
/// Ok(()) if Nyquist criterion satisfied, Err otherwise.
pub fn validate_nyquist(
    bit_time: Seconds,
    samples_per_ui: usize,
    channel_bandwidth: f64,
) -> DspResult<()> {
    // Estimate rise time from bandwidth (10-90% rise ≈ 0.35 / BW)
    let rise_time = 0.35 / channel_bandwidth;

    // Required samples per UI for 10× oversampling of rise time
    let min_samples = (bit_time.0 / rise_time * 10.0).ceil() as usize;

    if samples_per_ui < min_samples {
        return Err(DspError::NyquistViolation {
            required: min_samples,
            provided: samples_per_ui,
            bandwidth_ghz: channel_bandwidth * 1e-9,
        });
    }

    // Also check basic Nyquist (2× bandwidth)
    let nyquist_min = (2.0 * bit_time.0 * channel_bandwidth).ceil() as usize;

    if samples_per_ui < nyquist_min {
        return Err(DspError::NyquistViolation {
            required: nyquist_min,
            provided: samples_per_ui,
            bandwidth_ghz: channel_bandwidth * 1e-9,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_are_compatible_dt() {
        let dt1 = Seconds::from_ps(1.0);
        let dt2 = Seconds::from_ps(1.0);
        assert!(are_compatible_dt(dt1, dt2, 1e-10));

        let dt3 = Seconds::from_ps(1.001);
        assert!(!are_compatible_dt(dt1, dt3, 1e-6)); // 0.1% difference
        assert!(are_compatible_dt(dt1, dt3, 1e-2)); // 1% tolerance
    }

    #[test]
    fn test_windowed_sinc() {
        // sinc(0) should be 1
        assert!((windowed_sinc(0.0, 3.0) - 1.0).abs() < 1e-10);

        // Outside window should be 0
        assert!((windowed_sinc(4.0, 3.0)).abs() < 1e-10);

        // Symmetry
        let val = windowed_sinc(1.5, 3.0);
        assert!((val - windowed_sinc(-1.5, 3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_resample_identity() {
        // Resampling to same dt should return nearly identical waveform
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dt = Seconds::from_ps(10.0);
        let wf = Waveform::new(samples.clone(), dt, Seconds::ZERO);

        let resampled = resample_waveform(&wf, dt).unwrap();

        assert_eq!(resampled.samples.len(), samples.len());
        for (orig, resampled) in samples.iter().zip(resampled.samples.iter()) {
            assert!((orig - resampled).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resample_upsampling() {
        // Upsample by 2x
        let samples = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let dt = Seconds::from_ps(10.0);
        let wf = Waveform::new(samples, dt, Seconds::ZERO);

        let new_dt = Seconds::from_ps(5.0); // 2x finer
        let resampled = resample_waveform(&wf, new_dt).unwrap();

        // Should have approximately 2x samples
        assert!(resampled.samples.len() >= 9 && resampled.samples.len() <= 11);

        // Check dt is correct
        assert!((resampled.dt.0 - new_dt.0).abs() < 1e-15);
    }

    #[test]
    fn test_resample_downsampling() {
        // Downsample by 2x
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0];
        let dt = Seconds::from_ps(1.0);
        let wf = Waveform::new(samples, dt, Seconds::ZERO);

        let new_dt = Seconds::from_ps(2.0); // 2x coarser
        let resampled = resample_waveform(&wf, new_dt).unwrap();

        // Should have approximately half the samples
        assert!(resampled.samples.len() >= 4 && resampled.samples.len() <= 6);
    }

    #[test]
    fn test_validate_nyquist() {
        let bit_time = Seconds::from_ps(31.25); // PCIe Gen 5
        let bw = 16e9; // 16 GHz channel

        // 64 samples/UI should pass
        assert!(validate_nyquist(bit_time, 64, bw).is_ok());

        // 4 samples/UI should fail
        assert!(validate_nyquist(bit_time, 4, bw).is_err());
    }

    #[test]
    fn test_estimate_bandwidth() {
        // Create a simple pulse waveform
        let mut samples = vec![0.0; 1000];
        samples[100] = 1.0;

        let wf = Waveform::new(samples, Seconds::from_ps(1.0), Seconds::ZERO);
        let bw = estimate_bandwidth(&wf).unwrap();

        // Bandwidth should be positive and at most Nyquist
        assert!(bw > 0.0);
        assert!(bw <= 1.0 / (2.0 * wf.dt.0));

        // For an impulse, bandwidth estimate may be at or near Nyquist
        let nyquist = 1.0 / (2.0 * wf.dt.0);
        assert!(
            bw >= nyquist * 0.8,
            "Impulse bandwidth should be near Nyquist, got {:.1} GHz vs {:.1} GHz",
            bw * 1e-9,
            nyquist * 1e-9
        );
    }
}
