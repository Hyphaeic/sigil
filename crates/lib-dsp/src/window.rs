//! Windowing functions for spectral analysis.
//!
//! This module provides window functions to reduce spectral leakage and
//! Gibbs phenomenon artifacts when converting between frequency and time domains.
//!
//! # IEEE P370 Compliance
//!
//! Per IEEE P370-2020 Section 5.3.1: "A suitable windowing function (e.g.,
//! Kaiser-Bessel with beta=6) shall be applied before inverse transformation
//! to minimize truncation artifacts."

use num_complex::Complex64;
use std::f64::consts::PI;

/// Window function types for spectral processing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WindowType {
    /// No windowing (rectangular window).
    Rectangular,

    /// Hann (raised cosine) window - good general purpose.
    Hann,

    /// Hamming window - slightly better sidelobe rejection than Hann.
    Hamming,

    /// Blackman window - excellent sidelobe rejection.
    Blackman,

    /// Kaiser-Bessel window with configurable beta parameter.
    /// Beta = 6 is recommended by IEEE P370 for S-parameter processing.
    Kaiser { beta: f64 },
}

impl Default for WindowType {
    fn default() -> Self {
        // IEEE P370 Section 5.3.1 recommends Kaiser-Bessel with beta=6
        Self::Kaiser { beta: 6.0 }
    }
}

/// Compute the zeroth-order modified Bessel function of the first kind, I_0(x).
///
/// Uses the polynomial approximation for efficiency.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();

    if ax < 3.75 {
        // Polynomial approximation for small arguments
        let t = (x / 3.75).powi(2);
        1.0 + t * (3.5156229
            + t * (3.0899424
                + t * (1.2067492
                    + t * (0.2659732
                        + t * (0.0360768 + t * 0.0045813)))))
    } else {
        // Asymptotic expansion for large arguments
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537
                                        + t * (-0.01647633 + t * 0.00392377))))))))
    }
}

/// Generate window coefficients for a given window type and length.
///
/// The window is symmetric and normalized so that the center value is 1.0.
///
/// # Arguments
///
/// * `window_type` - Type of window function to generate
/// * `length` - Number of points in the window
///
/// # Returns
///
/// Vector of window coefficients, length `length`.
pub fn generate_window(window_type: WindowType, length: usize) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }

    let n = length as f64;
    let mut window = Vec::with_capacity(length);

    match window_type {
        WindowType::Rectangular => {
            window.resize(length, 1.0);
        }

        WindowType::Hann => {
            for i in 0..length {
                let x = i as f64 / (n - 1.0);
                window.push(0.5 * (1.0 - (2.0 * PI * x).cos()));
            }
        }

        WindowType::Hamming => {
            for i in 0..length {
                let x = i as f64 / (n - 1.0);
                window.push(0.54 - 0.46 * (2.0 * PI * x).cos());
            }
        }

        WindowType::Blackman => {
            for i in 0..length {
                let x = i as f64 / (n - 1.0);
                window.push(
                    0.42 - 0.5 * (2.0 * PI * x).cos() + 0.08 * (4.0 * PI * x).cos(),
                );
            }
        }

        WindowType::Kaiser { beta } => {
            let denom = bessel_i0(beta);
            for i in 0..length {
                let x = 2.0 * i as f64 / (n - 1.0) - 1.0; // Range [-1, 1]
                let arg = beta * (1.0 - x * x).max(0.0).sqrt();
                window.push(bessel_i0(arg) / denom);
            }
        }
    }

    window
}

/// Apply a window function to complex frequency-domain data.
///
/// This applies the window to the positive frequency portion of the spectrum,
/// tapering to zero at the edges to reduce Gibbs phenomenon.
///
/// # Arguments
///
/// * `spectrum` - Complex frequency-domain data (modified in place)
/// * `window_type` - Type of window to apply
/// * `positive_freq_count` - Number of positive frequency bins (typically N/2 for N-point FFT)
pub fn apply_window_to_spectrum(
    spectrum: &mut [Complex64],
    window_type: WindowType,
    positive_freq_count: usize,
) {
    if spectrum.is_empty() || positive_freq_count == 0 {
        return;
    }

    // Generate window for positive frequencies
    let window = generate_window(window_type, positive_freq_count);

    // Apply window to positive frequencies
    for (i, &w) in window.iter().enumerate() {
        if i < spectrum.len() {
            spectrum[i] *= w;
        }
    }

    // Zero out remaining spectrum beyond the windowed region
    // (This is handled by Hermitian symmetry application later)
}

/// Apply a frequency-domain taper window for S-parameter conversion.
///
/// This applies a smooth taper to the high-frequency edge of the data
/// to prevent abrupt truncation artifacts (Gibbs phenomenon).
///
/// # Arguments
///
/// * `spectrum` - Complex frequency-domain data (modified in place)
/// * `window_type` - Type of window to apply
/// * `data_length` - Number of actual data points (before zero-padding)
/// * `taper_fraction` - Fraction of data to taper (0.0 to 1.0, typically 0.1-0.2)
///
/// # IEEE P370 Compliance
///
/// Per IEEE P370-2020 Section 5.3.1, this reduces truncation artifacts
/// that would otherwise appear as ringing in the impulse response.
pub fn apply_edge_taper(
    spectrum: &mut [Complex64],
    window_type: WindowType,
    data_length: usize,
    taper_fraction: f64,
) {
    if spectrum.is_empty() || data_length == 0 {
        return;
    }

    let taper_len = ((data_length as f64 * taper_fraction).ceil() as usize).max(1);

    // Generate a half-window for the taper (descending from 1 to 0)
    let full_window = generate_window(window_type, taper_len * 2);

    // Apply taper to the high-frequency edge
    let start_idx = data_length.saturating_sub(taper_len);
    for i in 0..taper_len {
        let idx = start_idx + i;
        if idx < spectrum.len() && idx < data_length {
            // Use the descending half of the window
            let window_idx = taper_len + i;
            if window_idx < full_window.len() {
                spectrum[idx] *= full_window[window_idx];
            }
        }
    }
}

/// Configuration for windowing in S-parameter conversion.
#[derive(Clone, Debug)]
pub struct WindowConfig {
    /// Type of window function to use.
    pub window_type: WindowType,

    /// Fraction of bandwidth to taper at high frequencies (0.0 to 1.0).
    /// Default is 0.1 (10% taper).
    pub taper_fraction: f64,

    /// Whether to apply windowing at all.
    pub enabled: bool,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            window_type: WindowType::Kaiser { beta: 6.0 },
            taper_fraction: 0.1,
            enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bessel_i0() {
        // I_0(0) = 1
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-10);

        // I_0(1) ≈ 1.266
        assert!((bessel_i0(1.0) - 1.266).abs() < 0.001);

        // I_0(3) ≈ 4.881
        assert!((bessel_i0(3.0) - 4.881).abs() < 0.001);

        // Symmetry: I_0(-x) = I_0(x)
        assert!((bessel_i0(-2.0) - bessel_i0(2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rectangular_window() {
        let window = generate_window(WindowType::Rectangular, 10);
        assert_eq!(window.len(), 10);
        assert!(window.iter().all(|&w| (w - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_hann_window() {
        let window = generate_window(WindowType::Hann, 64);
        assert_eq!(window.len(), 64);

        // Hann window starts and ends at 0
        assert!(window[0].abs() < 1e-10);
        assert!(window[63].abs() < 1e-10);

        // Peak at center
        assert!(window[32] > 0.9);
    }

    #[test]
    fn test_kaiser_window() {
        let window = generate_window(WindowType::Kaiser { beta: 6.0 }, 64);
        assert_eq!(window.len(), 64);

        // Kaiser window has maximum at center
        let center = window[32];
        assert!((center - 1.0).abs() < 0.01);

        // Window tapers to edges
        assert!(window[0] < 0.1);
        assert!(window[63] < 0.1);
    }

    #[test]
    fn test_window_symmetry() {
        let window = generate_window(WindowType::Kaiser { beta: 6.0 }, 65);

        // Odd-length window should be symmetric
        for i in 0..32 {
            assert!(
                (window[i] - window[64 - i]).abs() < 1e-10,
                "Asymmetry at index {}: {} vs {}",
                i,
                window[i],
                window[64 - i]
            );
        }
    }

    #[test]
    fn test_apply_edge_taper() {
        let mut spectrum: Vec<Complex64> = (0..100)
            .map(|_| Complex64::new(1.0, 0.0))
            .collect();

        apply_edge_taper(
            &mut spectrum,
            WindowType::Kaiser { beta: 6.0 },
            100,
            0.1, // 10% taper
        );

        // First 90% should be mostly unchanged
        assert!((spectrum[0].re - 1.0).abs() < 0.01);
        assert!((spectrum[50].re - 1.0).abs() < 0.01);
        assert!((spectrum[80].re - 1.0).abs() < 0.01);

        // Last few samples should be tapered
        assert!(spectrum[99].re < 0.5);
    }

    #[test]
    fn test_window_config_default() {
        let config = WindowConfig::default();
        assert!(config.enabled);
        assert!((config.taper_fraction - 0.1).abs() < 1e-10);
        assert!(matches!(config.window_type, WindowType::Kaiser { beta } if (beta - 6.0).abs() < 1e-10));
    }
}
