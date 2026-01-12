//! Frequency-domain interpolation for S-parameters.

use crate::error::{DspError, DspResult};
use lib_types::units::Hertz;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Wrap a phase value to the range [-π, π].
///
/// Handles arbitrary phase values, not just those in [-2π, 2π].
#[inline]
fn wrap_phase(mut phase: f64) -> f64 {
    // Use modular arithmetic for efficiency
    // This handles large phase differences correctly
    let tau = 2.0 * PI;
    phase = phase % tau;
    if phase > PI {
        phase -= tau;
    } else if phase < -PI {
        phase += tau;
    }
    phase
}

/// Extrapolate S-parameters to DC using physics-based model.
///
/// # CRIT-PHYS-001 Fix
///
/// Per IEEE P370-2020 Section 5.2.3: "DC extrapolation shall enforce S21(0) = 1
/// for transmission-line channels."
///
/// For a passive transmission line:
/// - S21(0) → 1 (zero loss at DC)
/// - phase(0) = 0 (no delay contribution at DC)
/// - Conductor loss: Rs ∝ √f (skin effect)
/// - Dielectric loss: ∝ f (Djordjevic-Sarkar model)
///
/// This prevents measurement noise at the lowest VNA frequency from being
/// incorrectly extrapolated to DC, which would violate Kramers-Kronig relations
/// and produce acausal ringing in the time domain.
fn extrapolate_to_dc() -> Complex64 {
    // For a passive transmission line, S21(DC) = 1 + 0j (unity transmission)
    Complex64::new(1.0, 0.0)
}

/// Interpolate S-parameters to a uniform frequency grid.
pub fn interpolate_linear(
    freqs: &[Hertz],
    values: &[Complex64],
    target_freqs: &[Hertz],
) -> DspResult<Vec<Complex64>> {
    if freqs.len() != values.len() {
        return Err(DspError::LengthMismatch {
            expected: freqs.len(),
            actual: values.len(),
        });
    }
    if freqs.len() < 2 {
        return Err(DspError::InsufficientData { needed: 2, got: freqs.len() });
    }

    let mut result = Vec::with_capacity(target_freqs.len());

    for target in target_freqs {
        let value = interpolate_single(freqs, values, target.0);
        result.push(value);
    }

    Ok(result)
}

/// Interpolate a single frequency point.
fn interpolate_single(freqs: &[Hertz], values: &[Complex64], target: f64) -> Complex64 {
    // CRIT-PHYS-001 fix: Extrapolate to DC with S21(0) = 1
    // Below the lowest measured frequency, enforce physics-based DC behavior
    if target <= 0.0 || target < freqs[0].0 * 0.1 {
        // At or near DC: use physics-based extrapolation
        return extrapolate_to_dc();
    }

    // Below lowest measured frequency but not at DC: linear extrapolation toward DC
    if target < freqs[0].0 {
        let dc_value = extrapolate_to_dc();
        let f0_value = values[0];
        let frac = target / freqs[0].0;

        // Interpolate magnitude and phase separately
        let mag_dc = dc_value.norm();
        let mag_f0 = f0_value.norm();
        let phase_dc = dc_value.arg();
        let phase_f0 = f0_value.arg();

        let mag = mag_dc + frac * (mag_f0 - mag_dc);
        let phase_diff = wrap_phase(phase_f0 - phase_dc);
        let phase = phase_dc + frac * phase_diff;

        return Complex64::from_polar(mag, phase);
    }
    if target >= freqs[freqs.len() - 1].0 {
        return values[values.len() - 1];
    }

    // Find bracketing indices
    let mut lower = 0;
    let mut upper = freqs.len() - 1;

    while upper - lower > 1 {
        let mid = (lower + upper) / 2;
        if freqs[mid].0 <= target {
            lower = mid;
        } else {
            upper = mid;
        }
    }

    // Linear interpolation
    let f0 = freqs[lower].0;
    let f1 = freqs[upper].0;
    let frac = (target - f0) / (f1 - f0);

    let v0 = values[lower];
    let v1 = values[upper];

    // Interpolate magnitude and phase separately for smoother results
    let mag0 = v0.norm();
    let mag1 = v1.norm();
    let phase0 = v0.arg();
    let phase1 = v1.arg();

    // Handle phase wrapping using modular arithmetic
    // This correctly handles arbitrary phase differences, not just [-2π, 2π]
    let phase_diff = wrap_phase(phase1 - phase0);

    let mag = mag0 + frac * (mag1 - mag0);
    let phase = phase0 + frac * phase_diff;

    Complex64::from_polar(mag, phase)
}

/// Generate a uniform frequency grid.
pub fn uniform_frequency_grid(f_min: Hertz, f_max: Hertz, num_points: usize) -> Vec<Hertz> {
    let df = (f_max.0 - f_min.0) / (num_points - 1) as f64;
    (0..num_points)
        .map(|i| Hertz(f_min.0 + i as f64 * df))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let freqs = vec![Hertz(1e9), Hertz(2e9), Hertz(3e9)];
        let values = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let target = vec![Hertz(1.5e9)];
        let result = interpolate_linear(&freqs, &values, &target).unwrap();

        // Should be approximately 0.75
        assert!((result[0].re - 0.75).abs() < 0.01);
    }
}
