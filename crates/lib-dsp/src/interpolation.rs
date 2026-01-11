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
    // Handle out-of-range
    if target <= freqs[0].0 {
        return values[0];
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
