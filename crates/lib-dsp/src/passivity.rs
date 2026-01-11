//! Passivity enforcement for S-parameters.

use crate::error::{DspError, DspResult};
use lib_types::sparams::SParameters;
use ndarray::Array2;
use num_complex::Complex64;

/// Passivity enforcement result.
#[derive(Debug, Clone)]
pub struct PassivityReport {
    /// Number of frequency points that were non-passive.
    pub violations: usize,

    /// Maximum eigenvalue before enforcement.
    pub max_eigenvalue: f64,

    /// Scaling factors applied at each frequency.
    pub scale_factors: Vec<f64>,
}

/// Enforce passivity on S-parameters.
///
/// A passive network cannot generate energy, requiring ||S|| <= 1 at all frequencies.
/// For 2-port: |S11|² + |S21|² <= 1 and |S22|² + |S12|² <= 1.
pub fn enforce_passivity(sparams: &mut SParameters) -> DspResult<PassivityReport> {
    let mut violations = 0;
    let mut max_eigenvalue = 0.0f64;
    let mut scale_factors = Vec::with_capacity(sparams.len());

    for matrix in &mut sparams.matrices {
        let (is_passive, eigenvalue, scale) = enforce_passivity_matrix(matrix)?;

        if !is_passive {
            violations += 1;
        }
        max_eigenvalue = max_eigenvalue.max(eigenvalue);
        scale_factors.push(scale);
    }

    if violations > 0 {
        tracing::warn!(
            violations,
            max_eigenvalue,
            "Enforced passivity on S-parameters"
        );
    }

    Ok(PassivityReport {
        violations,
        max_eigenvalue,
        scale_factors,
    })
}

/// Enforce passivity on a single S-matrix.
///
/// Returns (was_passive, max_eigenvalue, scale_applied).
fn enforce_passivity_matrix(s: &mut Array2<Complex64>) -> DspResult<(bool, f64, f64)> {
    let n = s.nrows();

    // Compute S^H * S
    let s_h: Array2<Complex64> = s.t().mapv(|c| c.conj());
    let product = s_h.dot(s);

    // For 2x2, compute eigenvalues analytically
    let max_eigenvalue = if n == 2 {
        compute_max_eigenvalue_2x2(&product)
    } else {
        // For larger matrices, use simple bound: max diagonal
        product.diag().iter().map(|c| c.re).fold(0.0, f64::max)
    };

    // Use a tolerance appropriate for floating-point arithmetic
    // 1e-10 is too tight and causes false positives due to numerical errors
    let is_passive = max_eigenvalue <= 1.0 + 1e-6;

    let scale = if !is_passive {
        let scale = 1.0 / max_eigenvalue.sqrt();
        s.mapv_inplace(|c| c * scale);
        scale
    } else {
        1.0
    };

    Ok((is_passive, max_eigenvalue, scale))
}

/// Compute maximum eigenvalue of 2x2 Hermitian matrix.
fn compute_max_eigenvalue_2x2(m: &Array2<Complex64>) -> f64 {
    // For Hermitian 2x2 matrix:
    // eigenvalues = (trace ± sqrt(trace² - 4*det)) / 2

    let a = m[[0, 0]].re;
    let d = m[[1, 1]].re;
    let b = m[[0, 1]];
    let c = m[[1, 0]];

    let trace = a + d;
    let det = a * d - (b * c).re;

    let discriminant = trace * trace - 4.0 * det;

    if discriminant < 0.0 {
        // Shouldn't happen for Hermitian matrix
        trace / 2.0
    } else {
        (trace + discriminant.sqrt()) / 2.0
    }
}

/// Check if S-parameters are passive at all frequencies.
pub fn check_passivity(sparams: &SParameters) -> Vec<bool> {
    sparams
        .matrices
        .iter()
        .map(|m| {
            // Simple check: all |S_ij| <= 1
            m.iter().all(|c| c.norm() <= 1.0 + 1e-6)
        })
        .collect()
}

/// Compute the passivity margin (how much gain before becoming active).
pub fn passivity_margin(sparams: &SParameters) -> Vec<f64> {
    sparams
        .matrices
        .iter()
        .map(|m| {
            let max_mag = m.iter().map(|c| c.norm()).fold(0.0, f64::max);
            if max_mag > 0.0 {
                20.0 * (1.0 / max_mag).log10() // dB margin
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_passive_sparam_unchanged() {
        let mut sparams = SParameters::new(2, lib_types::units::Ohms::Z0_50);

        // Create passive matrix
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(0.1, 0.0);
        m[[0, 1]] = Complex64::new(0.0, 0.0);
        m[[1, 0]] = Complex64::new(0.9, 0.0);
        m[[1, 1]] = Complex64::new(0.1, 0.0);

        sparams.add_point(lib_types::units::Hertz::from_ghz(1.0), m.clone());

        let original_s21 = sparams.s21()[0];

        let report = enforce_passivity(&mut sparams).unwrap();
        assert_eq!(report.violations, 0);

        let new_s21 = sparams.s21()[0];
        assert!((original_s21.re - new_s21.re).abs() < 1e-10);
    }

    #[test]
    fn test_active_sparam_scaled() {
        let mut sparams = SParameters::new(2, lib_types::units::Ohms::Z0_50);

        // Create active matrix (|S21| > 1)
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(0.0, 0.0);
        m[[0, 1]] = Complex64::new(0.0, 0.0);
        m[[1, 0]] = Complex64::new(1.5, 0.0); // Gain!
        m[[1, 1]] = Complex64::new(0.0, 0.0);

        sparams.add_point(lib_types::units::Hertz::from_ghz(1.0), m);

        let report = enforce_passivity(&mut sparams).unwrap();
        assert_eq!(report.violations, 1);

        // After enforcement, should be passive
        let new_s21 = sparams.s21()[0];
        assert!(new_s21.norm() <= 1.0 + 1e-6);
    }
}
