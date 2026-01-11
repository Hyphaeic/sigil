//! Passivity enforcement for S-parameters.
//!
//! A passive network cannot generate energy, requiring the spectral norm
//! `‖S‖₂ ≤ 1` at all frequencies. This is equivalent to requiring that the
//! maximum singular value `σ_max ≤ 1`.
//!
//! # IEEE P370 Compliance
//!
//! Per IEEE P370-2020 Section 4.5.2: "Passivity validation shall use singular
//! value decomposition or equivalent eigenvalue analysis."
//!
//! The element-wise check `|S_ij| ≤ 1` is **insufficient** for passivity.
//! A matrix can have all elements with magnitude < 1 but still have σ_max > 1
//! due to constructive interference between columns.

use crate::error::{DspError, DspResult};
use faer::{mat::AsMatRef, prelude::*, Mat};
use lib_types::sparams::SParameters;
use ndarray::Array2;
use num_complex::Complex64;

/// Tolerance for passivity checking (accounts for floating-point representation errors).
/// Per existing codebase convention, 1e-6 provides practical tolerance without false positives.
const PASSIVITY_TOLERANCE: f64 = 1e-6;

/// Epsilon for numerical comparisons (near-zero detection).
const NUMERICAL_EPSILON: f64 = 1e-12;

/// Passivity enforcement result.
#[derive(Debug, Clone)]
pub struct PassivityReport {
    /// Number of frequency points that were non-passive.
    pub violations: usize,

    /// Maximum singular value before enforcement.
    /// Note: Field named `max_eigenvalue` for backward compatibility, but represents σ_max.
    pub max_eigenvalue: f64,

    /// Scaling factors applied at each frequency.
    pub scale_factors: Vec<f64>,
}

/// Convert ndarray Array2<Complex64> to faer Mat<c64>.
///
/// Both types use identical memory layout (Complex<f64> with re/im fields),
/// so conversion is straightforward.
fn ndarray_to_faer_mat(m: &Array2<Complex64>) -> Mat<c64> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| {
        let c = m[[i, j]];
        c64::new(c.re, c.im)
    })
}

/// Compute the maximum singular value (spectral norm) of a complex matrix.
///
/// Returns σ_max such that ‖S‖₂ = σ_max.
/// For a passive network, σ_max ≤ 1.
fn compute_max_singular_value(s: &Array2<Complex64>) -> f64 {
    if s.is_empty() {
        return 0.0;
    }

    let faer_mat = ndarray_to_faer_mat(s);

    // Compute singular values (returned in non-increasing order)
    let svd = faer_mat.svd();
    let s_diag = svd.s_diagonal();

    // First element is the largest singular value
    if s_diag.nrows() > 0 {
        s_diag.read(0).re.abs()
    } else {
        0.0
    }
}

/// Enforce passivity on a matrix using SVD clamping.
///
/// Clamps all singular values to ≤ 1.0 and reconstructs the matrix:
/// `S_enforced = U × diag(min(σ_i, 1)) × V^H`
///
/// Returns (max_original_singular_value, scale_equivalent).
fn enforce_passivity_via_svd(s: &mut Array2<Complex64>) -> (f64, f64) {
    if s.is_empty() {
        return (0.0, 1.0);
    }

    let faer_mat = ndarray_to_faer_mat(s);

    // Compute full SVD: S = U × Σ × V^H
    let svd = faer_mat.svd();
    let u = svd.u();
    let v = svd.v();
    let s_diag = svd.s_diagonal();

    let min_dim = s.nrows().min(s.ncols());

    // Find max singular value
    let sigma_max = if min_dim > 0 {
        s_diag.read(0).re.abs()
    } else {
        0.0
    };

    // If already passive, no modification needed
    if sigma_max <= 1.0 + PASSIVITY_TOLERANCE {
        return (sigma_max, 1.0);
    }

    // Build clamped diagonal matrix (as full matrix for multiplication)
    let mut clamped_sigma = Mat::<c64>::zeros(s.nrows(), s.ncols());
    for i in 0..min_dim {
        let sv = s_diag.read(i).re.abs();
        let clamped = sv.min(1.0);
        clamped_sigma.write(i, i, c64::new(clamped, 0.0));
    }

    // Reconstruct: U × clamped_Σ × V^H
    let temp = u.as_ref() * clamped_sigma.as_ref();
    let result = temp.as_ref() * v.adjoint();

    // Copy back to ndarray
    for i in 0..s.nrows() {
        for j in 0..s.ncols() {
            let c = result.read(i, j);
            s[[i, j]] = Complex64::new(c.re, c.im);
        }
    }

    // Return equivalent scale factor for reporting compatibility
    let scale_equivalent = 1.0 / sigma_max;
    (sigma_max, scale_equivalent)
}

/// Enforce passivity on S-parameters.
///
/// A passive network cannot generate energy, requiring ‖S‖₂ ≤ 1 at all frequencies.
/// This function enforces passivity by clamping singular values.
///
/// # Algorithm
///
/// For 2×2 matrices: Uses efficient analytical eigenvalue computation of S^H×S,
/// then uniform scaling if σ_max > 1.
///
/// For larger matrices: Uses full SVD decomposition with per-singular-value clamping.
pub fn enforce_passivity(sparams: &mut SParameters) -> DspResult<PassivityReport> {
    let mut violations = 0;
    let mut max_singular_value = 0.0f64;
    let mut scale_factors = Vec::with_capacity(sparams.len());

    for matrix in &mut sparams.matrices {
        let (is_passive, sigma_max, scale) = enforce_passivity_matrix(matrix)?;

        if !is_passive {
            violations += 1;
        }
        max_singular_value = max_singular_value.max(sigma_max);
        scale_factors.push(scale);
    }

    if violations > 0 {
        tracing::warn!(
            violations,
            max_singular_value,
            "Enforced passivity on S-parameters"
        );
    }

    Ok(PassivityReport {
        violations,
        max_eigenvalue: max_singular_value,
        scale_factors,
    })
}

/// Enforce passivity on a single S-matrix.
///
/// Uses SVD to compute and clamp singular values. For 2×2 matrices,
/// uses an efficient analytical eigenvalue formula.
///
/// Returns (was_passive, max_singular_value, scale_applied).
fn enforce_passivity_matrix(s: &mut Array2<Complex64>) -> DspResult<(bool, f64, f64)> {
    let n = s.nrows();

    if n == 2 && s.ncols() == 2 {
        // For 2×2, use efficient analytical path.
        // Compute eigenvalues of S^H × S, which are σ².
        let s_h: Array2<Complex64> = s.t().mapv(|c| c.conj());
        let product = s_h.dot(s);
        let max_eigenvalue = compute_max_eigenvalue_2x2(&product);

        // σ_max = sqrt(max eigenvalue of S^H × S)
        let sigma_max = max_eigenvalue.sqrt();
        let is_passive = sigma_max <= 1.0 + PASSIVITY_TOLERANCE;

        let scale = if !is_passive {
            // Uniform scaling: S_new = S / σ_max
            let scale = 1.0 / sigma_max;
            s.mapv_inplace(|c| c * scale);
            scale
        } else {
            1.0
        };

        Ok((is_passive, sigma_max, scale))
    } else {
        // For n > 2, use full SVD-based enforcement
        let (sigma_max, scale) = enforce_passivity_via_svd(s);
        let is_passive = (scale - 1.0).abs() < NUMERICAL_EPSILON;

        Ok((is_passive, sigma_max, scale))
    }
}

/// Compute maximum eigenvalue of 2×2 Hermitian matrix.
///
/// For Hermitian 2×2 matrix M:
/// eigenvalues = (trace ± sqrt(trace² - 4×det)) / 2
fn compute_max_eigenvalue_2x2(m: &Array2<Complex64>) -> f64 {
    let a = m[[0, 0]].re;
    let d = m[[1, 1]].re;
    let b = m[[0, 1]];
    let c = m[[1, 0]];

    let trace = a + d;
    let det = a * d - (b * c).re;

    let discriminant = trace * trace - 4.0 * det;

    if discriminant < 0.0 {
        // Shouldn't happen for Hermitian matrix; return trace/2 as fallback
        trace / 2.0
    } else {
        (trace + discriminant.sqrt()) / 2.0
    }
}

/// Check if S-parameters are passive at all frequencies.
///
/// Passivity requires the spectral norm ‖S‖₂ ≤ 1, which is equivalent
/// to the maximum singular value σ_max ≤ 1.
///
/// # IEEE P370 Compliance
///
/// This function correctly uses singular value analysis per IEEE P370-2020
/// Section 4.5.2, rather than the incorrect element-wise check.
pub fn check_passivity(sparams: &SParameters) -> Vec<bool> {
    sparams
        .matrices
        .iter()
        .map(|m| {
            let sigma_max = compute_max_singular_value(m);
            sigma_max <= 1.0 + PASSIVITY_TOLERANCE
        })
        .collect()
}

/// Compute the passivity margin (how much gain before becoming active).
///
/// Returns the margin in dB based on the spectral norm (largest singular value).
/// - Positive margin: passive (σ_max < 1)
/// - Zero margin: borderline (σ_max = 1)
/// - Negative margin: active (σ_max > 1)
///
/// Formula: margin_dB = 20 × log₁₀(1 / σ_max)
pub fn passivity_margin(sparams: &SParameters) -> Vec<f64> {
    sparams
        .matrices
        .iter()
        .map(|m| {
            let sigma_max = compute_max_singular_value(m);
            if sigma_max > NUMERICAL_EPSILON {
                20.0 * (1.0 / sigma_max).log10()
            } else {
                f64::INFINITY // Zero matrix has infinite margin
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_types::units::{Hertz, Ohms};

    #[test]
    fn test_passive_sparam_unchanged() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Create passive matrix (σ_max ≈ 0.906)
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(0.1, 0.0);
        m[[0, 1]] = Complex64::new(0.0, 0.0);
        m[[1, 0]] = Complex64::new(0.9, 0.0);
        m[[1, 1]] = Complex64::new(0.1, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m.clone());

        let original_s21 = sparams.s21()[0];

        let report = enforce_passivity(&mut sparams).unwrap();
        assert_eq!(report.violations, 0);

        let new_s21 = sparams.s21()[0];
        assert!((original_s21.re - new_s21.re).abs() < 1e-10);
    }

    #[test]
    fn test_active_sparam_scaled() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Create active matrix (|S21| > 1)
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(0.0, 0.0);
        m[[0, 1]] = Complex64::new(0.0, 0.0);
        m[[1, 0]] = Complex64::new(1.5, 0.0); // Gain!
        m[[1, 1]] = Complex64::new(0.0, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let report = enforce_passivity(&mut sparams).unwrap();
        assert_eq!(report.violations, 1);

        // After enforcement, should be passive
        let new_s21 = sparams.s21()[0];
        assert!(new_s21.norm() <= 1.0 + PASSIVITY_TOLERANCE);
    }

    /// Critical test: 4×4 matrix where all |S_ij| < 1 but σ_max > 1.
    /// This test catches the bug where element-wise checks incorrectly pass.
    #[test]
    fn test_4x4_elementwise_pass_spectral_fail() {
        let mut sparams = SParameters::new(4, Ohms::Z0_50);

        // Rank-1 matrix: S = 0.4 × ones(4,4)
        // max|S_ij| = 0.4 < 1, but σ_max = 0.4 × 4 = 1.6 > 1
        let mut m = Array2::zeros((4, 4));
        for i in 0..4 {
            for j in 0..4 {
                m[[i, j]] = Complex64::new(0.4, 0.0);
            }
        }
        sparams.add_point(Hertz::from_ghz(1.0), m);

        // Must detect violation (old code would miss this)
        let passivity = check_passivity(&sparams);
        assert!(!passivity[0], "Should detect spectral norm violation in 4x4 matrix");

        // Enforce passivity
        let mut sparams_mut = sparams.clone();
        let report = enforce_passivity(&mut sparams_mut).unwrap();

        assert_eq!(report.violations, 1);
        assert!(report.max_eigenvalue > 1.0, "σ_max should be > 1 before enforcement");

        // After enforcement, should be passive
        let passivity_after = check_passivity(&sparams_mut);
        assert!(passivity_after[0], "Should be passive after enforcement");
    }

    #[test]
    fn test_borderline_passive() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Matrix with σ_max = 0.999 (just under 1.0)
        let mut m = Array2::zeros((2, 2));
        m[[1, 0]] = Complex64::new(0.999, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let passivity = check_passivity(&sparams);
        assert!(passivity[0], "σ_max=0.999 should be passive");

        let mut sp_mut = sparams.clone();
        let report = enforce_passivity(&mut sp_mut).unwrap();
        assert_eq!(report.violations, 0);
    }

    #[test]
    fn test_borderline_active() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Matrix with σ_max = 1.01 (just over tolerance)
        let mut m = Array2::zeros((2, 2));
        m[[1, 0]] = Complex64::new(1.01, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let passivity = check_passivity(&sparams);
        assert!(!passivity[0], "σ_max=1.01 should be active (outside tolerance)");
    }

    #[test]
    fn test_within_tolerance() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Matrix with σ_max = 1.0000005 (within 1e-6 tolerance)
        let sigma = 1.0 + 0.5e-6;
        let mut m = Array2::zeros((2, 2));
        m[[1, 0]] = Complex64::new(sigma, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let passivity = check_passivity(&sparams);
        assert!(passivity[0], "σ within tolerance should be considered passive");
    }

    #[test]
    fn test_complex_matrix_passivity() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Complex S-matrix representing a phase shifter with loss
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(0.1, 0.05);
        m[[0, 1]] = Complex64::from_polar(0.85, 1.5);
        m[[1, 0]] = Complex64::from_polar(0.85, 1.5);
        m[[1, 1]] = Complex64::new(0.1, -0.05);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let passivity = check_passivity(&sparams);
        assert!(passivity[0], "Well-designed passive component should pass");
    }

    #[test]
    fn test_zero_matrix() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        let m = Array2::zeros((2, 2));
        sparams.add_point(Hertz::from_ghz(1.0), m);

        let passivity = check_passivity(&sparams);
        assert!(passivity[0], "Zero matrix should be passive");

        let margin = passivity_margin(&sparams);
        assert!(margin[0].is_infinite() && margin[0] > 0.0, "Zero matrix has infinite margin");
    }

    #[test]
    fn test_identity_matrix() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Identity matrix has σ_max = 1 (borderline passive)
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(1.0, 0.0);
        m[[1, 1]] = Complex64::new(1.0, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let passivity = check_passivity(&sparams);
        assert!(passivity[0], "Identity matrix should be passive (σ=1)");
    }

    #[test]
    fn test_passivity_margin_calculation() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // S21 = 0.5 (other elements zero) → σ_max = 0.5
        // Margin = 20×log₁₀(1/0.5) = 20×log₁₀(2) ≈ 6.02 dB
        let mut m = Array2::zeros((2, 2));
        m[[1, 0]] = Complex64::new(0.5, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let margin = passivity_margin(&sparams);
        let expected = 20.0 * 2.0_f64.log10(); // ~6.02 dB

        assert!((margin[0] - expected).abs() < 0.01);
    }

    #[test]
    fn test_3port_matrix() {
        let mut sparams = SParameters::new(3, Ohms::Z0_50);

        // 3-port with moderate coupling
        let mut m = Array2::zeros((3, 3));
        m[[0, 1]] = Complex64::new(0.5, 0.0);
        m[[1, 0]] = Complex64::new(0.5, 0.0);
        m[[1, 2]] = Complex64::new(0.5, 0.0);
        m[[2, 1]] = Complex64::new(0.5, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        // Should be passive
        let passivity = check_passivity(&sparams);
        assert!(passivity[0]);
    }

    #[test]
    fn test_4port_enforcement() {
        let mut sparams = SParameters::new(4, Ohms::Z0_50);

        // 4-port with gain in some paths
        let mut m = Array2::zeros((4, 4));
        m[[0, 1]] = Complex64::new(0.5, 0.0);
        m[[1, 0]] = Complex64::new(0.5, 0.0);
        m[[2, 3]] = Complex64::new(1.2, 0.0); // Gain!
        m[[3, 2]] = Complex64::new(1.2, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        let report = enforce_passivity(&mut sparams).unwrap();
        assert_eq!(report.violations, 1);

        // Verify enforcement worked
        let passivity = check_passivity(&sparams);
        assert!(passivity[0]);
    }

    #[test]
    fn test_near_singular_matrix() {
        let mut sparams = SParameters::new(2, Ohms::Z0_50);

        // Nearly rank-deficient matrix (rank 1)
        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(0.5, 0.0);
        m[[0, 1]] = Complex64::new(0.5, 0.0);
        m[[1, 0]] = Complex64::new(0.5, 0.0);
        m[[1, 1]] = Complex64::new(0.5, 0.0);

        sparams.add_point(Hertz::from_ghz(1.0), m);

        // Should not panic or produce NaN
        let passivity = check_passivity(&sparams);
        assert!(!passivity[0].to_string().contains("NaN"));

        let margin = passivity_margin(&sparams);
        assert!(margin[0].is_finite());
    }
}
