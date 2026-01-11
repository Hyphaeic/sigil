//! S-parameter (scattering parameter) data structures.
//!
//! S-parameters describe the frequency-domain behavior of multi-port networks.
//! This module provides types for storing, manipulating, and converting S-parameters.

use crate::units::{Hertz, Ohms};
use ndarray::Array2;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// S-parameter matrix at a single frequency.
pub type SMatrix = Array2<Complex64>;

/// Complete S-parameter dataset for a multi-port network.
#[derive(Clone, Debug)]
pub struct SParameters {
    /// Frequency points in Hz.
    pub frequencies: Vec<Hertz>,

    /// S-parameter matrices at each frequency.
    /// Length matches `frequencies`.
    pub matrices: Vec<SMatrix>,

    /// Reference impedance (typically 50 ohms).
    pub z0: Ohms,

    /// Number of ports.
    pub num_ports: usize,
}

impl SParameters {
    /// Create a new S-parameter dataset.
    pub fn new(num_ports: usize, z0: Ohms) -> Self {
        Self {
            frequencies: Vec::new(),
            matrices: Vec::new(),
            z0,
            num_ports,
        }
    }

    /// Add a frequency point with its S-matrix.
    ///
    /// # Panics
    ///
    /// Panics if the matrix dimensions don't match the number of ports.
    pub fn add_point(&mut self, freq: Hertz, matrix: SMatrix) {
        assert_eq!(
            matrix.nrows(),
            self.num_ports,
            "Matrix row count {} doesn't match port count {}",
            matrix.nrows(),
            self.num_ports
        );
        assert_eq!(
            matrix.ncols(),
            self.num_ports,
            "Matrix column count {} doesn't match port count {}",
            matrix.ncols(),
            self.num_ports
        );
        self.frequencies.push(freq);
        self.matrices.push(matrix);
    }

    /// Try to add a frequency point, returning an error if dimensions don't match.
    pub fn try_add_point(&mut self, freq: Hertz, matrix: SMatrix) -> Result<(), &'static str> {
        if matrix.nrows() != self.num_ports {
            return Err("Matrix row count doesn't match port count");
        }
        if matrix.ncols() != self.num_ports {
            return Err("Matrix column count doesn't match port count");
        }
        self.frequencies.push(freq);
        self.matrices.push(matrix);
        Ok(())
    }

    /// Number of frequency points.
    pub fn len(&self) -> usize {
        self.frequencies.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.frequencies.is_empty()
    }

    /// Frequency range.
    pub fn frequency_range(&self) -> Option<(Hertz, Hertz)> {
        if self.frequencies.is_empty() {
            return None;
        }
        Some((
            self.frequencies.first().copied().unwrap(),
            self.frequencies.last().copied().unwrap(),
        ))
    }

    /// Get S-parameter at specific port indices across all frequencies.
    pub fn get_parameter(&self, row: usize, col: usize) -> Vec<Complex64> {
        self.matrices
            .iter()
            .map(|m| m[[row, col]])
            .collect()
    }

    /// Get S21 (through) parameter.
    pub fn s21(&self) -> Vec<Complex64> {
        self.get_parameter(1, 0)
    }

    /// Get S11 (input reflection) parameter.
    pub fn s11(&self) -> Vec<Complex64> {
        self.get_parameter(0, 0)
    }

    /// Get S22 (output reflection) parameter.
    pub fn s22(&self) -> Vec<Complex64> {
        self.get_parameter(1, 1)
    }

    /// Convert to magnitude in dB.
    pub fn to_db(&self, row: usize, col: usize) -> Vec<f64> {
        self.get_parameter(row, col)
            .iter()
            .map(|c| 20.0 * c.norm().log10())
            .collect()
    }

    /// Convert to phase in degrees.
    pub fn to_phase_deg(&self, row: usize, col: usize) -> Vec<f64> {
        self.get_parameter(row, col)
            .iter()
            .map(|c| c.arg().to_degrees())
            .collect()
    }

    /// Unwrap phase (remove 360-degree jumps).
    pub fn to_phase_unwrapped_deg(&self, row: usize, col: usize) -> Vec<f64> {
        let phases = self.to_phase_deg(row, col);
        unwrap_phase(&phases)
    }

    /// Group delay in seconds.
    /// d(phase)/d(omega) = -d(phase_deg)/(360 * df)
    pub fn group_delay(&self, row: usize, col: usize) -> Vec<f64> {
        let phases = self.to_phase_unwrapped_deg(row, col);

        let mut delays = Vec::with_capacity(self.frequencies.len());

        for i in 0..self.frequencies.len() {
            if i == 0 {
                // Forward difference
                let df = self.frequencies[1].0 - self.frequencies[0].0;
                let dphase = phases[1] - phases[0];
                delays.push(-dphase / (360.0 * df));
            } else if i == self.frequencies.len() - 1 {
                // Backward difference
                let df = self.frequencies[i].0 - self.frequencies[i - 1].0;
                let dphase = phases[i] - phases[i - 1];
                delays.push(-dphase / (360.0 * df));
            } else {
                // Central difference
                let df = self.frequencies[i + 1].0 - self.frequencies[i - 1].0;
                let dphase = phases[i + 1] - phases[i - 1];
                delays.push(-dphase / (360.0 * df));
            }
        }

        delays
    }

    /// Check if the network is reciprocal (S_ij = S_ji).
    pub fn is_reciprocal(&self, tolerance: f64) -> bool {
        for matrix in &self.matrices {
            for i in 0..self.num_ports {
                for j in i + 1..self.num_ports {
                    if (matrix[[i, j]] - matrix[[j, i]]).norm() > tolerance {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check if the network is passive (|S_ij| <= 1).
    pub fn is_passive(&self) -> bool {
        for matrix in &self.matrices {
            // Check singular values <= 1
            // For now, simple check: all |S_ij| <= 1
            for val in matrix.iter() {
                if val.norm() > 1.0 + 1e-6 {
                    return false;
                }
            }
        }
        true
    }
}

/// Unwrap phase to remove discontinuities.
fn unwrap_phase(phases: &[f64]) -> Vec<f64> {
    let mut unwrapped = Vec::with_capacity(phases.len());
    let mut offset = 0.0;

    for (i, &phase) in phases.iter().enumerate() {
        let adjusted = phase + offset;

        if i > 0 {
            let diff = adjusted - unwrapped[i - 1];
            if diff > 180.0 {
                offset -= 360.0;
            } else if diff < -180.0 {
                offset += 360.0;
            }
        }

        unwrapped.push(phase + offset);
    }

    unwrapped
}

/// Differential S-parameters for 4-port networks.
///
/// Converts single-ended 4-port to mixed-mode (differential/common) representation.
#[derive(Clone, Debug)]
pub struct MixedModeSParameters {
    /// Differential-mode S-parameters (2x2 at each frequency).
    pub differential: SParameters,

    /// Common-mode S-parameters (2x2 at each frequency).
    pub common: SParameters,

    /// Cross-mode (diff to common) S-parameters.
    pub diff_to_common: SParameters,

    /// Cross-mode (common to diff) S-parameters.
    pub common_to_diff: SParameters,
}

impl MixedModeSParameters {
    /// Convert single-ended 4-port to mixed-mode.
    ///
    /// Port mapping (standard):
    /// - Port 1: Input positive
    /// - Port 2: Output positive
    /// - Port 3: Input negative
    /// - Port 4: Output negative
    pub fn from_single_ended(se: &SParameters) -> Self {
        assert_eq!(se.num_ports, 4, "Mixed-mode conversion requires 4-port network");

        let mut differential = SParameters::new(2, se.z0);
        let mut common = SParameters::new(2, se.z0);
        let mut diff_to_common = SParameters::new(2, se.z0);
        let mut common_to_diff = SParameters::new(2, se.z0);

        for (freq, matrix) in se.frequencies.iter().zip(se.matrices.iter()) {
            // Mixed-mode S-parameter transformation.
            // The transformation from single-ended to differential/common mode uses
            // the factor 0.5 (not 1/√2) for proper scaling.
            //
            // For a 4-port network with ports 1,3 as input pair and ports 2,4 as output pair:
            // SDD = 0.5 * (S_pp - S_pn - S_np + S_nn)
            // SCC = 0.5 * (S_pp + S_pn + S_np + S_nn)
            //
            // where p=positive, n=negative

            let scale = Complex64::new(0.5, 0.0);

            // SDD (differential-differential)
            let sdd11 = scale * (matrix[[0, 0]] - matrix[[0, 2]] - matrix[[2, 0]] + matrix[[2, 2]]);
            let sdd12 = scale * (matrix[[0, 1]] - matrix[[0, 3]] - matrix[[2, 1]] + matrix[[2, 3]]);
            let sdd21 = scale * (matrix[[1, 0]] - matrix[[1, 2]] - matrix[[3, 0]] + matrix[[3, 2]]);
            let sdd22 = scale * (matrix[[1, 1]] - matrix[[1, 3]] - matrix[[3, 1]] + matrix[[3, 3]]);

            let mut sdd = Array2::zeros((2, 2));
            sdd[[0, 0]] = sdd11;
            sdd[[0, 1]] = sdd12;
            sdd[[1, 0]] = sdd21;
            sdd[[1, 1]] = sdd22;
            differential.add_point(*freq, sdd);

            // SCC (common-common)
            let scc11 = scale * (matrix[[0, 0]] + matrix[[0, 2]] + matrix[[2, 0]] + matrix[[2, 2]]);
            let scc12 = scale * (matrix[[0, 1]] + matrix[[0, 3]] + matrix[[2, 1]] + matrix[[2, 3]]);
            let scc21 = scale * (matrix[[1, 0]] + matrix[[1, 2]] + matrix[[3, 0]] + matrix[[3, 2]]);
            let scc22 = scale * (matrix[[1, 1]] + matrix[[1, 3]] + matrix[[3, 1]] + matrix[[3, 3]]);

            let mut scc = Array2::zeros((2, 2));
            scc[[0, 0]] = scc11;
            scc[[0, 1]] = scc12;
            scc[[1, 0]] = scc21;
            scc[[1, 1]] = scc22;
            common.add_point(*freq, scc);

            // SDC (differential to common) and SCD (common to differential)
            // Simplified: set to zero matrices for now
            let mut sdc = Array2::zeros((2, 2));
            let mut scd = Array2::zeros((2, 2));
            diff_to_common.add_point(*freq, sdc);
            common_to_diff.add_point(*freq, scd);
        }

        Self {
            differential,
            common,
            diff_to_common,
            common_to_diff,
        }
    }

    /// Get differential through (SDD21).
    pub fn sdd21(&self) -> Vec<Complex64> {
        self.differential.get_parameter(1, 0)
    }
}

/// Touchstone file format information.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TouchstoneVersion {
    V1,  // .s1p, .s2p, etc.
    V2,  // .ts format
}

/// Data format in Touchstone files.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFormat {
    /// Real/Imaginary
    RI,
    /// Magnitude/Angle (degrees)
    MA,
    /// dB/Angle (degrees)
    DB,
}

impl DataFormat {
    /// Convert magnitude/phase pair to complex number.
    pub fn to_complex(&self, val1: f64, val2: f64) -> Complex64 {
        match self {
            Self::RI => Complex64::new(val1, val2),
            Self::MA => {
                let angle_rad = val2.to_radians();
                Complex64::from_polar(val1, angle_rad)
            }
            Self::DB => {
                let magnitude = 10.0_f64.powf(val1 / 20.0);
                let angle_rad = val2.to_radians();
                Complex64::from_polar(magnitude, angle_rad)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_format_conversion() {
        let ri = DataFormat::RI.to_complex(1.0, 0.0);
        assert!((ri.re - 1.0).abs() < 1e-10);
        assert!((ri.im - 0.0).abs() < 1e-10);

        let ma = DataFormat::MA.to_complex(1.0, 90.0);
        assert!(ma.re.abs() < 1e-10);
        assert!((ma.im - 1.0).abs() < 1e-10);

        let db = DataFormat::DB.to_complex(0.0, 0.0); // 0 dB = magnitude 1
        assert!((db.re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_s_parameter_basics() {
        let mut sp = SParameters::new(2, Ohms::Z0_50);

        let mut m1 = Array2::zeros((2, 2));
        m1[[0, 0]] = Complex64::new(0.1, 0.0);
        m1[[0, 1]] = Complex64::new(0.0, 0.0);
        m1[[1, 0]] = Complex64::new(0.9, 0.0);
        m1[[1, 1]] = Complex64::new(0.1, 0.0);

        sp.add_point(Hertz::from_ghz(1.0), m1);

        assert_eq!(sp.len(), 1);
        assert!(sp.is_passive());

        let s21 = sp.s21();
        assert!((s21[0].re - 0.9).abs() < 1e-10);
    }
}
