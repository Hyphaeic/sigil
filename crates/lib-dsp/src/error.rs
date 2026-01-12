//! Error types for DSP operations.

use thiserror::Error;

/// Errors that can occur during DSP operations.
#[derive(Debug, Error)]
pub enum DspError {
    /// FFT size is not a power of 2.
    #[error("FFT size must be power of 2, got {0}")]
    InvalidFftSize(usize),

    /// Input length mismatch.
    #[error("Input length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    /// Insufficient data for operation.
    #[error("Insufficient data: need at least {needed}, got {got}")]
    InsufficientData { needed: usize, got: usize },

    /// Interpolation failed.
    #[error("Interpolation failed: {0}")]
    InterpolationFailed(String),

    /// Causality enforcement failed.
    #[error("Causality enforcement failed: {0}")]
    CausalityFailed(String),

    /// Passivity enforcement failed.
    #[error("Passivity enforcement failed: {0}")]
    PassivityFailed(String),

    /// Invalid S-parameter data.
    #[error("Invalid S-parameter data: {0}")]
    InvalidSParams(String),

    /// Numerical instability detected.
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Operation not supported.
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Invalid configuration parameter.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Nyquist criterion violated (insufficient sampling).
    ///
    /// # HIGH-PHYS-006 Fix
    ///
    /// Indicates that the sampling rate is insufficient for the signal bandwidth,
    /// which would cause aliasing and incorrect convolution results.
    #[error("Nyquist violation: need {required} samples/UI for {bandwidth_ghz:.1} GHz bandwidth, got {provided}")]
    NyquistViolation {
        required: usize,
        provided: usize,
        bandwidth_ghz: f64,
    },
}

/// Result type for DSP operations.
pub type DspResult<T> = Result<T, DspError>;
