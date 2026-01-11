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
}

/// Result type for DSP operations.
pub type DspResult<T> = Result<T, DspError>;
