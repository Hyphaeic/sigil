//! Error types for AMI FFI operations.

use lib_types::ami::SessionState;
use std::time::Duration;
use thiserror::Error;

/// Errors that can occur during AMI model operations.
#[derive(Debug, Error)]
pub enum AmiError {
    /// Failed to load the shared library.
    #[error("Failed to load library '{path}': {source}")]
    LoadError {
        path: String,
        #[source]
        source: libloading::Error,
    },

    /// Required symbol not found in library.
    #[error("Symbol '{symbol}' not found in library")]
    SymbolNotFound { symbol: String },

    /// AMI_Init returned an error.
    #[error("AMI_Init failed with code {code}: {message}")]
    InitFailed { code: i64, message: String },

    /// AMI_GetWave returned an error.
    #[error("AMI_GetWave failed with code {code}")]
    GetWaveFailed { code: i64 },

    /// AMI_Close returned an error.
    #[error("AMI_Close failed with code {code}")]
    CloseFailed { code: i64 },

    /// Model execution timed out.
    #[error("Model execution timed out after {0:?}")]
    Timeout(Duration),

    /// Model panicked or crashed.
    #[error("Model panicked: {0}")]
    ModelPanicked(String),

    /// Model produced invalid output.
    #[error("Invalid output from model: {0}")]
    InvalidOutput(String),

    /// Invalid session state for operation.
    #[error("Invalid session state: expected {expected:?}, got {actual:?}")]
    InvalidState {
        expected: SessionState,
        actual: SessionState,
    },

    /// Operation not supported by this model.
    #[error("Operation '{operation}' not supported by model")]
    NotSupported { operation: String },

    /// Memory allocation failed.
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Invalid parameter.
    #[error("Invalid parameter '{name}': {reason}")]
    InvalidParameter { name: String, reason: String },

    /// Back-channel communication error.
    #[error("Back-channel error: {0}")]
    BackChannelError(String),

    /// Too many orphaned threads from previous timeouts.
    #[error("Too many orphaned threads ({count}), max allowed is {max}")]
    TooManyOrphanedThreads { count: usize, max: usize },

    /// Buffer overrun detected (model wrote beyond allocated size).
    ///
    /// # HIGH-FFI-004 Fix
    ///
    /// Per IBIS 7.2 Section 10.2.3: "wave_size indicates the maximum number
    /// of samples". Some buggy models write beyond this limit, causing memory
    /// corruption. This error is raised when sentinel values detect overrun.
    #[error("Buffer overrun detected: model '{model}' wrote beyond allocated {size} samples")]
    BufferOverrun {
        model: String,
        size: usize,
        detected_index: Option<usize>,
    },
}

impl AmiError {
    /// Create a load error.
    pub fn load_error(path: impl Into<String>, source: libloading::Error) -> Self {
        Self::LoadError {
            path: path.into(),
            source,
        }
    }

    /// Create a symbol not found error.
    pub fn symbol_not_found(symbol: impl Into<String>) -> Self {
        Self::SymbolNotFound {
            symbol: symbol.into(),
        }
    }

    /// Create an init failed error.
    pub fn init_failed(code: i64, message: impl Into<String>) -> Self {
        Self::InitFailed {
            code,
            message: message.into(),
        }
    }

    /// Create an invalid state error.
    pub fn invalid_state(expected: SessionState, actual: SessionState) -> Self {
        Self::InvalidState { expected, actual }
    }

    /// Check if this is a recoverable error.
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Timeout(_) | Self::GetWaveFailed { .. } | Self::InvalidOutput(_)
        )
    }

    /// Check if the session should be considered faulted.
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::ModelPanicked(_) | Self::InitFailed { .. } | Self::AllocationFailed(_)
        )
    }
}

/// Result type for AMI operations.
pub type AmiResult<T> = Result<T, AmiError>;
