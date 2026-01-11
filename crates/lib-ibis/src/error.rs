//! Error types for parsing operations.

use thiserror::Error;

/// Errors that can occur during file parsing.
#[derive(Debug, Error)]
pub enum ParseError {
    /// I/O error reading the file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Syntax error in the file.
    #[error("Syntax error at line {line}, column {column}: {message}")]
    Syntax {
        line: usize,
        column: usize,
        message: String,
    },

    /// Missing required section or keyword.
    #[error("Missing required {kind}: {name}")]
    Missing { kind: &'static str, name: String },

    /// Invalid value for a field.
    #[error("Invalid value for {field}: {message}")]
    InvalidValue { field: String, message: String },

    /// Unsupported file version.
    #[error("Unsupported {format} version: {version}")]
    UnsupportedVersion { format: String, version: String },

    /// Invalid file format.
    #[error("Invalid {format} format: {message}")]
    InvalidFormat { format: String, message: String },

    /// Nom parsing error (internal).
    #[error("Parse error: {0}")]
    Nom(String),
}

impl ParseError {
    /// Create a syntax error at a specific location.
    pub fn syntax(line: usize, column: usize, message: impl Into<String>) -> Self {
        Self::Syntax {
            line,
            column,
            message: message.into(),
        }
    }

    /// Create a missing section error.
    pub fn missing_section(name: impl Into<String>) -> Self {
        Self::Missing {
            kind: "section",
            name: name.into(),
        }
    }

    /// Create a missing keyword error.
    pub fn missing_keyword(name: impl Into<String>) -> Self {
        Self::Missing {
            kind: "keyword",
            name: name.into(),
        }
    }

    /// Create an invalid value error.
    pub fn invalid_value(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidValue {
            field: field.into(),
            message: message.into(),
        }
    }
}

/// Convert nom errors to our error type.
impl<'a> From<nom::Err<nom::error::Error<&'a str>>> for ParseError {
    fn from(err: nom::Err<nom::error::Error<&'a str>>) -> Self {
        match err {
            nom::Err::Incomplete(_) => ParseError::Nom("Incomplete input".to_string()),
            nom::Err::Error(e) | nom::Err::Failure(e) => {
                // LOW-004 FIX: Include error kind for better debugging context
                let preview: String = e.input.chars().take(20).collect();
                ParseError::Nom(format!("{:?} at '{}...'", e.code, preview))
            }
        }
    }
}
