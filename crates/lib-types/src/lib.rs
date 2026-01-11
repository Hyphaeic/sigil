//! # lib-types
//!
//! Core type definitions for SI-Kernel signal integrity simulation.
//!
//! This crate provides foundational types used throughout the SI-Kernel workspace:
//! - Physical units with compile-time safety
//! - Waveform representation for time-domain signals
//! - S-parameter structures for frequency-domain data
//! - AMI-specific types for algorithmic model interfaces

pub mod units;
pub mod waveform;
pub mod sparams;
pub mod ami;

pub use units::*;
pub use waveform::*;
pub use sparams::*;
pub use ami::*;

/// Re-export num_complex for convenience
pub use num_complex::Complex64;
