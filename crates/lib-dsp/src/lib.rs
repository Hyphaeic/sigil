//! # lib-dsp
//!
//! High-performance DSP engine for SI-Kernel signal integrity simulation.
//!
//! This crate provides the mathematical core for signal processing:
//!
//! - **FFT/IFFT**: Fast Fourier transforms for frequency-domain operations
//! - **Convolution**: High-performance overlap-save convolution with Rayon
//! - **S-parameter Processing**: Conversion to time domain, interpolation
//! - **Causality Enforcement**: Hilbert transform for causal responses
//! - **Passivity Enforcement**: Ensure physical realizability
//! - **PRBS Generation**: Pseudo-random bit sequences for simulation
//! - **Eye Diagram**: Statistical and time-domain eye analysis

pub mod error;
pub mod fft;
pub mod convolution;
pub mod interpolation;
pub mod causality;
pub mod passivity;
pub mod prbs;
pub mod eye;
pub mod sparam_convert;

pub use error::DspError;
pub use fft::FftEngine;
pub use convolution::ConvolutionEngine;
pub use prbs::PrbsGenerator;
pub use eye::{EyeAnalyzer, StatisticalEyeAnalyzer};
