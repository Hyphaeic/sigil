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
//! - **Windowing**: Kaiser-Bessel and other windows for spectral processing
//! - **Resampling**: Waveform time-step alignment (HIGH-PHYS-006)
//! - **PRBS Generation**: Pseudo-random bit sequences for simulation
//! - **Eye Diagram**: Statistical and time-domain eye analysis

pub mod error;
pub mod fft;
pub mod convolution;
pub mod interpolation;
pub mod causality;
pub mod passivity;
pub mod window;
pub mod resample;
pub mod prbs;
pub mod eye;
pub mod sparam_convert;

pub use error::DspError;
pub use fft::FftEngine;
pub use convolution::{ConvolutionEngine, FftSizeStrategy};
pub use prbs::PrbsGenerator;
pub use eye::{EyeAnalyzer, StatisticalEyeAnalyzer};
pub use causality::{
    apply_group_delay, enforce_causality, enforce_causality_with_delay_preservation,
    extract_reference_delay,
};
pub use sparam_convert::{sparam_to_impulse, sparam_to_pulse, ConversionConfig};
pub use resample::{are_compatible_dt, estimate_bandwidth, resample_waveform, validate_nyquist};
