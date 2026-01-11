//! Simple channel analysis example.
//!
//! This example demonstrates:
//! 1. Parsing a Touchstone S-parameter file
//! 2. Converting to pulse response
//! 3. Computing statistical eye diagram
//! 4. Running bit-by-bit simulation

use lib_dsp::convolution::ConvolutionEngine;
use lib_dsp::eye::StatisticalEyeAnalyzer;
use lib_dsp::prbs::PrbsGenerator;
use lib_dsp::sparam_convert::{sparam_to_pulse, ConversionConfig};
use lib_ibis::parse_touchstone;
use lib_types::units::{Hertz, Ohms, Seconds};

fn main() -> anyhow::Result<()> {
    // For this example, we'll create synthetic S-parameters
    // In practice, you'd load from a .s4p file
    let sparams = create_synthetic_channel();

    println!("=== SI-Kernel Simple Channel Example ===\n");

    // Convert to pulse response
    let config = ConversionConfig {
        num_fft_points: 4096,
        input_port: 0,
        output_port: 1,
        bit_time: Seconds::from_ps(31.25), // PCIe Gen 5
        enforce_causality: true,
        enforce_passivity: true,
    };

    println!("Converting S-parameters to pulse response...");
    let pulse = sparam_to_pulse(&sparams, &config)?;
    println!("  Pulse response: {} samples, {:.2} ns duration",
        pulse.len(), pulse.duration().as_ns());
    println!("  Peak amplitude: {:.4}", pulse.max_abs());

    // Statistical eye analysis
    println!("\nComputing statistical eye...");
    let samples_per_ui = 64;
    let analyzer = StatisticalEyeAnalyzer::new(samples_per_ui);
    let stat_eye = analyzer.analyze(&pulse);

    let eye_height = stat_eye.eye_height();
    let eye_width = stat_eye.eye_width_ui();
    let fom = analyzer.figure_of_merit(&stat_eye);

    println!("  Eye height: {:.4}", eye_height);
    println!("  Eye width: {:.2} UI", eye_width);
    println!("  Figure of merit: {:.4}", fom);

    // Bit-by-bit simulation (small sample)
    println!("\nRunning bit-by-bit simulation (10k bits)...");
    let conv_engine = ConvolutionEngine::from_waveform(&pulse)?;

    let mut prbs = PrbsGenerator::new(7); // PRBS-7 for quick demo
    let dt = Seconds(pulse.dt.0);
    let input = prbs.generate_nrz(10_000, samples_per_ui, dt);

    let output = conv_engine.convolve_waveform(&input);
    println!("  Output waveform: {} samples", output.len());
    println!("  Peak-to-peak: {:.4}", output.peak_to_peak());

    // Summary
    println!("\n=== Summary ===");
    if eye_height > 0.1 && eye_width > 0.3 {
        println!("Channel analysis: PASS - Eye is open");
    } else {
        println!("Channel analysis: MARGINAL - Eye may need equalization");
    }

    Ok(())
}

/// Create synthetic S-parameters for demonstration.
fn create_synthetic_channel() -> lib_types::sparams::SParameters {
    use lib_types::Complex64;
    use ndarray::Array2;

    let mut sp = lib_types::sparams::SParameters::new(2, Ohms::Z0_50);

    // Create a simple lossy transmission line model
    for i in 0..200 {
        let f = (i as f64 + 1.0) * 100e6; // 100 MHz to 20 GHz

        // Simple model: loss increases with sqrt(f), phase linear with f
        let loss_db = -0.5 * (f / 1e9).sqrt(); // dB per unit
        let loss = 10.0_f64.powf(loss_db / 20.0);
        let delay = 1e-9; // 1 ns delay
        let phase = -2.0 * std::f64::consts::PI * f * delay;

        let mut m = Array2::zeros((2, 2));
        m[[0, 0]] = Complex64::new(0.05, 0.0); // Small reflection
        m[[1, 0]] = Complex64::from_polar(loss, phase);
        m[[0, 1]] = Complex64::from_polar(loss, phase);
        m[[1, 1]] = Complex64::new(0.05, 0.0);

        sp.add_point(Hertz(f), m);
    }

    sp
}
