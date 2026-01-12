//! Simulation orchestration.

use crate::config::{SimulationConfig, SimulationMode};
use anyhow::{Context, Result};
use lib_dsp::convolution::ConvolutionEngine;
use lib_dsp::eye::{EyeAnalyzer, EyeMetrics, StatisticalEyeAnalyzer};
use lib_dsp::prbs::PrbsGenerator;
use lib_dsp::sparam_convert::{sparam_to_impulse, sparam_to_pulse, ConversionConfig};
use lib_dsp::window::WindowConfig;
use lib_ibis::parse_touchstone;
use lib_types::units::Seconds;
use lib_types::waveform::{StatisticalEye, Waveform};
use std::sync::Arc;

/// Simulation orchestrator.
pub struct Orchestrator {
    config: SimulationConfig,
    channel_pulse: Option<Waveform>,
    channel_impulse: Option<Waveform>,
}

impl Orchestrator {
    /// Create a new orchestrator.
    pub fn new(config: SimulationConfig) -> Result<Self> {
        Ok(Self {
            config,
            channel_pulse: None,
            channel_impulse: None,
        })
    }

    /// Run the simulation.
    pub fn run(&self) -> Result<SimulationResults> {
        tracing::info!("Starting simulation: {}", self.config.name);

        // Load and process channel
        let (channel_impulse, channel_pulse) = self.load_channel()?;

        // Run simulation based on mode
        let results = match self.config.simulation.mode {
            SimulationMode::Statistical => {
                self.run_statistical(&channel_pulse)?
            }
            SimulationMode::BitByBit => {
                self.run_bit_by_bit(&channel_impulse)?
            }
            SimulationMode::Hybrid => {
                let stat = self.run_statistical(&channel_pulse)?;
                let bbb = self.run_bit_by_bit(&channel_impulse)?;
                SimulationResults {
                    statistical_eye: stat.statistical_eye,
                    eye_metrics: bbb.eye_metrics.or(stat.eye_metrics),
                    channel_pulse: stat.channel_pulse,
                    output_waveform: bbb.output_waveform,
                    training_result: None,
                }
            }
        };

        tracing::info!("Simulation complete");
        Ok(results)
    }

    /// Load and process the channel S-parameters.
    fn load_channel(&self) -> Result<(Waveform, Waveform)> {
        tracing::info!("Loading channel from {:?}", self.config.channel.touchstone);

        let content = std::fs::read_to_string(&self.config.channel.touchstone)
            .context("Failed to read Touchstone file")?;

        let ts = parse_touchstone(&content)
            .context("Failed to parse Touchstone file")?;

        let conv_config = ConversionConfig {
            num_fft_points: 8192,
            input_port: self.config.channel.input_port - 1, // Convert to 0-based
            output_port: self.config.channel.output_port - 1,
            bit_time: self.config.bit_time(),
            enforce_causality: true,
            enforce_passivity: true,
            preserve_group_delay: true, // IBIS 7.2 compliant
            window_config: WindowConfig::default(), // HIGH-DSP-003 fix: IEEE P370 Kaiser beta=6
        };

        let impulse = sparam_to_impulse(&ts.sparams, &conv_config)
            .context("Failed to compute impulse response")?;

        let pulse = sparam_to_pulse(&ts.sparams, &conv_config)
            .context("Failed to compute pulse response")?;

        tracing::info!(
            "Channel loaded: {} samples, {:.2} ns duration",
            pulse.len(),
            pulse.duration().as_ns()
        );

        Ok((impulse, pulse))
    }

    /// Run statistical simulation.
    fn run_statistical(&self, channel_pulse: &Waveform) -> Result<SimulationResults> {
        tracing::info!("Running statistical analysis...");

        let analyzer = StatisticalEyeAnalyzer::new(self.config.simulation.samples_per_ui);
        let eye = analyzer.analyze(channel_pulse);

        let height = eye.eye_height();
        let width = eye.eye_width_ui();
        let fom = analyzer.figure_of_merit(&eye);

        tracing::info!(
            "Statistical eye: height={:.4}, width={:.2} UI, FOM={:.4}",
            height, width, fom
        );

        Ok(SimulationResults {
            statistical_eye: Some(eye),
            eye_metrics: Some(EyeMetrics {
                height,
                width_ui: width,
                jitter_rms: 0.0,
                snr: 0.0,
                ui_count: 0,
            }),
            channel_pulse: Some(channel_pulse.clone()),
            output_waveform: None,
            training_result: None,
        })
    }

    /// Run bit-by-bit simulation.
    fn run_bit_by_bit(&self, channel_impulse: &Waveform) -> Result<SimulationResults> {
        tracing::info!(
            "Running bit-by-bit simulation ({} bits)...",
            self.config.simulation.num_bits
        );

        // Create convolution engine
        let conv_engine = ConvolutionEngine::from_waveform(channel_impulse)
            .context("Failed to create convolution engine")?;

        // Generate PRBS
        let mut prbs = PrbsGenerator::new(self.config.simulation.prbs_order);
        let bit_time = self.config.bit_time();
        let samples_per_ui = self.config.simulation.samples_per_ui;
        let dt = Seconds(bit_time.0 / samples_per_ui as f64);

        let input_waveform = prbs.generate_nrz(
            self.config.simulation.num_bits,
            samples_per_ui,
            dt,
        );

        tracing::debug!("Generated {} input samples", input_waveform.len());

        // Convolve with channel
        let output_waveform = conv_engine.convolve_waveform(&input_waveform);

        tracing::debug!("Convolution complete: {} output samples", output_waveform.len());

        // Compute eye diagram
        // MED-006 FIX: Use configurable parameters instead of hard-coded values
        let eye_analyzer = EyeAnalyzer::new(
            samples_per_ui,
            self.config.simulation.voltage_bins,
            (self.config.simulation.voltage_min, self.config.simulation.voltage_max),
        );

        let eye_diagram = eye_analyzer.analyze(&output_waveform);
        let metrics = eye_analyzer.compute_metrics(&eye_diagram);

        tracing::info!(
            "Eye metrics: height={:.4}, width={:.2} UI",
            metrics.height, metrics.width_ui
        );

        Ok(SimulationResults {
            statistical_eye: None,
            eye_metrics: Some(metrics),
            channel_pulse: None,
            output_waveform: Some(output_waveform),
            training_result: None,
        })
    }
}

/// Simulation results.
#[derive(Debug)]
pub struct SimulationResults {
    /// Statistical eye (if computed).
    pub statistical_eye: Option<StatisticalEye>,

    /// Eye metrics.
    pub eye_metrics: Option<EyeMetrics>,

    /// Channel pulse response.
    pub channel_pulse: Option<Waveform>,

    /// Output waveform (for bit-by-bit).
    pub output_waveform: Option<Waveform>,

    /// Training result (if link training was performed).
    pub training_result: Option<TrainingResult>,
}

/// Link training result.
#[derive(Debug)]
pub struct TrainingResult {
    pub best_preset: u8,
    pub final_fom: f64,
    pub iterations: usize,
}
