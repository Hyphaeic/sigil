//! Simulation orchestration.

use crate::config::{SimulationConfig, SimulationMode};
use anyhow::{Context, Result};
use lib_dsp::convolution::ConvolutionEngine;
use lib_dsp::eye::{EyeAnalyzer, EyeMetrics, StatisticalEyeAnalyzer};
use lib_dsp::prbs::PrbsGenerator;
use lib_dsp::sparam_convert::{impulse_to_pulse, sparam_to_impulse, ConversionConfig};
use lib_dsp::window::WindowConfig;
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

        // HIGH-NEW-004: Validate link_training configuration
        if self.config.simulation.link_training {
            if self.config.tx.is_some() || self.config.rx.is_some() {
                tracing::error!(
                    "link_training=true with Tx/Rx models configured, but AMI integration not yet wired to orchestrator"
                );
                tracing::error!(
                    "See HIGH-NEW-004 in CRITICALISSUES.md for implementation status"
                );
                tracing::error!(
                    "Workaround: Set link_training=false for channel-only simulation"
                );
                anyhow::bail!(
                    "Link training requested but not yet implemented. \
                     Set 'simulation.link_training = false' or remove Tx/Rx model configs."
                );
            } else {
                tracing::warn!(
                    "link_training=true but no Tx/Rx models configured — ignoring (channel-only mode)"
                );
            }
        }

        // Load and process channel
        let (channel_impulse, channel_pulse, channel_response) = self.load_channel()?;

        // Run simulation based on mode
        let mut results = match self.config.simulation.mode {
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
                    eye_diagram: bbb.eye_diagram,
                    channel_pulse: stat.channel_pulse,
                    output_waveform: bbb.output_waveform,
                    channel_response: None,
                    training_result: None,
                }
            }
        };

        // Channel-level data is mode-independent; attach for outputs and the
        // TUI viewer regardless of which mode produced the eye.
        if results.channel_pulse.is_none() {
            results.channel_pulse = Some(channel_pulse);
        }
        results.channel_response = Some(channel_response);

        tracing::info!("Simulation complete");
        Ok(results)
    }

    /// Load and process the channel S-parameters.
    ///
    /// Returns the impulse response, pulse response, and the through-path
    /// frequency response (insertion loss curve) for reporting/visualization.
    fn load_channel(&self) -> Result<(Waveform, Waveform, ChannelResponse)> {
        tracing::info!("Loading channel from {:?}", self.config.channel.touchstone);

        // Path-aware parse: the .sNp extension authoritatively sets the
        // port count (wrapped multi-line 4-port exports are ambiguous to
        // shape inference).
        let ts = lib_ibis::parse_touchstone_file(&self.config.channel.touchstone)
            .context("Failed to parse Touchstone file")?;

        // HIGH-PHYS-004 FIX: Handle differential vs. single-ended modes
        let channel_mode = self.config.channel.get_mode();

        // Determine which S-parameters to use based on mode
        let (sparams_to_use, mode_description) = match channel_mode {
            crate::config::ChannelMode::SingleEnded { input_port, output_port } => {
                tracing::info!("Single-ended mode: S{}{}",  output_port, input_port);
                (ts.sparams.clone(), format!("Single-ended (S{}{})", output_port, input_port))
            }
            crate::config::ChannelMode::Differential { input_p, input_n, output_p, output_n } => {
                // Verify we have a 4-port network
                if ts.sparams.num_ports != 4 {
                    anyhow::bail!(
                        "Differential mode requires 4-port S-parameters, got {}-port",
                        ts.sparams.num_ports
                    );
                }

                tracing::info!(
                    "Differential mode: ports ({}+,{}-) → ({}+,{}-)",
                    input_p, input_n, output_p, output_n
                );

                // Verify port mapping matches standard differential configuration
                // Standard: (1+, 3-) → (2+, 4-)
                if (input_p, input_n, output_p, output_n) != (1, 3, 2, 4) {
                    tracing::warn!(
                        "Non-standard port mapping detected. \
                         Standard is (1+,3-) → (2+,4-), got ({}+,{}-) → ({}+,{}-)",
                        input_p, input_n, output_p, output_n
                    );
                }

                // Convert to mixed-mode S-parameters
                // Per IEEE P370-2020 Section 7.4
                let mixed_mode = lib_types::sparams::MixedModeSParameters::from_single_ended(&ts.sparams);

                // Log mode conversion metrics (use mid-band values).
                // mode_conversion_ratio() returns LINEAR |SDC21|/|SDD21|
                // ratios; convert to dB before display and thresholding.
                // (Previously the linear value was treated as dB, so the
                // "high mode conversion" warning fired for every
                // differential channel, including ones with zero coupling.)
                let mcr_ratios = mixed_mode.mode_conversion_ratio();
                if !mcr_ratios.is_empty() {
                    let mid_idx = mcr_ratios.len() / 2;
                    let (sdc_lin, scd_lin) = mcr_ratios[mid_idx];
                    let sdc_db = 20.0 * sdc_lin.max(1e-12).log10();
                    let scd_db = 20.0 * scd_lin.max(1e-12).log10();
                    tracing::info!(
                        "Mode conversion ratio (mid-band): SDC21={:.2} dB, SCD21={:.2} dB",
                        sdc_db, scd_db
                    );

                    if sdc_db > -40.0 || scd_db > -40.0 {
                        tracing::warn!(
                            "High mode conversion detected (SDC={:.2} dB, SCD={:.2} dB). \
                             This indicates impedance imbalance and will degrade signal integrity.",
                            sdc_db, scd_db
                        );
                    }
                }

                // Log effective insertion loss including mode conversion
                let eff_il_db_vec = mixed_mode.effective_insertion_loss_db();
                if !eff_il_db_vec.is_empty() {
                    let mid_idx = eff_il_db_vec.len() / 2;
                    let eff_il_db = eff_il_db_vec[mid_idx];
                    tracing::info!("Effective insertion loss (including mode conversion): {:.2} dB", eff_il_db);

                    // Compare to pure differential loss
                    let sdd21 = mixed_mode.differential.s21();
                    let pure_diff_loss_db = -20.0 * sdd21[mid_idx].norm().log10();
                    let mode_conversion_penalty = eff_il_db - pure_diff_loss_db;

                    if mode_conversion_penalty.abs() > 0.5 {
                        tracing::info!(
                            "Mode conversion adds {:.2} dB to insertion loss",
                            mode_conversion_penalty
                        );
                    }
                }

                // Use differential S-parameters (SDD)
                (mixed_mode.differential, "Differential (SDD21)".to_string())
            }
        };

        tracing::info!("Channel mode: {}", mode_description);

        // Determine port indices for conversion
        let (input_port, output_port) = match channel_mode {
            crate::config::ChannelMode::SingleEnded { input_port, output_port } => {
                (input_port - 1, output_port - 1) // Convert to 0-based
            }
            crate::config::ChannelMode::Differential { .. } => {
                // For mixed-mode, always use ports 0 → 1 (differential input → differential output)
                (0, 1)
            }
        };

        let conv_config = ConversionConfig {
            num_fft_points: 8192,
            input_port,
            output_port,
            bit_time: self.config.bit_time(),
            enforce_causality: true,
            enforce_passivity: true,
            preserve_group_delay: true, // IBIS 7.2 compliant
            window_config: WindowConfig::default(), // HIGH-DSP-003 fix: IEEE P370 Kaiser beta=6
            fft_strategy: lib_dsp::FftSizeStrategy::Auto, // HIGH-DSP-004 fix: configurable sizing
        };

        let impulse = sparam_to_impulse(&sparams_to_use, &conv_config)
            .context("Failed to compute impulse response")?;

        // Derive the pulse from the impulse we already computed instead of
        // calling sparam_to_pulse, which would redo the entire conversion
        // (passivity, causality, IFFT) and log every warning twice.
        let pulse = impulse_to_pulse(&impulse, self.config.bit_time());

        // Through-path magnitude vs frequency for reporting/visualization.
        let label = match channel_mode {
            crate::config::ChannelMode::SingleEnded { input_port: i, output_port: o } => {
                format!("S{}{}", o, i)
            }
            crate::config::ChannelMode::Differential { .. } => "SDD21".to_string(),
        };
        let transfer = sparams_to_use.get_parameter(output_port, input_port);
        let points: Vec<(f64, f64)> = sparams_to_use
            .frequencies
            .iter()
            .zip(transfer.iter())
            .map(|(f, s)| (f.0 * 1e-9, 20.0 * s.norm().max(1e-12).log10()))
            .collect();
        let channel_response = ChannelResponse { label, points };

        tracing::info!(
            "Channel loaded: {} samples, {:.2} ns duration",
            pulse.len(),
            pulse.duration().as_ns()
        );

        Ok((impulse, pulse, channel_response))
    }

    /// Run statistical simulation.
    fn run_statistical(&self, channel_pulse: &Waveform) -> Result<SimulationResults> {
        tracing::info!("Running statistical analysis...");

        // HIGH-NEW-003 FIX (revised with CRIT-NEW-005): Per IBIS 7.2 §11.2,
        // statistical folding requires uniformly spaced UI-aligned samples.
        // The pulse from sparam_to_pulse is sampled on the FFT grid, whose dt
        // generally does not divide the UI — so resample onto the UI grid at
        // the configured samples_per_ui instead of degrading the analysis to
        // whatever round(UI/dt) happens to be (previously collapsed to 1
        // sample/UI, destroying phase resolution).
        let bit_time = self.config.bit_time();
        let samples_per_ui = self.config.simulation.samples_per_ui;
        let target_dt = Seconds(bit_time.0 / samples_per_ui as f64);

        let pulse_aligned = if lib_dsp::are_compatible_dt(channel_pulse.dt, target_dt, 1e-6) {
            tracing::debug!(
                "Pulse dt={:.3e}s already UI-aligned ({} samples/UI)",
                channel_pulse.dt.0,
                samples_per_ui
            );
            channel_pulse.clone()
        } else {
            tracing::info!(
                "Resampling pulse for statistical analysis: dt={:.3e}s → {:.3e}s ({} samples/UI) per IBIS 7.2 §11.2",
                channel_pulse.dt.0,
                target_dt.0,
                samples_per_ui
            );
            lib_dsp::resample_waveform(channel_pulse, target_dt)
                .context("Failed to resample pulse response to UI grid")?
        };

        let analyzer = StatisticalEyeAnalyzer::new(samples_per_ui);
        let eye = analyzer.analyze(&pulse_aligned);

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
            eye_diagram: None,
            channel_pulse: Some(channel_pulse.clone()),
            output_waveform: None,
            channel_response: None,
            training_result: None,
        })
    }

    /// Run bit-by-bit simulation.
    fn run_bit_by_bit(&self, channel_impulse: &Waveform) -> Result<SimulationResults> {
        tracing::info!(
            "Running bit-by-bit simulation ({} bits)...",
            self.config.simulation.num_bits
        );

        let bit_time = self.config.bit_time();
        let samples_per_ui = self.config.simulation.samples_per_ui;
        let stimulus_dt = Seconds(bit_time.0 / samples_per_ui as f64);

        // HIGH-PHYS-006 FIX: Validate Nyquist criterion
        let channel_bw = lib_dsp::estimate_bandwidth(channel_impulse)
            .context("Failed to estimate channel bandwidth")?;

        lib_dsp::validate_nyquist(bit_time, samples_per_ui, channel_bw)
            .context("Nyquist criterion violated")?;

        tracing::debug!(
            "Nyquist check passed: {:.1} GHz bandwidth, {} samples/UI",
            channel_bw * 1e-9,
            samples_per_ui
        );

        // HIGH-PHYS-006 FIX: Ensure impulse and stimulus have compatible dt
        let impulse_for_conv = if !lib_dsp::are_compatible_dt(
            channel_impulse.dt,
            stimulus_dt,
            1e-6, // 0.0001% relative tolerance
        ) {
            tracing::warn!(
                "Resampling impulse: dt={:.3e}s → dt={:.3e}s (samples_per_ui={})",
                channel_impulse.dt.0,
                stimulus_dt.0,
                samples_per_ui
            );
            lib_dsp::resample_waveform(channel_impulse, stimulus_dt)
                .context("Failed to resample impulse response")?
        } else {
            tracing::debug!(
                "Impulse dt={:.3e}s matches stimulus dt (no resampling needed)",
                channel_impulse.dt.0
            );
            channel_impulse.clone()
        };

        // Create convolution engine with aligned impulse
        let conv_engine = ConvolutionEngine::from_waveform(&impulse_for_conv)
            .context("Failed to create convolution engine")?;

        // Generate PRBS
        let mut prbs = PrbsGenerator::new(self.config.simulation.prbs_order);

        let input_waveform = prbs.generate_nrz(
            self.config.simulation.num_bits,
            samples_per_ui,
            stimulus_dt,
        );

        tracing::debug!("Generated {} input samples", input_waveform.len());

        // CRIT-NEW-002 FIX: Convolve with channel and discard initial transient
        // Per IBIS 7.2 §11.3, bit-by-bit eyes must exclude ≥3× impulse warmup
        // to avoid over-reporting ISI from turn-on transients.
        let output_waveform = conv_engine.convolve_waveform_steady_state(&input_waveform, true);

        let warmup_discarded = conv_engine.warmup_samples();
        tracing::info!(
            "Convolution complete: {} output samples (discarded {} warmup samples per IBIS 7.2 §11.3)",
            output_waveform.len(),
            warmup_discarded
        );

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
            eye_diagram: Some(eye_diagram),
            channel_pulse: None,
            output_waveform: Some(output_waveform),
            channel_response: None,
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

    /// Bit-by-bit eye diagram histogram (if computed).
    pub eye_diagram: Option<lib_types::waveform::EyeDiagram>,

    /// Channel pulse response.
    pub channel_pulse: Option<Waveform>,

    /// Output waveform (for bit-by-bit).
    pub output_waveform: Option<Waveform>,

    /// Through-path frequency response (insertion loss) of the channel.
    pub channel_response: Option<ChannelResponse>,

    /// Training result (if link training was performed).
    pub training_result: Option<TrainingResult>,
}

/// Channel through-path frequency response for reporting.
#[derive(Clone, Debug)]
pub struct ChannelResponse {
    /// Which parameter this is (e.g. "S21", "SDD21").
    pub label: String,

    /// (frequency in GHz, magnitude in dB) points.
    pub points: Vec<(f64, f64)>,
}

/// Link training result.
#[derive(Debug)]
pub struct TrainingResult {
    pub best_preset: u8,
    pub final_fom: f64,
    pub iterations: usize,
}
