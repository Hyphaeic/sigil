//! Simulation configuration loading and validation.

use anyhow::{Context, Result};
use lib_types::ami::AmiConfig;
use lib_types::units::{BitsPerSecond, PcieGeneration, Seconds};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Top-level simulation configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Simulation name/description.
    pub name: String,

    /// PCIe generation.
    #[serde(default)]
    pub pcie_gen: PcieGen,

    /// Channel configuration.
    pub channel: ChannelConfig,

    /// Transmitter configuration.
    pub tx: Option<ModelConfig>,

    /// Receiver configuration.
    pub rx: Option<ModelConfig>,

    /// Simulation parameters.
    #[serde(default)]
    pub simulation: SimulationParams,

    /// Output configuration.
    #[serde(default)]
    pub output: OutputConfig,
}

/// PCIe generation selection.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PcieGen {
    Gen1,
    Gen2,
    Gen3,
    Gen4,
    #[default]
    Gen5,
    Gen6,
}

impl From<PcieGen> for PcieGeneration {
    fn from(gen: PcieGen) -> Self {
        match gen {
            PcieGen::Gen1 => PcieGeneration::Gen1,
            PcieGen::Gen2 => PcieGeneration::Gen2,
            PcieGen::Gen3 => PcieGeneration::Gen3,
            PcieGen::Gen4 => PcieGeneration::Gen4,
            PcieGen::Gen5 => PcieGeneration::Gen5,
            PcieGen::Gen6 => PcieGeneration::Gen6,
        }
    }
}

/// Channel mode: single-ended or differential.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ChannelMode {
    /// Single-ended channel (use single S-parameter entry).
    #[serde(rename = "single_ended")]
    SingleEnded {
        /// Input port (1-based in config, converted to 0-based).
        #[serde(default = "default_port")]
        input_port: usize,
        /// Output port (1-based in config, converted to 0-based).
        #[serde(default = "default_output_port")]
        output_port: usize,
    },
    /// Differential channel (use mixed-mode S-parameters).
    ///
    /// # IEEE P370 Compliance
    ///
    /// Per IEEE P370-2020 Section 7.4: "Full 4x4 mixed-mode analysis is
    /// required for differential channels operating above 16 Gbaud."
    ///
    /// This mode computes SDD (differential-differential), SDC (diff-to-common),
    /// and SCD (common-to-diff) terms to properly account for mode conversion.
    Differential {
        /// Input positive port (1-based in config).
        input_p: usize,
        /// Input negative port (1-based in config).
        input_n: usize,
        /// Output positive port (1-based in config).
        output_p: usize,
        /// Output negative port (1-based in config).
        output_n: usize,
    },
}

impl Default for ChannelMode {
    fn default() -> Self {
        Self::SingleEnded {
            input_port: 1,
            output_port: 2,
        }
    }
}

/// Channel configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Path to Touchstone file.
    pub touchstone: PathBuf,

    /// Channel mode (single-ended or differential).
    ///
    /// # Examples
    ///
    /// Single-ended:
    /// ```json
    /// "mode": {"type": "single_ended", "input_port": 1, "output_port": 2}
    /// ```
    ///
    /// Differential (4-port):
    /// ```json
    /// "mode": {"type": "differential", "input_p": 1, "input_n": 3, "output_p": 2, "output_n": 4}
    /// ```
    #[serde(default)]
    pub mode: ChannelMode,

    // Deprecated fields for backward compatibility
    /// Input port (1-based, deprecated - use mode instead).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_port: Option<usize>,

    /// Output port (1-based, deprecated - use mode instead).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_port: Option<usize>,
}

impl ChannelConfig {
    /// Get the channel mode, falling back to deprecated fields if needed.
    pub fn get_mode(&self) -> ChannelMode {
        // If legacy input_port/output_port are set, use them
        if let (Some(input), Some(output)) = (self.input_port, self.output_port) {
            tracing::warn!(
                "Using deprecated input_port/output_port fields. \
                 Please migrate to 'mode' configuration."
            );
            return ChannelMode::SingleEnded {
                input_port: input,
                output_port: output,
            };
        }
        self.mode.clone()
    }
}

fn default_port() -> usize { 1 }
fn default_output_port() -> usize { 2 }

/// AMI model configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to IBIS file.
    pub ibis: PathBuf,

    /// Model name within IBIS file.
    pub model: String,

    /// Path to AMI parameter file.
    pub ami: Option<PathBuf>,

    /// Path to shared library.
    pub library: Option<PathBuf>,

    /// Parameter overrides.
    #[serde(default)]
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
}

/// Simulation parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationParams {
    /// Simulation mode.
    #[serde(default)]
    pub mode: SimulationMode,

    /// Number of bits for bit-by-bit simulation.
    #[serde(default = "default_num_bits")]
    pub num_bits: u64,

    /// PRBS order for pattern generation.
    #[serde(default = "default_prbs_order")]
    pub prbs_order: u8,

    /// Samples per UI.
    #[serde(default = "default_samples_per_ui")]
    pub samples_per_ui: usize,

    /// Enable link training.
    #[serde(default)]
    pub link_training: bool,

    /// Maximum training iterations.
    #[serde(default = "default_max_training")]
    pub max_training_iterations: usize,

    // MED-006 FIX: Make eye diagram parameters configurable instead of hard-coded
    /// Number of voltage bins for eye diagram.
    #[serde(default = "default_voltage_bins")]
    pub voltage_bins: usize,

    /// Minimum voltage for eye diagram range.
    #[serde(default = "default_voltage_min")]
    pub voltage_min: f64,

    /// Maximum voltage for eye diagram range.
    #[serde(default = "default_voltage_max")]
    pub voltage_max: f64,
}

fn default_num_bits() -> u64 { 1_000_000 }
fn default_prbs_order() -> u8 { 31 }
fn default_samples_per_ui() -> usize { 64 }
fn default_max_training() -> usize { 100 }
fn default_voltage_bins() -> usize { 256 }
fn default_voltage_min() -> f64 { -1.5 }
fn default_voltage_max() -> f64 { 1.5 }

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            mode: SimulationMode::default(),
            num_bits: default_num_bits(),
            prbs_order: default_prbs_order(),
            samples_per_ui: default_samples_per_ui(),
            link_training: false,
            max_training_iterations: default_max_training(),
            voltage_bins: default_voltage_bins(),
            voltage_min: default_voltage_min(),
            voltage_max: default_voltage_max(),
        }
    }
}

/// Simulation mode.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimulationMode {
    /// Statistical analysis only.
    #[default]
    Statistical,
    /// Full bit-by-bit simulation.
    BitByBit,
    /// Statistical followed by bit-by-bit verification.
    Hybrid,
}

/// Output configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Generate eye diagram.
    #[serde(default = "default_true")]
    pub eye_diagram: bool,

    /// Generate bathtub curve.
    #[serde(default)]
    pub bathtub: bool,

    /// Generate waveform output.
    #[serde(default)]
    pub waveforms: bool,

    /// Generate pulse response.
    #[serde(default = "default_true")]
    pub pulse_response: bool,
}

fn default_true() -> bool { true }

/// Load configuration from a file.
pub fn load_config(path: &Path) -> Result<SimulationConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {:?}", path))?;

    let config: SimulationConfig = if path.extension().map_or(false, |e| e == "json") {
        serde_json::from_str(&content)?
    } else {
        // Assume TOML
        toml::from_str(&content)
            .with_context(|| "Failed to parse config as TOML")?
    };

    validate_config(&config)?;

    Ok(config)
}

/// Validate configuration.
fn validate_config(config: &SimulationConfig) -> Result<()> {
    // Check that touchstone file exists
    if !config.channel.touchstone.exists() {
        anyhow::bail!(
            "Touchstone file not found: {:?}",
            config.channel.touchstone
        );
    }

    // Validate TX model files if specified
    if let Some(ref tx) = config.tx {
        validate_model_config(tx, "TX")?;
    }

    // Validate RX model files if specified
    if let Some(ref rx) = config.rx {
        validate_model_config(rx, "RX")?;
    }

    // Validate port numbers (1-based) based on mode
    let mode = config.channel.get_mode();
    match mode {
        ChannelMode::SingleEnded { input_port, output_port } => {
            if input_port == 0 || output_port == 0 {
                anyhow::bail!(
                    "Port numbers must be 1-based (got input={}, output={})",
                    input_port,
                    output_port
                );
            }
        }
        ChannelMode::Differential { input_p, input_n, output_p, output_n } => {
            if input_p == 0 || input_n == 0 || output_p == 0 || output_n == 0 {
                anyhow::bail!(
                    "Port numbers must be 1-based (got input_p={}, input_n={}, output_p={}, output_n={})",
                    input_p, input_n, output_p, output_n
                );
            }
        }
    }

    // Validate PRBS order
    if ![7, 9, 11, 15, 23, 31].contains(&config.simulation.prbs_order) {
        anyhow::bail!(
            "Invalid PRBS order: {}. Must be 7, 9, 11, 15, 23, or 31",
            config.simulation.prbs_order
        );
    }

    Ok(())
}

/// Validate model configuration file paths.
fn validate_model_config(model: &ModelConfig, label: &str) -> Result<()> {
    // Check IBIS file exists
    if !model.ibis.exists() {
        anyhow::bail!("{} IBIS file not found: {:?}", label, model.ibis);
    }

    // Check AMI file exists if specified
    if let Some(ref ami) = model.ami {
        if !ami.exists() {
            anyhow::bail!("{} AMI file not found: {:?}", label, ami);
        }
    }

    // Check library file exists if specified
    if let Some(ref library) = model.library {
        if !library.exists() {
            anyhow::bail!("{} library file not found: {:?}", label, library);
        }
    }

    Ok(())
}

/// Get the bit time for the configured PCIe generation.
impl SimulationConfig {
    pub fn bit_time(&self) -> Seconds {
        let gen: PcieGeneration = self.pcie_gen.into();
        gen.ui()
    }

    pub fn data_rate(&self) -> BitsPerSecond {
        let gen: PcieGeneration = self.pcie_gen.into();
        gen.data_rate()
    }

    pub fn ami_config(&self) -> AmiConfig {
        AmiConfig::for_data_rate(self.data_rate(), self.simulation.samples_per_ui)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcie_gen_conversion() {
        let gen5: PcieGeneration = PcieGen::Gen5.into();
        assert!((gen5.ui().as_ps() - 31.25).abs() < 0.01);
    }
}
