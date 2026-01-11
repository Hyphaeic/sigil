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

/// Channel configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Path to Touchstone file.
    pub touchstone: PathBuf,

    /// Input port (1-based in config, converted to 0-based).
    #[serde(default = "default_port")]
    pub input_port: usize,

    /// Output port (1-based in config, converted to 0-based).
    #[serde(default = "default_output_port")]
    pub output_port: usize,
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
}

fn default_num_bits() -> u64 { 1_000_000 }
fn default_prbs_order() -> u8 { 31 }
fn default_samples_per_ui() -> usize { 64 }
fn default_max_training() -> usize { 100 }

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            mode: SimulationMode::default(),
            num_bits: default_num_bits(),
            prbs_order: default_prbs_order(),
            samples_per_ui: default_samples_per_ui(),
            link_training: false,
            max_training_iterations: default_max_training(),
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

    // Validate port numbers
    if config.channel.input_port == 0 || config.channel.output_port == 0 {
        anyhow::bail!("Port numbers must be >= 1");
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
