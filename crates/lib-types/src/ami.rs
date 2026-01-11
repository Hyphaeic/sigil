//! AMI (Algorithmic Modeling Interface) types.
//!
//! These types define the interface for IBIS-AMI models, including
//! parameter handling, session state, and simulation configuration.

use crate::units::{BitsPerSecond, Hertz, Seconds};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::c_void;

/// AMI parameter value types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AmiValue {
    /// Floating-point value.
    Float(f64),

    /// Integer value.
    Integer(i64),

    /// String value.
    String(String),

    /// Boolean value.
    Boolean(bool),

    /// List of values.
    List(Vec<AmiValue>),

    /// Tap values (for equalizer coefficients).
    Taps(Vec<f64>),

    /// Corner case (Typ, Min, Max).
    Corner(f64, f64, f64),

    /// Range with default.
    Range {
        default: f64,
        min: f64,
        max: f64,
    },

    /// Table data (2D).
    Table {
        row_labels: Vec<f64>,
        col_labels: Vec<f64>,
        data: Vec<Vec<f64>>,
    },
}

impl AmiValue {
    /// Try to extract as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Try to extract as i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to extract as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to extract as bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            Self::Integer(i) => Some(*i != 0),
            _ => None,
        }
    }
}

/// Collection of AMI parameters.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AmiParameters {
    /// Parameter name to value mapping.
    pub params: HashMap<String, AmiValue>,
}

impl AmiParameters {
    /// Create empty parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a parameter by name.
    pub fn get(&self, name: &str) -> Option<&AmiValue> {
        self.params.get(name)
    }

    /// Set a parameter.
    pub fn set(&mut self, name: impl Into<String>, value: AmiValue) {
        self.params.insert(name.into(), value);
    }

    /// Check if parameter exists.
    pub fn contains(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }

    /// Convert to AMI parameter string format.
    ///
    /// Returns a string in the IBIS-AMI format:
    /// `(param1 value1) (param2 value2) ...`
    pub fn to_ami_string(&self) -> String {
        let mut parts = Vec::new();

        for (name, value) in &self.params {
            let value_str = match value {
                AmiValue::Float(v) => format!("{v}"),
                AmiValue::Integer(v) => format!("{v}"),
                AmiValue::String(s) => format!("\"{s}\""),
                AmiValue::Boolean(b) => if *b { "True" } else { "False" }.to_string(),
                AmiValue::List(items) => {
                    let item_strs: Vec<String> = items
                        .iter()
                        .map(|v| match v {
                            AmiValue::Float(f) => format!("{f}"),
                            AmiValue::Integer(i) => format!("{i}"),
                            AmiValue::String(s) => format!("\"{s}\""),
                            _ => String::new(),
                        })
                        .collect();
                    item_strs.join(" ")
                }
                AmiValue::Taps(taps) => {
                    let tap_strs: Vec<String> = taps.iter().map(|t| format!("{t}")).collect();
                    tap_strs.join(" ")
                }
                _ => continue, // Skip complex types for now
            };

            parts.push(format!("({name} {value_str})"));
        }

        parts.join(" ")
    }

    /// Parse from AMI parameter string.
    pub fn from_ami_string(s: &str) -> Result<Self, AmiParseError> {
        // Simple parser for AMI parameter strings
        // Full implementation would use nom
        let mut params = HashMap::new();

        // Basic parsing: find (name value) pairs
        let mut chars = s.chars().peekable();
        let mut depth = 0;
        let mut current_token = String::new();
        let mut tokens = Vec::new();

        while let Some(c) = chars.next() {
            match c {
                '(' => {
                    depth += 1;
                    if depth == 1 {
                        current_token.clear();
                    } else {
                        current_token.push(c);
                    }
                }
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        tokens.push(std::mem::take(&mut current_token));
                    } else {
                        current_token.push(c);
                    }
                }
                _ if depth > 0 => {
                    current_token.push(c);
                }
                _ => {}
            }
        }

        // Parse each token as "name value"
        for token in tokens {
            let token = token.trim();
            if let Some(space_idx) = token.find(char::is_whitespace) {
                let name = token[..space_idx].trim().to_string();
                let value_str = token[space_idx..].trim();

                let value = if let Ok(i) = value_str.parse::<i64>() {
                    AmiValue::Integer(i)
                } else if let Ok(f) = value_str.parse::<f64>() {
                    AmiValue::Float(f)
                } else if value_str.eq_ignore_ascii_case("true") {
                    AmiValue::Boolean(true)
                } else if value_str.eq_ignore_ascii_case("false") {
                    AmiValue::Boolean(false)
                } else {
                    AmiValue::String(value_str.trim_matches('"').to_string())
                };

                params.insert(name, value);
            }
        }

        Ok(Self { params })
    }
}

/// Error parsing AMI parameters.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AmiParseError {
    #[error("Unbalanced parentheses")]
    UnbalancedParentheses,

    #[error("Invalid parameter format: {0}")]
    InvalidFormat(String),
}

/// AMI model type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmiModelType {
    /// Transmitter model.
    Tx,
    /// Receiver model.
    Rx,
}

/// AMI session state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    /// Session created but AMI_Init not called.
    Uninitialized,
    /// AMI_Init completed successfully.
    Initialized,
    /// AMI_GetWave has been called at least once.
    Active,
    /// AMI_Close has been called.
    Closed,
    /// Model faulted (crash, timeout, etc.).
    Faulted,
}

/// Result from AMI_Init call.
#[derive(Clone, Debug)]
pub struct AmiInitResult {
    /// Return code from AMI_Init (0 = success).
    pub return_code: i64,

    /// Output parameters from the model.
    pub output_params: AmiParameters,

    /// Message from the model (info or error).
    pub message: Option<String>,

    /// Whether the model supports GetWave.
    pub supports_getwave: bool,

    /// Whether the model modified the impulse response.
    pub modified_impulse: bool,
}

/// Result from AMI_GetWave call.
#[derive(Clone, Debug)]
pub struct AmiGetWaveResult {
    /// Return code from AMI_GetWave (0 = success).
    pub return_code: i64,

    /// Clock edge times from CDR.
    pub clock_times: Vec<f64>,

    /// Output parameters from the model.
    pub output_params: AmiParameters,
}

/// Configuration for AMI simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AmiConfig {
    /// Bit time (UI duration).
    pub bit_time: Seconds,

    /// Sample interval for waveforms.
    pub sample_interval: Seconds,

    /// Number of samples per UI.
    pub samples_per_ui: usize,

    /// Number of aggressors (for crosstalk).
    pub num_aggressors: usize,

    /// Model-specific input parameters.
    pub input_params: AmiParameters,
}

impl AmiConfig {
    /// Create config for a given data rate.
    pub fn for_data_rate(data_rate: BitsPerSecond, samples_per_ui: usize) -> Self {
        let bit_time = data_rate.ui();
        let sample_interval = Seconds(bit_time.0 / samples_per_ui as f64);

        Self {
            bit_time,
            sample_interval,
            samples_per_ui,
            num_aggressors: 0,
            input_params: AmiParameters::new(),
        }
    }

    /// Create config for PCIe Gen 5.
    pub fn pcie_gen5(samples_per_ui: usize) -> Self {
        Self::for_data_rate(BitsPerSecond::from_gtps(32.0), samples_per_ui)
    }

    /// Create config for PCIe Gen 6.
    pub fn pcie_gen6(samples_per_ui: usize) -> Self {
        Self::for_data_rate(BitsPerSecond::from_gtps(64.0), samples_per_ui)
    }
}

/// Tx equalization preset (PCIe Gen 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TxPreset {
    /// Preset number (P0-P10).
    pub preset: u8,

    /// Pre-cursor coefficient (typically negative).
    pub pre_cursor: i8,

    /// Main cursor (typically positive, largest).
    pub main_cursor: i8,

    /// Post-cursor coefficient (typically negative).
    pub post_cursor: i8,
}

impl TxPreset {
    /// PCIe Gen 5 preset definitions.
    pub const PCIE_GEN5_PRESETS: [TxPreset; 11] = [
        TxPreset { preset: 0, pre_cursor: 0, main_cursor: 24, post_cursor: 0 },
        TxPreset { preset: 1, pre_cursor: 0, main_cursor: 20, post_cursor: -4 },
        TxPreset { preset: 2, pre_cursor: 0, main_cursor: 18, post_cursor: -6 },
        TxPreset { preset: 3, pre_cursor: 0, main_cursor: 16, post_cursor: -8 },
        TxPreset { preset: 4, pre_cursor: 0, main_cursor: 14, post_cursor: -10 },
        TxPreset { preset: 5, pre_cursor: -2, main_cursor: 22, post_cursor: 0 },
        TxPreset { preset: 6, pre_cursor: -2, main_cursor: 18, post_cursor: -4 },
        TxPreset { preset: 7, pre_cursor: -3, main_cursor: 17, post_cursor: -4 },
        TxPreset { preset: 8, pre_cursor: -4, main_cursor: 16, post_cursor: -4 },
        TxPreset { preset: 9, pre_cursor: -5, main_cursor: 15, post_cursor: -4 },
        TxPreset { preset: 10, pre_cursor: -6, main_cursor: 18, post_cursor: 0 },
    ];

    /// Get preset by number.
    pub fn get(preset: u8) -> Option<TxPreset> {
        Self::PCIE_GEN5_PRESETS.get(preset as usize).copied()
    }

    /// Convert to tap values normalized by absolute sum.
    ///
    /// Returns taps where the sum of absolute values equals 1.0.
    /// This is useful when the taps represent a constraint on maximum swing.
    ///
    /// For preset P5 (-2, 22, 0):
    /// - Sum of absolutes = 24
    /// - Returns (-0.0833, 0.917, 0.0)
    /// - |pre| + |main| + |post| = 1.0
    pub fn to_normalized_taps_absolute(&self) -> (f64, f64, f64) {
        let sum = (self.pre_cursor.abs() + self.main_cursor.abs() + self.post_cursor.abs()) as f64;
        if sum == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        (
            self.pre_cursor as f64 / sum,
            self.main_cursor as f64 / sum,
            self.post_cursor as f64 / sum,
        )
    }

    /// Convert to tap values normalized by signed sum (DC gain = 1).
    ///
    /// Returns taps where the algebraic sum equals 1.0.
    /// This is useful when the taps represent FIR filter coefficients
    /// and you want unity DC gain.
    ///
    /// For preset P5 (-2, 22, 0):
    /// - Signed sum = 20
    /// - Returns (-0.1, 1.1, 0.0)
    /// - pre + main + post = 1.0
    pub fn to_normalized_taps_signed(&self) -> Option<(f64, f64, f64)> {
        let sum = (self.pre_cursor + self.main_cursor + self.post_cursor) as f64;
        if sum == 0.0 {
            return None; // Cannot normalize if sum is zero
        }
        Some((
            self.pre_cursor as f64 / sum,
            self.main_cursor as f64 / sum,
            self.post_cursor as f64 / sum,
        ))
    }

    /// Convert to raw tap values as f64.
    ///
    /// Returns the tap values without any normalization.
    pub fn to_raw_taps(&self) -> (f64, f64, f64) {
        (
            self.pre_cursor as f64,
            self.main_cursor as f64,
            self.post_cursor as f64,
        )
    }

    /// Legacy method - alias for to_normalized_taps_absolute.
    #[deprecated(since = "0.2.0", note = "Use to_normalized_taps_absolute() instead")]
    pub fn to_normalized_taps(&self) -> (f64, f64, f64) {
        self.to_normalized_taps_absolute()
    }
}

/// Back-channel message types for Tx-Rx communication.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BackChannelMessage {
    /// Rx requests Tx to change preset.
    PresetRequest { preset: u8 },

    /// Rx reports figure of merit.
    FigureOfMerit { value: f64 },

    /// Tx reports current coefficients.
    TxCoefficients {
        pre_cursor: f64,
        main_cursor: f64,
        post_cursor: f64,
    },

    /// Rx reports eye margin.
    EyeMargin {
        height_mv: f64,
        width_ps: f64,
    },

    /// Generic parameter update.
    ParameterUpdate {
        name: String,
        value: AmiValue,
    },

    /// Training complete.
    TrainingComplete,
}

/// Training state for link adaptation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingState {
    /// Not started.
    Idle,
    /// Sweeping Tx presets.
    PresetSweep,
    /// Coarse coefficient adaptation.
    CoarseAdaptation,
    /// Fine coefficient adaptation.
    FineAdaptation,
    /// Training converged.
    Converged,
    /// Training failed.
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ami_parameters_roundtrip() {
        let mut params = AmiParameters::new();
        params.set("data_rate", AmiValue::Float(32e9));
        params.set("preset", AmiValue::Integer(5));
        params.set("model_name", AmiValue::String("test".to_string()));

        let s = params.to_ami_string();
        let parsed = AmiParameters::from_ami_string(&s).unwrap();

        assert_eq!(
            parsed.get("data_rate").and_then(|v| v.as_f64()),
            Some(32e9)
        );
        assert_eq!(
            parsed.get("preset").and_then(|v| v.as_i64()),
            Some(5)
        );
    }

    #[test]
    fn test_tx_preset_normalization_absolute() {
        let p5 = TxPreset::get(5).unwrap();
        let (pre, main, post) = p5.to_normalized_taps_absolute();

        // Sum of absolute values should be 1
        assert!((pre.abs() + main.abs() + post.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tx_preset_normalization_signed() {
        let p5 = TxPreset::get(5).unwrap();
        let (pre, main, post) = p5.to_normalized_taps_signed().unwrap();

        // Signed sum should be 1
        assert!((pre + main + post - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ami_config_pcie_gen5() {
        let config = AmiConfig::pcie_gen5(64);

        // 32 GT/s = 31.25 ps UI
        assert!((config.bit_time.as_ps() - 31.25).abs() < 0.01);
        assert_eq!(config.samples_per_ui, 64);
    }
}
