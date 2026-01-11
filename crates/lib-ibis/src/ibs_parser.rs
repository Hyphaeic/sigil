//! IBIS (.ibs) file parser.
//!
//! Parses IBIS I/O buffer model files according to the IBIS 7.2 specification.
//! This is a partial implementation covering the most common keywords needed
//! for signal integrity simulation.
//!
//! Reference: IBIS Specification Version 7.2

use crate::error::ParseError;
use lib_types::units::{Volts, Ohms, Seconds};
use std::collections::HashMap;

/// Parsed IBIS file.
#[derive(Clone, Debug, Default)]
pub struct IbisFile {
    /// File header information.
    pub header: IbisHeader,

    /// Component definitions.
    pub components: Vec<Component>,

    /// Model definitions.
    pub models: Vec<Model>,

    /// Model selector definitions.
    pub model_selectors: Vec<ModelSelector>,
}

/// IBIS file header section.
#[derive(Clone, Debug, Default)]
pub struct IbisHeader {
    /// IBIS version.
    pub ibis_ver: String,

    /// File name.
    pub file_name: String,

    /// File revision.
    pub file_rev: String,

    /// Date.
    pub date: String,

    /// Source (vendor name).
    pub source: String,

    /// Notes.
    pub notes: String,

    /// Copyright.
    pub copyright: String,
}

/// Component definition.
#[derive(Clone, Debug, Default)]
pub struct Component {
    /// Component name.
    pub name: String,

    /// Manufacturer.
    pub manufacturer: String,

    /// Package information.
    pub package: Option<Package>,

    /// Pin definitions.
    pub pins: Vec<Pin>,
}

/// Package parasitics.
#[derive(Clone, Debug, Default)]
pub struct Package {
    /// R_pkg (typical, min, max) in Ohms.
    pub r_pkg: Option<(f64, f64, f64)>,

    /// L_pkg (typical, min, max) in Henries.
    pub l_pkg: Option<(f64, f64, f64)>,

    /// C_pkg (typical, min, max) in Farads.
    pub c_pkg: Option<(f64, f64, f64)>,
}

/// Pin definition.
#[derive(Clone, Debug, Default)]
pub struct Pin {
    /// Pin name/number.
    pub name: String,

    /// Signal name.
    pub signal_name: String,

    /// Associated model name.
    pub model_name: String,

    /// R_pin (optional).
    pub r_pin: Option<f64>,

    /// L_pin (optional).
    pub l_pin: Option<f64>,

    /// C_pin (optional).
    pub c_pin: Option<f64>,
}

/// I/O buffer model.
#[derive(Clone, Debug, Default)]
pub struct Model {
    /// Model name.
    pub name: String,

    /// Model type.
    pub model_type: ModelType,

    /// Enable (for 3-state outputs).
    pub enable: Enable,

    /// Voltage range (typical, min, max).
    pub voltage_range: Option<(f64, f64, f64)>,

    /// Temperature range (typical, min, max).
    pub temperature_range: Option<(f64, f64, f64)>,

    /// Pull-up reference voltage.
    pub pullup_reference: Option<f64>,

    /// Pull-down reference voltage.
    pub pulldown_reference: Option<f64>,

    /// Input threshold high (typical, min, max).
    pub vih: Option<(f64, f64, f64)>,

    /// Input threshold low (typical, min, max).
    pub vil: Option<(f64, f64, f64)>,

    /// Pullup I-V table.
    pub pullup: Option<IVTable>,

    /// Pulldown I-V table.
    pub pulldown: Option<IVTable>,

    /// Power clamp I-V table.
    pub power_clamp: Option<IVTable>,

    /// Ground clamp I-V table.
    pub gnd_clamp: Option<IVTable>,

    /// Rising waveform tables.
    pub rising_waveform: Vec<WaveformTable>,

    /// Falling waveform tables.
    pub falling_waveform: Vec<WaveformTable>,

    /// Algorithmic model reference.
    pub algorithmic_model: Option<AlgorithmicModel>,
}

/// Model type enumeration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ModelType {
    #[default]
    Input,
    Output,
    IO,
    ThreeState,
    OpenDrain,
    OpenSink,
    OpenSource,
    InputECL,
    OutputECL,
    IOECL,
    ThreeStateECL,
    Terminator,
    Series,
    SeriesSwitch,
}

impl ModelType {
    /// Parse from string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "input" => Some(Self::Input),
            "output" => Some(Self::Output),
            "i/o" | "io" => Some(Self::IO),
            "3-state" | "three-state" => Some(Self::ThreeState),
            "open_drain" => Some(Self::OpenDrain),
            "open_sink" => Some(Self::OpenSink),
            "open_source" => Some(Self::OpenSource),
            "input_ecl" => Some(Self::InputECL),
            "output_ecl" => Some(Self::OutputECL),
            "i/o_ecl" | "io_ecl" => Some(Self::IOECL),
            "3-state_ecl" => Some(Self::ThreeStateECL),
            "terminator" => Some(Self::Terminator),
            "series" => Some(Self::Series),
            "series_switch" => Some(Self::SeriesSwitch),
            _ => None,
        }
    }
}

/// Enable configuration for 3-state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Enable {
    #[default]
    Active,
    ActiveLow,
    ActiveHigh,
}

/// I-V table data.
#[derive(Clone, Debug, Default)]
pub struct IVTable {
    /// (Voltage, I_typical, I_min, I_max) points.
    pub points: Vec<(f64, f64, f64, f64)>,
}

impl IVTable {
    /// Interpolate current at a given voltage.
    pub fn current_at(&self, voltage: f64, corner: Corner) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        // Find bracketing points
        let mut lower_idx = 0;
        let mut upper_idx = self.points.len() - 1;

        for (i, point) in self.points.iter().enumerate() {
            if point.0 <= voltage {
                lower_idx = i;
            }
            if point.0 >= voltage {
                upper_idx = i;
                break;
            }
        }

        if lower_idx == upper_idx {
            let point = &self.points[lower_idx];
            return match corner {
                Corner::Typical => point.1,
                Corner::Min => point.2,
                Corner::Max => point.3,
            };
        }

        // Linear interpolation
        let lower = &self.points[lower_idx];
        let upper = &self.points[upper_idx];
        let frac = (voltage - lower.0) / (upper.0 - lower.0);

        let (i_lower, i_upper) = match corner {
            Corner::Typical => (lower.1, upper.1),
            Corner::Min => (lower.2, upper.2),
            Corner::Max => (lower.3, upper.3),
        };

        i_lower + frac * (i_upper - i_lower)
    }
}

/// Corner case selector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Corner {
    #[default]
    Typical,
    Min,
    Max,
}

/// Waveform table (V-T data).
#[derive(Clone, Debug, Default)]
pub struct WaveformTable {
    /// R_fixture value.
    pub r_fixture: f64,

    /// V_fixture value.
    pub v_fixture: f64,

    /// V_fixture_min (optional).
    pub v_fixture_min: Option<f64>,

    /// V_fixture_max (optional).
    pub v_fixture_max: Option<f64>,

    /// C_fixture (optional).
    pub c_fixture: Option<f64>,

    /// L_fixture (optional).
    pub l_fixture: Option<f64>,

    /// (Time, V_typical, V_min, V_max) points.
    pub points: Vec<(f64, f64, f64, f64)>,
}

/// Algorithmic model reference.
#[derive(Clone, Debug, Default)]
pub struct AlgorithmicModel {
    /// Executable file name (without extension).
    pub executable: String,

    /// AMI file name.
    pub ami_file: String,

    /// Supported model types.
    pub supported_types: Vec<AlgorithmicModelType>,
}

/// Algorithmic model type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlgorithmicModelType {
    /// Transmitter
    Tx,
    /// Receiver
    Rx,
}

/// Model selector.
#[derive(Clone, Debug, Default)]
pub struct ModelSelector {
    /// Selector name.
    pub name: String,

    /// Available models with descriptions.
    pub models: Vec<(String, String)>,
}

/// Parse an IBIS file from a string.
pub fn parse_ibs_file(content: &str) -> Result<IbisFile, ParseError> {
    let mut file = IbisFile::default();
    let mut current_section = Section::None;
    let mut current_component: Option<Component> = None;
    let mut current_model: Option<Model> = None;
    let mut current_table: Option<TableType> = None;

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('|') {
            continue;
        }

        // Check for section headers
        if line.starts_with('[') {
            // Save any pending section
            if let Some(comp) = current_component.take() {
                file.components.push(comp);
            }
            if let Some(model) = current_model.take() {
                file.models.push(model);
            }

            current_section = parse_section_header(line);
            current_table = None;

            match &current_section {
                Section::Component(name) => {
                    current_component = Some(Component {
                        name: name.clone(),
                        ..Default::default()
                    });
                }
                Section::Model(name) => {
                    current_model = Some(Model {
                        name: name.clone(),
                        ..Default::default()
                    });
                }
                _ => {}
            }
            continue;
        }

        // Parse line content based on current section
        match &current_section {
            Section::IbisVer => {
                file.header.ibis_ver = line.to_string();
            }
            Section::FileName => {
                file.header.file_name = line.to_string();
            }
            Section::FileRev => {
                file.header.file_rev = line.to_string();
            }
            Section::Date => {
                file.header.date = line.to_string();
            }
            Section::Source => {
                file.header.source = line.to_string();
            }
            Section::Component(_) => {
                if let Some(ref mut comp) = current_component {
                    parse_component_line(line, comp, &mut current_table);
                }
            }
            Section::Model(_) => {
                if let Some(ref mut model) = current_model {
                    parse_model_line(line, model, &mut current_table);
                }
            }
            _ => {}
        }
    }

    // Save any remaining sections
    if let Some(comp) = current_component {
        file.components.push(comp);
    }
    if let Some(model) = current_model {
        file.models.push(model);
    }

    Ok(file)
}

#[derive(Clone, Debug)]
enum Section {
    None,
    IbisVer,
    FileName,
    FileRev,
    Date,
    Source,
    Component(String),
    Model(String),
    ModelSelector(String),
    End,
}

#[derive(Clone, Copy, Debug)]
enum TableType {
    Pullup,
    Pulldown,
    PowerClamp,
    GndClamp,
    RisingWaveform,
    FallingWaveform,
    Pin,
}

fn parse_section_header(line: &str) -> Section {
    let line = line.trim_start_matches('[').trim_end_matches(']').trim();

    // Check for parameterized sections
    if line.to_lowercase().starts_with("component") {
        let name = line.split_whitespace().nth(1).unwrap_or("").to_string();
        return Section::Component(name);
    }
    if line.to_lowercase().starts_with("model") && !line.to_lowercase().starts_with("model_selector") {
        let name = line.split_whitespace().nth(1).unwrap_or("").to_string();
        return Section::Model(name);
    }
    if line.to_lowercase().starts_with("model_selector") {
        let name = line.split_whitespace().nth(1).unwrap_or("").to_string();
        return Section::ModelSelector(name);
    }

    match line.to_lowercase().as_str() {
        "ibis ver" | "ibis_ver" => Section::IbisVer,
        "file name" | "file_name" => Section::FileName,
        "file rev" | "file_rev" => Section::FileRev,
        "date" => Section::Date,
        "source" => Section::Source,
        "end" => Section::End,
        _ => Section::None,
    }
}

fn parse_component_line(line: &str, comp: &mut Component, current_table: &mut Option<TableType>) {
    let lower = line.to_lowercase();

    if lower.starts_with("manufacturer") {
        comp.manufacturer = line.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
    } else if lower.starts_with("[pin]") {
        *current_table = Some(TableType::Pin);
    } else if matches!(current_table, Some(TableType::Pin)) {
        // Parse pin line
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            comp.pins.push(Pin {
                name: parts[0].to_string(),
                signal_name: parts[1].to_string(),
                model_name: parts[2].to_string(),
                ..Default::default()
            });
        }
    }
}

fn parse_model_line(line: &str, model: &mut Model, current_table: &mut Option<TableType>) {
    let lower = line.to_lowercase();

    // Check for sub-section keywords
    if lower.starts_with("model_type") {
        let type_str = line.split_whitespace().nth(1).unwrap_or("");
        model.model_type = ModelType::from_str(type_str).unwrap_or_default();
    } else if lower.starts_with("[pullup]") {
        *current_table = Some(TableType::Pullup);
        model.pullup = Some(IVTable::default());
    } else if lower.starts_with("[pulldown]") {
        *current_table = Some(TableType::Pulldown);
        model.pulldown = Some(IVTable::default());
    } else if lower.starts_with("[power clamp]") || lower.starts_with("[power_clamp]") {
        *current_table = Some(TableType::PowerClamp);
        model.power_clamp = Some(IVTable::default());
    } else if lower.starts_with("[gnd clamp]") || lower.starts_with("[gnd_clamp]") {
        *current_table = Some(TableType::GndClamp);
        model.gnd_clamp = Some(IVTable::default());
    } else if lower.starts_with("[rising waveform]") || lower.starts_with("[rising_waveform]") {
        *current_table = Some(TableType::RisingWaveform);
        model.rising_waveform.push(WaveformTable::default());
    } else if lower.starts_with("[falling waveform]") || lower.starts_with("[falling_waveform]") {
        *current_table = Some(TableType::FallingWaveform);
        model.falling_waveform.push(WaveformTable::default());
    } else if lower.starts_with("[algorithmic model]") || lower.starts_with("[algorithmic_model]") {
        model.algorithmic_model = Some(AlgorithmicModel::default());
    } else if lower.starts_with("executable") {
        if let Some(ref mut am) = model.algorithmic_model {
            if let Some((_, val)) = line.split_once(char::is_whitespace) {
                am.executable = val.trim().to_string();
            }
        }
    } else if matches!(current_table, Some(TableType::Pullup | TableType::Pulldown | TableType::PowerClamp | TableType::GndClamp)) {
        // Parse I-V table line
        if let Some(point) = parse_iv_line(line) {
            match current_table {
                Some(TableType::Pullup) => {
                    if let Some(ref mut table) = model.pullup {
                        table.points.push(point);
                    }
                }
                Some(TableType::Pulldown) => {
                    if let Some(ref mut table) = model.pulldown {
                        table.points.push(point);
                    }
                }
                Some(TableType::PowerClamp) => {
                    if let Some(ref mut table) = model.power_clamp {
                        table.points.push(point);
                    }
                }
                Some(TableType::GndClamp) => {
                    if let Some(ref mut table) = model.gnd_clamp {
                        table.points.push(point);
                    }
                }
                _ => {}
            }
        }
    }
}

fn parse_iv_line(line: &str) -> Option<(f64, f64, f64, f64)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() >= 4 {
        let v = parse_value_with_suffix(parts[0])?;
        let i_typ = parse_value_with_suffix(parts[1])?;
        let i_min = parse_value_with_suffix(parts[2])?;
        let i_max = parse_value_with_suffix(parts[3])?;
        Some((v, i_typ, i_min, i_max))
    } else {
        None
    }
}

fn parse_value_with_suffix(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("na") || s == "-" {
        return Some(0.0); // NA values
    }

    // Handle SI suffixes
    let (num_part, suffix) = if let Some(idx) = s.find(|c: char| c.is_alphabetic()) {
        (&s[..idx], &s[idx..])
    } else {
        (s, "")
    };

    let num: f64 = num_part.parse().ok()?;
    let suffix_lower = suffix.to_lowercase();
    let multiplier = match suffix_lower.as_str() {
        "t" => 1e12,
        "g" => 1e9,
        "meg" => 1e6,  // "meg" specifically for mega (to avoid confusion with milli)
        "k" => 1e3,
        "" => 1.0,
        "m" => 1e-3,   // Single "m" is milli
        "u" | "μ" => 1e-6,
        "n" => 1e-9,
        "p" => 1e-12,
        "f" => 1e-15,
        _ => return None,  // Unknown suffix should fail, not silently default
    };

    Some(num * multiplier)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_IBS: &str = r#"
[IBIS Ver]      7.2
[File Name]     sample.ibs
[File Rev]      1.0
[Date]          January 1, 2024
[Source]        Test Vendor

[Component]     TEST_CHIP
Manufacturer    Test Vendor Inc

[Pin] signal_name  model_name
1     TXDATA       TX_MODEL
2     RXDATA       RX_MODEL

[Model]         TX_MODEL
Model_type      Output

[End]
"#;

    #[test]
    fn test_parse_sample_ibs() {
        let result = parse_ibs_file(SAMPLE_IBS).unwrap();

        assert_eq!(result.header.ibis_ver, "7.2");
        assert_eq!(result.header.file_name, "sample.ibs");
        assert_eq!(result.components.len(), 1);
        assert_eq!(result.components[0].name, "TEST_CHIP");
        assert_eq!(result.models.len(), 1);
        assert_eq!(result.models[0].name, "TX_MODEL");
        assert_eq!(result.models[0].model_type, ModelType::Output);
    }

    #[test]
    fn test_parse_value_with_suffix() {
        assert!((parse_value_with_suffix("1.5m").unwrap() - 1.5e-3).abs() < 1e-15);
        assert!((parse_value_with_suffix("100n").unwrap() - 100e-9).abs() < 1e-18);
        assert!((parse_value_with_suffix("3.3").unwrap() - 3.3).abs() < 1e-10);
        assert!((parse_value_with_suffix("10k").unwrap() - 10e3).abs() < 1e-6);
    }
}
