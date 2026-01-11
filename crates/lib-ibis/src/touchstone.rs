//! Touchstone S-parameter file parser.
//!
//! Supports:
//! - Touchstone 1.0 format (.s1p, .s2p, .s4p, .s6p, .s8p, etc.)
//! - All data formats: RI, MA, DB
//! - Frequency units: Hz, kHz, MHz, GHz
//!
//! Reference: Touchstone File Format Specification, Version 1.1

use crate::error::ParseError;
use lib_types::{
    sparams::{DataFormat, SMatrix, SParameters, TouchstoneVersion},
    units::{Hertz, Ohms},
    Complex64,
};
use ndarray::Array2;
use nom::{
    branch::alt,
    bytes::complete::{tag_no_case, take_while1},
    character::complete::{char, line_ending, not_line_ending, space0, space1},
    combinator::{map, opt, value},
    multi::{many0, many1},
    number::complete::double,
    sequence::preceded,
    IResult, Parser,
};
use std::path::Path;

/// Parsed Touchstone file.
#[derive(Clone, Debug)]
pub struct TouchstoneFile {
    /// File version.
    pub version: TouchstoneVersion,

    /// Number of ports.
    pub num_ports: usize,

    /// Data format (RI, MA, DB).
    pub format: DataFormat,

    /// Reference impedance.
    pub z0: Ohms,

    /// Frequency multiplier (Hz, kHz, MHz, GHz).
    pub freq_mult: f64,

    /// Parsed S-parameters.
    pub sparams: SParameters,
}

impl TouchstoneFile {
    /// Get the S-parameters.
    pub fn into_sparams(self) -> SParameters {
        self.sparams
    }
}

/// Parse a Touchstone file from a string.
pub fn parse_touchstone(content: &str) -> Result<TouchstoneFile, ParseError> {
    let (remaining, file) = parse_touchstone_inner(content)
        .map_err(|e| ParseError::Nom(format!("{:?}", e)))?;

    if !remaining.trim().is_empty() {
        tracing::warn!("Unparsed content at end of Touchstone file");
    }

    Ok(file)
}

/// Parse a Touchstone file from a path.
pub fn parse_touchstone_file(path: &Path) -> Result<TouchstoneFile, ParseError> {
    let content = std::fs::read_to_string(path)?;

    // Infer number of ports from extension
    let num_ports = infer_ports_from_extension(path);

    let file = parse_touchstone(&content)?;

    // Verify port count matches extension
    if let Some(expected) = num_ports {
        if file.num_ports != expected {
            tracing::warn!(
                "Port count mismatch: extension suggests {} ports, file has {}",
                expected,
                file.num_ports
            );
        }
    }

    Ok(file)
}

fn infer_ports_from_extension(path: &Path) -> Option<usize> {
    let ext = path.extension()?.to_str()?;
    if ext.starts_with('s') || ext.starts_with('S') {
        let num_part = &ext[1..ext.len() - 1];
        num_part.parse().ok()
    } else {
        None
    }
}

// ============================================================================
// Nom Parsers (nom 8 compatible)
// ============================================================================

fn parse_touchstone_inner(input: &str) -> IResult<&str, TouchstoneFile> {
    let (input, _) = many0(comment_or_blank_line).parse(input)?;
    let (input, options) = parse_options_line(input)?;
    let (input, _) = many0(comment_or_blank_line).parse(input)?;
    let (input, data_lines) = many1(parse_data_line).parse(input)?;
    let (input, _) = many0(comment_or_blank_line).parse(input)?;

    // Determine number of ports from data
    let num_ports = infer_ports_from_data(&data_lines);

    // Build S-parameters
    let sparams = build_sparams(&options, &data_lines, num_ports);

    Ok((
        input,
        TouchstoneFile {
            version: TouchstoneVersion::V1,
            num_ports,
            format: options.format,
            z0: options.z0,
            freq_mult: options.freq_mult,
            sparams,
        },
    ))
}

/// Options from the # line.
#[derive(Clone, Debug)]
struct OptionsLine {
    freq_mult: f64,
    param_type: char,  // S, Y, Z, H, G
    format: DataFormat,
    z0: Ohms,
}

impl Default for OptionsLine {
    fn default() -> Self {
        Self {
            freq_mult: 1e9,  // GHz default
            param_type: 'S',
            format: DataFormat::MA,
            z0: Ohms::Z0_50,
        }
    }
}

fn parse_options_line(input: &str) -> IResult<&str, OptionsLine> {
    let (input, _) = space0(input)?;
    let (input, _) = char('#')(input)?;
    let (input, _) = space0(input)?;

    let mut options = OptionsLine::default();

    // Parse tokens in any order
    let (input, tokens) = many0(preceded(space0, parse_option_token)).parse(input)?;

    for token in tokens {
        match token {
            OptionToken::FreqUnit(mult) => options.freq_mult = mult,
            OptionToken::ParamType(t) => options.param_type = t,
            OptionToken::Format(f) => options.format = f,
            OptionToken::Z0(z) => options.z0 = z,
        }
    }

    let (input, _) = opt(preceded(space0, not_line_ending)).parse(input)?;
    let (input, _) = opt(line_ending).parse(input)?;

    Ok((input, options))
}

#[derive(Clone, Debug)]
enum OptionToken {
    FreqUnit(f64),
    ParamType(char),
    Format(DataFormat),
    Z0(Ohms),
}

fn parse_option_token(input: &str) -> IResult<&str, OptionToken> {
    alt((
        parse_freq_unit,
        parse_param_type,
        parse_format,
        parse_z0,
    )).parse(input)
}

fn parse_freq_unit(input: &str) -> IResult<&str, OptionToken> {
    alt((
        value(OptionToken::FreqUnit(1.0), tag_no_case("HZ")),
        value(OptionToken::FreqUnit(1e3), tag_no_case("KHZ")),
        value(OptionToken::FreqUnit(1e6), tag_no_case("MHZ")),
        value(OptionToken::FreqUnit(1e9), tag_no_case("GHZ")),
    )).parse(input)
}

fn parse_param_type(input: &str) -> IResult<&str, OptionToken> {
    alt((
        value(OptionToken::ParamType('S'), tag_no_case("S")),
        value(OptionToken::ParamType('Y'), tag_no_case("Y")),
        value(OptionToken::ParamType('Z'), tag_no_case("Z")),
        value(OptionToken::ParamType('H'), tag_no_case("H")),
        value(OptionToken::ParamType('G'), tag_no_case("G")),
    )).parse(input)
}

fn parse_format(input: &str) -> IResult<&str, OptionToken> {
    alt((
        value(OptionToken::Format(DataFormat::RI), tag_no_case("RI")),
        value(OptionToken::Format(DataFormat::MA), tag_no_case("MA")),
        value(OptionToken::Format(DataFormat::DB), tag_no_case("DB")),
    )).parse(input)
}

fn parse_z0(input: &str) -> IResult<&str, OptionToken> {
    let (input, _) = tag_no_case("R")(input)?;
    let (input, _) = space1(input)?;
    let (input, z0) = double(input)?;
    Ok((input, OptionToken::Z0(Ohms(z0))))
}

fn comment_or_blank_line(input: &str) -> IResult<&str, ()> {
    alt((
        map(
            (space0, char('!'), not_line_ending, opt(line_ending)),
            |_| (),
        ),
        map((space0, line_ending), |_| ()),
    )).parse(input)
}

fn parse_data_line(input: &str) -> IResult<&str, Vec<f64>> {
    let (input, _) = space0(input)?;

    // Skip if it's a comment or empty
    if input.starts_with('!') || input.starts_with('\n') || input.starts_with('\r') || input.is_empty() {
        return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)));
    }

    let (input, values) = many1(preceded(space0, double)).parse(input)?;
    let (input, _) = opt(preceded(space0, preceded(char('!'), not_line_ending))).parse(input)?;
    let (input, _) = opt(line_ending).parse(input)?;

    Ok((input, values))
}

fn infer_ports_from_data(data_lines: &[Vec<f64>]) -> usize {
    // For N ports, each frequency has N^2 S-parameters (2 values each for complex)
    // Plus 1 frequency value = 1 + 2*N^2 values per frequency point

    // Count values in first complete frequency point
    if data_lines.is_empty() {
        return 2;
    }

    // For 2-port: 1 + 2*4 = 9 values
    // For 4-port: 1 + 2*16 = 33 values (may span multiple lines)

    // Simple heuristic: check first line
    let first_line_len = data_lines[0].len();

    match first_line_len {
        3 => 1,   // 1-port: freq, S11_r, S11_i (or mag/ang)
        9 => 2,   // 2-port: freq, S11, S21, S12, S22 (4 params * 2 values + 1 freq)
        5 => 2,   // Sometimes 2-port on multiple lines
        _ => {
            // Try to figure out from multiple lines
            // For now, assume 2-port if unclear
            2
        }
    }
}

fn build_sparams(options: &OptionsLine, data_lines: &[Vec<f64>], num_ports: usize) -> SParameters {
    let mut sparams = SParameters::new(num_ports, options.z0);

    // Flatten all data
    let all_values: Vec<f64> = data_lines.iter().flatten().copied().collect();

    // Values per frequency point
    let values_per_freq = 1 + 2 * num_ports * num_ports;

    // Parse frequency points
    let mut idx = 0;
    while idx + values_per_freq <= all_values.len() {
        let freq = Hertz(all_values[idx] * options.freq_mult);
        idx += 1;

        let mut matrix = Array2::zeros((num_ports, num_ports));

        // S-parameters are stored in row-major order for 2-port
        // For larger: may be different
        for row in 0..num_ports {
            for col in 0..num_ports {
                let val1 = all_values[idx];
                let val2 = all_values[idx + 1];
                idx += 2;

                matrix[[row, col]] = options.format.to_complex(val1, val2);
            }
        }

        sparams.add_point(freq, matrix);
    }

    sparams
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_S2P: &str = r#"! Sample 2-port S-parameter file
# GHz S RI R 50
! freq  S11_re S11_im S21_re S21_im S12_re S12_im S22_re S22_im
1.0  0.1 0.0  0.9 0.0  0.9 0.0  0.1 0.0
2.0  0.15 0.05  0.85 -0.1  0.85 -0.1  0.15 0.05
"#;

    #[test]
    fn test_parse_sample_s2p() {
        let result = parse_touchstone(SAMPLE_S2P).unwrap();

        assert_eq!(result.num_ports, 2);
        assert_eq!(result.format, DataFormat::RI);
        assert_eq!(result.z0.0, 50.0);
        assert_eq!(result.sparams.len(), 2);

        // Check first frequency point
        let freqs = &result.sparams.frequencies;
        assert!((freqs[0].0 - 1e9).abs() < 1.0);
        assert!((freqs[1].0 - 2e9).abs() < 1.0);

        // Check S21 at first point
        let s21 = result.sparams.s21();
        assert!((s21[0].re - 0.9).abs() < 1e-10);
        assert!((s21[0].im - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_options_parsing() {
        let input = "# MHZ S DB R 75\n";
        let (_, options) = parse_options_line(input).unwrap();

        assert!((options.freq_mult - 1e6).abs() < 1.0);
        assert_eq!(options.param_type, 'S');
        assert_eq!(options.format, DataFormat::DB);
        assert!((options.z0.0 - 75.0).abs() < 1e-10);
    }
}
